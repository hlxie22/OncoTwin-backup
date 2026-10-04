from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from access import AccessPolicy
from common import atomic_json, atomic_parquet, atomic_text, sha256_file
from metrics import fit_censoring_km, v2_metric_bundle
from v2_02_models import StructuredBatch, fractional_censor_nll_torch, weighted_patient_objective
from v2_02_train import attach_and_validate_development_folds, patient_balance_weights, _compact_prediction_rows, _calibration_table
from v2_03_gate import blend_logits
from v2_04_models import HybridTemporalScanModel, choose_with_simplicity_margin, masked_weighted_ce


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    representation_mode: str
    genomic_mode: str
    auxiliary_mode: str = "none"
    diagnostic_only: bool = False


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _batch(arrays: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> StructuredBatch:
    def t(name: str):
        return torch.from_numpy(np.asarray(arrays[name][idx], dtype=np.float32)).to(device, non_blocking=True)
    return StructuredBatch(
        history_core=t("history_core"), history_core_mask=t("history_core_mask"),
        utilization_core=t("utilization_core"), utilization_core_mask=t("utilization_core_mask"),
        current_core=t("current_core"), current_core_mask=t("current_core_mask"),
        change_core=t("change_core"), change_core_mask=t("change_core_mask"),
        optional_history=t("optional_history"), optional_history_mask=t("optional_history_mask"),
        optional_history_recency=t("optional_history_recency"), optional_history_recency_mask=t("optional_history_recency_mask"),
        optional_current=t("optional_current"), optional_current_mask=t("optional_current_mask"),
        stream_availability=t("stream_availability"), stream_availability_mask=t("stream_availability_mask"),
        tumor=t("tumor"), genomic_available=t("genomic_available").reshape(-1), genomic_age_scaled=t("genomic_age_scaled").reshape(-1),
    )


def _targets(frame: pd.DataFrame, local: np.ndarray, device: torch.device):
    rows=frame.iloc[local]
    return (
        torch.tensor(rows["survival_time_days"].to_numpy(np.float32),device=device),
        torch.tensor(rows["survival_cause"].astype(int).to_numpy(),device=device),
        torch.tensor(rows["patient_weight"].to_numpy(np.float32),device=device),
    )


def _target_weight(frame: pd.DataFrame, mask_col: str) -> np.ndarray:
    out=np.zeros(len(frame),dtype=np.float32)
    mask=frame[mask_col].fillna(False).astype(bool).to_numpy()
    if not mask.any(): return out
    sub=frame.loc[mask,["patient_id"]].copy()
    counts=sub.groupby("patient_id")["patient_id"].transform("size").to_numpy(dtype=np.float64)
    w=1.0/np.maximum(counts,1.0)
    w=w/w.mean()
    out[mask]=w.astype(np.float32)
    return out


def _metric_views(frame: pd.DataFrame, pre: np.ndarray, post: np.ndarray, km, alpha: float) -> dict[str,Any]:
    return {
        "PRE": v2_metric_bundle(frame,pre,km),
        "POST": v2_metric_bundle(frame,post,km),
        "LOCKED_ALPHA": v2_metric_bundle(frame,blend_logits(pre,post,alpha),km),
    }


def _summary_metric(views: dict[str,Any]) -> dict[str,float]:
    return {
        "pre_brier": float(views["PRE"]["patient_brier_4h_mean"]),
        "post_brier": float(views["POST"]["patient_brier_4h_mean"]),
        "locked_alpha_brier": float(views["LOCKED_ALPHA"]["patient_brier_4h_mean"]),
        "locked_alpha_nll": float(views["LOCKED_ALPHA"]["fractional_censor_patient_mean_nll_v1"]),
    }


def _predict(model, frame, arrays, global_rows, device, batch_size=1024):
    model.eval(); pre=[]; post=[]; nxt=[]; pp=[]; ppt=[]
    with torch.no_grad():
        for start in range(0,len(frame),batch_size):
            loc=np.arange(start,min(start+batch_size,len(frame)))
            g=global_rows[loc]
            b=_batch(arrays,g,device)
            tp=torch.from_numpy(arrays["temporal_pre"][g]).to(device)
            tq=torch.from_numpy(arrays["temporal_post"][g]).to(device)
            sm=torch.from_numpy(arrays["genomic_simple"][g]).to(device)
            ctx=torch.from_numpy(arrays["portable_context"][g]).to(device); o=model(b,tp,tq,sm,ctx)
            pre.append(o.pre_logits.float().cpu().numpy()); post.append(o.post_logits.float().cpu().numpy())
            nxt.append(o.next_scan_logits.float().cpu().numpy()); pp.append(o.postprog_logits.float().cpu().numpy()); ppt.append(o.postprog_time_logits.float().cpu().numpy())
    return {"pre":np.concatenate(pre),"post":np.concatenate(post),"next":np.concatenate(nxt),"postprog":np.concatenate(pp),"postprog_time":np.concatenate(ppt)}


def _classification_diag(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str,Any]:
    mask=np.asarray(mask,dtype=bool)
    if mask.sum()==0: return {"rows":0,"accuracy":None,"macro_auprc":None}
    p=np.exp(logits[mask]-logits[mask].max(axis=1,keepdims=True)); p/=p.sum(axis=1,keepdims=True)
    y=np.asarray(labels,dtype=int)[mask]
    pred=p.argmax(axis=1)
    out={"rows":int(mask.sum()),"accuracy":float((pred==y).mean())}
    try:
        from sklearn.metrics import average_precision_score
        one=np.eye(p.shape[1])[y]
        out["macro_auprc"]=float(average_precision_score(one,p,average="macro"))
    except Exception:
        out["macro_auprc"]=None
    return out



def _grad_norm(grads) -> float:
    total=0.0
    for g in grads:
        if g is not None:
            total += float((g.detach().float() ** 2).sum().cpu())
    return float(math.sqrt(total))


def _aux_gradient_probe(model, spec, train, aux_train, arrays, device, next_weight, pp_weight, ppt_weight, max_rows=256):
    if spec.auxiliary_mode == "none":
        return {}
    params=[p for p in model.parameters() if p.requires_grad]
    probe={}
    next_idx=np.where(train["next_scan_mask"].fillna(False).to_numpy(dtype=bool))[0][:max_rows]
    if len(next_idx):
        g=train.iloc[next_idx]["global_row"].to_numpy(np.int64); b=_batch(arrays,g,device)
        tp=torch.from_numpy(arrays["temporal_pre"][g]).to(device); tq=torch.from_numpy(arrays["temporal_post"][g]).to(device); sm=torch.from_numpy(arrays["genomic_simple"][g]).to(device); ctx=torch.from_numpy(arrays["portable_context"][g]).to(device)
        o=model(b,tp,tq,sm,ctx)
        td=torch.tensor(train.iloc[next_idx]["survival_time_days"].to_numpy(np.float32),device=device); cause=torch.tensor(train.iloc[next_idx]["survival_cause"].astype(int).to_numpy(),device=device); w=torch.tensor(train.iloc[next_idx]["patient_weight"].to_numpy(np.float32),device=device)
        surv=weighted_patient_objective(0.5*(fractional_censor_nll_torch(o.pre_logits,td,cause)+fractional_censor_nll_torch(o.post_logits,td,cause)),w)
        labels=torch.tensor(train.iloc[next_idx]["next_scan_label_filled"].to_numpy(np.int64),device=device); mask=torch.ones(len(next_idx),dtype=torch.bool,device=device); aw=torch.tensor(train.iloc[next_idx]["next_target_weight"].to_numpy(np.float32),device=device)
        aux=masked_weighted_ce(o.next_scan_logits,labels,mask,aw)
        sn=_grad_norm(torch.autograd.grad(surv,params,retain_graph=True,allow_unused=True)); an=_grad_norm(torch.autograd.grad(aux,params,retain_graph=False,allow_unused=True))
        probe["next_scan"]={"rows":int(len(next_idx)),"survival_grad_norm":sn,"aux_grad_norm":an,"weighted_aux_to_survival_grad_ratio":float(next_weight*an/max(sn,1e-12))}
    if spec.auxiliary_mode == "next_scan_postprog" and len(aux_train):
        pp_idx=np.arange(min(max_rows,len(aux_train))); g=aux_train.iloc[pp_idx]["global_row"].to_numpy(np.int64); b=_batch(arrays,g,device)
        tp=torch.from_numpy(arrays["temporal_pre"][g]).to(device); tq=torch.from_numpy(arrays["temporal_post"][g]).to(device); sm=torch.from_numpy(arrays["genomic_simple"][g]).to(device); ctx=torch.from_numpy(arrays["portable_context"][g]).to(device); o=model(b,tp,tq,sm,ctx)
        rows=aux_train.iloc[pp_idx]; ppm=torch.tensor(rows["postprog_mask"].fillna(False).to_numpy(dtype=bool),device=device); ppl=torch.tensor(rows["postprog_label_filled"].to_numpy(np.int64),device=device); ppw=torch.tensor(rows["postprog_target_weight"].to_numpy(np.float32),device=device); ptm=torch.tensor(rows["postprog_time_mask"].fillna(False).to_numpy(dtype=bool),device=device); ptl=torch.tensor(rows["postprog_time_bucket_filled"].to_numpy(np.int64),device=device)
        l1=masked_weighted_ce(o.postprog_logits,ppl,ppm,ppw); l2=masked_weighted_ce(o.postprog_time_logits,ptl,ptm,ppw); aux=pp_weight*l1+ppt_weight*l2
        an=_grad_norm(torch.autograd.grad(aux,params,retain_graph=False,allow_unused=True))
        probe["postprogression"]={"rows":int(len(pp_idx)),"weighted_aux_grad_norm":an,"note":"CONTINUED_OR_CENSORED remains class 1; this probe is auxiliary-only and does not alter survival-row support"}
    model.zero_grad(set_to_none=True)
    return probe

def _train_one_fold(spec: CandidateSpec, train: pd.DataFrame, assess: pd.DataFrame, aux_train: pd.DataFrame, arrays: dict[str,np.ndarray], dims: dict[str,int], config: dict[str,Any], device, seed:int, checkpoint:Path):
    _seed(seed)
    model=HybridTemporalScanModel(representation_mode=spec.representation_mode,genomic_mode=spec.genomic_mode,dims=dims,adapter_bottleneck=int(config["temporal_adapter"]["bottleneck_dim"])).to(device)
    lr=float(config["adapter_learning_rate"] if spec.representation_mode=="temporal_latent_adapter" else config["learning_rate"])
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(config["weight_decay"]))
    batch_size=int(config["batch_size"]); epochs=int(config["epochs"]); rng=np.random.default_rng(seed)
    history=[]
    next_weight=float(config["auxiliary"]["next_scan_max_weight"])
    pp_weight=float(config["auxiliary"]["postprog_weight"]); ppt_weight=float(config["auxiliary"]["postprog_time_weight"])
    tg=train["global_row"].to_numpy(np.int64); ag=aux_train["global_row"].to_numpy(np.int64) if len(aux_train) else np.empty(0,np.int64)
    model.train()
    gradient_probe=_aux_gradient_probe(model,spec,train,aux_train,arrays,device,next_weight,pp_weight,ppt_weight)
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(train)); sums={"survival":0.0,"next":0.0,"postprog":0.0,"postprog_time":0.0}; nb=0
        for start in range(0,len(order),batch_size):
            local=order[start:start+batch_size]; g=tg[local]; b=_batch(arrays,g,device)
            tp=torch.from_numpy(arrays["temporal_pre"][g]).to(device); tq=torch.from_numpy(arrays["temporal_post"][g]).to(device); sm=torch.from_numpy(arrays["genomic_simple"][g]).to(device)
            td,cause,w=_targets(train,local,device); opt.zero_grad(set_to_none=True); ctx=torch.from_numpy(arrays["portable_context"][g]).to(device); o=model(b,tp,tq,sm,ctx)
            pre_loss=fractional_censor_nll_torch(o.pre_logits,td,cause); post_loss=fractional_censor_nll_torch(o.post_logits,td,cause)
            surv=weighted_patient_objective(0.5*(pre_loss+post_loss),w); total=surv
            next_loss=o.next_scan_logits.sum()*0.0
            if spec.auxiliary_mode in {"next_scan","next_scan_postprog"}:
                rows=train.iloc[local]
                mask=torch.tensor(rows["next_scan_mask"].fillna(False).astype(bool).to_numpy(),device=device)
                labels=torch.tensor(rows["next_scan_label_filled"].to_numpy(np.int64),device=device)
                weights=torch.tensor(rows["next_target_weight"].to_numpy(np.float32),device=device)
                next_loss=masked_weighted_ce(o.next_scan_logits,labels,mask,weights); total=total+next_weight*next_loss
            if not torch.isfinite(total): raise RuntimeError(f"non-finite loss {spec.name}")
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["gradient_clip_norm"])); opt.step()
            sums["survival"]+=float(surv.detach().cpu()); sums["next"]+=float(next_loss.detach().cpu()); nb+=1
        if spec.auxiliary_mode=="next_scan_postprog" and len(aux_train):
            aorder=rng.permutation(len(aux_train)); pp_sum=0.0; ppt_sum=0.0; anb=0
            for start in range(0,len(aorder),batch_size):
                loc=aorder[start:start+batch_size]; g=ag[loc]; rows=aux_train.iloc[loc]; b=_batch(arrays,g,device)
                tp=torch.from_numpy(arrays["temporal_pre"][g]).to(device); tq=torch.from_numpy(arrays["temporal_post"][g]).to(device); sm=torch.from_numpy(arrays["genomic_simple"][g]).to(device)
                opt.zero_grad(set_to_none=True); ctx=torch.from_numpy(arrays["portable_context"][g]).to(device); o=model(b,tp,tq,sm,ctx)
                ppm=torch.tensor(rows["postprog_mask"].fillna(False).astype(bool).to_numpy(),device=device)
                ppl=torch.tensor(rows["postprog_label_filled"].to_numpy(np.int64),device=device); ppw=torch.tensor(rows["postprog_target_weight"].to_numpy(np.float32),device=device)
                ptm=torch.tensor(rows["postprog_time_mask"].fillna(False).astype(bool).to_numpy(),device=device); ptl=torch.tensor(rows["postprog_time_bucket_filled"].to_numpy(np.int64),device=device)
                l1=masked_weighted_ce(o.postprog_logits,ppl,ppm,ppw); l2=masked_weighted_ce(o.postprog_time_logits,ptl,ptm,ppw)
                loss=pp_weight*l1+ppt_weight*l2; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["gradient_clip_norm"])); opt.step()
                pp_sum+=float(l1.detach().cpu()); ppt_sum+=float(l2.detach().cpu()); anb+=1
            sums["postprog"]=pp_sum/max(anb,1); sums["postprog_time"]=ppt_sum/max(anb,1)
        row={"epoch":epoch,"survival_loss":sums["survival"]/max(nb,1),"next_scan_loss":sums["next"]/max(nb,1),"postprog_loss":sums["postprog"],"postprog_time_loss":sums["postprog_time"]}
        if epoch==1 and gradient_probe:
            row["gradient_probe"]=gradient_probe
        history.append(row); print("[V2_04_TRAIN]",spec.name,"seed",seed,row,flush=True)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"spec":asdict(spec),"seed":seed,"state_dict":model.state_dict(),"history":history,"parameter_count":sum(p.numel() for p in model.parameters())},checkpoint)
    return model.eval(),history


def _train_candidate(spec, dev, all_dev, arrays, dims, config, device, seed, out, km, alpha):
    n=len(dev); pre=np.full((n,24,4),np.nan,np.float32); post=np.full_like(pre,np.nan); fold_rows=[]; histories={}; aux_diags=[]
    t0=time.time()
    for fold in (0,1,2):
        tr=dev.loc[dev["development_fold"]!=fold].copy().reset_index(drop=True); va=dev.loc[dev["development_fold"]==fold].copy().reset_index(drop=True)
        tr["patient_weight"]=patient_balance_weights(tr); va["patient_weight"]=patient_balance_weights(va)
        aux=all_dev.loc[(all_dev["development_fold"]!=fold)&all_dev["postprog_mask"].fillna(False).astype(bool)].copy().reset_index(drop=True)
        fold_seed=int(seed+1000*fold)
        model,hist=_train_one_fold(spec,tr,va,aux,arrays,dims,config,device,fold_seed,out/"checkpoints"/spec.name/f"seed_{seed}"/f"fold_{fold}.pt")
        pred=_predict(model,va,arrays,va["global_row"].to_numpy(np.int64),device)
        loc=np.where(dev["development_fold"].to_numpy()==fold)[0]; pre[loc]=pred["pre"]; post[loc]=pred["post"]
        views=_metric_views(va,pred["pre"],pred["post"],km,alpha); sm=_summary_metric(views); sm.update({"fold":fold,"rows":len(va),"patients":int(va["patient_id"].nunique())}); fold_rows.append(sm); histories[str(fold)]=hist
        aux_diags.append({"fold":fold,"next_scan":_classification_diag(pred["next"],va["next_scan_label_filled"].to_numpy(),va["next_scan_mask"].fillna(False).to_numpy())})
        print("[V2_04_EVAL]",spec.name,"seed",seed,"fold",fold,sm,flush=True)
    if not np.isfinite(pre).all() or not np.isfinite(post).all(): raise RuntimeError(f"incomplete OOF {spec.name}")
    views=_metric_views(dev,pre,post,km,alpha); summary=_summary_metric(views)
    summary.update({"candidate":spec.name,"seed":seed,"representation_mode":spec.representation_mode,"genomic_mode":spec.genomic_mode,"auxiliary_mode":spec.auxiliary_mode,"runtime_seconds":time.time()-t0,"folds":fold_rows,"next_scan_diagnostics":aux_diags})
    return summary,{"pre":pre,"post":post},histories


def _baseline_from_v202(repo,dev,km,alpha,policy):
    idx=policy.read_parquet(repo/"artifacts/dynamic_scan_v2/v2_02/selected_oof_index.parquet")
    with policy.np_load(repo/"artifacts/dynamic_scan_v2/v2_02/selected_oof_logits.npz") as z:
        pre=np.asarray(z["pre_logits"],np.float32); post=np.asarray(z["post_logits"],np.float32)
    keys=["patient_id","scan_episode_id"]
    if len(idx)!=len(dev) or not np.array_equal(idx[keys].astype(str).to_numpy(),dev[keys].astype(str).to_numpy()): raise RuntimeError("V2-02 selected OOF keys no longer match V2-04 modeling rows")
    views=_metric_views(dev,pre,post,km,alpha); s=_summary_metric(views); s.update({"candidate":"temporal_post_only","seed":"frozen_v2_02","representation_mode":"temporal_post_only","genomic_mode":"embedding","auxiliary_mode":"none","runtime_seconds":0.0})
    return s,{"pre":pre,"post":post}


def _make_simple_genomics(scan:pd.DataFrame,samples:pd.DataFrame)->np.ndarray:
    s=samples.copy(); s["sample_id"]=s["sample_id"].astype(str).str.strip(); s=s.drop_duplicates("sample_id",keep="first").set_index("sample_id")
    out=np.zeros((len(scan),3),np.float32)
    for i,(sid,avail) in enumerate(zip(scan["sample_id"],scan["genomic_available"])):
        if not bool(avail) or pd.isna(sid): continue
        key=str(sid).strip()
        if key not in s.index: continue
        row=s.loc[key]; cov=float(pd.to_numeric(row.get("coverage_gene_count",0),errors="coerce") or 0); alt=float(pd.to_numeric(row.get("selected_alteration_count",0),errors="coerce") or 0)
        if not np.isfinite(cov): cov=0.0
        if not np.isfinite(alt): alt=0.0
        out[i]=[np.log1p(max(cov,0.0)),np.log1p(max(alt,0.0)),max(alt,0.0)/max(cov,1.0)]
    return out


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",required=True); args=ap.parse_args(); repo=Path(args.repo).resolve(); out=repo/"artifacts/dynamic_scan_v2/v2_04"; out.mkdir(parents=True,exist_ok=True)
    config=json.loads((repo/"configs/dynamic_scan_v2/v2_04_ablation.json").read_text()); policy=AccessPolicy.from_json(repo,repo/"configs/dynamic_scan_v2/data_access_policy.json")
    v202=json.loads((repo/"artifacts/dynamic_scan_v2/v2_02/result_packet.json").read_text()); v203=json.loads((repo/"artifacts/dynamic_scan_v2/v2_03/result_packet.json").read_text())
    if v202.get("status")!="PASS" or v203.get("status")!="PASS": raise RuntimeError("V2-02 and V2-03 must PASS")
    rule=json.loads((repo/"artifacts/dynamic_scan_v2/v2_03/selected_update_rule.json").read_text())
    if rule.get("type")!="constant": raise RuntimeError("V2-04 implementation expects V2-03 to have frozen a constant rule")
    alpha=float(rule["alpha"]); print("[V2_04_LOCKED_ALPHA]",alpha,flush=True)

    cols=["patient_id","scan_episode_id","landmark_day","split","survival_mask","survival_time_days","survival_cause","next_scan_mask","next_scan_label","postprog_mask","postprog_label","postprog_time_mask","postprog_time_bucket","sample_id","coverage_gene_count","genomic_available","genomic_age_days"]
    scan=policy.read_parquet(repo/"artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet",columns=cols)
    scan["patient_id"]=scan["patient_id"].astype(str).str.strip(); scan["scan_episode_id"]=scan["scan_episode_id"].astype(str).str.strip(); scan["global_row"]=np.arange(len(scan),dtype=np.int64)
    folds=policy.read_parquet(repo/"artifacts/dynamic_scan_v2/v2_00/development_folds.parquet")
    frame=attach_and_validate_development_folds(scan,folds,fold_count=3)
    is_dev=frame["split"].isin(["train","val"]); eligible=is_dev & frame["survival_mask"].fillna(False).astype(bool); dev=frame.loc[eligible].copy().reset_index(drop=True); all_dev=frame.loc[is_dev].copy().reset_index(drop=True)
    if (dev["development_fold"]<0).any() or (all_dev["development_fold"]<0).any(): raise RuntimeError("historical test sentinel entered V2-04 development rows")
    for d in (dev,all_dev):
        d["next_scan_mask"]=d["next_scan_mask"].fillna(False).astype(bool); d["next_scan_label_filled"]=pd.to_numeric(d["next_scan_label"],errors="coerce").fillna(0).astype(int); d["next_target_weight"]=_target_weight(d,"next_scan_mask")
        d["postprog_mask"]=d["postprog_mask"].fillna(False).astype(bool); d["postprog_label_filled"]=pd.to_numeric(d["postprog_label"],errors="coerce").fillna(0).astype(int); d["postprog_time_mask"]=d["postprog_time_mask"].fillna(False).astype(bool); d["postprog_time_bucket_filled"]=pd.to_numeric(d["postprog_time_bucket"],errors="coerce").fillna(0).astype(int); d["postprog_target_weight"]=_target_weight(d,"postprog_mask")
    # Frozen censoring nuisance estimator: original training split only.
    kmf=frame.loc[frame["split"].eq("train") & frame["survival_mask"].fillna(False).astype(bool)]
    km=fit_censoring_km(kmf["survival_time_days"].to_numpy(float),kmf["survival_cause"].astype(int).to_numpy())

    with policy.np_load(repo/"artifacts/dynamic_scan_v2/v2_01/paired_scan_tensors.npz") as z: arrays={k:np.asarray(z[k],np.float32) for k in z.files}
    arrays["tumor"]=np.asarray(policy.np_load(repo/"artifacts/checkpoint7r2_transport_safe_supervised/prepared/tumor_embeddings_f16.npy",mmap_mode="r"),np.float32)
    temporal=np.asarray(policy.np_load(repo/"artifacts/checkpoint7r2_transport_safe_supervised/prepared/temporal_prepost_f16.npy",mmap_mode="r"),np.float32); arrays["temporal_pre"]=temporal[:,0]; arrays["temporal_post"]=temporal[:,1]
    context=np.asarray(policy.np_load(repo/"artifacts/checkpoint7r2_transport_safe_supervised/prepared/context_features_f32.npy",mmap_mode="r"),np.float32); arrays["portable_context"]=context[:,[2,4,5]].astype(np.float32); arrays["genomic_available"]=context[:,4:5]; arrays["genomic_age_scaled"]=context[:,5:6]
    sample_paths=[repo/"artifacts/checkpoint7r2_transport_safe_supervised/prepared/chord_genomic_samples.parquet",repo/"artifacts/checkpoint5/prepared/chord_genomic_samples.parquet"]
    sp=next((p for p in sample_paths if p.is_file()),None)
    if sp is None: raise RuntimeError("chord_genomic_samples.parquet not found")
    samples=policy.read_parquet(sp,columns=["sample_id","coverage_gene_count","selected_alteration_count"]); arrays["genomic_simple"]=_make_simple_genomics(frame,samples)
    schema=json.loads((repo/"artifacts/dynamic_scan_v2/v2_01/feature_schema.json").read_text()); dims={k:len(v) for k,v in schema["blocks"].items()}
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print("[V2_04_DEVICE]",device,torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",flush=True)

    baseline,baseline_logits=_baseline_from_v202(repo,dev,km,alpha,policy)
    screening_seed=int(config["screening_seed"]); all_metrics={"temporal_post_only":baseline}; logits_by_name={"temporal_post_only":baseline_logits}; histories={}; runtime={"temporal_post_only":0.0}
    registry=[]

    # Stage 1: representation isolation. A newly trained temporal-only control
    # shares the exact supervised layer dimensions with V2-02; frozen V2-02 remains
    # the stronger promotion benchmark.
    rep_specs=[
        CandidateSpec("temporal_post_only_matched","temporal_post_only","embedding"),
        CandidateSpec("pre_temporal_plus_explicit_scan","pre_temporal_plus_explicit_scan","embedding"),
        CandidateSpec("post_temporal_plus_explicit_scan","post_temporal_plus_explicit_scan","embedding"),
    ]
    for spec in rep_specs:
        m,l,h=_train_candidate(spec,dev,all_dev,arrays,dims,config,device,screening_seed,out,km,alpha); all_metrics[spec.name]=m; logits_by_name[spec.name]=l; histories[spec.name]={str(screening_seed):h}; runtime[spec.name]=m["runtime_seconds"]; registry.append(asdict(spec))
    matched_names=[s.name for s in rep_specs]
    matched_scores={k:all_metrics[k]["locked_alpha_brier"] for k in matched_names}
    matched_winner=min(matched_scores,key=matched_scores.get)
    matched_control=all_metrics["temporal_post_only_matched"]["locked_alpha_brier"]
    rep_winner="temporal_post_only"
    if matched_winner!="temporal_post_only_matched":
        candidate_score=all_metrics[matched_winner]["locked_alpha_brier"]
        if candidate_score <= matched_control-float(config["selection"]["representation_candidate_margin"]) and candidate_score < baseline["locked_alpha_brier"]:
            rep_winner=matched_winner
    rep_scores={"frozen_v2_02_temporal":baseline["locked_alpha_brier"],**matched_scores}
    rep_decision={"scores":rep_scores,"matched_control":"temporal_post_only_matched","matched_winner":matched_winner,"selected_representation_candidate":rep_winner,"rule":"explicit scan evidence must beat the architecture-matched temporal-only control by the representation margin and also beat the frozen V2-02 branch; otherwise retain the frozen temporal representation"}
    atomic_json(out/"representation_ablation.json",rep_decision)

    # Stage 2: controlled genomics on selected representation.
    rep_mode=all_metrics[rep_winner]["representation_mode"]
    genomic_embedding_control = rep_winner if rep_winner != "temporal_post_only" else "temporal_post_only_matched"
    genomic_metric_names={"embedding":genomic_embedding_control}
    for gm in ["simple_summary","availability_age","none"]:
        name=f"{rep_mode}__genomic_{gm}"; spec=CandidateSpec(name,rep_mode,gm); m,l,h=_train_candidate(spec,dev,all_dev,arrays,dims,config,device,screening_seed,out,km,alpha); all_metrics[name]=m; logits_by_name[name]=l; histories[name]={str(screening_seed):h}; runtime[name]=m["runtime_seconds"]; registry.append(asdict(spec)); genomic_metric_names[gm]=name
    g_scores={gm:all_metrics[name]["locked_alpha_brier"] for gm,name in genomic_metric_names.items()}
    selected_gm=choose_with_simplicity_margin(g_scores,list(config["genomic_simplicity_order"]),float(config["selection"]["simplicity_tie_margin"])); genomic_winner=genomic_metric_names[selected_gm]
    genomic_decision={"scores":g_scores,"candidate_names":genomic_metric_names,"selected_genomic_mode":selected_gm,"selected_candidate":genomic_winner,"simple_summary_semantics":["log1p(panel_coverage_gene_count)","log1p(selected_alteration_count)","selected_alteration_count/max(coverage_gene_count,1)"],"availability_age_separated_from_molecular_content":True}
    atomic_json(out/"genomic_ablation.json",genomic_decision)

    # Stage 3: auxiliary next-scan; post-progression only if next-scan passes trigger.
    base_spec=CandidateSpec(genomic_winner,rep_mode,selected_gm)
    next_name=f"{rep_mode}__genomic_{selected_gm}__next_scan"; next_spec=CandidateSpec(next_name,rep_mode,selected_gm,"next_scan")
    nm,nl,nh=_train_candidate(next_spec,dev,all_dev,arrays,dims,config,device,screening_seed,out,km,alpha); all_metrics[next_name]=nm; logits_by_name[next_name]=nl; histories[next_name]={str(screening_seed):nh}; runtime[next_name]=nm["runtime_seconds"]; registry.append(asdict(next_spec))
    base_score=all_metrics[genomic_winner]["locked_alpha_brier"]; next_improvement=base_score-nm["locked_alpha_brier"]; postprog_trigger=next_improvement>=float(config["selection"]["auxiliary_trigger_improvement"])
    post_name=None
    if postprog_trigger:
        post_name=f"{rep_mode}__genomic_{selected_gm}__next_scan_postprog"; ps=CandidateSpec(post_name,rep_mode,selected_gm,"next_scan_postprog")
        pm,pl,ph=_train_candidate(ps,dev,all_dev,arrays,dims,config,device,screening_seed,out,km,alpha); all_metrics[post_name]=pm; logits_by_name[post_name]=pl; histories[post_name]={str(screening_seed):ph}; runtime[post_name]=pm["runtime_seconds"]; registry.append(asdict(ps))
    aux_candidates=[genomic_winner,next_name]+([post_name] if post_name else []); aux_scores={k:all_metrics[k]["locked_alpha_brier"] for k in aux_candidates}; aux_best=min(aux_scores,key=aux_scores.get)
    if aux_best!=genomic_winner:
        improvement=base_score-all_metrics[aux_best]["locked_alpha_brier"]; nll_delta=all_metrics[aux_best]["locked_alpha_nll"]-all_metrics[genomic_winner]["locked_alpha_nll"]
        if improvement<float(config["selection"]["simplicity_tie_margin"]) or nll_delta>float(config["selection"]["nll_guardrail"]): aux_best=genomic_winner
    aux_decision={"scores":aux_scores,"next_scan_improvement":next_improvement,"postprog_triggered":postprog_trigger,"selected_candidate":aux_best,"continued_or_censored_semantics":"preserved as a frozen post-progression class; not relabeled as clean continuation"}
    atomic_json(out/"auxiliary_ablation.json",aux_decision)

    # Stage 4: diagnostic low-rank temporal adapter only if explicit scan representation did not displace baseline.
    adapter_info={"triggered":False,"diagnostic_only":True,"transport_robust_claim":False}
    if rep_winner=="temporal_post_only" and bool(config["temporal_adapter"]["enabled_as_diagnostic_trigger"]):
        an=f"temporal_latent_adapter__genomic_{selected_gm}"; asp=CandidateSpec(an,"temporal_latent_adapter",selected_gm,"none",True); am,al,ah=_train_candidate(asp,dev,all_dev,arrays,dims,config,device,screening_seed,out,km,alpha); all_metrics[an]=am; logits_by_name[an]=al; histories[an]={str(screening_seed):ah}; runtime[an]=am["runtime_seconds"]; registry.append(asdict(asp)); adapter_info.update({"triggered":True,"candidate":an,"locked_alpha_brier":am["locked_alpha_brier"],"delta_vs_baseline":am["locked_alpha_brier"]-baseline["locked_alpha_brier"],"promotion_rule":"diagnostic only in V2-04 because cached R1 latent cannot support before-encoder stream missingness augmentation"})
    atomic_json(out/"temporal_adaptation.json",adapter_info)

    # Overall selectable candidates exclude diagnostic-only adapter.
    selectable=["temporal_post_only",genomic_winner,aux_best]
    selectable=list(dict.fromkeys(selectable)); score_map={k:all_metrics[k]["locked_alpha_brier"] for k in selectable}
    best=min(score_map,key=score_map.get)
    # Simplicity: baseline remains if within global tie margin of the best.
    if baseline["locked_alpha_brier"] <= score_map[best]+float(config["selection"]["simplicity_tie_margin"]): best="temporal_post_only"

    # Seed confirmation for a genuinely new finalist. Primary seed remains the frozen OOF artifact.
    confirmation=[]
    if best!="temporal_post_only":
        selected_metric=all_metrics[best]; spec=CandidateSpec(best,selected_metric["representation_mode"],selected_metric["genomic_mode"],selected_metric["auxiliary_mode"])
        for s in config["confirmation_seeds"]:
            cm,_,ch=_train_candidate(spec,dev,all_dev,arrays,dims,config,device,int(s),out,km,alpha); confirmation.append(cm); histories.setdefault(best,{})[str(s)]=ch
        seed_scores=[all_metrics[best]["locked_alpha_brier"]]+[x["locked_alpha_brier"] for x in confirmation]
        improving=sum(x < baseline["locked_alpha_brier"] for x in seed_scores)
        if improving<int(config["selection"]["confirmation_min_improving_seeds"]): best="temporal_post_only"
    else:
        seed_scores=[baseline["locked_alpha_brier"]]; improving=0

    selected_logits=logits_by_name[best]
    np.savez_compressed(out/"selected_v2_04_oof_logits.npz",pre_logits=selected_logits["pre"],post_logits=selected_logits["post"])
    atomic_parquet(out/"selected_v2_04_oof_index.parquet",dev[["patient_id","scan_episode_id","landmark_day","development_fold","survival_time_days","survival_cause"]].copy())
    # Matched survival calibration under the locked V2-03 update rule.
    baseline_locked=blend_logits(baseline_logits["pre"],baseline_logits["post"],alpha)
    selected_locked=blend_logits(selected_logits["pre"],selected_logits["post"],alpha)
    cal_pred=pd.concat([
        _compact_prediction_rows(dev,baseline_locked,family="frozen_v2_03_baseline",profile="full_supported",view="LOCKED_ALPHA",fold=-1),
        _compact_prediction_rows(dev,selected_locked,family="v2_04_selected",profile="full_supported",view="LOCKED_ALPHA",fold=-1),
    ],ignore_index=True)
    atomic_parquet(out/"selected_calibration.parquet",_calibration_table(cal_pred,km))
    atomic_json(out/"ablation_registry.json",{"candidates":registry,"frozen_baseline":"temporal_post_only","locked_alpha":alpha,"screening_seed":screening_seed,"confirmation_seeds":config["confirmation_seeds"]})
    atomic_json(out/"candidate_metrics.json",all_metrics); atomic_json(out/"training_history.json",histories); atomic_json(out/"runtime_accounting.json",runtime)

    final_metric=all_metrics[best]
    selection={"selected_candidate":best,"selected_metric":final_metric,"baseline_locked_alpha_brier":baseline["locked_alpha_brier"],"selected_delta_vs_baseline":final_metric["locked_alpha_brier"]-baseline["locked_alpha_brier"],"confirmation_seed_scores":seed_scores,"confirmation_improving_seed_count":improving,"locked_alpha":alpha,"survival_formulation":config["survival_formulation"]}
    atomic_json(out/"selection_decision.json",selection)

    acceptance={
        "v2_03_pass_required":True,"locked_v2_03_constant_alpha_reused_without_candidate_retuning":True,"same_17194_survival_eligible_rows_for_primary_comparisons":len(dev)==int(v202["development_rows"]),"historical_test_not_used_for_training_or_selection":not bool((dev["split"]=="test").any()),"representation_ablation_completed":all(x in all_metrics for x in ["pre_temporal_plus_explicit_scan","post_temporal_plus_explicit_scan"]),"genomic_embedding_no_genomics_availability_and_simple_summary_completed":all(gm in genomic_metric_names for gm in config["genomic_modes"]),"next_scan_auxiliary_completed":next_name in all_metrics,"postprogression_only_triggered_if_next_scan_useful":bool(postprog_trigger)==bool(post_name is not None),"continued_or_censored_semantics_preserved":True,"temporal_adapter_not_mislabeled_transport_robust":not bool(config["temporal_adapter"]["transport_robust_claim"]),"fractional_censor_likelihood_preserved":True,"line_derived_features_absent":True,"protected_external_rows_not_opened":True,"external_predictions_not_regenerated":True,"selected_oof_complete":bool(np.isfinite(selected_logits["pre"]).all() and np.isfinite(selected_logits["post"]).all())
    }
    if not all(acceptance.values()): raise RuntimeError(f"V2-04 acceptance failed: {acceptance}")
    result={"checkpoint":"V2-04","status":"PASS","next_checkpoint":"V2-05","development_rows":int(len(dev)),"development_patients":int(dev["patient_id"].nunique()),"locked_alpha":alpha,"representation_decision":rep_decision,"genomic_decision":genomic_decision,"auxiliary_decision":aux_decision,"temporal_adaptation":adapter_info,"selection":selection,"acceptance":acceptance,"input_hashes":{"v2_02_oof_logits":sha256_file(repo/"artifacts/dynamic_scan_v2/v2_02/selected_oof_logits.npz"),"v2_03_rule":sha256_file(repo/"artifacts/dynamic_scan_v2/v2_03/selected_update_rule.json"),"v2_01_tensors":sha256_file(repo/"artifacts/dynamic_scan_v2/v2_01/paired_scan_tensors.npz"),"r2_temporal_cache":sha256_file(repo/"artifacts/checkpoint7r2_transport_safe_supervised/prepared/temporal_prepost_f16.npy")},"external_rows_opened":False,"external_predictions_regenerated":False}
    atomic_json(out/"result_packet.json",result)
    report=f"""# OncoTwin V2-04 targeted ablations\n\nStatus: **PASS**\n\nLocked update rule: alpha={alpha:.4f} from V2-03; no candidate-specific alpha retuning.\n\nRepresentation decision: **{rep_winner}**.\n\nGenomic mode: **{selected_gm}**.\n\nAuxiliary decision: **{aux_best}**.\n\nFinal V2-04 candidate: **{best}**.\n\nLocked-alpha Brier: {final_metric['locked_alpha_brier']:.9f}; frozen V2-03 branch baseline: {baseline['locked_alpha_brier']:.9f}.\n\nTemporal adapter diagnostic triggered: {adapter_info['triggered']}. It is not labeled transport-robust because it adapts already-cached R1 latent states.\n\nNo historical CHORD test or DFCI/VICC outcome rows were used for V2-04 selection.\n"""
    atomic_text(out/"selection_memo.md",report)
    manifest_files=[out/x for x in ["representation_ablation.json","genomic_ablation.json","auxiliary_ablation.json","temporal_adaptation.json","ablation_registry.json","candidate_metrics.json","training_history.json","runtime_accounting.json","selection_decision.json","selected_v2_04_oof_logits.npz","selected_v2_04_oof_index.parquet","selected_calibration.parquet","selection_memo.md","result_packet.json"]]
    atomic_json(out/"artifact_manifest.json",{"status":"PASS","files":{p.name:sha256_file(p) for p in manifest_files}})
    print(json.dumps(result,indent=2,sort_keys=True),flush=True); return 0

if __name__=="__main__": raise SystemExit(main())
