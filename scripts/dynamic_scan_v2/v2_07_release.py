from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch

from access import AccessPolicy
from v2_02_models import TemporalFixedSurvivalModel, fractional_censor_nll_torch, weighted_patient_objective
from v2_07_inference import OncoTwinV2Predictor, PreparedLandmarkBatch, sha256_file


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_torch(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".pt", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_text(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def patient_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("patient_id")["patient_id"].transform("size").to_numpy(dtype=np.float64)
    if (count <= 0).any():
        raise RuntimeError("Invalid patient landmark count")
    return (1.0 / count).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_hashes(repo: Path) -> dict[str, str]:
    targets = [
        "artifacts/checkpoint7r1_line_agnostic_temporal/temporal_encoder.pt",
        "artifacts/checkpoint3/genomic_encoder.pt",
        "artifacts/checkpoint3/prepared/gene_space_468.txt",
        "artifacts/checkpoint3/prepared/graph_edges.npy",
        "artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet",
        "artifacts/checkpoint7r2_transport_safe_supervised/prepared/temporal_prepost_f16.npy",
        "artifacts/checkpoint7r2_transport_safe_supervised/prepared/tumor_embeddings_f16.npy",
        "artifacts/checkpoint7r2_transport_safe_supervised/prepared/context_features_f32.npy",
        "artifacts/dynamic_scan_v2/v2_05/candidate_lock.json",
        "artifacts/dynamic_scan_v2/v2_06/candidate_bundle_lock.json",
        "artifacts/dynamic_scan_v2/v2_06/locked_external_protocol.json",
        "scripts/dynamic_scan_v2/v2_02_models.py",
    ]
    out = {}
    for rel in targets:
        p = repo / rel
        if not p.is_file():
            raise FileNotFoundError(p)
        out[rel] = sha256_file(p)
    return out


def verify_v206_lock(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((repo / "artifacts/dynamic_scan_v2/v2_06/result_packet.json").read_text())
    lock = json.loads((repo / "artifacts/dynamic_scan_v2/v2_06/candidate_bundle_lock.json").read_text())
    if result.get("status") != "PASS" or not all(result.get("acceptance", {}).values()):
        raise RuntimeError("V2-06 must PASS before V2-07")
    selected = result["selected_candidate"]
    if selected["selected_rule"] != config["selected_rule"] or selected["branch_family"] != config["branch_family"]:
        raise RuntimeError("V2-06 selected specification changed")
    if abs(float(selected["locked_alpha"]) - float(config["locked_alpha"])) > 1e-12:
        raise RuntimeError("V2-06 alpha changed")
    if lock["status"] != "SPECIFICATION_FROZEN_DEPLOY_FIT_PENDING_V2_07":
        raise RuntimeError("Unexpected V2-06 deploy boundary state")
    return {"v2_06_result": result, "v2_06_lock": lock}


def load_training_data(repo: Path, policy: AccessPolicy) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    idx = policy.read_parquet(
        repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/scan_index.parquet",
        columns=["patient_id", "scan_episode_id", "landmark_day", "split", "survival_mask", "survival_time_days", "survival_cause"],
    ).copy()
    idx["patient_id"] = idx["patient_id"].astype(str).str.strip()
    idx["scan_episode_id"] = idx["scan_episode_id"].astype(str).str.strip()
    eligible = idx["split"].astype(str).str.lower().isin(["train", "val"]) & idx["survival_mask"].fillna(False).astype(bool)
    dev = idx.loc[eligible].copy()
    dev["global_row"] = np.where(eligible.to_numpy())[0].astype(np.int64)
    dev = dev.reset_index(drop=True)
    if set(dev["split"].astype(str).str.lower()) - {"train", "val"}:
        raise RuntimeError("Non-development row entered deploy fit")
    if len(dev) != 17194 or dev["patient_id"].nunique() != 2443:
        raise RuntimeError(f"V2-07 development population drift: rows={len(dev)} patients={dev['patient_id'].nunique()}")
    dev["patient_weight"] = patient_weights(dev)

    temporal = np.asarray(policy.np_load(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/temporal_prepost_f16.npy", mmap_mode="r"), dtype=np.float32)
    tumor = np.asarray(policy.np_load(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/tumor_embeddings_f16.npy", mmap_mode="r"), dtype=np.float32)
    context = np.asarray(policy.np_load(repo / "artifacts/checkpoint7r2_transport_safe_supervised/prepared/context_features_f32.npy", mmap_mode="r"), dtype=np.float32)
    n = len(idx)
    if temporal.shape != (n, 2, 192) or tumor.shape != (n, 128) or context.shape != (n, 6):
        raise RuntimeError(f"Unexpected prepared shapes: temporal={temporal.shape} tumor={tumor.shape} context={context.shape}")
    if np.max(np.abs(context[:, [0, 1, 3]])) != 0.0:
        raise RuntimeError("Permanently neutralized line-derived context coordinates are not exact zero")
    arrays = {
        "temporal_pre": temporal[:, 0],
        "temporal_post": temporal[:, 1],
        "tumor": tumor,
        "portable_context": context[:, [2, 4, 5]].astype(np.float32),
    }
    return dev, arrays


def train_or_resume(
    out: Path,
    dev: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    hashes: dict[str, str],
    device: torch.device,
) -> tuple[TemporalFixedSurvivalModel, list[dict[str, Any]], bool]:
    model = TemporalFixedSurvivalModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["torch_learning_rate"]),
        weight_decay=float(config["torch_weight_decay"]),
    )
    state_path = out / "training_state.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    resumed = False
    config_hash = sha256_text(config)
    source_hash = sha256_text(hashes)
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_hash") != config_hash or state.get("source_hash") != source_hash:
            raise RuntimeError("Existing V2-07 training state does not match frozen config/source hashes")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        history = list(state.get("history", []))
        start_epoch = int(state["epoch"]) + 1
        resumed = True
        print(f"[V2_07_RESUME] completed_epoch={state['epoch']}", flush=True)

    rows = dev["global_row"].to_numpy(dtype=np.int64)
    epochs = int(config["torch_epochs"])
    batch_size = int(config["torch_batch_size"])
    base_seed = int(config["deploy_seed"])
    model.train()
    for epoch in range(start_epoch, epochs + 1):
        epoch_seed = base_seed + 100000 * epoch
        set_seed(epoch_seed)
        rng = np.random.default_rng(epoch_seed)
        order = rng.permutation(len(dev))
        numerator = 0.0
        denominator = 0.0
        for start in range(0, len(order), batch_size):
            local = order[start:start + batch_size]
            g = rows[local]
            tp = torch.from_numpy(arrays["temporal_pre"][g]).to(device)
            tq = torch.from_numpy(arrays["temporal_post"][g]).to(device)
            tumor = torch.from_numpy(arrays["tumor"][g]).to(device)
            ctx = torch.from_numpy(arrays["portable_context"][g]).to(device)
            time_days = torch.tensor(dev.iloc[local]["survival_time_days"].to_numpy(dtype=np.float32), device=device)
            cause = torch.tensor(dev.iloc[local]["survival_cause"].astype(int).to_numpy(), device=device)
            weight = torch.tensor(dev.iloc[local]["patient_weight"].to_numpy(dtype=np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            pre, post = model(tp, tq, tumor, ctx)
            pair = 0.5 * (fractional_censor_nll_torch(pre, time_days, cause) + fractional_censor_nll_torch(post, time_days, cause))
            loss = weighted_patient_objective(pair, weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite V2-07 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            numerator += float((pair.detach() * weight).sum().cpu())
            denominator += float(weight.sum().cpu())
        row = {"epoch": epoch, "patient_balanced_pair_fractional_nll": numerator / max(denominator, 1e-12)}
        history.append(row)
        print("[V2_07_TRAIN]", row, flush=True)
        atomic_torch(
            state_path,
            {
                "checkpoint": "V2-07-TRAINING-STATE",
                "epoch": epoch,
                "config_hash": config_hash,
                "source_hash": source_hash,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            },
        )
    return model.eval(), history, resumed


def make_fixture(out: Path, checkpoint: Path) -> dict[str, Any]:
    rng = np.random.default_rng(20261007)
    n = 23
    tp = rng.normal(size=(n, 192)).astype(np.float32)
    tq = rng.normal(size=(n, 192)).astype(np.float32)
    tumor = rng.normal(size=(n, 128)).astype(np.float32)
    scan_number = rng.uniform(0, 5, size=n).astype(np.float32)
    available = rng.integers(0, 2, size=n).astype(np.float32)
    age = rng.uniform(0, 10, size=n).astype(np.float32) * available
    current_known = np.ones(n, dtype=np.uint8)
    clock_known = np.ones(n, dtype=np.uint8)
    fixture_in = out / "synthetic_inference_fixture_input.npz"
    atomic_npz(
        fixture_in,
        temporal_pre=tp,
        temporal_post=tq,
        tumor_embedding=tumor,
        scan_number_patient_log=scan_number,
        genomic_available=available,
        genomic_age_scaled=age,
        current_scan_state_known=current_known,
        required_clock_known=clock_known,
    )
    predictor = OncoTwinV2Predictor(checkpoint, device="cpu")
    batch = PreparedLandmarkBatch(tp, tq, tumor, scan_number, available, age, current_known, clock_known)

    # Float32 GEMM kernels may use different accumulation paths for different
    # batch shapes. Require numerical invariance at the clinically exposed
    # probability/curve levels, while allowing a few ULPs at the raw-logit level.
    # Check both a full batch and two smaller chunkings, including batch_size=1.
    batch_sizes = (23, 7, 1)
    predictions = {bs: predictor.predict(batch, batch_size=bs) for bs in batch_sizes}
    one = predictions[23]
    batch_logit_delta = 0.0
    batch_probability_delta = 0.0
    batch_curve_delta = 0.0
    for bs in batch_sizes[1:]:
        other = predictions[bs]
        batch_logit_delta = max(
            batch_logit_delta,
            float(np.nanmax(np.abs(one["selected_logits"] - other["selected_logits"]))),
        )
        batch_probability_delta = max(
            batch_probability_delta,
            float(np.nanmax(np.abs(one["selected_conditional_probabilities"] - other["selected_conditional_probabilities"]))),
        )
        batch_curve_delta = max(
            batch_curve_delta,
            float(np.nanmax(np.abs(one["selected_curves"]["pfs_survival"] - other["selected_curves"]["pfs_survival"]))),
            float(np.nanmax(np.abs(one["selected_curves"]["progression_cif"] - other["selected_curves"]["progression_cif"]))),
            float(np.nanmax(np.abs(one["selected_curves"]["death_cif"] - other["selected_curves"]["death_cif"]))),
            float(np.nanmax(np.abs(one["selected_curves"]["switch_cif"] - other["selected_curves"]["switch_cif"]))),
        )
    logit_atol = 3e-6
    probability_atol = 5e-7
    curve_atol = 5e-7
    if batch_logit_delta > logit_atol or batch_probability_delta > probability_atol or batch_curve_delta > curve_atol:
        raise RuntimeError(
            "Inference batch-size numerical invariance failed: "
            f"logit={batch_logit_delta} prob={batch_probability_delta} curve={batch_curve_delta}"
        )

    # Genomic-unavailable invariance is a different property from batch-size
    # invariance. Hold batch shape fixed so this test measures only whether the
    # unavailable genomic embedding is correctly gated out.
    modified = tumor.copy()
    modified[available == 0] = rng.normal(loc=100.0, scale=20.0, size=modified[available == 0].shape)
    changed = predictor.predict(
        PreparedLandmarkBatch(tp, tq, modified, scan_number, available, age, current_known, clock_known),
        batch_size=23,
    )
    unavailable = available == 0
    genomic_delta = 0.0 if not unavailable.any() else float(np.max(np.abs(one["selected_logits"][unavailable] - changed["selected_logits"][unavailable])))
    if genomic_delta > 1e-7:
        raise RuntimeError(f"Unavailable-genomics invariance failed: {genomic_delta}")
    fixture_out = out / "synthetic_inference_fixture_expected.npz"
    atomic_npz(
        fixture_out,
        selected_logits=one["selected_logits"],
        selected_conditional_probabilities=one["selected_conditional_probabilities"],
        event_free_survival=one["selected_curves"]["event_free_survival"],
        progression_cif=one["selected_curves"]["progression_cif"],
        death_cif=one["selected_curves"]["death_cif"],
        switch_cif=one["selected_curves"]["switch_cif"],
        progression_or_death_cif=one["selected_curves"]["progression_or_death_cif"],
        pfs_survival=one["selected_curves"]["pfs_survival"],
    )
    return {
        "rows": n,
        "batch_sizes_checked": list(batch_sizes),
        "batch_size_invariance_max_abs": batch_logit_delta,
        "batch_size_invariance_logit_max_abs": batch_logit_delta,
        "batch_size_invariance_probability_max_abs": batch_probability_delta,
        "batch_size_invariance_curve_max_abs": batch_curve_delta,
        "batch_size_invariance_tolerances": {
            "logit_atol": logit_atol,
            "probability_atol": probability_atol,
            "curve_atol": curve_atol,
        },
        "unavailable_genomics_invariance_max_abs": genomic_delta,
        "unavailable_genomics_batch_size": 23,
        "input_sha256": sha256_file(fixture_in),
        "expected_sha256": sha256_file(fixture_out),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    args = p.parse_args()
    repo = Path(args.repo).resolve()
    out = repo / "artifacts/dynamic_scan_v2/v2_07"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((repo / "configs/dynamic_scan_v2/v2_07_release.json").read_text())
    if config["selected_rule"] != "v2_03_frozen_temporal_constant_alpha" or config["branch_family"] != "r1_temporal_fixed" or abs(float(config["locked_alpha"]) - 0.52) > 1e-12:
        raise RuntimeError("V2-07 frozen model specification changed")
    upstream = verify_v206_lock(repo, config)
    policy = AccessPolicy.from_json(repo, repo / "configs/dynamic_scan_v2/data_access_policy.json")
    hashes = source_hashes(repo)

    # Verify the most important V2-06 freeze targets again at the release boundary.
    frozen_targets = {x["path"]: x["sha256"] for x in upstream["v2_06_lock"]["freeze_targets"] if x.get("present")}
    for rel in (
        "artifacts/checkpoint7r1_line_agnostic_temporal/temporal_encoder.pt",
        "artifacts/checkpoint3/genomic_encoder.pt",
        "artifacts/checkpoint3/prepared/gene_space_468.txt",
        "artifacts/checkpoint3/prepared/graph_edges.npy",
        "artifacts/dynamic_scan_v2/v2_05/candidate_lock.json",
    ):
        if rel not in frozen_targets or hashes[rel] != frozen_targets[rel]:
            raise RuntimeError(f"V2-06 frozen target hash changed: {rel}")

    dev, arrays = load_training_data(repo, policy)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[V2_07_DEVICE]", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    model, history, resumed = train_or_resume(out, dev, arrays, config, hashes, device)

    # Architecture identity check against one V2-02 CV head: keys/shapes only.
    cv_path = repo / "artifacts/dynamic_scan_v2/v2_02/checkpoints/r1_temporal_fixed/fold_0/model.pt"
    if not cv_path.is_file():
        raise FileNotFoundError(cv_path)
    cv = torch.load(cv_path, map_location="cpu", weights_only=False)
    state = model.state_dict()
    cv_state = cv["model_state"]
    if list(state) != list(cv_state) or any(tuple(state[k].shape) != tuple(cv_state[k].shape) for k in state):
        raise RuntimeError("Final deploy head architecture differs from frozen V2-02 family")

    deploy_path = out / "deploy_model.pt"
    payload = {
        "checkpoint": "V2-07",
        "version": config["version"],
        "family": "r1_temporal_fixed",
        "selected_rule": config["selected_rule"],
        "locked_alpha": float(config["locked_alpha"]),
        "training_scope": "all_v2_development_survival_eligible",
        "is_crossfit_fold_head": False,
        "deploy_seed": int(config["deploy_seed"]),
        "epochs": int(config["torch_epochs"]),
        "model_dimensions": {"temporal": 192, "tumor": 128, "portable_context": 3, "months": 24, "causes": 4},
        "portable_context_names": list(config["portable_context_names"]),
        "cause_order": list(config["cause_order"]),
        "training_rows": int(len(dev)),
        "training_patients": int(dev["patient_id"].nunique()),
        "training_split_counts": {str(k): int(v) for k, v in dev["split"].astype(str).value_counts().sort_index().items()},
        "training_objective": config["training_objective"],
        "performance_claim_policy": config["performance_claim_policy"],
        "model_state": {k: v.detach().cpu() for k, v in state.items()},
        "parameter_count": int(sum(x.numel() for x in model.parameters())),
        "source_hashes": {
            k: v for k, v in hashes.items()
            if k in {
                "artifacts/checkpoint7r1_line_agnostic_temporal/temporal_encoder.pt",
                "artifacts/checkpoint3/genomic_encoder.pt",
                "artifacts/checkpoint3/prepared/gene_space_468.txt",
                "artifacts/checkpoint3/prepared/graph_edges.npy",
            }
        },
    }
    atomic_torch(deploy_path, payload)

    # Reload through public inference API; this is the release-integrity boundary.
    predictor = OncoTwinV2Predictor(deploy_path, device="cpu")
    predictor.verify_upstream_files(repo)
    fixture = make_fixture(out, deploy_path)

    training_population = {
        "rows": int(len(dev)),
        "patients": int(dev["patient_id"].nunique()),
        "split_counts": {str(k): int(v) for k, v in dev["split"].astype(str).value_counts().sort_index().items()},
        "cause_counts": {str(int(k)): int(v) for k, v in dev["survival_cause"].astype(int).value_counts().sort_index().items()},
        "historical_test_rows_used": 0,
        "historical_test_patients_used": 0,
        "external_rows_used": 0,
        "eligibility": "split in {train,val} AND survival_mask=True",
    }
    atomic_json(out / "training_population.json", training_population)
    atomic_json(out / "training_history.json", history)
    atomic_json(out / "inference_fixture_report.json", fixture)

    prepared_schema = {
        "version": "v2_07_prepared_landmark_v1",
        "boundary": config["inference_boundary"],
        "required_arrays": {
            "temporal_pre": ["N", 192],
            "temporal_post": ["N", 192],
            "tumor_embedding": ["N", 128],
            "scan_number_patient_log": ["N"],
            "genomic_available": ["N"],
            "genomic_age_scaled": ["N"],
        },
        "optional_arrays": {"current_scan_state_known": ["N"], "required_clock_known": ["N"]},
        "portable_context_order": list(config["portable_context_names"]),
        "line_derived_context_coordinates": "not accepted by V2-07 inference API",
        "raw_cohort_preflight": "V2-06 contracts govern upstream predictor construction before this boundary",
    }
    atomic_json(out / "prepared_input_schema.json", prepared_schema)

    release_contract = {
        "checkpoint": "V2-07",
        "status": "RESEARCH_RELEASE_READY_EXTERNAL_CONFIRMATION_PENDING",
        "selected_rule": config["selected_rule"],
        "alpha": float(config["locked_alpha"]),
        "cause_order": list(config["cause_order"]),
        "time_bins": 24,
        "horizons_months": [3, 6, 12, 18],
        "output_semantics": {
            "PRE": "forecast from repaired line-agnostic temporal state before current scan evidence",
            "POST": "forecast from repaired line-agnostic temporal state after current scan evidence",
            "selected": "PRE logits + 0.52*(POST logits-PRE logits)",
            "CIFs": "cause-specific cumulative incidence for progression, death, switch; progression_or_death is their progression+death sum",
            "PFS": "1 - progression CIF - death CIF; switch remains a competing event and is not counted as a PFS event",
        },
        "abstention": {
            "unknown_current_scan_state": "ABSTAIN_UNKNOWN_CURRENT_SCAN_STATE",
            "unknown_required_clock": "ABSTAIN_UNKNOWN_REQUIRED_CLOCK",
        },
        "external_confirmation": "PENDING_NEW_INDEPENDENT_COHORT",
        "v2_06_protocol": "artifacts/dynamic_scan_v2/v2_06/locked_external_protocol.json",
    }
    atomic_json(out / "deployment_contract.json", release_contract)

    model_card = f"""# OncoTwin V2 research model card\n\n## Frozen release\n\n- Checkpoint: V2-07\n- Internal specification: `r1_temporal_fixed`\n- Scan update: fixed logit interpolation alpha = 0.52\n- Final supervised head: one all-development fit on {len(dev):,} survival-eligible landmarks from {dev['patient_id'].nunique():,} patients\n- Temporal encoder: repaired CKPT7R1 line-agnostic encoder (frozen)\n- Genomic encoder: CKPT3 genomic encoder (frozen)\n\n## Evidence status\n\nThe deploy head is fitted after internal model selection and is **not** evaluated on its own training rows as validation evidence. The original CHORD historical test and completed DFCI/VICC external outcomes are not reopened in V2-07. Independent V2 external confirmation remains pending.\n\n## Input boundary\n\nThe public inference interface consumes prepared PRE/POST temporal states, the frozen 128-D tumor embedding, patient-level scan count context, genomic availability, and genomic age. Raw-cohort predictor construction is upstream and must satisfy the frozen V2-06 protocol. Treatment-line identity is not accepted.\n\n## Intended use\n\nResearch forecasting and prospective evaluation under the frozen protocol. This artifact is not a clinical device and is not validated for autonomous clinical decision-making.\n"""
    (out / "model_card.md").write_text(model_card)
    data_card = f"""# OncoTwin V2 data card\n\n## Final supervised fitting population\n\n- Rows: {len(dev):,}\n- Patients: {dev['patient_id'].nunique():,}\n- Eligibility: original CHORD train/val and `survival_mask=True`\n- Original CHORD test used in V2-07 fit: no\n- External outcomes used in V2-07 fit: no\n\nThe final head reuses frozen upstream representations prepared under earlier checkpoints. Genomics must satisfy the strict availability-before-landmark rule inherited from the frozen pipeline.\n"""
    (out / "data_card.md").write_text(data_card)
    changelog = """# CKPT7 to V2 release change log\n\n1. Preserved the repaired CKPT7R1 line-agnostic temporal encoder.\n2. Rebuilt V2 metric contracts and development-only folds without reopening protected external outcomes.\n3. Added source-transparent paired-scan representations and missingness-aware experiments.\n4. Selected the repaired R1 temporal branch in V2-02.\n5. Rejected adaptive gating in V2-03; froze alpha=0.52.\n6. Found promising explicit-scan gains in V2-04 but did not promote them after V2-05 specificity testing.\n7. Froze new-cohort external protocol in V2-06.\n8. V2-07 fits one all-development deploy head for the locked V2-03 specification and exposes a typed prepared-landmark inference API.\n\nCompleted historical DFCI/VICC evidence remains immutable and is not reclassified as new V2 confirmation.\n"""
    (out / "CHANGELOG_CKPT7_TO_V2.md").write_text(changelog)

    project_state = f"""# OncoTwin project state after V2-07\n\nStatus: **V2 research release complete; new independent external confirmation pending.**\n\nLocked internal specification: `r1_temporal_fixed` with alpha=0.52.\n\nFinal deploy head: `artifacts/dynamic_scan_v2/v2_07/deploy_model.pt`\n\nThe deploy head was fitted on all {len(dev):,} survival-eligible V2 development landmarks ({dev['patient_id'].nunique():,} patients). No historical CHORD test rows or DFCI/VICC outcomes were used for this fit. The fit is deployment-only; its in-sample performance is not validation evidence.\n\nFor a genuinely new cohort, follow the frozen V2-06 stage order: outcome-blind intake/overlap -> immutable V2-07 prediction freeze -> endpoint reveal and one-shot primary evaluation -> optional separately labeled recalibration.\n"""
    (out / "PROJECT_STATE_V2.md").write_text(project_state)

    manifest_files = [
        "deploy_model.pt", "training_population.json", "training_history.json",
        "inference_fixture_report.json", "synthetic_inference_fixture_input.npz",
        "synthetic_inference_fixture_expected.npz", "prepared_input_schema.json",
        "deployment_contract.json", "model_card.md", "data_card.md", "CHANGELOG_CKPT7_TO_V2.md",
        "PROJECT_STATE_V2.md",
    ]
    release_code = [
        "scripts/dynamic_scan_v2/v2_07_inference.py",
        "scripts/dynamic_scan_v2/v2_07_predict_prepared.py",
        "scripts/dynamic_scan_v2/v2_07_release.py",
        "configs/dynamic_scan_v2/v2_07_release.json",
    ]
    manifest = {
        "checkpoint": "V2-07",
        "version": config["version"],
        "deploy_model_sha256": sha256_file(deploy_path),
        "files": {name: sha256_file(out / name) for name in manifest_files},
        "upstream_source_hashes": hashes,
        "release_code_hashes": {rel: sha256_file(repo / rel) for rel in release_code},
        "training_state_sha256": sha256_file(out / "training_state.pt"),
        "status": "FROZEN",
    }
    atomic_json(out / "release_manifest.json", manifest)

    acceptance = {
        "v2_06_pass_required": True,
        "selected_rule_locked": True,
        "locked_alpha_is_0_52": abs(float(config["locked_alpha"]) - 0.52) < 1e-12,
        "all_development_survival_eligible_population_exact": len(dev) == 17194 and dev["patient_id"].nunique() == 2443,
        "historical_test_excluded_from_fit": True,
        "external_outcomes_not_opened": True,
        "cv_fold_head_not_used_as_deploy_model": True,
        "deploy_architecture_matches_frozen_v2_02_family": True,
        "line_derived_context_not_model_facing": True,
        "frozen_upstream_hashes_verified": True,
        "resume_safe_epoch_checkpoint_written": (out / "training_state.pt").is_file(),
        "typed_prepared_inference_interface_written": True,
        "deterministic_batch_invariance_pass": (
            fixture["batch_size_invariance_logit_max_abs"] <= fixture["batch_size_invariance_tolerances"]["logit_atol"]
            and fixture["batch_size_invariance_probability_max_abs"] <= fixture["batch_size_invariance_tolerances"]["probability_atol"]
            and fixture["batch_size_invariance_curve_max_abs"] <= fixture["batch_size_invariance_tolerances"]["curve_atol"]
        ),
        "unavailable_genomics_invariance_pass": fixture["unavailable_genomics_invariance_max_abs"] <= 1e-7,
        "release_manifest_frozen": True,
        "external_confirmation_explicitly_pending": True,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"V2-07 acceptance failed: {acceptance}")

    result = {
        "checkpoint": "V2-07",
        "status": "PASS",
        "project_status": "V2_RESEARCH_RELEASE_COMPLETE_EXTERNAL_CONFIRMATION_PENDING",
        "selected_rule": config["selected_rule"],
        "branch_family": config["branch_family"],
        "locked_alpha": float(config["locked_alpha"]),
        "deploy_model": "artifacts/dynamic_scan_v2/v2_07/deploy_model.pt",
        "deploy_model_sha256": manifest["deploy_model_sha256"],
        "training_rows": int(len(dev)),
        "training_patients": int(dev["patient_id"].nunique()),
        "training_resumed": bool(resumed),
        "parameter_count": int(payload["parameter_count"]),
        "external_confirmation": "PENDING_NEW_INDEPENDENT_COHORT",
        "prediction_execution_readiness": "READY_AFTER_V2_06_STAGE_A_PREFLIGHT",
        "historical_test_rows_opened": False,
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
        "in_sample_performance_used_as_evidence": False,
        "acceptance": acceptance,
    }
    atomic_json(out / "result_packet.json", result)

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
