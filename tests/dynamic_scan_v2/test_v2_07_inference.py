import numpy as np
import pytest
import torch

from v2_02_models import TemporalFixedSurvivalModel
from v2_07_inference import (
    PreparedLandmarkBatch,
    blend_logits,
    conditional_probs_to_curves,
    softmax_np,
)


def batch(n=7, seed=1):
    r = np.random.default_rng(seed)
    return PreparedLandmarkBatch(
        temporal_pre=r.normal(size=(n,192)).astype("float32"),
        temporal_post=r.normal(size=(n,192)).astype("float32"),
        tumor_embedding=r.normal(size=(n,128)).astype("float32"),
        scan_number_patient_log=r.uniform(0,4,size=n).astype("float32"),
        genomic_available=np.asarray([i % 2 for i in range(n)], dtype="float32"),
        genomic_age_scaled=r.uniform(0,8,size=n).astype("float32") * np.asarray([i % 2 for i in range(n)], dtype="float32"),
    )


def test_blend_endpoints_and_shape():
    r=np.random.default_rng(3)
    a=r.normal(size=(5,24,4)).astype("float32")
    b=r.normal(size=(5,24,4)).astype("float32")
    assert np.array_equal(blend_logits(a,b,0), a)
    assert np.array_equal(blend_logits(a,b,1), b)
    c=blend_logits(a,b,.52)
    assert c.shape==(5,24,4)


def test_softmax_and_curve_mass():
    r=np.random.default_rng(4)
    p=softmax_np(r.normal(size=(11,24,4)).astype("float32"))
    assert np.max(np.abs(p.sum(-1)-1)) < 1e-6
    c=conditional_probs_to_curves(p)
    total=c["event_free_survival"] + c["progression_cif"] + c["death_cif"] + c["switch_cif"]
    assert np.max(np.abs(total-1)) < 2e-5
    assert np.min(np.diff(c["progression_cif"],axis=1)) >= -2e-6


def test_prepared_contract_rejects_wrong_shapes():
    b=batch()
    assert b.validate()==7
    with pytest.raises(ValueError):
        PreparedLandmarkBatch(
            temporal_pre=np.zeros((2,191),dtype="float32"), temporal_post=np.zeros((2,192),dtype="float32"),
            tumor_embedding=np.zeros((2,128),dtype="float32"), scan_number_patient_log=np.zeros(2),
            genomic_available=np.zeros(2), genomic_age_scaled=np.zeros(2),
        ).validate()


def test_abstention_contract():
    b=batch(3)
    x=PreparedLandmarkBatch(
        b.temporal_pre,b.temporal_post,b.tumor_embedding,b.scan_number_patient_log,b.genomic_available,b.genomic_age_scaled,
        current_scan_state_known=np.array([1,0,1]), required_clock_known=np.array([1,1,0])
    )
    ok,reasons=x.applicability()
    assert ok.tolist()==[True,False,False]
    assert reasons==["SUPPORTED","ABSTAIN_UNKNOWN_CURRENT_SCAN_STATE","ABSTAIN_UNKNOWN_REQUIRED_CLOCK"]


def test_model_genomics_unavailable_invariance():
    torch.manual_seed(8)
    m=TemporalFixedSurvivalModel().eval()
    r=np.random.default_rng(8)
    n=4
    tp=torch.tensor(r.normal(size=(n,192)).astype("float32"))
    tq=torch.tensor(r.normal(size=(n,192)).astype("float32"))
    tumor=torch.tensor(r.normal(size=(n,128)).astype("float32"))
    ctx=torch.tensor(np.stack([np.arange(n),np.zeros(n),np.zeros(n)],axis=1).astype("float32"))
    with torch.no_grad():
        a=m(tp,tq,tumor,ctx)
        b=m(tp,tq,tumor+999,ctx)
    assert torch.max(torch.abs(a[0]-b[0])).item() < 1e-6
    assert torch.max(torch.abs(a[1]-b[1])).item() < 1e-6


def _dummy_deploy_checkpoint(path, crossfit=False):
    m = TemporalFixedSurvivalModel()
    torch.save({
        "checkpoint":"V2-07",
        "family":"r1_temporal_fixed",
        "selected_rule":"v2_03_frozen_temporal_constant_alpha",
        "locked_alpha":0.52,
        "training_scope":"all_v2_development_survival_eligible",
        "is_crossfit_fold_head":crossfit,
        "model_state":m.state_dict(),
        "model_dimensions":{"temporal":192,"tumor":128,"portable_context":3,"months":24,"causes":4},
        "source_hashes":{},
    }, path)


def test_deploy_checkpoint_load_and_predict(tmp_path):
    from v2_07_inference import OncoTwinV2Predictor
    p=tmp_path/'deploy.pt'
    _dummy_deploy_checkpoint(p)
    predictor=OncoTwinV2Predictor(p,device='cpu')
    x=predictor.predict(batch(5),batch_size=2)
    assert x['selected_logits'].shape==(5,24,4)
    assert x['selected_curves']['pfs_survival'].shape==(5,24)
    assert np.isfinite(x['selected_horizon_pfs_survival']).all()


def test_crossfit_checkpoint_rejected(tmp_path):
    from v2_07_inference import OncoTwinV2Predictor
    p=tmp_path/'fold.pt'
    _dummy_deploy_checkpoint(p,crossfit=True)
    with pytest.raises(RuntimeError, match='Cross-fit'):
        OncoTwinV2Predictor(p,device='cpu')


def test_predictor_batch_size_numerical_invariance_and_genomic_mask(tmp_path):
    from v2_07_inference import OncoTwinV2Predictor
    p = tmp_path / "deploy.pt"
    _dummy_deploy_checkpoint(p)
    predictor = OncoTwinV2Predictor(p, device="cpu")
    x = batch(23, seed=19)
    full = predictor.predict(x, batch_size=23)
    single = predictor.predict(x, batch_size=1)
    assert np.max(np.abs(full["selected_logits"] - single["selected_logits"])) <= 3e-6
    assert np.max(np.abs(full["selected_conditional_probabilities"] - single["selected_conditional_probabilities"])) <= 5e-7
    assert np.max(np.abs(full["selected_curves"]["pfs_survival"] - single["selected_curves"]["pfs_survival"])) <= 5e-7

    unavailable = np.asarray(x.genomic_available).reshape(-1) == 0
    modified = np.asarray(x.tumor_embedding).copy()
    modified[unavailable] += np.float32(999.0)
    y = PreparedLandmarkBatch(
        x.temporal_pre, x.temporal_post, modified, x.scan_number_patient_log,
        x.genomic_available, x.genomic_age_scaled, x.current_scan_state_known, x.required_clock_known,
    )
    changed = predictor.predict(y, batch_size=23)
    assert np.array_equal(full["selected_logits"][unavailable], changed["selected_logits"][unavailable])
