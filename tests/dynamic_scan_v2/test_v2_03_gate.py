from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from metrics import fractional_censor_nll_per_row
from v2_03_gate import (
    LinearSigmoidGate,
    NonlinearSigmoidGate,
    apply_standardizer,
    assert_class_offset_invariant_features,
    blend_logits,
    build_gate_features,
    fit_standardizer,
    gate_distribution,
    patient_balance_weights,
    torch_fractional_nll_per_row,
)


def _meta(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "clinical_state": np.where(np.arange(n) % 3 == 0, "INDETERMINATE", "NON_PROGRESSIVE"),
        "days_since_previous_assessment": np.arange(n) + 10,
        "days_since_previous_comparable_assessment": np.arange(n) + 5,
        "prior_scan_count": np.arange(n) % 10,
        "prior_nonprogressive_run": np.arange(n) % 5,
        "coverage_jaccard": np.linspace(0, 1, n),
        "current_coverage_count": np.arange(n) % 6,
        "coverage_gained_count": np.arange(n) % 3,
        "coverage_lost_count": np.arange(n) % 2,
        "coverage_comparable_to_previous": np.arange(n) % 2 == 0,
        "current_modality_observed": True,
        "current_site_positive_stream_observed": np.arange(n) % 2 == 0,
        "genomic_available": np.arange(n) % 2 == 1,
        "genomic_age_days": np.arange(n) * 3.0,
    })


def _logits(n: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    pre = rng.normal(size=(n, 24, 4)).astype(np.float32)
    post = pre + rng.normal(scale=0.2, size=pre.shape).astype(np.float32)
    return pre, post


def test_alpha_endpoints_exact():
    pre, post = _logits(12)
    assert np.array_equal(blend_logits(pre, post, 0.0), pre.astype(np.float64))
    assert np.allclose(blend_logits(pre, post, 1.0), post)


def test_gate_feature_shared_logit_offset_invariance():
    n = 25
    pre, post = _logits(n)
    assert assert_class_offset_invariant_features(_meta(n), pre, post)


def test_gate_feature_policy_has_no_outcomes_or_identifiers():
    n = 10
    pre, post = _logits(n)
    x, names, _ = build_gate_features(_meta(n), pre, post)
    assert x.shape[0] == n
    forbidden = ["survival", "survival_cause", "censor", "outcome", "patient_id", "site_id", "development_fold"]
    for name in names:
        assert not any(token in name.lower() for token in forbidden)


def test_standardization_is_fit_then_apply():
    x = np.asarray([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]], dtype=np.float32)
    s = fit_standardizer(x)
    z = apply_standardizer(x, s)
    assert np.allclose(z[:, 0].mean(), 0.0, atol=1e-6)
    assert np.all(np.isfinite(z))
    assert np.allclose(z[:, 1], 0.0)


def test_patient_balance_equal_total_weight_per_patient():
    ids = ["a", "a", "a", "b", "c", "c"]
    w = patient_balance_weights(ids)
    totals = {}
    for pid, wi in zip(ids, w):
        totals[pid] = totals.get(pid, 0.0) + float(wi)
    assert all(abs(v - 1.0) < 1e-6 for v in totals.values())


def test_torch_fractional_nll_matches_numpy_reference():
    n = 32
    pre, _ = _logits(n, seed=8)
    rng = np.random.default_rng(9)
    time = rng.uniform(1.0, 729.0, size=n).astype(np.float32)
    cause = rng.integers(0, 4, size=n, dtype=np.int64)
    ref = fractional_censor_nll_per_row(pre, time, cause)
    got = torch_fractional_nll_per_row(
        torch.tensor(pre), torch.tensor(time), torch.tensor(cause)
    ).detach().numpy()
    assert np.allclose(ref, got, atol=2e-5, rtol=2e-5)


def test_gate_outputs_are_bounded_and_initialized_near_half():
    x = torch.randn(20, 7)
    for model in [LinearSigmoidGate(7), NonlinearSigmoidGate(7, hidden_dim=8)]:
        a = model(x).detach().numpy()
        assert ((a >= 0.0) & (a <= 1.0)).all()
        assert np.allclose(a, 0.5, atol=1e-6)


def test_gate_distribution_reports_saturation():
    a = np.asarray([0.01, 0.2, 0.5, 0.8, 0.99])
    d = gate_distribution(a)
    assert abs(d["saturation_fraction"] - 0.4) < 1e-12
    assert d["std"] > 0


def test_torch_fractional_nll_boundary_and_interior_censoring_matches_numpy():
    rng = np.random.default_rng(77)
    logits = rng.normal(size=(6, 24, 4)).astype(np.float32)
    month = 730.0 / 24.0
    time = np.asarray([
        month,
        2.0 * month,
        2.5 * month,
        6.0 * month,
        6.25 * month,
        729.0,
    ], dtype=np.float32)
    cause = np.asarray([0, 0, 0, 1, 2, 3], dtype=np.int64)
    ref = fractional_censor_nll_per_row(logits, time, cause)
    got = torch_fractional_nll_per_row(
        torch.tensor(logits), torch.tensor(time), torch.tensor(cause)
    ).detach().numpy()
    assert np.allclose(ref, got, atol=3e-5, rtol=3e-5)
