from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts" / "dynamic_scan_v2"
if str(MODULE) not in sys.path:
    sys.path.insert(0, str(MODULE))

from access import AccessDeniedError, AccessPolicy
from common import stable_fold
from metrics import (
    CAUSE_DEATH,
    CAUSE_NO_EVENT_OR_CENSOR,
    CAUSE_PROGRESSION,
    CAUSE_SWITCH,
    MONTH_DAYS,
    brier_at_horizon,
    censor_exposure,
    curves_from_logits,
    fit_censoring_km,
    fractional_censor_nll_per_row,
    resample_patient_clusters,
    validate_curve_invariants,
)

from legacy_metrics import legacy_nll_per_row


def constant_logits(q0: float = 0.8) -> np.ndarray:
    qevent = (1.0 - q0) / 3.0
    logits = np.zeros((4, 24, 4), dtype=float)
    logits[:, :, 0] = np.log(q0)
    logits[:, :, 1:] = np.log(qevent)
    return logits


def test_probability_mass_and_monotonicity():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(32, 24, 4))
    curves = curves_from_logits(logits)
    result = validate_curve_invariants(curves)
    assert result["status"] == "PASS"
    assert np.allclose(curves["total_mass"], 1.0, atol=1e-10)


def test_switch_is_distinct_competing_cause():
    logits = np.full((1, 24, 4), -8.0)
    logits[:, :, 0] = 4.0
    logits[:, 0, CAUSE_SWITCH] = 8.0
    curves = curves_from_logits(logits)
    assert curves["switch_cif"][0, 0] > 0.9
    assert curves["progression_cif"][0, 0] < 0.01
    assert curves["death_cif"][0, 0] < 0.01
    # Switch is not silently relabeled as progression/death.
    assert curves["pfs"][0, 0] > 0.98


def test_zero_time_rejected_by_metric_contract():
    logits = constant_logits()[:1]
    with pytest.raises(ValueError):
        legacy_nll_per_row(logits, np.asarray([0.0]), np.asarray([CAUSE_PROGRESSION]))
    with pytest.raises(ValueError):
        fractional_censor_nll_per_row(logits, np.asarray([0.0]), np.asarray([CAUSE_NO_EVENT_OR_CENSOR]))


def test_exact_boundary_censor_matches_legacy():
    logits = constant_logits()[:1]
    t = np.asarray([MONTH_DAYS])
    c = np.asarray([CAUSE_NO_EVENT_OR_CENSOR])
    legacy = legacy_nll_per_row(logits, t, c)[0]
    corrected = fractional_censor_nll_per_row(logits, t, c)[0]
    assert np.isclose(legacy, corrected, atol=1e-12)
    completed, fraction = censor_exposure(MONTH_DAYS)
    assert completed == 1
    assert fraction == 0.0


def test_interior_bin_censor_uses_fractional_exposure():
    logits = constant_logits()[:1]
    t = np.asarray([0.5 * MONTH_DAYS])
    c = np.asarray([CAUSE_NO_EVENT_OR_CENSOR])
    legacy = legacy_nll_per_row(logits, t, c)[0]
    corrected = fractional_censor_nll_per_row(logits, t, c)[0]
    assert np.isclose(corrected, 0.5 * legacy, atol=1e-12)
    assert corrected < legacy


def test_event_interval_semantics_at_exact_boundary():
    logits = constant_logits()[:2]
    times = np.asarray([MONTH_DAYS, MONTH_DAYS + 1e-6])
    causes = np.asarray([CAUSE_PROGRESSION, CAUSE_PROGRESSION])
    nll = legacy_nll_per_row(logits, times, causes)
    # Exactly one interval width is an event in interval 1; just after is interval 2.
    expected_gap = -np.log(0.8)
    assert np.isclose(nll[1] - nll[0], expected_gap, atol=1e-10)


def test_duplicate_bootstrap_patient_draws_are_not_collapsed():
    frame = pd.DataFrame({"patient_id": ["a", "b", "c"], "x": [1, 2, 3]})
    sample = resample_patient_clusters(frame, np.random.default_rng(0))
    assert sample["_bootstrap_cluster_draw"].nunique() == 3
    # Seed 0 draws one source patient more than once; all three draws must remain.
    assert sample["_bootstrap_source_patient"].nunique() < 3
    assert len(sample) == 3


def test_synthetic_brier_orders_good_prediction_before_bad():
    n = 1000
    case = np.arange(n) < 300
    frame = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "survival_time_days": np.where(case, 60.0, 730.0),
        "survival_cause": np.where(case, CAUSE_PROGRESSION, CAUSE_NO_EVENT_OR_CENSOR),
    })
    km = fit_censoring_km(frame["survival_time_days"].to_numpy(), frame["survival_cause"].to_numpy())
    good_pfs = np.where(case, 0.1, 0.9)
    bad_pfs = 1.0 - good_pfs
    good = brier_at_horizon(frame, good_pfs, 3, km, patient_balanced=True)["brier"]
    bad = brier_at_horizon(frame, bad_pfs, 3, km, patient_balanced=True)["brier"]
    assert good < bad


def test_outcome_blind_fold_assignment_is_stable_and_patient_level():
    ids = ["p1", "p2", "p1", "p3"]
    folds = [stable_fold(x, 3, "fixed-salt") for x in ids]
    assert folds[0] == folds[2]
    assert all(0 <= x < 3 for x in folds)


def test_access_policy_blocks_external_and_hard_outcomes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = tmp_path / "policy.json"
    config.write_text(json.dumps({
        "blocked_path_tokens": ["dfci", "vicc", "vanderbilt"],
        "blocked_relative_roots": ["artifacts/checkpoint7r3g1_external_prediction_freeze"],
        "blocked_basenames": ["regimen_cancer_level_dataset.csv"],
        "allowed_schema_positive_controls": ["artifacts/checkpoint1/canonical/bpc_msk_scan_audit.parquet"],
    }))
    policy = AccessPolicy.from_json(repo, config)
    with pytest.raises(AccessDeniedError):
        policy.assert_allowed(repo / "data/dfci/predictors.parquet")
    with pytest.raises(AccessDeniedError):
        policy.assert_allowed(repo / "data/x/regimen_cancer_level_dataset.csv")
    with pytest.raises(AccessDeniedError):
        policy.assert_allowed(repo / "artifacts/checkpoint7r3g1_external_prediction_freeze/pred.parquet")
    allowed = policy.assert_allowed(repo / "artifacts/checkpoint1/canonical/bpc_msk_scan_audit.parquet")
    assert allowed.name == "bpc_msk_scan_audit.parquet"
