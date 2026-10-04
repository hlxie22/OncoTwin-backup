from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import fit_censoring_km
from v2_05_controls import (
    decide_candidate_lock,
    deterministic_stratified_permutation,
    mask_explicit_scan_arrays,
    patient_bootstrap_brier_delta,
    seed_robustness,
    window_spread,
)


def _arrays(n=8):
    rng = np.random.default_rng(1)
    return {
        "current_core": rng.normal(size=(n, 8)).astype("f4"),
        "current_core_mask": np.ones((n, 8), "f4"),
        "change_core": rng.normal(size=(n, 22)).astype("f4"),
        "change_core_mask": np.ones((n, 22), "f4"),
        "optional_current": rng.normal(size=(n, 16)).astype("f4"),
        "optional_current_mask": np.ones((n, 16), "f4"),
        "stream_availability": np.ones((n, 8), "f4"),
        "stream_availability_mask": np.ones((n, 8), "f4"),
    }


def test_stratified_permutation_stays_within_strata_and_is_deterministic():
    strata = [("a", 1)] * 4 + [("b", 2)] * 4
    p1 = deterministic_stratified_permutation(strata, 7)
    p2 = deterministic_stratified_permutation(strata, 7)
    assert np.array_equal(p1, p2)
    assert any(p1 != np.arange(8))
    for i, j in enumerate(p1):
        assert strata[i] == strata[j]


def test_coverage_timing_mask_removes_state_and_optional_content():
    a = _arrays()
    b = mask_explicit_scan_arrays(a, "coverage_timing_only")
    assert np.all(b["current_core"][:, :3] == 0)
    assert np.all(b["current_core_mask"][:, :3] == 0)
    assert np.all(b["change_core"][:, :9] == 0)
    assert np.all(b["change_core"][:, 19:22] == 0)
    assert np.all(b["optional_current"] == 0)
    assert not np.shares_memory(a["current_core"], b["current_core"])
    assert np.allclose(a["current_core"][:, 3:], b["current_core"][:, 3:])


def test_clinical_content_mask_removes_coverage_timing():
    a = _arrays()
    b = mask_explicit_scan_arrays(a, "clinical_content_only")
    assert np.all(b["current_core"][:, 3:] == 0)
    assert np.all(b["change_core"][:, 9:19] == 0)
    assert np.allclose(a["current_core"][:, :3], b["current_core"][:, :3])


def test_seed_robustness_rules():
    cfg = {"minimum_seed_mean_brier_gain": 0.0005, "minimum_improving_seeds": 2, "maximum_single_seed_regression": 0.0005}
    r = seed_robustness([0.1699, 0.1695, 0.1714], 0.1712, cfg)
    assert r["status"] == "PASS"
    assert r["improving_seeds"] == 2


def test_window_spread():
    assert window_spread({"w0": .17, "w3": .171, "w7": .172}, .015)["pass"]
    assert not window_spread({"w0": .17, "w3": .171, "w7": .19}, .015)["pass"]


def test_candidate_lock_fallback_and_keep():
    seed = {"status": "PASS"}
    keep = decide_candidate_lock(seed_report=seed, nll_delta=-.01, nll_guardrail=.01, reassignment_gain=.001, minimum_reassignment_gain=.0001, content_margin=.001, minimum_content_margin=.00025, calibration_regression=0.0, maximum_calibration_regression=.02, window_pass=True)
    assert keep.candidate_passes
    fail = decide_candidate_lock(seed_report=seed, nll_delta=-.01, nll_guardrail=.01, reassignment_gain=0.0, minimum_reassignment_gain=.0001, content_margin=.001, minimum_content_margin=.00025, calibration_regression=0.0, maximum_calibration_regression=.02, window_pass=True)
    assert not fail.candidate_passes
    assert "v2_03" in fail.selected


def test_patient_bootstrap_identical_predictions_zero_delta():
    n = 20
    frame = pd.DataFrame({
        "patient_id": [f"p{i//2}" for i in range(n)],
        "survival_time_days": np.linspace(40, 700, n),
        "survival_cause": np.where(np.arange(n) % 4 == 0, 1, 0),
    })
    km = fit_censoring_km(frame["survival_time_days"].to_numpy(), frame["survival_cause"].to_numpy())
    logits = np.zeros((n, 24, 4), dtype=np.float32)
    r = patient_bootstrap_brier_delta(frame, logits, logits.copy(), km, repetitions=50, seed=3)
    assert abs(r["delta_candidate_minus_baseline"]) < 1e-12
    assert abs(r["ci_low"]) < 1e-12 and abs(r["ci_high"]) < 1e-12


def test_mask_does_not_mutate_input():
    a = _arrays()
    original = a["optional_current"].copy()
    _ = mask_explicit_scan_arrays(a, "coverage_timing_only")
    assert np.array_equal(a["optional_current"], original)


def test_window_alignment_uses_patient_landmark_not_episode_id():
    from v2_05_robustness import _patient_landmark_index

    w3 = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "scan_episode_id": ["p1::SCAN::0001::W3", "p1::SCAN::0002::W3", "p2::SCAN::0001::W3"],
        "landmark_day": [10, 30, 12],
    })
    w0 = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "scan_episode_id": ["p1::SCAN::0001::W0", "p1::SCAN::0003::W0", "p2::SCAN::0001::W0"],
        "landmark_day": [10, 30, 12],
    })
    _, m3 = _patient_landmark_index(w3, label="w3")
    _, m0 = _patient_landmark_index(w0, label="w0")
    assert set(m3) == set(m0)
    assert all("::W3" in x for x in w3["scan_episode_id"])
    assert all("::W0" in x for x in w0["scan_episode_id"])


def test_window_alignment_rejects_duplicate_patient_landmark():
    import pytest
    from v2_05_robustness import _patient_landmark_index

    bad = pd.DataFrame({
        "patient_id": ["p1", "p1"],
        "scan_episode_id": ["a::W0", "b::W0"],
        "landmark_day": [10, 10],
    })
    with pytest.raises(RuntimeError, match="not unique"):
        _patient_landmark_index(bad, label="w0")


def test_rep_to_local_arrays_uses_featureblock_masks_contract():
    from types import SimpleNamespace
    from scan_representation import FeatureBlock
    from v2_05_robustness import _rep_to_local_arrays

    n = 3

    def block(dim: int, offset: float):
        values = (np.arange(n * dim, dtype=np.float32).reshape(n, dim) + offset)
        masks = np.ones((n, dim), dtype=np.float32)
        masks[0, 0] = 0.0
        return FeatureBlock(
            values=values,
            masks=masks,
            names=tuple(f"f{i}" for i in range(dim)),
        )

    rep = SimpleNamespace(
        index=pd.DataFrame({"patient_id": ["a", "b", "c"]}),
        history_core=block(13, 1),
        utilization_core=block(3, 2),
        current_core=block(8, 3),
        change_core=block(22, 4),
        optional_history=block(10, 5),
        optional_history_recency=block(10, 6),
        optional_current=block(16, 7),
        stream_availability=block(8, 8),
    )
    temporal = np.zeros((n, 2, 192), dtype=np.float32)

    arrays = _rep_to_local_arrays(rep, temporal)

    for name in [
        "history_core",
        "utilization_core",
        "current_core",
        "change_core",
        "optional_history",
        "optional_history_recency",
        "optional_current",
        "stream_availability",
    ]:
        source = getattr(rep, name)
        assert np.array_equal(arrays[name], source.values)
        assert np.array_equal(arrays[f"{name}_mask"], source.masks)

    assert arrays["temporal_pre"].shape == (n, 192)
    assert arrays["temporal_post"].shape == (n, 192)
    assert arrays["tumor"].shape == (n, 128)
