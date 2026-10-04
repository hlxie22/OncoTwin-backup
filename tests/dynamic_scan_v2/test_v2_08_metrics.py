import numpy as np

from v2_08_metrics import (
    calibration_ece,
    calibration_intercept_slope,
    effective_sample_size,
    weighted_auc,
    weighted_brier,
    weighted_logistic_fit,
    weighted_mean,
)


def test_weighted_mean_basic():
    x = np.array([1.0, 3.0])
    w = np.array([1.0, 3.0])
    assert abs(weighted_mean(x, w) - 2.5) < 1e-12


def test_weighted_auc_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    w = np.ones(4)
    assert abs(weighted_auc(y, s, w) - 1.0) < 1e-12


def test_weighted_auc_reverse():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    w = np.ones(4)
    assert abs(weighted_auc(y, s, w) - 0.0) < 1e-12


def test_weighted_auc_ties_half_credit():
    y = np.array([0, 1])
    s = np.array([0.5, 0.5])
    w = np.ones(2)
    assert abs(weighted_auc(y, s, w) - 0.5) < 1e-12


def test_weighted_auc_invariant_to_positive_weight_scaling():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.2, 0.8, 0.4, 0.6])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(weighted_auc(y, s, w) - weighted_auc(y, s, 7.0 * w)) < 1e-12


def test_weighted_brier_known_value():
    y = np.array([0.0, 1.0])
    p = np.array([0.25, 0.75])
    w = np.ones(2)
    assert abs(weighted_brier(y, p, w) - 0.0625) < 1e-12


def test_effective_sample_size_uniform():
    assert abs(effective_sample_size(np.ones(10)) - 10.0) < 1e-12


def test_calibration_fit_recovers_ideal_probabilities():
    p = np.linspace(0.05, 0.95, 1000)
    # Deterministic weighted pseudo-sample: duplicate each probability into y=0/1
    # with weights proportional to its Bernoulli mass. This exactly represents a
    # perfectly calibrated population without Monte Carlo noise.
    pp = np.repeat(p, 2)
    y = np.tile(np.array([0.0, 1.0]), len(p))
    w = np.column_stack([1.0 - p, p]).reshape(-1)
    cal = calibration_intercept_slope(y, pp, w)
    assert abs(cal["intercept"]) < 1e-5
    assert abs(cal["slope"] - 1.0) < 1e-5


def test_calibration_ece_zero_for_exact_group_probabilities():
    p = np.array([0.25, 0.25, 0.75, 0.75])
    y = np.array([0.0, 0.5, 0.5, 1.0])
    w = np.ones(4)
    # calibration_ece expects binary y; use exact Bernoulli expansion instead.
    pp = np.repeat(np.array([0.25, 0.75]), 4)
    yy = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)
    ww = np.ones(8)
    ece, rows = calibration_ece(yy, pp, ww, bins=2)
    assert len(rows) == 2
    assert abs(ece) < 1e-12


def test_weighted_logistic_fit_finite():
    x = np.linspace(-2, 2, 200)[:, None]
    y = (x[:, 0] > 0).astype(float)
    w = np.ones(len(y))
    beta = weighted_logistic_fit(x, y, w, ridge=1e-3)
    assert np.isfinite(beta).all()
    assert beta.shape == (2,)


def test_antolini_style_concordance_perfect_and_same_patient_exclusion():
    from v2_08_metrics import antolini_style_concordance
    # p1 fails at day 20 and has higher 1m risk than p2/p3, p2 fails at day 50
    # and has higher 2m risk than p3. Extra p1 landmark is excluded as a
    # same-patient comparator for the first p1 event.
    t = np.array([20.0, 80.0, 50.0, 100.0])
    c = np.array([1, 0, 2, 0])
    pid = np.array(["p1", "p1", "p2", "p3"])
    risk = np.zeros((4, 24), dtype=float)
    risk[0, :] = 0.9
    risk[1, :] = 0.1
    risk[2, :] = 0.7
    risk[3, :] = 0.2
    out = antolini_style_concordance(t, c, pid, risk, 30.0)
    assert abs(out["concordance"] - 1.0) < 1e-12
    assert out["pfs_event_rows_within_24m"] == 2
