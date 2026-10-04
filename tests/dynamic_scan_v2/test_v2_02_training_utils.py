from __future__ import annotations

import numpy as np
import pandas as pd

from v2_02_train import (
    patient_balance_weights,
    build_person_period_rows,
    standardizer_fit,
    standardizer_apply,
    attach_and_validate_development_folds,
)


def test_patient_balance_sums_to_patient_count():
    frame = pd.DataFrame({"patient_id": ["a", "a", "a", "b", "c", "c"]})
    w = patient_balance_weights(frame)
    assert np.isclose(w.sum(), 3.0)
    assert np.isclose(w[:3].sum(), 1.0)
    assert np.isclose(w[3:4].sum(), 1.0)
    assert np.isclose(w[4:].sum(), 1.0)


def test_standardizer_train_only_roundtrip_shape():
    x = np.array([[1, 2], [3, 2], [5, 2]], dtype=np.float32)
    stat = standardizer_fit(x)
    y = standardizer_apply(x, stat)
    assert y.shape == x.shape
    assert np.allclose(y[:, 0].mean(), 0.0, atol=1e-6)
    assert np.allclose(y[:, 1], 0.0)


def test_person_period_fractional_censor_weight():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    # 1.5 months censored: one full no-event interval + half a no-event interval.
    t = np.array([730.0 / 24.0 * 1.5])
    c = np.array([0])
    w = np.array([1.0])
    xp, yp, wp = build_person_period_rows(x, t, c, w)
    assert xp.shape[0] == 2
    assert yp.tolist() == [0, 0]
    assert np.allclose(wp, [1.0, 0.5], atol=1e-6)


def _fold_frames():
    frame = pd.DataFrame({
        "patient_id": ["a", "a", "b", "c", "d", "d"],
        "split": ["train", "train", "val", "test", "test", "test"],
    })
    folds = pd.DataFrame({
        "patient_id": ["a", "b", "c", "d"],
        "original_split": ["train", "val", "test", "test"],
        "v2_fold": [0, 2, -1, -1],
        "v2_role": [
            "v2_development",
            "v2_development",
            "historical_test_read_only",
            "historical_test_read_only",
        ],
    })
    return frame, folds


def test_frozen_fold_minus_one_is_valid_historical_test_sentinel():
    frame, folds = _fold_frames()
    out = attach_and_validate_development_folds(frame, folds, fold_count=3)
    assert set(out.loc[out["split"].eq("train"), "development_fold"]) == {0}
    assert set(out.loc[out["split"].eq("val"), "development_fold"]) == {2}
    assert set(out.loc[out["split"].eq("test"), "development_fold"]) == {-1}


def test_historical_test_positive_fold_is_rejected():
    import pytest
    frame, folds = _fold_frames()
    folds.loc[folds["patient_id"].eq("c"), "v2_fold"] = 1
    with pytest.raises(RuntimeError, match="Historical test patient assigned"):
        attach_and_validate_development_folds(frame, folds, fold_count=3)


def test_fold_registry_role_or_split_drift_is_rejected():
    import pytest
    frame, folds = _fold_frames()
    folds.loc[folds["patient_id"].eq("b"), "v2_role"] = "historical_test_read_only"
    with pytest.raises(RuntimeError, match="wrong V2 role"):
        attach_and_validate_development_folds(frame, folds, fold_count=3)

    frame, folds = _fold_frames()
    folds.loc[folds["patient_id"].eq("b"), "original_split"] = "train"
    with pytest.raises(RuntimeError, match="original_split disagrees"):
        attach_and_validate_development_folds(frame, folds, fold_count=3)
