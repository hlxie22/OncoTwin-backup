import pandas as pd

from v2_02_finalize_existing import modeling_eligible_mask


def test_modeling_population_is_survival_eligible_train_val_not_all_representation_rows():
    frame = pd.DataFrame({
        "split": ["train", "val", "test", "train", "val", "test"],
        "survival_mask": [True, True, True, False, False, False],
    })
    got = modeling_eligible_mask(frame).tolist()
    assert got == [True, True, False, False, False, False]
    assert sum(got) == 2
    assert len(frame) == 6


def test_modeling_population_is_case_normalized_and_null_mask_is_false():
    frame = pd.DataFrame({
        "split": ["TRAIN", "Val", "test", "train"],
        "survival_mask": [True, True, True, None],
    })
    assert modeling_eligible_mask(frame).tolist() == [True, True, False, False]
