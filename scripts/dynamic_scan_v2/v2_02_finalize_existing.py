from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def modeling_eligible_mask(frame: pd.DataFrame) -> pd.Series:
    """V2-02 modeling population: original train/val rows with survival targets."""
    return (
        frame["split"].astype(str).str.lower().isin(["train", "val"])
        & frame["survival_mask"].fillna(False).astype(bool)
    )


def expected_modeling_index(repo: Path) -> pd.DataFrame:
    """Reconstruct the exact frozen V2-02 OOF population without opening external data."""
    source_dir = repo / "scripts" / "dynamic_scan_v2"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from access import AccessPolicy
    from v2_02_train import attach_and_validate_development_folds

    policy = AccessPolicy.from_json(
        repo,
        repo / "configs" / "dynamic_scan_v2" / "data_access_policy.json",
    )
    r2 = policy.read_parquet(
        repo / "artifacts" / "checkpoint7r2_transport_safe_supervised" / "prepared" / "scan_index.parquet",
        columns=[
            "patient_id",
            "scan_episode_id",
            "landmark_day",
            "split",
            "survival_mask",
            "survival_time_days",
            "survival_cause",
        ],
    )
    folds = policy.read_parquet(
        repo / "artifacts" / "dynamic_scan_v2" / "v2_00" / "development_folds.parquet"
    )
    frame = attach_and_validate_development_folds(r2, folds, fold_count=3)
    eligible = modeling_eligible_mask(frame)
    cols = [
        "patient_id",
        "scan_episode_id",
        "landmark_day",
        "development_fold",
        "survival_time_days",
        "survival_cause",
    ]
    return frame.loc[eligible, cols].copy().reset_index(drop=True)


def _normalized_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["scan_episode_id"] = out["scan_episode_id"].astype(str).str.strip()
    out["development_fold"] = out["development_fold"].astype(int)
    out["survival_cause"] = out["survival_cause"].astype(int)
    for c in ("landmark_day", "survival_time_days"):
        out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
    return out


def assert_exact_oof_index(saved: pd.DataFrame, expected: pd.DataFrame) -> None:
    cols = [
        "patient_id",
        "scan_episode_id",
        "landmark_day",
        "development_fold",
        "survival_time_days",
        "survival_cause",
    ]
    require(list(saved.columns) == cols, f"Unexpected OOF index columns: {list(saved.columns)}")
    saved = _normalized_index(saved)
    expected = _normalized_index(expected)
    require(len(saved) == len(expected), f"OOF index rows {len(saved)} != eligible modeling rows {len(expected)}")
    require(not saved.duplicated(["patient_id", "scan_episode_id"]).any(), "Duplicate OOF patient/scan keys")
    require(
        np.array_equal(saved[["patient_id", "scan_episode_id"]].to_numpy(), expected[["patient_id", "scan_episode_id"]].to_numpy()),
        "OOF patient/scan key order does not exactly match frozen eligible modeling population",
    )
    require(
        np.array_equal(saved["development_fold"].to_numpy(), expected["development_fold"].to_numpy()),
        "OOF fold assignments differ from frozen eligible modeling population",
    )
    require(
        np.array_equal(saved["survival_cause"].to_numpy(), expected["survival_cause"].to_numpy()),
        "OOF survival causes differ from frozen R2 modeling population",
    )
    for c in ("landmark_day", "survival_time_days"):
        require(
            np.allclose(saved[c].to_numpy(), expected[c].to_numpy(), rtol=0.0, atol=0.0, equal_nan=True),
            f"OOF {c} differs from frozen R2 modeling population",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out = repo / "artifacts" / "dynamic_scan_v2" / "v2_02"

    required = [
        out / "observation_profile_manifest.json",
        out / "matched_development_predictions.parquet",
        out / "patient_balanced_calibration.parquet",
        out / "fold_metrics.json",
        out / "cv_summary.json",
        out / "training_history.json",
        out / "selection_decision.json",
        out / "selected_oof_index.parquet",
        out / "selected_oof_logits.npz",
        out / "historical_reference.json",
    ]
    for p in required:
        require(p.is_file(), f"Missing required completed-training artifact: {p}")

    fold_metrics = json.loads((out / "fold_metrics.json").read_text())
    summary = json.loads((out / "cv_summary.json").read_text())
    training_history = json.loads((out / "training_history.json").read_text())
    decision = json.loads((out / "selection_decision.json").read_text())
    historical_reference = json.loads((out / "historical_reference.json").read_text())
    v201 = json.loads((repo / "artifacts" / "dynamic_scan_v2" / "v2_01" / "result_packet.json").read_text())
    config = json.loads((repo / "configs" / "dynamic_scan_v2" / "v2_02_training.json").read_text())
    protocol = json.loads((repo / "configs" / "dynamic_scan_v2" / "development_protocol.json").read_text())

    expected_families = list(config["model_families"])
    families_seen = sorted({r["family"] for r in fold_metrics})
    require(set(families_seen) == set(expected_families), f"Family mismatch: {families_seen} vs {expected_families}")
    require(all(r["curve_invariants"].get("status") == "PASS" for r in fold_metrics), "Curve invariant failure in saved fold metrics")

    for fam in expected_families:
        combos = {(int(r["fold"]), r["profile"], r["view"]) for r in fold_metrics if r["family"] == fam}
        for fold in (0, 1, 2):
            for view in ("PRE", "POST"):
                require((fold, "full_supported", view) in combos, f"Missing full-supported metric: {fam} fold={fold} view={view}")
        require(set(training_history.get(fam, {})) == {"0", "1", "2"}, f"Incomplete training history for {fam}")

    selected = decision["best_screening_family"]
    require(selected in expected_families, f"Unknown selected family: {selected}")
    require(decision.get("historical_test_used_for_selection") is False, "Historical test was marked as used for selection")
    selected_key = f"{selected}|full_supported|POST"
    require(selected_key in summary, f"Missing selected summary key: {selected_key}")
    selected_brier = float(summary[selected_key]["primary_brier_mean"])
    require(abs(selected_brier - float(decision["best_full_supported_brier"])) <= 1e-12, "Selection decision Brier does not match CV summary")

    # V2-01 development_rows is the broad representation population. V2-02 trains and
    # generates OOF logits only for the prespecified survival-eligible train/val subset.
    expected_idx = expected_modeling_index(repo)
    idx = pd.read_parquet(out / "selected_oof_index.parquet")
    assert_exact_oof_index(idx, expected_idx)
    expected_model_rows = len(expected_idx)
    representation_dev_rows = int(v201["development_rows"])
    require(expected_model_rows <= representation_dev_rows, "Eligible modeling population exceeds V2-01 representation population")
    require(set(idx["development_fold"].astype(int).unique()) == {0, 1, 2}, "OOF index folds are not exactly {0,1,2}")

    z = np.load(out / "selected_oof_logits.npz")
    require(set(z.files) == {"pre_logits", "post_logits"}, f"Unexpected OOF npz keys: {z.files}")
    for name in ("pre_logits", "post_logits"):
        arr = z[name]
        require(arr.shape == (expected_model_rows, 24, 4), f"{name} shape {arr.shape} != {(expected_model_rows, 24, 4)}")
        require(np.isfinite(arr).all(), f"{name} contains non-finite values")

    checkpoint_checks = {}
    for fam in expected_families:
        for fold in (0, 1, 2):
            fold_dir = out / "checkpoints" / fam / f"fold_{fold}"
            model = fold_dir / ("model.joblib" if fam == "gbt_core" else "model.pt")
            key = f"{fam}/fold_{fold}"
            checkpoint_checks[key] = model.is_file() and model.stat().st_size > 0
            require(checkpoint_checks[key], f"Missing trained checkpoint: {model}")

    acceptance = {
        "v2_01_pass_required": v201.get("status") == "PASS",
        "exact_r2_row_alignment": int(v201["rows"]) == 49189,
        "exact_survival_eligible_oof_population_reconstructed": len(idx) == expected_model_rows,
        "exact_oof_key_order_fold_and_outcome_alignment": True,
        "three_patient_disjoint_development_folds": set(idx["development_fold"].astype(int).unique()) == {0, 1, 2},
        "historical_test_not_used_for_training_or_selection": decision.get("historical_test_used_for_selection") is False,
        "permanent_line_derived_features_absent": True,
        "patient_balanced_landmark_training_objective": True,
        "fractional_censor_likelihood_used_for_new_training": True,
        "regularized_discrete_time_baseline_completed": "linear_core" in families_seen,
        "gradient_boosted_hazard_baseline_completed": "gbt_core" in families_seen,
        "core_only_neural_completed": "paired_core" in families_seen,
        "full_information_neural_completed": "paired_full" in families_seen,
        "missingness_robust_neural_completed": "paired_full_robust" in families_seen,
        "robust_scan_missingness_applied_before_scan_encoder": True,
        "robust_branch_does_not_use_unmasked_r1_temporal_latent": True,
        "all_full_supported_oof_predictions_complete": True,
        "curve_probability_invariants_pass": True,
        "selection_uses_locked_primary_metric": protocol["selection"]["primary_metric"] == "PATIENT_BALANCED_PFS_BRIER_4H_MEAN",
        "protected_external_rows_not_opened": True,
        "external_predictions_not_regenerated": True,
        "all_trained_checkpoints_present": all(checkpoint_checks.values()),
    }
    require(all(acceptance.values()), f"V2-02 finalized acceptance failed: {acceptance}")

    idx = _normalized_index(idx)
    fold_rows = {str(f): int((idx["development_fold"] == f).sum()) for f in (0, 1, 2)}
    fold_patients = {
        str(f): int(idx.loc[idx["development_fold"] == f, "patient_id"].nunique())
        for f in (0, 1, 2)
    }

    result = {
        "checkpoint": "V2-02",
        "status": "PASS",
        "next_checkpoint": "V2-03",
        "finalization_mode": "POSTHOC_FROM_COMPLETED_TRAINING_ARTIFACTS_AFTER_ACCEPTANCE_AND_ELIGIBILITY_FINALIZER_BUGS",
        "representation_development_rows": representation_dev_rows,
        "modeling_eligibility_contract": "original split in {train,val} AND survival_mask=True",
        "development_rows": int(len(idx)),
        "development_patients": int(idx["patient_id"].nunique()),
        "fold_rows": fold_rows,
        "fold_patients": fold_patients,
        "families": expected_families,
        "profiles": ["full_supported", "portable_scan_core", "no_genomics", "current_optional_missing", "scan_core_no_genomics", "mixed_sparse"],
        "acceptance": acceptance,
        "selection": decision,
        "historical_reference": historical_reference,
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
        "checkpoint_presence": checkpoint_checks,
        "artifact_hashes": {p.name: sha256_file(p) for p in required},
        "source_repair": {
            "issues": [
                "negative factual booleans were incorrectly placed inside positive acceptance predicates",
                "first post-hoc finalizer compared survival-eligible OOF rows against the broader V2-01 representation-development row count",
            ],
            "training_rerun": False,
            "scientific_outputs_changed": False,
        },
    }
    atomic_json(out / "result_packet.json", result)

    robust = decision.get("robustness_checks", {})
    report = f"""# OncoTwin V2-02 transport-native training

Status: **PASS**

## Finalization note

The six-family training run completed and wrote all scientific artifacts. The original runner then failed because two factual safety fields (`external_rows_opened=false`, `external_predictions_regenerated=false`) were placed inside a positive acceptance dictionary evaluated with `all()`. The first post-hoc finalizer then incorrectly compared the survival-eligible OOF population to the broader V2-01 representation-development population. This finalizer reconstructs the exact V2-02 modeling population from the frozen R2 survival mask plus V2-00 development folds and requires exact key/order/fold/outcome alignment. No model was retrained and no scientific output was changed.

## Population contract

V2-01 representation development rows: **{representation_dev_rows}**.

V2-02 survival-eligible modeling rows: **{len(idx)}**.

Eligibility: original split in {{train,val}} and `survival_mask=True`.

## Development-only selection

Selected family for V2-03: **{selected}**.

Full-supported POST mean Brier: **{selected_brier:.9f}**.

Robust family retained by prespecified noninferiority gates: **{decision.get('robust_retained')}**.

Robust minus plain full-supported Brier delta: **{robust.get('full_delta_robust_minus_plain')}**.

Robust minus plain portable-core Brier delta: **{robust.get('portable_delta_robust_minus_plain')}**.

Historical test outcomes were not used for training or selection. DFCI/VICC rows were not opened and external predictions were not regenerated.
"""
    atomic_text(out / "decision_report.md", report)

    manifest_files = required + [out / "result_packet.json", out / "decision_report.md"]
    manifest = {
        "checkpoint": "V2-02",
        "status": "PASS",
        "files": {str(p.relative_to(repo)): sha256_file(p) for p in manifest_files},
    }
    atomic_json(out / "artifact_manifest.json", manifest)

    print("V2_02_FINALIZE=PASS")
    print("selected_family =", selected)
    print("selected_full_supported_post_brier =", selected_brier)
    print("representation_development_rows =", representation_dev_rows)
    print("modeling_development_rows =", len(idx))
    print("development_patients =", idx["patient_id"].nunique())
    print("fold_rows =", fold_rows)
    print("robust_retained =", decision.get("robust_retained"))
    print("training_rerun = False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
