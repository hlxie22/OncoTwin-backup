#!/usr/bin/env python3
"""Train and serialize the frozen deployable Layer 6 ensemble."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evals.prior_stack.v1_real_data_eval as evalmod
from experiments.prior_builder.layer6_deployable_artifact import (
    LAYER6_DEPLOYABLE_ARTIFACT_VERSION,
    save_fitted_layer6_member,
    sha256_file,
    write_json_atomic,
)
from experiments.prior_builder.layer6_late_response import (
    LAYER6_DECODER_VERSION,
    LAYER6_FEATURE_NAMES,
    late_response_target_per_day,
    layer6_activation,
    layer6_feature_vector,
)
from experiments.prior_builder.layer6_neural_inference import (
    LAYER6_NEURAL_VERSION,
    fit_layer6_neural_model,
)


EXPECTED_CONFIRMATORY_SHA256 = (
    "c910c99eb8b129794faa0db7215cebf669ad922bf6dfe5dd2df09e70146eec48"
)
DEFAULT_CONFIRMATORY_ARTIFACT = Path(
    "artifacts/prior_builder/layer6/locked_confirmatory_evaluation_v1.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/prior_builder/layer6/v1_late_response_ensemble"
)
CALIBRATION_SEED_OFFSET = 900_000
FINAL_ENSEMBLE_SEED_OFFSET = 1_000_000

DISCOVERY_ARTIFACTS = (
    Path("artifacts/prior_builder/layer6/reduced_rate_screen_diagnostic_v1.json"),
    Path("artifacts/prior_builder/layer6/neural_late_rate_screen_diagnostic_v1.json"),
    Path(
        "artifacts/prior_builder/layer6/"
        "neural_late_rate_calibrated_screen_diagnostic_v2.json"
    ),
    Path(
        "artifacts/prior_builder/layer6/"
        "neural_late_rate_blend_screen_diagnostic_v3.json"
    ),
)


def _load_locked_runner() -> ModuleType:
    path = REPO_ROOT / "scripts/evals/run_layer6_locked_confirmatory.py"
    spec = importlib.util.spec_from_file_location(
        "_oncotwin_layer6_locked_confirmatory",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load locked confirmatory runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LOCKED = _load_locked_runner()


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def _verify_locked_artifact(path: Path) -> dict[str, object]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_CONFIRMATORY_SHA256:
        raise RuntimeError(
            "locked confirmatory artifact hash mismatch: "
            f"expected {EXPECTED_CONFIRMATORY_SHA256}, got {actual_sha256}"
        )

    payload = _read_json_object(path)
    if payload.get("analysis_version") != "oncotwin_layer6_locked_confirmatory_v1":
        raise RuntimeError("unexpected locked confirmatory analysis_version")
    if payload.get("status") != "locked_confirmatory_evaluation":
        raise RuntimeError("unexpected locked confirmatory status")

    design = payload.get("design")
    if not isinstance(design, Mapping) or design.get("selection_performed") is not False:
        raise RuntimeError("locked confirmatory artifact must record no selection")

    expected_specification = {
        "neural_config": asdict(LOCKED.LOCKED_CONFIG),
        "fixed_epochs": LOCKED.FIXED_EPOCHS,
        "ensemble_size": LOCKED.ENSEMBLE_SIZE,
        "neural_weight": LOCKED.NEURAL_WEIGHT,
        "baseline_weight": LOCKED.BASELINE_WEIGHT,
        "blend_space": "volume",
        "point_calibration": "inner_oof_log_multiplicative",
        "interval_calibration": (
            "inner_oof_standardized_absolute_rate_residual"
        ),
        "inactive_fallback": evalmod.EARLY_RESPONSE_CANDIDATE,
    }
    if payload.get("locked_specification") != expected_specification:
        raise RuntimeError(
            "current locked constants do not match the preserved confirmatory artifact"
        )

    if tuple(payload.get("feature_names", ())) != tuple(LAYER6_FEATURE_NAMES):
        raise RuntimeError("locked confirmatory feature order mismatch")

    return payload


def _build_training_data() -> dict[str, object]:
    evaluation = evalmod.run_real_data_eval(
        LOCKED.COHORT,
        n_samples=2000,
        seed=LOCKED.BASE_SEED,
        interval_calibrator_path=LOCKED.STATIC_CALIBRATOR,
        allow_candidate_interval_calibrator=True,
        early_response_updater_path=LOCKED.EARLY_UPDATER,
        early_response_interval_calibrator_path=LOCKED.EARLY_CALIBRATOR,
        allow_candidate_early_response_updater=True,
    )

    evaluated_by_id = {
        str(row["case_id"]): row
        for row in evaluation["case_predictions"]
    }
    cases = evalmod.load_real_cohort(
        LOCKED.COHORT,
        allow_demo_data=False,
    )

    records: list[dict[str, object]] = []
    feature_rows: list[Sequence[float]] = []
    target_rows: list[float] = []

    for case in cases:
        case_id = str(case["case_id"])
        evaluated = evaluated_by_id[case_id]
        predictions = evaluated["predictions"]
        activation = layer6_activation(case)

        record: dict[str, object] = {
            "case": case,
            "case_id": case_id,
            "baseline_volume_ml": float(case["baseline_volume_ml"]),
            "early_volume_ml": (
                float(case["early_volume_ml"])
                if activation["active"]
                else None
            ),
            "final_day": int(round(float(case["final_day"]))),
            "observed": float(evaluated["observed_final_volume_ml"]),
            "active": bool(activation["active"]),
            "activation_status": activation["status"],
            "current_point": float(
                predictions["layer4_mri_qc"]["point_ml"]
            ),
            "early_candidate_point": float(
                predictions[evalmod.EARLY_RESPONSE_CANDIDATE]["point_ml"]
            ),
            "early_candidate_lower_80": float(
                predictions[evalmod.EARLY_RESPONSE_CANDIDATE]["lower_80_ml"]
            ),
            "early_candidate_upper_80": float(
                predictions[evalmod.EARLY_RESPONSE_CANDIDATE]["upper_80_ml"]
            ),
            "early_candidate_lower_95": float(
                predictions[evalmod.EARLY_RESPONSE_CANDIDATE]["lower_95_ml"]
            ),
            "early_candidate_upper_95": float(
                predictions[evalmod.EARLY_RESPONSE_CANDIDATE]["upper_95_ml"]
            ),
        }
        records.append(record)

        if not activation["active"]:
            feature_rows.append(
                [float("nan")] * len(LAYER6_FEATURE_NAMES)
            )
            target_rows.append(float("nan"))
            continue

        static_prediction = predictions[evalmod.CALIBRATED_LAYER4_CANDIDATE]
        feature_rows.append(
            layer6_feature_vector(case, static_prediction)
        )
        target_rows.append(
            late_response_target_per_day(
                case,
                float(record["observed"]),
            )
            * LOCKED.TARGET_RATE_REFERENCE_DAYS
        )

    feature_matrix = np.asarray(feature_rows, dtype=float)
    targets = np.asarray(target_rows, dtype=float)
    active_indices = [
        index
        for index, record in enumerate(records)
        if record["active"]
    ]
    inactive_indices = [
        index
        for index, record in enumerate(records)
        if not record["active"]
    ]
    early_candidate_predictions = {
        index: float(record["early_candidate_point"])
        for index, record in enumerate(records)
    }

    if not np.isfinite(feature_matrix[active_indices]).all():
        raise RuntimeError("active Layer 6 feature matrix contains non-finite values")
    if not np.isfinite(targets[active_indices]).all():
        raise RuntimeError("active Layer 6 targets contain non-finite values")

    return {
        "records": records,
        "feature_matrix": feature_matrix,
        "targets": targets,
        "active_indices": active_indices,
        "inactive_indices": inactive_indices,
        "early_candidate_predictions": early_candidate_predictions,
    }


def _artifact_provenance(path: Path) -> dict[str, object]:
    payload = _read_json_object(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "analysis_version": payload.get("analysis_version"),
        "status": payload.get("status"),
    }


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; "
            "pass --overwrite to replace it"
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    return staging


def _commit_output_directory(
    staging: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    staging.replace(output_dir)


def train_deployable_ensemble(
    *,
    confirmatory_artifact: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    confirmatory = _verify_locked_artifact(confirmatory_artifact)

    cohort_sha256 = sha256_file(LOCKED.COHORT)
    if cohort_sha256 != confirmatory.get("cohort_sha256"):
        raise RuntimeError(
            "cohort hash does not match the locked confirmatory artifact"
        )

    data = _build_training_data()
    records = data["records"]
    feature_matrix = data["feature_matrix"]
    targets = data["targets"]
    active_indices = data["active_indices"]
    inactive_indices = data["inactive_indices"]
    early_candidate_predictions = data["early_candidate_predictions"]

    assert isinstance(records, list)
    assert isinstance(feature_matrix, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert isinstance(active_indices, list)
    assert isinstance(inactive_indices, list)
    assert isinstance(early_candidate_predictions, dict)

    expected_dataset = confirmatory.get("dataset")
    actual_dataset = {
        "case_count": len(records),
        "active_count": len(active_indices),
        "inactive_count": len(inactive_indices),
    }
    if expected_dataset != actual_dataset:
        raise RuntimeError(
            "current training-data counts do not match the locked confirmatory artifact"
        )

    calibration_seed = LOCKED.BASE_SEED + CALIBRATION_SEED_OFFSET
    calibration = LOCKED.estimate_training_calibration(
        records,
        feature_matrix,
        targets,
        active_indices,
        early_candidate_predictions,
        seed=calibration_seed,
    )

    calibration_folds = LOCKED.make_folds(
        records,
        active_indices,
        LOCKED.N_INNER_FOLDS,
        calibration_seed,
    )

    staging = _prepare_output_directory(output_dir, overwrite)
    try:
        member_metadata: list[dict[str, object]] = []
        final_seed_base = LOCKED.BASE_SEED + FINAL_ENSEMBLE_SEED_OFFSET

        for member_index in range(LOCKED.ENSEMBLE_SIZE):
            seed = final_seed_base + member_index
            fitted = fit_layer6_neural_model(
                feature_matrix[active_indices],
                targets[active_indices],
                config=LOCKED.LOCKED_CONFIG,
                seed=seed,
                fixed_epochs=LOCKED.FIXED_EPOCHS,
            )
            member_path = staging / f"ensemble_member_{member_index:02d}.pt"
            member_metadata.append(
                save_fitted_layer6_member(
                    fitted,
                    member_path,
                    member_index=member_index,
                    feature_names=LAYER6_FEATURE_NAMES,
                )
            )

        provenance = [
            {
                "path": str(confirmatory_artifact),
                "sha256": EXPECTED_CONFIRMATORY_SHA256,
                "analysis_version": confirmatory.get("analysis_version"),
                "status": confirmatory.get("status"),
            }
        ]
        for path in DISCOVERY_ARTIFACTS:
            if path.is_file():
                provenance.append(_artifact_provenance(path))

        manifest: dict[str, object] = {
            "artifact_version": LAYER6_DEPLOYABLE_ARTIFACT_VERSION,
            "status": "trained_internal_research_only",
            "model_family": "probabilistic_neural_mechanistic_late_response",
            "neural_model_version": LAYER6_NEURAL_VERSION,
            "decoder_version": LAYER6_DECODER_VERSION,
            "cohort": {
                "path": str(LOCKED.COHORT),
                "sha256": cohort_sha256,
            },
            "feature_names": list(LAYER6_FEATURE_NAMES),
            "target": {
                "name": "post_early_mri_log_volume_rate_per_day",
                "training_scale_days": LOCKED.TARGET_RATE_REFERENCE_DAYS,
                "stored_model_units": (
                    "late_log_response_rate_per_day_scaled_by_84_days"
                ),
            },
            "network": {
                "architecture": (
                    "linear residual mean plus one-hidden-layer SiLU MLP "
                    "with heteroscedastic Gaussian standard deviation"
                ),
                "input_dim": len(LAYER6_FEATURE_NAMES),
                "config": asdict(LOCKED.LOCKED_CONFIG),
                "fixed_epochs": LOCKED.FIXED_EPOCHS,
                "ensemble_size": LOCKED.ENSEMBLE_SIZE,
            },
            "blend": {
                "space": "volume",
                "neural_weight": LOCKED.NEURAL_WEIGHT,
                "fallback_weight": LOCKED.BASELINE_WEIGHT,
                "fallback_candidate": evalmod.EARLY_RESPONSE_CANDIDATE,
            },
            "calibration": {
                "method": (
                    "four_fold_cross_fitted_locked_ensemble_on_all_active_cases"
                ),
                "ordinary_in_sample_residuals_used": False,
                "fold_count": LOCKED.N_INNER_FOLDS,
                "ensemble_size_per_fold": LOCKED.ENSEMBLE_SIZE,
                "seed": calibration_seed,
                "log_shift": float(calibration["log_shift"]),
                "q80": float(calibration["q80"]),
                "q95": float(calibration["q95"]),
                "fold_case_ids": [
                    [str(records[index]["case_id"]) for index in fold]
                    for fold in calibration_folds
                ],
            },
            "activation": {
                "function": "layer6_activation",
                "requirements": [
                    "valid early MRI day and volume",
                    "prediction day strictly after early MRI day",
                ],
                "inactive_behavior": (
                    "return the existing D1 early-response candidate unchanged"
                ),
            },
            "training_data": actual_dataset,
            "ensemble_members": member_metadata,
            "provenance": provenance,
            "intended_use": (
                "Internal research forecasting of effective post-early-MRI "
                "tumor log-volume response under observed treatment trajectories."
            ),
            "limitations": [
                "Internal repeated cross-validation only; no external or prospective validation.",
                "Predicts an effective late response rate, not a uniquely identified biological resistance parameter.",
                "Does not establish causal treatment effects or support treatment recommendations.",
                "Early-missing cases use the previously validated D1 fallback.",
                "Residual negative volume bias observed in locked confirmation remains documented and was not retuned.",
            ],
        }

        write_json_atomic(staging / "manifest.json", manifest)
        manifest_sha256 = sha256_file(staging / "manifest.json")
        write_json_atomic(
            staging / "checksums.json",
            {
                "manifest.json": manifest_sha256,
                **{
                    str(member["file"]): str(member["sha256"])
                    for member in member_metadata
                },
            },
        )

        _commit_output_directory(
            staging,
            output_dir,
            overwrite=overwrite,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "calibration": calibration,
        "training_data": actual_dataset,
        "members": member_metadata,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen five-member Layer 6 ensemble, estimate "
            "cross-fitted calibration, and serialize an inspectable artifact."
        )
    )
    parser.add_argument(
        "--confirmatory-artifact",
        type=Path,
        default=DEFAULT_CONFIRMATORY_ARTIFACT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after all training succeeds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_deployable_ensemble(
        confirmatory_artifact=args.confirmatory_artifact,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print("=== LAYER 6 DEPLOYABLE ARTIFACT ===")
    print("output_dir:", result["output_dir"])
    print("manifest:", result["manifest"])
    print("manifest_sha256:", result["manifest_sha256"])
    print("training_data:", json.dumps(result["training_data"], sort_keys=True))
    print("calibration:", json.dumps(result["calibration"], sort_keys=True))
    for member in result["members"]:
        print(
            f"member_{int(member['member_index']):02d}: "
            f"seed={member['seed']} sha256={member['sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
