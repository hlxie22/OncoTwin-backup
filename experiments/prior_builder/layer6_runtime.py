"""Artifact-backed production inference for the frozen Layer 6 ensemble."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.prior_builder.layer6_deployable_artifact import (
    LAYER6_DEPLOYABLE_ARTIFACT_VERSION,
    load_fitted_layer6_member,
    sha256_file,
)
from experiments.prior_builder.layer6_late_response import (
    LAYER6_FEATURE_NAMES,
    layer6_activation,
    layer6_feature_vector,
    predict_volume_from_late_rate,
)
from experiments.prior_builder.layer6_neural_inference import (
    FittedLayer6NeuralModel,
    combine_gaussian_ensemble,
    predict_layer6_distribution,
)


LAYER6_RUNTIME_VERSION = "oncotwin_layer6_runtime_v1"
SUPPORTED_ARTIFACT_STATUS = "trained_internal_research_only"

# These are runtime safety guards, not retrained model parameters.
# They catch catastrophic extrapolation while leaving ordinary
# in-support predictions unchanged.
# With n=252 and population standardization, an actual training
# observation can attain abs(z) up to sqrt(n - 1) ~= 15.843.
MAX_ABS_FEATURE_Z = 16.0
MAX_ABS_TARGET_Z = 12.0
MAX_PREDICTIVE_SD_MULTIPLIER = 12.0


@dataclass(frozen=True)
class LoadedLayer6Ensemble:
    """Validated deployable Layer 6 artifact loaded in memory."""

    artifact_dir: Path
    manifest: Mapping[str, object]
    members: tuple[FittedLayer6NeuralModel, ...]
    feature_names: tuple[str, ...]
    training_scale_days: float
    neural_weight: float
    fallback_weight: float
    fallback_candidate: str
    log_shift: float
    q80: float
    q95: float
    manifest_sha256: str
    member_sha256s: tuple[str, ...]
    limitations: tuple[str, ...]


def _require_mapping(
    value: object,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_sequence(
    value: object,
    name: str,
) -> Sequence[object]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise ValueError(f"{name} must be a list")
    return value


def _require_text(
    value: object,
    name: str,
) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be nonempty")
    return text


def _require_finite(
    value: object,
    name: str,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")

    return number


def _require_positive(
    value: object,
    name: str,
) -> float:
    number = _require_finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _read_json_mapping(
    path: Path,
    name: str,
) -> Mapping[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{name} is not valid JSON: {path}"
        ) from exc

    return _require_mapping(payload, name)


def _resolved_child(
    directory: Path,
    filename: object,
) -> Path:
    name = _require_text(filename, "ensemble member file")
    candidate = (directory / name).resolve()
    root = directory.resolve()

    if candidate.parent != root:
        raise ValueError(
            "Layer 6 member files must be direct children "
            "of the artifact directory"
        )

    return candidate


def _prediction_values(
    prediction: Mapping[str, object],
    name: str,
) -> dict[str, float]:
    values = {
        key: _require_positive(
            prediction.get(key),
            f"{name}.{key}",
        )
        for key in (
            "point_ml",
            "lower_80_ml",
            "upper_80_ml",
            "lower_95_ml",
            "upper_95_ml",
        )
    }

    if not (
        values["lower_95_ml"]
        <= values["lower_80_ml"]
        <= values["point_ml"]
        <= values["upper_80_ml"]
        <= values["upper_95_ml"]
    ):
        raise ValueError(
            f"{name} intervals must be nested around point_ml"
        )

    return values


def load_layer6_ensemble(
    artifact_dir: Path,
) -> LoadedLayer6Ensemble:
    """Load and strictly validate one deployable Layer 6 ensemble."""

    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    checksums_path = artifact_dir / "checksums.json"

    manifest = _read_json_mapping(
        manifest_path,
        "Layer 6 manifest",
    )
    checksums = _read_json_mapping(
        checksums_path,
        "Layer 6 checksums",
    )

    manifest_sha256 = sha256_file(manifest_path)
    expected_manifest_sha256 = _require_text(
        checksums.get("manifest.json"),
        "checksums.manifest.json",
    )

    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Layer 6 manifest hash mismatch: "
            f"expected {expected_manifest_sha256}, "
            f"got {manifest_sha256}"
        )

    if (
        manifest.get("artifact_version")
        != LAYER6_DEPLOYABLE_ARTIFACT_VERSION
    ):
        raise ValueError(
            "Unsupported Layer 6 deployable artifact version"
        )

    if manifest.get("status") != SUPPORTED_ARTIFACT_STATUS:
        raise ValueError(
            "Layer 6 artifact does not have the supported "
            "internal-research status"
        )

    feature_names = tuple(
        str(value)
        for value in _require_sequence(
            manifest.get("feature_names"),
            "feature_names",
        )
    )

    if feature_names != tuple(LAYER6_FEATURE_NAMES):
        raise ValueError(
            "Layer 6 manifest feature order does not match "
            "the frozen feature contract"
        )

    target = _require_mapping(
        manifest.get("target"),
        "target",
    )
    training_scale_days = _require_positive(
        target.get("training_scale_days"),
        "target.training_scale_days",
    )

    network = _require_mapping(
        manifest.get("network"),
        "network",
    )
    input_dim = int(
        _require_positive(
            network.get("input_dim"),
            "network.input_dim",
        )
    )
    ensemble_size = int(
        _require_positive(
            network.get("ensemble_size"),
            "network.ensemble_size",
        )
    )

    if input_dim != len(feature_names):
        raise ValueError(
            "network.input_dim does not match feature_names"
        )

    blend = _require_mapping(
        manifest.get("blend"),
        "blend",
    )
    if blend.get("space") != "volume":
        raise ValueError(
            "Layer 6 runtime supports only ordinary-volume blending"
        )

    neural_weight = _require_finite(
        blend.get("neural_weight"),
        "blend.neural_weight",
    )
    fallback_weight = _require_finite(
        blend.get("fallback_weight"),
        "blend.fallback_weight",
    )

    if neural_weight < 0 or fallback_weight < 0:
        raise ValueError(
            "Layer 6 blend weights must be nonnegative"
        )

    if not math.isclose(
        neural_weight + fallback_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Layer 6 blend weights must sum to one"
        )

    fallback_candidate = _require_text(
        blend.get("fallback_candidate"),
        "blend.fallback_candidate",
    )

    calibration = _require_mapping(
        manifest.get("calibration"),
        "calibration",
    )

    if (
        calibration.get(
            "ordinary_in_sample_residuals_used"
        )
        is not False
    ):
        raise ValueError(
            "Layer 6 calibration must be cross-fitted"
        )

    log_shift = _require_finite(
        calibration.get("log_shift"),
        "calibration.log_shift",
    )
    q80 = _require_positive(
        calibration.get("q80"),
        "calibration.q80",
    )
    q95 = _require_positive(
        calibration.get("q95"),
        "calibration.q95",
    )

    if q95 < q80:
        raise ValueError(
            "calibration.q95 must be at least calibration.q80"
        )

    raw_member_entries = _require_sequence(
        manifest.get("ensemble_members"),
        "ensemble_members",
    )
    member_entries = [
        _require_mapping(
            value,
            f"ensemble_members[{index}]",
        )
        for index, value in enumerate(raw_member_entries)
    ]
    member_entries.sort(
        key=lambda value: int(value["member_index"])
    )

    if len(member_entries) != ensemble_size:
        raise ValueError(
            "ensemble member count does not match network metadata"
        )

    expected_indices = list(range(ensemble_size))
    actual_indices = [
        int(entry["member_index"])
        for entry in member_entries
    ]
    if actual_indices != expected_indices:
        raise ValueError(
            "ensemble member indices must be contiguous from zero"
        )

    members = []
    member_sha256s = []

    for entry in member_entries:
        member_path = _resolved_child(
            artifact_dir,
            entry.get("file"),
        )
        expected_sha256 = _require_text(
            entry.get("sha256"),
            "ensemble member sha256",
        )

        checksum_sha256 = _require_text(
            checksums.get(member_path.name),
            f"checksums.{member_path.name}",
        )
        if checksum_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum metadata disagrees for {member_path.name}"
            )

        members.append(
            load_fitted_layer6_member(
                member_path,
                expected_sha256=expected_sha256,
                expected_feature_names=feature_names,
            )
        )
        member_sha256s.append(expected_sha256)

    limitations_value = manifest.get("limitations", [])
    limitations = tuple(
        str(value)
        for value in _require_sequence(
            limitations_value,
            "limitations",
        )
    )

    return LoadedLayer6Ensemble(
        artifact_dir=artifact_dir,
        manifest=dict(manifest),
        members=tuple(members),
        feature_names=feature_names,
        training_scale_days=training_scale_days,
        neural_weight=neural_weight,
        fallback_weight=fallback_weight,
        fallback_candidate=fallback_candidate,
        log_shift=log_shift,
        q80=q80,
        q95=q95,
        manifest_sha256=manifest_sha256,
        member_sha256s=tuple(member_sha256s),
        limitations=limitations,
    )


def _blend_volume(
    neural_value: float,
    fallback_value: float,
    ensemble: LoadedLayer6Ensemble,
) -> float:
    return max(
        ensemble.neural_weight * float(neural_value)
        + ensemble.fallback_weight * float(fallback_value),
        1e-12,
    )


def _apply_log_calibration(
    value: float,
    log_shift: float,
) -> float:
    return max(
        float(value) * math.exp(float(log_shift)),
        1e-12,
    )


def predict_layer6(
    case: Mapping[str, object],
    static_layer4_prediction: Mapping[str, object],
    fallback_prediction: Mapping[str, object],
    ensemble: LoadedLayer6Ensemble,
) -> dict[str, object]:
    """Run frozen Layer 6 inference or return the D1 fallback unchanged."""

    fallback = _prediction_values(
        fallback_prediction,
        "fallback_prediction",
    )
    activation = layer6_activation(case)

    provenance = {
        "artifact_dir": str(ensemble.artifact_dir),
        "manifest_sha256": ensemble.manifest_sha256,
        "member_sha256s": list(
            ensemble.member_sha256s
        ),
    }

    if not bool(activation["active"]):
        output = dict(fallback_prediction)
        output["layer6"] = {
            "runtime_version": LAYER6_RUNTIME_VERSION,
            "artifact_version": (
                LAYER6_DEPLOYABLE_ARTIFACT_VERSION
            ),
            "status": "inactive_fallback",
            "activation": dict(activation),
            "fallback_used": True,
            "fallback_candidate": (
                ensemble.fallback_candidate
            ),
            "provenance": provenance,
            "limitations": list(
                ensemble.limitations
            ),
        }
        return output

    vector = layer6_feature_vector(
        case,
        static_layer4_prediction,
    )
    feature_matrix = np.asarray(
        [vector],
        dtype=float,
    )

    feature_z_rows = [
        np.abs(
            (
                feature_matrix[0]
                - member.feature_mean
            )
            / member.feature_sd
        )
        for member in ensemble.members
    ]

    max_abs_feature_z_by_name = {
        name: float(
            max(
                row[index]
                for row in feature_z_rows
            )
        )
        for index, name in enumerate(
            ensemble.feature_names
        )
    }
    max_abs_feature_z = max(
        max_abs_feature_z_by_name.values()
    )

    if max_abs_feature_z > MAX_ABS_FEATURE_Z:
        output = dict(fallback_prediction)
        output["layer6"] = {
            "runtime_version": LAYER6_RUNTIME_VERSION,
            "artifact_version": (
                LAYER6_DEPLOYABLE_ARTIFACT_VERSION
            ),
            "status": "active_ood_fallback",
            "activation": dict(activation),
            "fallback_used": True,
            "fallback_candidate": (
                ensemble.fallback_candidate
            ),
            "reason": (
                "Layer 6 inputs are outside the "
                "deployable feature-support guard."
            ),
            "support_diagnostics": {
                "max_abs_feature_z": (
                    max_abs_feature_z
                ),
                "max_abs_feature_z_by_name": (
                    max_abs_feature_z_by_name
                ),
                "max_allowed_abs_feature_z": (
                    MAX_ABS_FEATURE_Z
                ),
            },
            "provenance": provenance,
            "limitations": list(
                ensemble.limitations
            ),
        }
        return output

    member_means = []
    member_sds = []
    member_target_z = []

    for member in ensemble.members:
        mean, sd = predict_layer6_distribution(
            member,
            feature_matrix,
        )
        member_means.append(mean)
        member_sds.append(sd)
        member_target_z.append(
            abs(
                (
                    float(mean[0])
                    - float(member.target_mean)
                )
                / float(member.target_sd)
            )
        )

    combined_mean, combined_sd = (
        combine_gaussian_ensemble(
            member_means,
            member_sds,
        )
    )

    scaled_mean = float(combined_mean[0])
    scaled_sd = max(float(combined_sd[0]), 1e-8)

    median_target_sd = float(
        np.median(
            [
                float(member.target_sd)
                for member in ensemble.members
            ]
        )
    )
    max_abs_target_z = max(member_target_z)
    predictive_sd_multiplier = (
        scaled_sd / median_target_sd
    )

    support_diagnostics = {
        "max_abs_feature_z": (
            max_abs_feature_z
        ),
        "max_abs_feature_z_by_name": (
            max_abs_feature_z_by_name
        ),
        "max_abs_member_target_z": (
            max_abs_target_z
        ),
        "predictive_sd_multiplier": (
            predictive_sd_multiplier
        ),
        "max_allowed_abs_feature_z": (
            MAX_ABS_FEATURE_Z
        ),
        "max_allowed_abs_target_z": (
            MAX_ABS_TARGET_Z
        ),
        "max_allowed_predictive_sd_multiplier": (
            MAX_PREDICTIVE_SD_MULTIPLIER
        ),
    }

    if (
        max_abs_target_z > MAX_ABS_TARGET_Z
        or predictive_sd_multiplier
        > MAX_PREDICTIVE_SD_MULTIPLIER
    ):
        output = dict(fallback_prediction)
        output["layer6"] = {
            "runtime_version": LAYER6_RUNTIME_VERSION,
            "artifact_version": (
                LAYER6_DEPLOYABLE_ARTIFACT_VERSION
            ),
            "status": "active_ood_fallback",
            "activation": dict(activation),
            "fallback_used": True,
            "fallback_candidate": (
                ensemble.fallback_candidate
            ),
            "reason": (
                "Layer 6 neural output is outside "
                "the deployable target-support guard."
            ),
            "support_diagnostics": (
                support_diagnostics
            ),
            "provenance": provenance,
            "limitations": list(
                ensemble.limitations
            ),
        }
        return output

    scale = ensemble.training_scale_days

    def decode(scaled_rate: float) -> float:
        return predict_volume_from_late_rate(
            case,
            float(scaled_rate) / scale,
        )

    neural = {
        "point_ml": decode(scaled_mean),
        "lower_80_ml": decode(
            scaled_mean - ensemble.q80 * scaled_sd
        ),
        "upper_80_ml": decode(
            scaled_mean + ensemble.q80 * scaled_sd
        ),
        "lower_95_ml": decode(
            scaled_mean - ensemble.q95 * scaled_sd
        ),
        "upper_95_ml": decode(
            scaled_mean + ensemble.q95 * scaled_sd
        ),
    }

    blended = {
        key: _blend_volume(
            neural[key],
            fallback[key],
            ensemble,
        )
        for key in fallback
    }

    calibrated = {
        key: _apply_log_calibration(
            value,
            ensemble.log_shift,
        )
        for key, value in blended.items()
    }

    _prediction_values(
        calibrated,
        "layer6_prediction",
    )

    return {
        **calibrated,
        "layer6": {
            "runtime_version": LAYER6_RUNTIME_VERSION,
            "artifact_version": (
                LAYER6_DEPLOYABLE_ARTIFACT_VERSION
            ),
            "status": "active",
            "activation": dict(activation),
            "fallback_used": False,
            "fallback_candidate": (
                ensemble.fallback_candidate
            ),
            "feature_names": list(
                ensemble.feature_names
            ),
            "feature_vector": [
                float(value) for value in vector
            ],
            "support_diagnostics": (
                support_diagnostics
            ),
            "target_distribution": {
                "scaled_rate_mean": scaled_mean,
                "scaled_rate_sd": scaled_sd,
                "training_scale_days": scale,
                "late_rate_mean_per_day": (
                    scaled_mean / scale
                ),
                "late_rate_sd_per_day": (
                    scaled_sd / scale
                ),
            },
            "neural_mechanistic_prediction": neural,
            "fallback_prediction": fallback,
            "blend": {
                "space": "volume",
                "neural_weight": (
                    ensemble.neural_weight
                ),
                "fallback_weight": (
                    ensemble.fallback_weight
                ),
                "precalibration_prediction": blended,
            },
            "calibration": {
                "log_shift": ensemble.log_shift,
                "q80": ensemble.q80,
                "q95": ensemble.q95,
                "method": (
                    "cross_fitted_log_shift_and_"
                    "standardized_rate_residuals"
                ),
            },
            "provenance": provenance,
            "limitations": list(
                ensemble.limitations
            ),
        },
    }
