"""Baseline-only mechanistic prior particles for Layer 7.

This module constructs simulator-ready particles using only information
available at baseline. Early-treatment measurements and held-out outcomes are
removed recursively before any prior rule is evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from experiments.prior_builder.adapter_to_volume_ode import (
    adapt_v1_prior_samples_to_volume_ode,
)
from experiments.prior_builder.bounds import (
    validate_parameter_bounds,
)
from experiments.prior_builder.mri_feature_rules import (
    apply_mri_feature_rules,
    sample_mri_feature_prior,
)
from experiments.prior_builder.parameter_contract import (
    FIXED_PARAMETER_NAMES,
    TNBC_CHEMO_CONTRACT_ID,
    resolve_parameter_contract,
)
from experiments.prior_builder.pathology_biomarker_rules import (
    apply_pathology_biomarker_rules,
)
from experiments.prior_builder.population_prior import (
    resolve_population_prior,
)


LAYER7_PARTICLE_BUILDER_VERSION = (
    "oncotwin_layer7_baseline_particles_v0_1"
)
DEFAULT_POLICY_PATH = Path(
    "configs/prior/"
    "layer7_baseline_particle_policy_v0_1.json"
)

_BASELINE_TOP_LEVEL_FIELDS = (
    "subtype",
    "disease_context",
    "cancer_subtype",
    "treatment_context",
    "treatment_regimen",
    "regimen_name",
    "schedule_type",
    "er_status",
    "pr_status",
    "her2_status",
    "grade",
    "ki67_percent",
    "brca_status",
    "hrd_status",
    "baseline_day",
    "baseline_volume_ml",
)

_REQUIRED_POLICY_KEYS = {
    "analysis_version",
    "status",
    "scope",
    "particle_counts",
    "future_field_fragments",
    "fixed_parameters",
    "rules",
    "sampling",
    "limitations",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
        default=str,
    )
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_layer7_particle_policy(
    path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Layer 7 particle policy must be a JSON object"
        )

    missing = sorted(
        _REQUIRED_POLICY_KEYS - set(payload)
    )
    if missing:
        raise ValueError(
            "Layer 7 particle policy is missing: "
            + ", ".join(missing)
        )

    fixed = payload["fixed_parameters"]
    if not isinstance(fixed, Mapping):
        raise ValueError(
            "fixed_parameters must be a JSON object"
        )

    expected_fixed = set(FIXED_PARAMETER_NAMES)
    actual_fixed = set(fixed)
    if actual_fixed != expected_fixed:
        raise ValueError(
            "fixed_parameters must exactly match the "
            "Layer 0 contract; expected "
            f"{sorted(expected_fixed)}, got "
            f"{sorted(actual_fixed)}"
        )

    fragments = payload["future_field_fragments"]
    if (
        not isinstance(fragments, Sequence)
        or isinstance(fragments, (str, bytes))
        or not fragments
    ):
        raise ValueError(
            "future_field_fragments must be a nonempty list"
        )

    return dict(payload)


def _normalized_key(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _is_forbidden_key(
    key: object,
    forbidden_fragments: Sequence[str],
) -> bool:
    normalized = _normalized_key(key)
    return any(
        fragment in normalized
        for fragment in forbidden_fragments
    )


def _sanitize_value(
    value: object,
    *,
    forbidden_fragments: Sequence[str],
    path: str,
) -> tuple[object, list[str]]:
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        removed: list[str] = []

        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = (
                f"{path}.{key}"
                if path
                else key
            )

            if _is_forbidden_key(
                key,
                forbidden_fragments,
            ):
                removed.append(child_path)
                continue

            clean_child, child_removed = (
                _sanitize_value(
                    child,
                    forbidden_fragments=(
                        forbidden_fragments
                    ),
                    path=child_path,
                )
            )
            cleaned[key] = clean_child
            removed.extend(child_removed)

        return cleaned, removed

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
    ):
        cleaned_items = []
        removed: list[str] = []

        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            clean_child, child_removed = (
                _sanitize_value(
                    child,
                    forbidden_fragments=(
                        forbidden_fragments
                    ),
                    path=child_path,
                )
            )
            cleaned_items.append(clean_child)
            removed.extend(child_removed)

        return cleaned_items, removed

    if isinstance(value, float) and not math.isfinite(value):
        return None, []

    return value, []


def _assert_no_forbidden_keys(
    value: object,
    *,
    forbidden_fragments: Sequence[str],
    path: str = "$",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"

            if _is_forbidden_key(
                key,
                forbidden_fragments,
            ):
                raise RuntimeError(
                    "Future-derived field survived baseline "
                    f"sanitization: {child_path}"
                )

            _assert_no_forbidden_keys(
                child,
                forbidden_fragments=(
                    forbidden_fragments
                ),
                path=child_path,
            )

    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
    ):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(
                child,
                forbidden_fragments=(
                    forbidden_fragments
                ),
                path=f"{path}[{index}]",
            )


def baseline_only_context(
    case: Mapping[str, object],
    *,
    policy: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Return baseline-only rule context and removed field paths."""

    raw_context = case.get("context", {})
    if not isinstance(raw_context, Mapping):
        raise ValueError(
            "case.context must be a mapping"
        )

    fragments = tuple(
        _normalized_key(value)
        for value in policy[
            "future_field_fragments"
        ]
    )

    cleaned_value, removed = _sanitize_value(
        raw_context,
        forbidden_fragments=fragments,
        path="$.context",
    )
    cleaned = dict(cleaned_value)

    for field in _BASELINE_TOP_LEVEL_FIELDS:
        if field not in case:
            continue

        if _is_forbidden_key(field, fragments):
            removed.append(f"$.{field}")
            continue

        if (
            field not in cleaned
            and case[field] is not None
        ):
            cleaned[field] = case[field]

    # Baseline volume is allowed as an MRI-volume input. These aliases make
    # the baseline measurement visible to the existing Layer 4 feature rules
    # without exposing any follow-up measurement.
    baseline_volume = case.get(
        "baseline_volume_ml"
    )
    if baseline_volume not in (None, ""):
        cleaned.setdefault(
            "baseline_volume_ml",
            baseline_volume,
        )
        cleaned.setdefault(
            "volume_ml",
            baseline_volume,
        )
        cleaned.setdefault(
            "tumor_volume_ml",
            baseline_volume,
        )

    # Record top-level future fields as explicitly excluded provenance.
    for key in case:
        if _is_forbidden_key(key, fragments):
            removed.append(f"$.{key}")

    _assert_no_forbidden_keys(
        cleaned,
        forbidden_fragments=fragments,
    )

    return cleaned, tuple(sorted(set(removed)))


def _positive_integer(
    value: object,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive integer"
        )

    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer"
        ) from exc

    if number < 1:
        raise ValueError(
            f"{name} must be a positive integer"
        )

    return number


def _rejection_parameter(
    error: ValueError,
) -> str:
    message = str(error)
    if "=" in message:
        parameter = message.split("=", 1)[0].strip()
        if parameter:
            return parameter
    return "unknown"


def _sample_bounded_layer4_prior(
    layer4,
    *,
    n_samples: int,
    seed: int,
    policy: Mapping[str, object],
) -> tuple[
    list[dict[str, float]],
    dict[str, object],
]:
    """Sample the correlated prior conditional on hard parameter bounds.

    Values outside the existing Layer 1 hard bounds are rejected, never
    clipped. This preserves a continuous truncated prior and avoids creating
    artificial point mass at a simulator boundary.
    """

    sampling = policy.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(
            "policy.sampling must be a JSON object"
        )

    if (
        sampling.get("hard_bound_policy")
        != "deterministic_rejection_sampling"
    ):
        raise ValueError(
            "Unsupported Layer 7 hard-bound policy"
        )

    oversampling_factor = _positive_integer(
        sampling.get("oversampling_factor"),
        "sampling.oversampling_factor",
    )
    minimum_batch_size = _positive_integer(
        sampling.get("minimum_batch_size"),
        "sampling.minimum_batch_size",
    )
    max_rounds = _positive_integer(
        sampling.get("max_rounds"),
        "sampling.max_rounds",
    )

    accepted: list[dict[str, float]] = []
    total_examined = 0
    rejected_by_parameter: dict[str, int] = {}
    rounds_used = 0

    for round_index in range(max_rounds):
        remaining = n_samples - len(accepted)
        if remaining <= 0:
            break

        rounds_used = round_index + 1
        batch_size = max(
            minimum_batch_size,
            remaining * oversampling_factor,
        )

        # Each retry uses a deterministic, non-overlapping seed stream.
        round_seed = (
            int(seed)
            + round_index * 1_000_003
        )

        batch = sample_mri_feature_prior(
            layer4,
            n_samples=batch_size,
            seed=round_seed,
        ).samples

        for raw_sample in batch:
            sample = {
                str(name): float(value)
                for name, value
                in raw_sample.items()
            }
            total_examined += 1

            try:
                validate_parameter_bounds(sample)
            except ValueError as exc:
                parameter = _rejection_parameter(
                    exc
                )
                rejected_by_parameter[parameter] = (
                    rejected_by_parameter.get(
                        parameter,
                        0,
                    )
                    + 1
                )
                continue

            accepted.append(sample)

            if len(accepted) == n_samples:
                break

    if len(accepted) != n_samples:
        raise RuntimeError(
            "Layer 7 rejection sampling exhausted "
            f"{max_rounds} rounds: accepted "
            f"{len(accepted)} of {n_samples} requested "
            f"after examining {total_examined} draws"
        )

    rejected_count = (
        total_examined - len(accepted)
    )

    diagnostics = {
        "method": (
            "deterministic_rejection_sampling"
        ),
        "requested_count": n_samples,
        "accepted_count": len(accepted),
        "examined_count": total_examined,
        "rejected_count": rejected_count,
        "rejection_fraction": (
            rejected_count / total_examined
        ),
        "rejected_by_parameter": dict(
            sorted(
                rejected_by_parameter.items()
            )
        ),
        "rounds_used": rounds_used,
        "oversampling_factor": (
            oversampling_factor
        ),
        "minimum_batch_size": (
            minimum_batch_size
        ),
        "max_rounds": max_rounds,
        "clipping_performed": False,
        "hard_bounds_source": (
            "experiments.prior_builder.bounds."
            "PARAMETER_BOUNDS"
        ),
    }

    return accepted, diagnostics


def build_layer7_prior_particles(
    case: Mapping[str, object],
    *,
    n_particles: int,
    seed: int,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, object]:
    """Construct deterministic baseline-only simulator particles."""

    if n_particles < 1:
        raise ValueError(
            "n_particles must be positive"
        )
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    policy = load_layer7_particle_policy(
        policy_path
    )
    context, removed_paths = (
        baseline_only_context(
            case,
            policy=policy,
        )
    )

    contract = resolve_parameter_contract(
        context
    )
    if (
        contract.contract_id
        != TNBC_CHEMO_CONTRACT_ID
    ):
        raise ValueError(
            "Layer 7 baseline particles require the "
            "in-scope TNBC chemotherapy contract; got "
            f"{contract.contract_id}"
        )

    layer2 = resolve_population_prior(
        contract
    )
    layer3 = apply_pathology_biomarker_rules(
        layer2,
        context,
    )
    layer4 = apply_mri_feature_rules(
        layer3,
        context,
    )

    sampled, sampling_diagnostics = (
        _sample_bounded_layer4_prior(
            layer4,
            n_samples=n_particles,
            seed=int(seed),
            policy=policy,
        )
    )

    fixed_parameters = dict(
        policy["fixed_parameters"]
    )
    case_id = str(
        case.get("case_id", "unknown_case")
    )
    prefix = f"{case_id}_layer7_prior"

    adapted = adapt_v1_prior_samples_to_volume_ode(
        contract,
        sampled,
        fixed_parameters,
        particle_id_prefix=prefix,
    )

    particles = [
        dict(result.parameters)
        for result in adapted
    ]

    adapter_warnings = sorted(
        {
            warning
            for result in adapted
            for warning in result.warnings
        }
    )
    prior_warnings = sorted(
        {
            *contract.warnings,
            *layer2.warnings,
            *layer3.warnings,
            *layer4.warnings,
            *adapter_warnings,
        }
    )

    prior_samples = [
        {
            "particle_id": particles[index][
                "particle_id"
            ],
            **{
                key: float(value)
                for key, value in sample.items()
            },
        }
        for index, sample in enumerate(sampled)
    ]

    particle_sha256 = canonical_sha256(
        particles
    )
    prior_sample_sha256 = canonical_sha256(
        prior_samples
    )

    payload: dict[str, object] = {
        "artifact_version": (
            LAYER7_PARTICLE_BUILDER_VERSION
        ),
        "status": (
            "baseline_prior_particles_exploratory"
        ),
        "case_id": case_id,
        "seed": int(seed),
        "particle_count": n_particles,
        "contract": {
            "contract_id": contract.contract_id,
            "base_group": contract.base_group,
            "learnable_parameters": list(
                contract.learnable_parameters
            ),
            "fixed_parameters": list(
                contract.fixed_parameters
            ),
            "active_treatment_drugs": list(
                contract.active_treatment_drugs
            ),
        },
        "baseline_context": {
            "sha256": canonical_sha256(context),
            "top_level_fields": sorted(
                context.keys()
            ),
            "removed_future_field_paths": list(
                removed_paths
            ),
            "future_information_used": False,
        },
        "policy": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
            "analysis_version": policy[
                "analysis_version"
            ],
            "status": policy["status"],
        },
        "layer_contributions": [
            layer2.layer_contribution(),
            layer3.layer_contribution(),
            layer4.layer_contribution(),
        ],
        "sampling": sampling_diagnostics,
        "prior_samples": prior_samples,
        "parameter_particles": particles,
        "prior_sample_sha256": (
            prior_sample_sha256
        ),
        "particle_sha256": particle_sha256,
        "initial_particle_weight": (
            1.0 / n_particles
        ),
        "warnings": prior_warnings,
        "limitations": list(
            policy["limitations"]
        ),
    }

    payload["artifact_sha256"] = (
        canonical_sha256(payload)
    )

    return payload


def write_layer7_prior_particle_artifact(
    path: Path,
    artifact: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
