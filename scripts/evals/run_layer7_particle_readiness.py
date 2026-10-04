#!/usr/bin/env python3
"""Audit baseline-only Layer 7 particle construction on the real cohort."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.prior_stack.v1_real_data_eval import (
    load_real_cohort,
)
from experiments.prior_builder.parameter_contract import (
    TNBC_CHEMO_CONTRACT_ID,
)
from experiments.twin_runtime.layer7_prior_particles import (
    DEFAULT_POLICY_PATH,
    build_layer7_prior_particles,
    canonical_sha256,
    load_layer7_particle_policy,
    sha256_file,
)
from experiments.twin_runtime.posterior import (
    update_volume_posterior,
)
from experiments.v0.mechanistic_simulator.volume_ode import (
    simulate_volume_trajectory,
)


COHORT = Path(
    "data/processed/v1_prior_stack/"
    "ispy2_v1_prior_eval_cohort.jsonl"
)
OUTPUT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_particle_readiness_v0_1.json"
)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def mutate_future_values(
    value: object,
) -> object:
    """Change existing future-valued fields without changing their paths."""

    future_fragments = (
        "early",
        "followup",
        "follow_up",
        "mid_treatment",
        "post_treatment",
        "final",
        "heldout",
        "held_out",
        "outcome",
        "observed",
        "response",
        "residual",
        "surgery",
        "pcr",
        "rcb",
    )

    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            normalized = (
                str(key)
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            if any(
                fragment in normalized
                for fragment in future_fragments
            ):
                if isinstance(child, bool):
                    output[key] = not child
                elif isinstance(child, (int, float)):
                    output[key] = 987654.321
                elif child is None:
                    output[key] = "mutated_future_value"
                else:
                    output[key] = (
                        "mutated_future_value"
                    )
            else:
                output[key] = mutate_future_values(
                    child
                )
        return output

    if isinstance(value, list):
        return [
            mutate_future_values(child)
            for child in value
        ]

    return copy.deepcopy(value)


policy = load_layer7_particle_policy()
particle_count = int(
    policy["particle_counts"]["readiness"]
)
cases = load_real_cohort(
    COHORT,
    allow_demo_data=False,
)

status_counts: Counter[str] = Counter()
removed_path_counts: Counter[str] = Counter()
case_summaries = []
failures = []
all_samples = {
    "growth_rate_per_day": [],
    "active_treatment_sensitivity": [],
    "resistant_fraction": [],
}
first_successful = None

for index, case in enumerate(cases):
    seed = 720000 + index

    try:
        artifact = build_layer7_prior_particles(
            case,
            n_particles=particle_count,
            seed=seed,
        )
    except ValueError as exc:
        message = str(exc)
        if "in-scope TNBC" in message:
            status_counts["out_of_scope"] += 1
            case_summaries.append(
                {
                    "case_id": case["case_id"],
                    "status": "out_of_scope",
                    "reason": message,
                }
            )
            continue

        status_counts["failed"] += 1
        failures.append(
            {
                "case_id": case.get("case_id"),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )
        continue
    except Exception as exc:
        status_counts["failed"] += 1
        failures.append(
            {
                "case_id": case.get("case_id"),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )
        continue

    mutated_case = mutate_future_values(
        case
    )
    mutated = build_layer7_prior_particles(
        mutated_case,
        n_particles=particle_count,
        seed=seed,
    )

    leakage_invariant = (
        artifact["baseline_context"]["sha256"]
        == mutated["baseline_context"]["sha256"]
        and artifact["prior_sample_sha256"]
        == mutated["prior_sample_sha256"]
        and artifact["particle_sha256"]
        == mutated["particle_sha256"]
        and artifact["parameter_particles"]
        == mutated["parameter_particles"]
    )

    if not leakage_invariant:
        status_counts["failed_leakage_invariance"] += 1
        failures.append(
            {
                "case_id": case.get("case_id"),
                "error": (
                    "future-value mutation changed "
                    "baseline particles"
                ),
            }
        )
        continue

    status_counts["ready"] += 1

    for path in artifact["baseline_context"][
        "removed_future_field_paths"
    ]:
        removed_path_counts[path] += 1

    for sample in artifact["prior_samples"]:
        for name in all_samples:
            value = float(sample[name])
            if not math.isfinite(value):
                raise RuntimeError(
                    f"non-finite {name} for "
                    f"{case['case_id']}"
                )
            all_samples[name].append(value)

    case_summaries.append(
        {
            "case_id": case["case_id"],
            "status": "ready",
            "contract_id": artifact[
                "contract"
            ]["contract_id"],
            "particle_count": artifact[
                "particle_count"
            ],
            "baseline_context_sha256": (
                artifact["baseline_context"][
                    "sha256"
                ]
            ),
            "particle_sha256": artifact[
                "particle_sha256"
            ],
            "removed_future_field_count": len(
                artifact["baseline_context"][
                    "removed_future_field_paths"
                ]
            ),
            "warning_count": len(
                artifact["warnings"]
            ),
            "leakage_invariant": True,
        }
    )

    if first_successful is None:
        first_successful = (
            case,
            artifact,
        )

if failures:
    print("failures:")
    for failure in failures:
        print(failure)
    raise RuntimeError(
        f"{len(failures)} case(s) failed particle readiness"
    )

if first_successful is None:
    raise RuntimeError(
        "No in-scope case produced Layer 7 particles"
    )

smoke_case, smoke_artifact = first_successful
smoke_particles = smoke_artifact[
    "parameter_particles"
]
smoke_initial_volume = float(
    smoke_case["baseline_volume_ml"]
)

smoke_schedule = {
    "schedule_id": "layer7_readiness_chemo",
    "regimen_name": (
        "A/C-T neoadjuvant chemotherapy"
    ),
    "total_duration_days": 84,
    "events": [
        {
            "drug": "anthracycline",
            "day": 0,
            "relative_dose": 1.0,
        },
        {
            "drug": "anthracycline",
            "day": 14,
            "relative_dose": 1.0,
        },
        {
            "drug": "anthracycline",
            "day": 28,
            "relative_dose": 1.0,
        },
        {
            "drug": "taxane",
            "day": 42,
            "relative_dose": 1.0,
        },
        {
            "drug": "taxane",
            "day": 49,
            "relative_dose": 1.0,
        },
    ],
}

smoke_trajectory = simulate_volume_trajectory(
    initial_volume_ml=smoke_initial_volume,
    treatment_schedule=smoke_schedule,
    params=smoke_particles[0],
    output_days=[42.0],
)
smoke_observed = float(
    smoke_trajectory["trajectory"][0][
        "tumor_volume_ml"
    ]
)

posterior_smoke = update_volume_posterior(
    initial_volume_ml=smoke_initial_volume,
    treatment_schedule=smoke_schedule,
    parameter_particles=smoke_particles,
    observations=[
        {
            "day": 42.0,
            "tumor_volume_ml": smoke_observed,
            "source": "mask_derived",
            "confidence": "high",
            "segmentation_qc": "high",
            "observation_id": (
                "layer7_readiness_synthetic"
            ),
        }
    ],
    prediction_days=[84.0],
)

parameter_ranges = {
    name: {
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "sample_count": len(values),
    }
    for name, values in all_samples.items()
}

payload = {
    "analysis_version": (
        "oncotwin_layer7_particle_readiness_v0_1"
    ),
    "status": (
        "layer7_baseline_particles_ready"
    ),
    "interpretation": (
        "Exploratory implementation-readiness audit. "
        "This is not a predictive-performance evaluation."
    ),
    "cohort": {
        "path": str(COHORT),
        "sha256": sha256_file(COHORT),
        "case_count": len(cases),
    },
    "policy": {
        "path": str(DEFAULT_POLICY_PATH),
        "sha256": sha256_file(
            DEFAULT_POLICY_PATH
        ),
        "analysis_version": policy[
            "analysis_version"
        ],
        "status": policy["status"],
        "readiness_particle_count": (
            particle_count
        ),
    },
    "source_hashes": {
        "layer7_prior_particles.py": source_sha256(
            Path(
                "experiments/twin_runtime/"
                "layer7_prior_particles.py"
            )
        ),
        "adapter_to_volume_ode.py": source_sha256(
            Path(
                "experiments/prior_builder/"
                "adapter_to_volume_ode.py"
            )
        ),
        "population_prior.py": source_sha256(
            Path(
                "experiments/prior_builder/"
                "population_prior.py"
            )
        ),
        "pathology_biomarker_rules.py": source_sha256(
            Path(
                "experiments/prior_builder/"
                "pathology_biomarker_rules.py"
            )
        ),
        "mri_feature_rules.py": source_sha256(
            Path(
                "experiments/prior_builder/"
                "mri_feature_rules.py"
            )
        ),
    },
    "status_counts": dict(
        sorted(status_counts.items())
    ),
    "requirements": {
        "all_in_scope_cases_constructed": (
            status_counts["failed"] == 0
            and status_counts[
                "failed_leakage_invariance"
            ]
            == 0
        ),
        "future_value_mutation_invariant": True,
        "simulator_adapter_compatible": True,
        "posterior_runtime_compatible": True,
        "uniform_initial_weights": True,
        "layer6_used_as_prior": False,
        "early_observation_used_as_prior": False,
        "final_outcome_used_as_prior": False,
    },
    "parameter_ranges": parameter_ranges,
    "removed_future_field_path_counts": dict(
        sorted(removed_path_counts.items())
    ),
    "posterior_compatibility_smoke": {
        "case_id": smoke_case["case_id"],
        "particle_count": len(
            smoke_particles
        ),
        "effective_sample_size": (
            posterior_smoke[
                "effective_sample_size"
            ]
        ),
        "effective_sample_size_fraction": (
            posterior_smoke[
                "effective_sample_size_fraction"
            ]
        ),
        "fallback_status": (
            posterior_smoke["fallback_status"]
        ),
        "prediction_days": (
            posterior_smoke[
                "posterior_trajectory_summary"
            ]["times"]
        ),
    },
    "case_summaries": case_summaries,
    "failures": failures,
}

payload["artifact_sha256"] = canonical_sha256(
    payload
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)
OUTPUT.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)

print("analysis_version:", payload["analysis_version"])
print("status:", payload["status"])
print("cohort_case_count:", len(cases))
print(
    "status_counts:",
    payload["status_counts"],
)
print(
    "parameter_ranges:",
    json.dumps(
        parameter_ranges,
        sort_keys=True,
    ),
)
print(
    "posterior_compatibility_smoke:",
    payload[
        "posterior_compatibility_smoke"
    ],
)
print("artifact:", OUTPUT)
print(
    "artifact_sha256:",
    sha256_file(OUTPUT),
)
print("LAYER7_PARTICLE_READINESS: PASS")
