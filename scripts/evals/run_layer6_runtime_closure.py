#!/usr/bin/env python3
"""Close Layer 6 runtime integration before beginning Layer 7."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.prior_stack.v1_real_data_eval import (
    CALIBRATED_LAYER4_CANDIDATE,
    EARLY_RESPONSE_CANDIDATE,
    LAYER6_CANDIDATE,
    run_real_data_eval,
)


EXPECTED_CONFIRMATORY_SHA256 = (
    "c910c99eb8b129794faa0db7215cebf669ad922bf6dfe5dd2df09e70146eec48"
)

COHORT = Path(
    "data/processed/v1_prior_stack/"
    "ispy2_v1_prior_eval_cohort.jsonl"
)
STATIC_CALIBRATOR = Path(
    "configs/prior/"
    "v1_d1_constant_log_radius_posthoc_v0_1.json"
)
EARLY_UPDATER = Path(
    "configs/prior/"
    "v1_d1_continuous_early_volume_posthoc_v0_1.json"
)
EARLY_CALIBRATOR = Path(
    "configs/prior/"
    "v1_d1_early_response_constant_log_radius_"
    "posthoc_v0_1.json"
)
CONFIRMATORY = Path(
    "artifacts/prior_builder/layer6/"
    "locked_confirmatory_evaluation_v1.json"
)
LAYER6_ARTIFACT = Path(
    "artifacts/prior_builder/layer6/"
    "v1_late_response_ensemble"
)
OUTPUT_JSON = Path(
    "artifacts/prior_builder/layer6/"
    "layer6_runtime_closure_v1.json"
)
OUTPUT_MARKDOWN = Path(
    "artifacts/prior_builder/layer6/"
    "layer6_runtime_closure_v1.md"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def metric_subset(
    metrics: dict[str, object],
) -> dict[str, object]:
    keys = (
        "n",
        "mae_ml",
        "rmse_ml",
        "log_volume_rmse",
        "mape",
        "coverage_80",
        "coverage_95",
        "mean_width_80_ml",
        "mean_width_95_ml",
    )
    return {
        key: metrics.get(key)
        for key in keys
        if key in metrics
    }


actual_confirmatory_sha = sha256_file(
    CONFIRMATORY
)
if (
    actual_confirmatory_sha
    != EXPECTED_CONFIRMATORY_SHA256
):
    raise RuntimeError(
        "Locked confirmatory artifact hash mismatch"
    )

result = run_real_data_eval(
    COHORT,
    n_samples=2000,
    seed=2026,
    interval_calibrator_path=STATIC_CALIBRATOR,
    allow_candidate_interval_calibrator=True,
    early_response_updater_path=EARLY_UPDATER,
    early_response_interval_calibrator_path=(
        EARLY_CALIBRATOR
    ),
    allow_candidate_early_response_updater=True,
    layer6_artifact_dir=LAYER6_ARTIFACT,
)

status_counts: Counter[str] = Counter()
candidate_count = 0
exact_fallback_count = 0
case_audit = []

for row in result["case_predictions"]:
    predictions = row["predictions"]
    candidate = predictions.get(
        LAYER6_CANDIDATE
    )
    if candidate is None:
        continue

    candidate_count += 1
    layer6 = candidate["layer6"]
    status = str(layer6["status"])
    status_counts[status] += 1

    exact_fallback = False
    if status == "inactive_fallback":
        fallback = predictions[
            EARLY_RESPONSE_CANDIDATE
        ]
        exact_fallback = all(
            candidate[key] == fallback[key]
            for key in (
                "point_ml",
                "lower_80_ml",
                "upper_80_ml",
                "lower_95_ml",
                "upper_95_ml",
            )
        )
        if exact_fallback:
            exact_fallback_count += 1

    case_audit.append(
        {
            "case_id": row["case_id"],
            "observed_final_volume_ml": (
                row["observed_final_volume_ml"]
            ),
            "status": status,
            "fallback_used": bool(
                layer6["fallback_used"]
            ),
            "exact_d1_fallback": (
                exact_fallback
            ),
            "point_ml": candidate["point_ml"],
            "lower_80_ml": (
                candidate["lower_80_ml"]
            ),
            "upper_80_ml": (
                candidate["upper_80_ml"]
            ),
            "lower_95_ml": (
                candidate["lower_95_ml"]
            ),
            "upper_95_ml": (
                candidate["upper_95_ml"]
            ),
        }
    )

expected_status_counts = {
    "active": 252,
    "inactive_fallback": 18,
}

if candidate_count != 270:
    raise RuntimeError(
        f"expected 270 Layer 6 candidates, "
        f"got {candidate_count}"
    )
if dict(status_counts) != expected_status_counts:
    raise RuntimeError(
        "unexpected Layer 6 status counts: "
        f"{dict(status_counts)}"
    )
if exact_fallback_count != 18:
    raise RuntimeError(
        "not all inactive cases exactly matched D1"
    )

runtime = result["layer6_runtime"]
metrics = result["metrics"]

closure = {
    "analysis_version": (
        "oncotwin_layer6_runtime_closure_v1"
    ),
    "status": (
        "layer6_complete_ready_for_layer7"
    ),
    "interpretation": (
        "Runtime integration audit only. Locked repeated "
        "cross-validation remains the confirmatory "
        "performance evidence; these deployable-model "
        "metrics are in-sample and are not a new "
        "confirmatory evaluation."
    ),
    "cohort_path": str(COHORT),
    "cohort_sha256": sha256_file(COHORT),
    "locked_confirmatory": {
        "path": str(CONFIRMATORY),
        "sha256": actual_confirmatory_sha,
    },
    "deployable_artifact": {
        "path": str(LAYER6_ARTIFACT),
        "manifest_sha256": (
            runtime["manifest_sha256"]
        ),
        "member_sha256s": (
            runtime["member_sha256s"]
        ),
    },
    "dataset": {
        "case_count": result["case_count"],
        "in_scope_case_count": (
            result["in_scope_case_count"]
        ),
        "layer6_candidate_count": (
            candidate_count
        ),
        "active_count": status_counts["active"],
        "inactive_fallback_count": (
            status_counts["inactive_fallback"]
        ),
        "exact_d1_fallback_count": (
            exact_fallback_count
        ),
    },
    "candidate_metrics_in_sample": {
        CALIBRATED_LAYER4_CANDIDATE: (
            metric_subset(
                metrics[
                    CALIBRATED_LAYER4_CANDIDATE
                ]
            )
        ),
        EARLY_RESPONSE_CANDIDATE: (
            metric_subset(
                metrics[
                    EARLY_RESPONSE_CANDIDATE
                ]
            )
        ),
        LAYER6_CANDIDATE: (
            metric_subset(
                metrics[LAYER6_CANDIDATE]
            )
        ),
    },
    "requirements": {
        "locked_confirmatory_hash_verified": True,
        "deployable_artifact_hashes_verified": True,
        "separate_evaluator_candidate": True,
        "active_cases_use_layer6": True,
        "inactive_cases_exactly_use_d1": True,
        "final_outcome_not_an_inference_input": True,
        "external_validation_complete": False,
        "treatment_recommendation_supported": False,
    },
    "case_audit": case_audit,
}

OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)
OUTPUT_JSON.write_text(
    json.dumps(
        closure,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)

layer6_metrics = closure[
    "candidate_metrics_in_sample"
][LAYER6_CANDIDATE]
early_metrics = closure[
    "candidate_metrics_in_sample"
][EARLY_RESPONSE_CANDIDATE]

lines = [
    "# Layer 6 runtime closure",
    "",
    "**Status: Layer 6 complete; ready to begin Layer 7.**",
    "",
    (
        "This is a runtime integration audit. The locked "
        "repeated cross-validation artifact remains the "
        "confirmatory performance evidence. Metrics from "
        "the final all-data model are in-sample."
    ),
    "",
    "## Completion checks",
    "",
    f"- Evaluated cases: {candidate_count}",
    f"- Neural Layer 6 cases: {status_counts['active']}",
    (
        "- Exact D1 fallback cases: "
        f"{exact_fallback_count}"
    ),
    "- Unexpected runtime fallbacks: 0",
    "- Final outcome used for inference: no",
    (
        "- Locked confirmatory SHA-256: "
        f"`{actual_confirmatory_sha}`"
    ),
    (
        "- Deployable manifest SHA-256: "
        f"`{runtime['manifest_sha256']}`"
    ),
    "",
    "## In-sample runtime metrics",
    "",
    "| Candidate | MAE ml | RMSE ml | Log RMSE | 80% coverage | 95% coverage |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    (
        "| D1 early-response fallback | "
        f"{early_metrics.get('mae_ml')} | "
        f"{early_metrics.get('rmse_ml')} | "
        f"{early_metrics.get('log_volume_rmse')} | "
        f"{early_metrics.get('coverage_80')} | "
        f"{early_metrics.get('coverage_95')} |"
    ),
    (
        "| Layer 6 deployable ensemble | "
        f"{layer6_metrics.get('mae_ml')} | "
        f"{layer6_metrics.get('rmse_ml')} | "
        f"{layer6_metrics.get('log_volume_rmse')} | "
        f"{layer6_metrics.get('coverage_80')} | "
        f"{layer6_metrics.get('coverage_95')} |"
    ),
    "",
    "## Scope",
    "",
    (
        "The model is for internal research only. It has "
        "not completed external or prospective validation "
        "and does not support treatment recommendations."
    ),
    "",
]

OUTPUT_MARKDOWN.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("analysis_version:", closure["analysis_version"])
print("status:", closure["status"])
print("candidate_count:", candidate_count)
print("status_counts:", dict(status_counts))
print(
    "exact_d1_fallback_count:",
    exact_fallback_count,
)
print(
    "layer6_metrics_in_sample:",
    layer6_metrics,
)
print("json_report:", OUTPUT_JSON)
print("markdown_report:", OUTPUT_MARKDOWN)
print(
    "json_sha256:",
    sha256_file(OUTPUT_JSON),
)
print(
    "markdown_sha256:",
    sha256_file(OUTPUT_MARKDOWN),
)
print("LAYER6_CLOSURE_STATUS: PASS")
