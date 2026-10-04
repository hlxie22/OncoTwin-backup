from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.evals.audit_layer8_temporal_landmarks import (
    run_audit,
    write_outputs,
)


def _write_jsonl(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_table(
    path: Path,
    rows: list[dict[str, object]],
    *,
    delimiter: str,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter=delimiter,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_audit_separates_raw_t2_from_usable_volume(
    tmp_path: Path,
) -> None:
    cohort = tmp_path / "cohort.jsonl"
    manifest = tmp_path / "manifest.tsv"
    clinical = tmp_path / "clinical.csv"
    features = tmp_path / "features.csv"
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"

    _write_jsonl(
        cohort,
        [
            {
                "case_id": "CASE-A",
                "baseline_day": 0.0,
                "baseline_volume_ml": 10.0,
                "early_day": 21.0,
                "early_volume_ml": 7.0,
                "final_day": 140.0,
                "final_volume_ml": 2.0,
            },
            {
                "case_id": "CASE-B",
                "baseline_day": 0.0,
                "baseline_volume_ml": 12.0,
                "early_day": 21.0,
                "early_volume_ml": 9.0,
                "final_day": 140.0,
                "final_volume_ml": 3.0,
                "measurements": [
                    {"day": 0.0, "tumor_volume_ml": 12.0},
                    {"day": 21.0, "tumor_volume_ml": 9.0},
                    {"day": 84.0, "tumor_volume_ml": 5.0},
                    {"day": 140.0, "tumor_volume_ml": 3.0},
                ],
            },
        ],
    )

    _write_table(
        manifest,
        [
            {
                "case_id": "CASE-A",
                "timepoint": "T0",
                "series_description": "DCE",
            },
            {
                "case_id": "CASE-A",
                "timepoint": "T1",
                "series_description": "DCE",
            },
            {
                "case_id": "CASE-A",
                "timepoint": "T2",
                "series_description": "DCE",
            },
            {
                "case_id": "CASE-A",
                "timepoint": "T3",
                "series_description": "DCE",
            },
            {
                "case_id": "CASE-B",
                "timepoint": "T0",
                "series_description": "DCE",
            },
            {
                "case_id": "CASE-B",
                "timepoint": "T1",
                "series_description": "DCE",
            },
            {
                "case_id": "CASE-B",
                "timepoint": "T3",
                "series_description": "DCE",
            },
        ],
        delimiter="\t",
    )

    _write_table(
        clinical,
        [
            {
                "case_id": "CASE-A",
                "visit_label": "inter-regimen",
            },
            {
                "case_id": "CASE-B",
                "visit_label": "presurgery",
            },
        ],
        delimiter=",",
    )

    _write_table(
        features,
        [
            {
                "case_id": "CASE-A",
                "timepoint": "T2",
                "tumor_volume_ml": 4.5,
            },
            {
                "case_id": "CASE-B",
                "timepoint": "T3",
                "tumor_volume_ml": 3.0,
            },
        ],
        delimiter=",",
    )

    result = run_audit(
        cohort_path=cohort,
        manifest_path=manifest,
        clinical_path=clinical,
        longitudinal_features_path=features,
    )

    assert (
        result["cohort"][
            "layer6_eligible_case_count"
        ]
        == 2
    )
    assert (
        result["cohort"][
            "strict_post_early_pre_final_volume_case_count"
        ]
        == 1
    )
    assert (
        result["derived_manifest"][
            "layer6_eligible_explicit_t2_case_count"
        ]
        == 1
    )
    assert (
        result["longitudinal_features"][
            "usable_post_early_pre_final_volume_case_count"
        ]
        == 1
    )
    assert (
        result["landmark_assessment"][
            "usable_post_early_pre_final_volume_case_count"
        ]
        == 2
    )
    assert (
        result["landmark_assessment"][
            "usable_new_observation_status"
        ]
        == "established"
    )
    assert (
        result["landmark_assessment"]["decision"]
        == (
            "proceed_to_frozen_layer6_t1_to_t3_baseline_"
            "and_t2_incremental_value_screen"
        )
    )

    write_outputs(
        result,
        output_json=output_json,
        output_md=output_md,
    )

    assert output_json.stat().st_size > 0
    assert output_md.stat().st_size > 0
    assert json.loads(
        output_json.read_text(encoding="utf-8")
    )["status"] == "complete"


def test_raw_t2_without_processed_volume_requires_extraction(
    tmp_path: Path,
) -> None:
    cohort = tmp_path / "cohort.jsonl"
    manifest = tmp_path / "manifest.tsv"
    missing_features = tmp_path / "missing.csv"

    _write_jsonl(
        cohort,
        [
            {
                "case_id": "CASE-A",
                "baseline_day": 0.0,
                "baseline_volume_ml": 10.0,
                "early_day": 21.0,
                "early_volume_ml": 7.0,
                "final_day": 140.0,
                "final_volume_ml": 2.0,
            },
        ],
    )

    _write_table(
        manifest,
        [
            {
                "case_id": "CASE-A",
                "timepoint": "T0",
            },
            {
                "case_id": "CASE-A",
                "timepoint": "T1",
            },
            {
                "case_id": "CASE-A",
                "timepoint": "T2",
            },
            {
                "case_id": "CASE-A",
                "timepoint": "T3",
            },
        ],
        delimiter="\t",
    )

    result = run_audit(
        cohort_path=cohort,
        manifest_path=manifest,
        clinical_path=None,
        longitudinal_features_path=missing_features,
    )

    assert (
        result["landmark_assessment"][
            "usable_new_observation_status"
        ]
        == "not_established_missing_processed_t2_volume"
    )
    assert (
        result["landmark_assessment"]["decision"]
        == "build_t2_volume_extraction_and_case_crosswalk"
    )
    assert (
        result["requirements"][
            "frozen_layer6_model_modified"
        ]
        is False
    )
