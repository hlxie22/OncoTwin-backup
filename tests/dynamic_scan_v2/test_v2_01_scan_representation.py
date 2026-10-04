from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dynamic_scan_v2.paired_scan_encoder import PairedScanEncoder
from dynamic_scan_v2.scan_representation import (
    REGIONS,
    _bool_or_missing,
    build_paired_scan_representation,
)


def scan_row(
    pid: str,
    eid: str,
    start: int,
    end: int,
    state: str,
    coverage: tuple[int, int, int, int, int],
    modalities: list[str] | None = None,
    sites: list[str] | None = None,
    site_component_count: int = 0,
    progression_count: int = 1,
    cancer_count: int = 1,
) -> dict:
    row = {
        "patient_id": pid,
        "scan_episode_id": eid,
        "episode_start_day": start,
        "episode_end_day": end,
        "landmark_day": end,
        "source_days_json": json.dumps(list(range(start, end + 1))),
        "progression_state_3": state,
        "modalities_json": json.dumps(modalities or []),
        "tumor_sites_json": json.dumps(sites or []),
        "component_counts_json": json.dumps(
            {
                "progression": progression_count,
                "cancer_presence": cancer_count,
                "tumor_sites": site_component_count,
            }
        ),
    }
    for region, value in zip(REGIONS, coverage):
        row[f"coverage_{region}"] = bool(value)
    return row


def fixture_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    scans = pd.DataFrame(
        [
            scan_row("P1", "S1", 10, 11, "NON_PROGRESSIVE", (1, 1, 0, 0, 0), ["CT"], ["BONE"], 1),
            scan_row("P1", "S2", 30, 31, "NON_PROGRESSIVE", (1, 1, 1, 0, 0), ["CT"], ["BONE"], 1),
            scan_row("P1", "S3", 60, 62, "INDETERMINATE", (1, 0, 1, 0, 0), [], [], 0),
            scan_row("P1", "S4", 95, 95, "PROGRESSIVE", (1, 0, 1, 0, 0), ["PET-CT"], ["LIVER"], 1),
            scan_row("P2", "T1", 15, 15, "NON_PROGRESSIVE", (0, 1, 0, 0, 0), ["MRI"], [], 0),
            scan_row("P2", "T2", 45, 45, "NON_PROGRESSIVE", (0, 1, 0, 0, 1), ["MRI"], ["PLEURA"], 1),
        ]
    )
    anchors = scans[["patient_id", "scan_episode_id", "landmark_day"]].copy()
    anchors["split"] = ["train", "train", "val", "val", "train", "val"]
    anchors["survival_mask"] = [True, True, True, False, True, True]
    return scans, anchors


def arrays(result):
    return [
        result.history_core.values,
        result.history_core.masks,
        result.utilization_core.values,
        result.utilization_core.masks,
        result.current_core.values,
        result.current_core.masks,
        result.change_core.values,
        result.change_core.masks,
        result.optional_history.values,
        result.optional_history.masks,
        result.optional_history_recency.values,
        result.optional_history_recency.masks,
        result.optional_current.values,
        result.optional_current.masks,
        result.stream_availability.values,
        result.stream_availability.masks,
    ]


def test_future_append_and_input_order_invariance():
    scans, anchors = fixture_tables()
    base = build_paired_scan_representation(scans, anchors)

    future_row = scan_row("P1", "S99", 5000, 5000, "PROGRESSIVE", (1, 1, 1, 1, 1), ["PET"], ["BRAIN"], 1)
    future = build_paired_scan_representation(pd.concat([scans, pd.DataFrame([future_row])], ignore_index=True), anchors)
    shuffled = build_paired_scan_representation(scans.sample(frac=1, random_state=7), anchors)

    for x, y, z in zip(arrays(base), arrays(future), arrays(shuffled)):
        assert np.array_equal(x, y, equal_nan=True)
        assert np.array_equal(x, z, equal_nan=True)
    pd.testing.assert_frame_equal(base.index, future.index)
    pd.testing.assert_frame_equal(base.index, shuffled.index)


def test_current_scan_perturbation_is_post_only():
    scans, anchors = fixture_tables()
    anchor = anchors[anchors["scan_episode_id"] == "S3"].copy()
    base = build_paired_scan_representation(scans, anchor)

    changed = scans.copy()
    mask = changed["scan_episode_id"] == "S3"
    changed.loc[mask, "progression_state_3"] = "PROGRESSIVE"
    # Make current coverage nonoverlapping with all prior P1 scans. This changes
    # current-relative comparability, which must remain POST-only.
    for region in REGIONS:
        changed.loc[mask, f"coverage_{region}"] = False
    changed.loc[mask, "coverage_head"] = True
    post = build_paired_scan_representation(changed, anchor)

    assert np.array_equal(base.history_core.values, post.history_core.values)
    assert np.array_equal(base.utilization_core.values, post.utilization_core.values)
    assert np.array_equal(base.optional_history.values, post.optional_history.values)
    assert np.array_equal(base.optional_history_recency.values, post.optional_history_recency.values)
    assert not np.array_equal(base.current_core.values, post.current_core.values)
    assert not np.array_equal(base.change_core.values, post.change_core.values)


def test_strict_episode_start_cutoff_excludes_current_component_days():
    scans, _ = fixture_tables()
    # Prior scan ending on the current episode start is not PRE-safe under the strict rule.
    current = scan_row("P3", "C2", 100, 102, "NON_PROGRESSIVE", (1, 0, 0, 0, 0), ["CT"], [], 0)
    prior_touching = scan_row("P3", "C1", 99, 100, "NON_PROGRESSIVE", (1, 0, 0, 0, 0), ["CT"], [], 0)
    local = pd.concat([scans, pd.DataFrame([prior_touching, current])], ignore_index=True)
    anchor = pd.DataFrame([{"patient_id": "P3", "scan_episode_id": "C2", "landmark_day": 102}])
    result = build_paired_scan_representation(local, anchor)
    assert result.index.iloc[0]["previous_scan_episode_id"] is None
    assert int(result.index.iloc[0]["prior_scan_count"]) == 0


def test_line_labels_extra_fields_and_duplicate_counts_do_not_change_features():
    scans, anchors = fixture_tables()
    base = build_paired_scan_representation(scans, anchors)

    anchors2 = anchors.copy()
    anchors2["line"] = np.arange(len(anchors2)) + 100
    anchors2["treatment_line"] = np.arange(len(anchors2))[::-1] + 200
    scans2 = scans.copy()
    scans2["unrelated_same_day_payload"] = "adversarial"
    counts = json.loads(scans2.loc[0, "component_counts_json"])
    scans2.loc[0, "component_counts_json"] = json.dumps({k: (v * 99 if v else 0) for k, v in counts.items()})

    changed = build_paired_scan_representation(scans2, anchors2)
    for x, y in zip(arrays(base), arrays(changed)):
        assert np.array_equal(x, y, equal_nan=True)


def test_first_observed_positive_mention_is_literal_not_metastasis_label():
    scans, anchors = fixture_tables()
    result = build_paired_scan_representation(scans, anchors[anchors["scan_episode_id"] == "S4"])
    names = list(result.optional_current.names)
    liver = names.index("current_site_positive_liver")
    first_liver = names.index("first_observed_positive_mention_liver")
    bone_first = names.index("first_observed_positive_mention_bone")
    assert result.optional_current.values[0, liver] == 1.0
    assert result.optional_current.values[0, first_liver] == 1.0
    assert result.optional_current.values[0, bone_first] == 0.0



def test_bone_scan_modality_is_not_a_positive_bone_site():
    scans, _ = fixture_tables()
    row = scan_row(
        "P4", "B1", 10, 10, "NON_PROGRESSIVE", (1, 0, 0, 0, 0),
        modalities=["BONE SCAN"], sites=[], site_component_count=0,
    )
    local = pd.concat([scans, pd.DataFrame([row])], ignore_index=True)
    anchor = pd.DataFrame([{"patient_id": "P4", "scan_episode_id": "B1", "landmark_day": 10}])
    result = build_paired_scan_representation(local, anchor)
    names = list(result.optional_current.names)
    assert result.optional_current.values[0, names.index("current_modality_bone_scan")] == 1.0
    assert result.optional_current.values[0, names.index("current_site_positive_bone")] == 0.0
    assert result.optional_current.masks[0, names.index("current_site_positive_bone")] == 0.0

def test_zero_is_distinct_from_missing():
    assert _bool_or_missing(False) == (0.0, 1.0)
    assert _bool_or_missing(None) == (0.0, 0.0)
    assert _bool_or_missing(True) == (1.0, 1.0)


def test_mask_aware_encoder_and_pre_current_isolation():
    scans, anchors = fixture_tables()
    result = build_paired_scan_representation(scans, anchors)
    torch.manual_seed(11)
    model = PairedScanEncoder(
        history_dim=result.history_core.values.shape[1],
        utilization_dim=result.utilization_core.values.shape[1],
        optional_history_dim=result.optional_history.values.shape[1],
        optional_history_recency_dim=result.optional_history_recency.values.shape[1],
        current_dim=result.current_core.values.shape[1],
        change_dim=result.change_core.values.shape[1],
        optional_current_dim=result.optional_current.values.shape[1],
        stream_dim=result.stream_availability.values.shape[1],
        hidden_dim=16,
        output_dim=8,
    ).eval()

    t = lambda x: torch.tensor(x, dtype=torch.float32)
    common = dict(
        history_values=t(result.history_core.values),
        history_masks=t(result.history_core.masks),
        utilization_values=t(result.utilization_core.values),
        utilization_masks=t(result.utilization_core.masks),
        optional_history_values=t(result.optional_history.values),
        optional_history_masks=t(result.optional_history.masks),
        optional_history_recency_values=t(result.optional_history_recency.values),
        optional_history_recency_masks=t(result.optional_history_recency.masks),
    )
    with torch.no_grad():
        a = model(
            **common,
            current_values=t(result.current_core.values),
            current_masks=t(result.current_core.masks),
            change_values=t(result.change_core.values),
            change_masks=t(result.change_core.masks),
        )
        b = model(
            **common,
            current_values=torch.randn_like(t(result.current_core.values)) * 100,
            current_masks=t(result.current_core.masks),
            change_values=torch.randn_like(t(result.change_core.values)) * 100,
            change_masks=t(result.change_core.masks),
        )
    assert torch.equal(a.history_context, b.history_context)

    n = len(anchors)
    zoh = torch.zeros((n, result.optional_history.values.shape[1]))
    zohr = torch.zeros((n, result.optional_history_recency.values.shape[1]))
    zoc = torch.zeros((n, result.optional_current.values.shape[1]))
    zs = torch.zeros((n, result.stream_availability.values.shape[1]))
    core = dict(
        history_values=t(result.history_core.values),
        history_masks=t(result.history_core.masks),
        utilization_values=t(result.utilization_core.values),
        utilization_masks=t(result.utilization_core.masks),
        current_values=t(result.current_core.values),
        current_masks=t(result.current_core.masks),
        change_values=t(result.change_core.values),
        change_masks=t(result.change_core.masks),
    )
    with torch.no_grad():
        x = model(
            **core,
            optional_history_values=torch.randn_like(zoh) * 1000,
            optional_history_masks=torch.zeros_like(zoh),
            optional_history_recency_values=torch.randn_like(zohr) * 1000,
            optional_history_recency_masks=torch.zeros_like(zohr),
            optional_current_values=torch.randn_like(zoc) * 1000,
            optional_current_masks=torch.zeros_like(zoc),
            stream_values=torch.randn_like(zs) * 1000,
            stream_masks=torch.zeros_like(zs),
        )
        y = model(
            **core,
            optional_history_values=torch.randn_like(zoh) * 1000,
            optional_history_masks=torch.zeros_like(zoh),
            optional_history_recency_values=torch.randn_like(zohr) * 1000,
            optional_history_recency_masks=torch.zeros_like(zohr),
            optional_current_values=torch.randn_like(zoc) * 1000,
            optional_current_masks=torch.zeros_like(zoc),
            stream_values=torch.randn_like(zs) * 1000,
            stream_masks=torch.zeros_like(zs),
        )
    assert torch.equal(x.history_context, y.history_context)
    assert torch.equal(x.current_evidence, y.current_evidence)
    assert torch.equal(x.change_evidence, y.change_evidence)
    assert torch.equal(x.post_evidence, y.post_evidence)
