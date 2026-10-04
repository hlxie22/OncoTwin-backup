from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


STATE_NAMES = ("NON_PROGRESSIVE", "INDETERMINATE", "PROGRESSIVE")
REGIONS = ("chest", "abdomen", "pelvis", "head", "other")

MODALITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "ct": re.compile(r"\bCT\b|COMPUTED TOMOGRAPH", re.I),
    "pet": re.compile(r"\bPET\b|PET[- ]?CT", re.I),
    "mr": re.compile(r"\bMRI?\b|MAGNETIC", re.I),
    "bone_scan": re.compile(r"BONE SCAN", re.I),
}

SITE_PATTERNS: dict[str, re.Pattern[str]] = {
    "bone": re.compile(r"\bBONE\b|SKELET|VERTEBR|RIB", re.I),
    "liver": re.compile(r"\bLIVER\b", re.I),
    "lung": re.compile(r"\bLUNG\b|PULMON", re.I),
    "brain": re.compile(r"\bBRAIN\b|\bCNS\b|CEREBR", re.I),
    "lymph": re.compile(r"LYMPH", re.I),
    "pleura": re.compile(r"PLEURA", re.I),
}


@dataclass(frozen=True)
class FeatureBlock:
    values: np.ndarray
    masks: np.ndarray
    names: tuple[str, ...]


@dataclass(frozen=True)
class RepresentationResult:
    index: pd.DataFrame
    history_core: FeatureBlock
    utilization_core: FeatureBlock
    current_core: FeatureBlock
    change_core: FeatureBlock
    optional_history: FeatureBlock
    optional_history_recency: FeatureBlock
    optional_current: FeatureBlock
    stream_availability: FeatureBlock


def _json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if parsed is None:
        return []
    return [str(parsed)]


def _json_dict(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _bool_or_missing(value: Any) -> tuple[float, float]:
    """Return (value, source_available_mask), preserving zero != missing."""
    if value is None:
        return 0.0, 0.0
    if isinstance(value, float) and math.isnan(value):
        return 0.0, 0.0
    if isinstance(value, (np.bool_, bool)):
        return float(bool(value)), 1.0
    if isinstance(value, (np.integer, int)):
        return float(int(value) != 0), 1.0
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return 0.0, 0.0
    if text in {"1", "true", "t", "yes", "y", "imaged"}:
        return 1.0, 1.0
    if text in {"0", "false", "f", "no", "n", "not_imaged", "not imaged"}:
        return 0.0, 1.0
    raise ValueError(f"Unrecognized boolean-like value: {value!r}")


def _state_one_hot(state: Any) -> np.ndarray:
    out = np.zeros(len(STATE_NAMES), dtype=np.float32)
    text = str(state).strip().upper()
    if text not in STATE_NAMES:
        raise ValueError(f"Unsupported scan state: {state!r}")
    out[STATE_NAMES.index(text)] = 1.0
    return out


def _pattern_vector(items: Iterable[str], patterns: dict[str, re.Pattern[str]]) -> np.ndarray:
    text = " | ".join(str(x) for x in items)
    return np.asarray([float(bool(pattern.search(text))) for pattern in patterns.values()], dtype=np.float32)


def _source_component_flags(component_counts: dict[str, Any]) -> dict[str, bool]:
    # Intentionally collapse counts to booleans so duplicate raw component rows cannot
    # inflate model-facing burden/utilization features.
    def present(name: str) -> bool:
        raw = component_counts.get(name, 0)
        try:
            return float(raw) > 0.0
        except (TypeError, ValueError):
            return False

    return {
        "progression": present("progression"),
        "cancer_presence": present("cancer_presence"),
        "tumor_sites": present("tumor_sites"),
    }


def _normalize_scan_table(scans: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id",
        "scan_episode_id",
        "episode_start_day",
        "episode_end_day",
        "landmark_day",
        "source_days_json",
        "progression_state_3",
        "modalities_json",
        "tumor_sites_json",
        "component_counts_json",
        *(f"coverage_{region}" for region in REGIONS),
    }
    missing = sorted(required.difference(scans.columns))
    if missing:
        raise RuntimeError(f"Authoritative scan table missing required columns: {missing}")

    out = scans.copy()
    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["scan_episode_id"] = out["scan_episode_id"].astype(str).str.strip()
    out["progression_state_3"] = out["progression_state_3"].astype(str).str.strip().str.upper()

    for column in ["episode_start_day", "episode_end_day", "landmark_day"]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(np.int64)

    if out[["patient_id", "scan_episode_id"]].duplicated().any():
        raise RuntimeError("Authoritative scan episode keys are not unique")
    if not np.array_equal(out["episode_end_day"].to_numpy(), out["landmark_day"].to_numpy()):
        raise RuntimeError("V2-01 requires landmark_day == episode_end_day")
    if (out["episode_start_day"] > out["episode_end_day"]).any():
        raise RuntimeError("Found scan episode with start after end")
    if not out["progression_state_3"].isin(STATE_NAMES).all():
        bad = out.loc[~out["progression_state_3"].isin(STATE_NAMES), "progression_state_3"].value_counts().to_dict()
        raise RuntimeError(f"Unexpected scan states: {bad}")

    out = out.sort_values(["patient_id", "landmark_day", "scan_episode_id"], kind="mergesort").reset_index(drop=True)
    return out


def _normalize_anchor_table(anchors: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "scan_episode_id", "landmark_day"}
    missing = sorted(required.difference(anchors.columns))
    if missing:
        raise RuntimeError(f"Anchor table missing required columns: {missing}")
    out = anchors.copy()
    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["scan_episode_id"] = out["scan_episode_id"].astype(str).str.strip()
    out["landmark_day"] = pd.to_numeric(out["landmark_day"], errors="raise").astype(np.int64)
    if out[["patient_id", "scan_episode_id"]].duplicated().any():
        raise RuntimeError("Anchor keys are not unique")
    return out.reset_index(drop=True)


def _coverage_from_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = []
    masks = []
    for region in REGIONS:
        value, mask = _bool_or_missing(row[f"coverage_{region}"])
        values.append(value)
        masks.append(mask)
    return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=np.float32)


def _safe_log1p_days(days: float | int | None, cap_days: float = 3650.0) -> float:
    if days is None:
        return 0.0
    value = max(0.0, min(float(days), cap_days))
    return float(np.log1p(value) / np.log1p(cap_days))


def _safe_log1p_count(count: int, cap_count: int = 100) -> float:
    value = max(0, min(int(count), cap_count))
    return float(np.log1p(value) / np.log1p(cap_count))


def _jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int, int]:
    aa = a > 0.5
    bb = b > 0.5
    overlap = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    gained = int(np.logical_and(aa, ~bb).sum())
    lost = int(np.logical_and(~aa, bb).sum())
    return (float(overlap / union) if union > 0 else 0.0, overlap, gained, lost)


def build_paired_scan_representation(scans: pd.DataFrame, anchors: pd.DataFrame) -> RepresentationResult:
    """Build causal paired-scan features for exact anchor episodes.

    The history cutoff is strict: only episodes with landmark_day < current
    episode_start_day are eligible for PRE/history features. Therefore no source
    component belonging to a multi-day current episode can leak into PRE.
    """
    scans = _normalize_scan_table(scans)
    anchors = _normalize_anchor_table(anchors)

    scan_lookup = {
        (str(row["patient_id"]), str(row["scan_episode_id"])): row
        for _, row in scans.iterrows()
    }
    anchor_keys = list(zip(anchors["patient_id"], anchors["scan_episode_id"]))
    missing_keys = [key for key in anchor_keys if key not in scan_lookup]
    if missing_keys:
        raise RuntimeError(f"Anchor episodes absent from authoritative scan table; sample={missing_keys[:5]}")

    # Build per-patient canonical histories once.
    histories: dict[str, pd.DataFrame] = {
        str(pid): group.reset_index(drop=True)
        for pid, group in scans.groupby("patient_id", sort=False)
    }

    history_names = tuple(
        [f"prev1_state_{s.lower()}" for s in STATE_NAMES]
        + [f"prev2_state_{s.lower()}" for s in STATE_NAMES]
        + [f"prev1_coverage_{r}" for r in REGIONS]
        + ["prior_nonprogressive_run_scaled", "last3_nonprogressive_fraction"]
    )
    utilization_names = (
        "prior_scan_count_scaled",
        "days_since_prev1_scaled",
        "days_since_prev2_scaled",
    )
    current_names = tuple([f"current_state_{s.lower()}" for s in STATE_NAMES] + [f"current_coverage_{r}" for r in REGIONS])
    transition_names = tuple(
        f"transition_{a.lower()}_to_{b.lower()}" for a in STATE_NAMES for b in STATE_NAMES
    )
    change_names = transition_names + (
        "coverage_jaccard",
        "coverage_overlap_fraction_of_current",
        "coverage_overlap_fraction_of_prior",
        "coverage_gained_fraction",
        "coverage_lost_fraction",
        "coverage_exact_match",
        "coverage_any_overlap",
        "coverage_comparable_previous_exists",
        "any_prior_comparable_assessment",
        "days_since_last_comparable_scaled",
        "last_comparable_state_non_progressive",
        "last_comparable_state_indeterminate",
        "last_comparable_state_progressive",
    )
    optional_history_names = tuple(
        [f"prior_observed_modality_{x}" for x in MODALITY_PATTERNS]
        + [f"prior_observed_positive_site_{x}" for x in SITE_PATTERNS]
    )
    optional_history_recency_names = tuple(
        [f"days_since_prior_modality_{x}_scaled" for x in MODALITY_PATTERNS]
        + [f"days_since_prior_site_positive_{x}_scaled" for x in SITE_PATTERNS]
    )
    optional_current_names = tuple(
        [f"current_modality_{x}" for x in MODALITY_PATTERNS]
        + [f"current_site_positive_{x}" for x in SITE_PATTERNS]
        + [f"first_observed_positive_mention_{x}" for x in SITE_PATTERNS]
    )
    stream_names = (
        "state_source_available",
        "coverage_source_available",
        "modality_source_available",
        "modality_observed",
        "site_positive_source_available",
        "site_positive_observed",
        "prior_modality_stream_observed",
        "prior_site_positive_stream_observed",
    )

    n = len(anchors)
    h_val = np.zeros((n, len(history_names)), dtype=np.float32)
    h_mask = np.zeros_like(h_val)
    u_val = np.zeros((n, len(utilization_names)), dtype=np.float32)
    u_mask = np.zeros_like(u_val)
    c_val = np.zeros((n, len(current_names)), dtype=np.float32)
    c_mask = np.zeros_like(c_val)
    d_val = np.zeros((n, len(change_names)), dtype=np.float32)
    d_mask = np.zeros_like(d_val)
    oh_val = np.zeros((n, len(optional_history_names)), dtype=np.float32)
    oh_mask = np.zeros_like(oh_val)
    r_val = np.zeros((n, len(optional_history_recency_names)), dtype=np.float32)
    r_mask = np.zeros_like(r_val)
    oc_val = np.zeros((n, len(optional_current_names)), dtype=np.float32)
    oc_mask = np.zeros_like(oc_val)
    s_val = np.zeros((n, len(stream_names)), dtype=np.float32)
    s_mask = np.ones_like(s_val)

    index_rows: list[dict[str, Any]] = []

    modality_keys = list(MODALITY_PATTERNS)
    site_keys = list(SITE_PATTERNS)

    for i, anchor in anchors.iterrows():
        pid = str(anchor["patient_id"])
        episode_id = str(anchor["scan_episode_id"])
        current = scan_lookup[(pid, episode_id)]

        patient_history = histories[pid]
        strict_prior = patient_history[patient_history["landmark_day"] < int(current["episode_start_day"])]
        prior_count = int(len(strict_prior))
        prev1 = strict_prior.iloc[-1] if prior_count >= 1 else None
        prev2 = strict_prior.iloc[-2] if prior_count >= 2 else None

        current_state = str(current["progression_state_3"])
        current_state_vec = _state_one_hot(current_state)
        current_cov, current_cov_mask = _coverage_from_row(current)
        current_modalities = _json_list(current["modalities_json"])
        current_sites = _json_list(current["tumor_sites_json"])
        current_mod_vec = _pattern_vector(current_modalities, MODALITY_PATTERNS)
        current_site_vec = _pattern_vector(current_sites, SITE_PATTERNS)
        current_counts = _json_dict(current["component_counts_json"])
        current_component_flags = _source_component_flags(current_counts)

        # Current portable core: clinical state and anatomy observation coverage.
        c_val[i, :3] = current_state_vec
        c_mask[i, :3] = 1.0
        c_val[i, 3:8] = current_cov
        c_mask[i, 3:8] = current_cov_mask

        # Stream-level availability deliberately separates source support from
        # whether a positive/descriptor observation is present in this episode.
        modality_observed = float(len(current_modalities) > 0)
        site_observed = float(current_component_flags["tumor_sites"])
        # Optional current stream values. Values are always multiplied by masks;
        # adversarial changes under mask=0 cannot alter downstream encodings.
        c0 = 0
        c1 = c0 + len(modality_keys)
        c2 = c1 + len(site_keys)
        c3 = c2 + len(site_keys)
        oc_val[i, c0:c1] = current_mod_vec
        oc_mask[i, c0:c1] = modality_observed
        oc_val[i, c1:c2] = current_site_vec
        oc_mask[i, c1:c2] = site_observed

        prior_positive_sites = np.zeros(len(site_keys), dtype=np.float32)
        prior_positive_modalities = np.zeros(len(modality_keys), dtype=np.float32)
        last_site_day = np.full(len(site_keys), np.nan, dtype=np.float64)
        last_modality_day = np.full(len(modality_keys), np.nan, dtype=np.float64)
        prior_modality_stream_observed = False
        prior_site_stream_observed = False

        for _, historical_scan in strict_prior.iterrows():
            day = float(historical_scan["landmark_day"])
            historical_modalities = _json_list(historical_scan["modalities_json"])
            historical_sites = _json_list(historical_scan["tumor_sites_json"])
            historical_counts = _source_component_flags(_json_dict(historical_scan["component_counts_json"]))
            prior_modality_stream_observed = prior_modality_stream_observed or bool(historical_modalities)
            prior_site_stream_observed = prior_site_stream_observed or historical_counts["tumor_sites"]
            mod_vec = _pattern_vector(historical_modalities, MODALITY_PATTERNS)
            site_vec = _pattern_vector(historical_sites, SITE_PATTERNS)
            prior_positive_modalities = np.maximum(prior_positive_modalities, mod_vec)
            prior_positive_sites = np.maximum(prior_positive_sites, site_vec)
            for j, value in enumerate(mod_vec):
                if value > 0.5:
                    last_modality_day[j] = day
            for j, value in enumerate(site_vec):
                if value > 0.5:
                    last_site_day[j] = day

        h0 = 0
        h1 = h0 + len(modality_keys)
        h2 = h1 + len(site_keys)
        oh_val[i, h0:h1] = prior_positive_modalities
        oh_mask[i, h0:h1] = float(prior_modality_stream_observed)
        oh_val[i, h1:h2] = prior_positive_sites
        oh_mask[i, h1:h2] = float(prior_site_stream_observed)

        first_observed_sites = (current_site_vec > 0.5) & (prior_positive_sites < 0.5)
        oc_val[i, c2:c3] = first_observed_sites.astype(np.float32)
        oc_mask[i, c2:c3] = site_observed

        s_val[i] = np.asarray([
            1.0,
            float(current_cov_mask.min() > 0.5),
            1.0,
            modality_observed,
            1.0,
            site_observed,
            float(prior_modality_stream_observed),
            float(prior_site_stream_observed),
        ], dtype=np.float32)

        # Optional recency: only defined when a prior positive descriptor exists.
        for j in range(len(modality_keys)):
            if not np.isnan(last_modality_day[j]):
                r_val[i, j] = _safe_log1p_days(int(current["episode_start_day"]) - last_modality_day[j])
                r_mask[i, j] = 1.0
        for j in range(len(site_keys)):
            k = len(modality_keys) + j
            if not np.isnan(last_site_day[j]):
                r_val[i, k] = _safe_log1p_days(int(current["episode_start_day"]) - last_site_day[j])
                r_mask[i, k] = 1.0

        # Utilization/history descriptors; all are PRE-safe.
        u_val[i, 0] = _safe_log1p_count(prior_count)
        u_mask[i, 0] = 1.0

        prev1_cov = np.zeros(len(REGIONS), dtype=np.float32)
        prev1_cov_mask = np.zeros(len(REGIONS), dtype=np.float32)
        prev1_state = None
        prev2_state = None
        days_since_prev1 = None
        days_since_prev2 = None

        if prev1 is not None:
            prev1_state = str(prev1["progression_state_3"])
            prev1_state_vec = _state_one_hot(prev1_state)
            h_val[i, 0:3] = prev1_state_vec
            h_mask[i, 0:3] = 1.0
            prev1_cov, prev1_cov_mask = _coverage_from_row(prev1)
            h_val[i, 6:11] = prev1_cov
            h_mask[i, 6:11] = prev1_cov_mask
            days_since_prev1 = int(current["episode_start_day"]) - int(prev1["landmark_day"])
            if days_since_prev1 <= 0:
                raise RuntimeError("Strict PRE history contains nonpositive previous-scan interval")
            u_val[i, 1] = _safe_log1p_days(days_since_prev1)
            u_mask[i, 1] = 1.0

        if prev2 is not None:
            prev2_state = str(prev2["progression_state_3"])
            h_val[i, 3:6] = _state_one_hot(prev2_state)
            h_mask[i, 3:6] = 1.0
            days_since_prev2 = int(current["episode_start_day"]) - int(prev2["landmark_day"])
            if days_since_prev2 <= 0:
                raise RuntimeError("Strict PRE history contains nonpositive second-previous interval")
            u_val[i, 2] = _safe_log1p_days(days_since_prev2)
            u_mask[i, 2] = 1.0

        # Consecutive PRE nonprogressive run and last-three fraction.
        run = 0
        for state in reversed(strict_prior["progression_state_3"].astype(str).tolist()):
            if state == "NON_PROGRESSIVE":
                run += 1
            else:
                break
        h_val[i, 11] = _safe_log1p_count(run, cap_count=20)
        h_mask[i, 11] = 1.0
        if prior_count > 0:
            last3 = strict_prior.tail(3)["progression_state_3"].astype(str)
            h_val[i, 12] = float((last3 == "NON_PROGRESSIVE").mean())
            h_mask[i, 12] = 1.0

        # Previous comparable assessment = most recent prior scan with at least
        # one anatomically overlapping covered region.
        last_comp = None
        if prior_count > 0 and current_cov_mask.min() > 0.5 and current_cov.sum() > 0:
            for _, historical_scan in strict_prior.iloc[::-1].iterrows():
                hc, hm = _coverage_from_row(historical_scan)
                if hm.min() > 0.5 and np.logical_and(hc > 0.5, current_cov > 0.5).any():
                    last_comp = historical_scan
                    break

        if last_comp is not None:
            comp_state = str(last_comp["progression_state_3"])
            days_since_comp = int(current["episode_start_day"]) - int(last_comp["landmark_day"])
            if days_since_comp <= 0:
                raise RuntimeError("Comparable prior scan is not strictly before current episode")
        else:
            comp_state = None
            days_since_comp = None

        # Change features are POST-only because they use current scan content.
        if prev1 is not None:
            transition_index = STATE_NAMES.index(prev1_state) * len(STATE_NAMES) + STATE_NAMES.index(current_state)
            d_val[i, transition_index] = 1.0
            d_mask[i, : len(transition_names)] = 1.0

            comparable = bool(
                current_cov_mask.min() > 0.5
                and prev1_cov_mask.min() > 0.5
                and current_cov.sum() > 0
                and prev1_cov.sum() > 0
                and np.logical_and(current_cov > 0.5, prev1_cov > 0.5).any()
            )
            jaccard, overlap, gained, lost = _jaccard(current_cov, prev1_cov)
            current_count = int((current_cov > 0.5).sum())
            prior_count_cov = int((prev1_cov > 0.5).sum())
            union_count = int(np.logical_or(current_cov > 0.5, prev1_cov > 0.5).sum())
            off = len(transition_names)
            d_val[i, off + 0] = jaccard
            d_val[i, off + 1] = float(overlap / current_count) if current_count else 0.0
            d_val[i, off + 2] = float(overlap / prior_count_cov) if prior_count_cov else 0.0
            d_val[i, off + 3] = float(gained / len(REGIONS))
            d_val[i, off + 4] = float(lost / len(REGIONS))
            d_val[i, off + 5] = float(np.array_equal(current_cov > 0.5, prev1_cov > 0.5))
            d_val[i, off + 6] = float(overlap > 0)
            d_val[i, off + 7] = float(comparable)
            if current_cov_mask.min() > 0.5 and prev1_cov_mask.min() > 0.5:
                d_mask[i, off : off + 8] = 1.0

            d_val[i, off + 8] = float(last_comp is not None)
            d_mask[i, off + 8] = 1.0
            if last_comp is not None:
                d_val[i, off + 9] = _safe_log1p_days(days_since_comp)
                d_mask[i, off + 9] = 1.0
                d_val[i, off + 10 : off + 13] = _state_one_hot(comp_state)
                d_mask[i, off + 10 : off + 13] = 1.0
        else:
            current_count = int((current_cov > 0.5).sum())
            prior_count_cov = 0
            union_count = current_count
            overlap = 0
            gained = current_count
            lost = 0
            comparable = False
            jaccard = 0.0
            off = len(transition_names)
            d_val[i, off + 8] = 0.0
            d_mask[i, off + 8] = 1.0

        source_days = [int(float(x)) for x in _json_list(current["source_days_json"])]
        if source_days:
            if min(source_days) < int(current["episode_start_day"]) or max(source_days) > int(current["episode_end_day"]):
                raise RuntimeError("Current source day falls outside current episode bounds")
            if prev1 is not None and int(prev1["landmark_day"]) >= min(source_days):
                raise RuntimeError("PRE/current episode source-component separation violated")

        index_rows.append(
            {
                "patient_id": pid,
                "scan_episode_id": episode_id,
                "episode_start_day": int(current["episode_start_day"]),
                "episode_end_day": int(current["episode_end_day"]),
                "landmark_day": int(current["landmark_day"]),
                "source_days_json": current["source_days_json"],
                "clinical_state": current_state,
                "prior_scan_count": prior_count,
                "previous_scan_episode_id": None if prev1 is None else str(prev1["scan_episode_id"]),
                "previous_landmark_day": None if prev1 is None else int(prev1["landmark_day"]),
                "previous_clinical_state": prev1_state,
                "previous_comparable_scan_episode_id": None if last_comp is None else str(last_comp["scan_episode_id"]),
                "previous_comparable_landmark_day": None if last_comp is None else int(last_comp["landmark_day"]),
                "previous_comparable_clinical_state": None if last_comp is None else str(last_comp["progression_state_3"]),
                "days_since_previous_assessment": days_since_prev1,
                "days_since_previous_comparable_assessment": days_since_comp,
                "prior_nonprogressive_run": run,
                "current_coverage_count": current_count,
                "previous_coverage_count": prior_count_cov,
                "coverage_overlap_count": overlap,
                "coverage_union_count": union_count,
                "coverage_jaccard": jaccard,
                "coverage_gained_count": gained,
                "coverage_lost_count": lost,
                "coverage_comparable_to_previous": comparable,
                "current_modality_observed": bool(modality_observed),
                "current_site_positive_stream_observed": bool(site_observed),
                "source_component_progression_present": bool(current_component_flags["progression"]),
                "source_component_cancer_presence_present": bool(current_component_flags["cancer_presence"]),
                "source_component_tumor_sites_present": bool(current_component_flags["tumor_sites"]),
            }
        )

    index = pd.DataFrame(index_rows)
    # Preserve anchor-side fields useful for downstream exact alignment, while
    # explicitly excluding prohibited line identity from model-facing arrays.
    passthrough = [
        c
        for c in [
            "split",
            "survival_mask",
            "next_scan_mask",
            "postprog_mask",
            "genomic_available",
            "genomic_age_days",
            "sample_id",
        ]
        if c in anchors.columns
    ]
    if passthrough:
        merged = anchors[["patient_id", "scan_episode_id", *passthrough]].copy()
        index = index.merge(merged, on=["patient_id", "scan_episode_id"], how="left", validate="one_to_one")

    # Enforce masking algebra once at materialization. Downstream models receive
    # values and masks separately, but stored values under unavailable masks are
    # already neutralized to exact zero as a defense in depth.
    for values, masks in [
        (h_val, h_mask),
        (u_val, u_mask),
        (c_val, c_mask),
        (d_val, d_mask),
        (oh_val, oh_mask),
        (r_val, r_mask),
        (oc_val, oc_mask),
    ]:
        values *= masks

    return RepresentationResult(
        index=index,
        history_core=FeatureBlock(h_val, h_mask, history_names),
        utilization_core=FeatureBlock(u_val, u_mask, utilization_names),
        current_core=FeatureBlock(c_val, c_mask, current_names),
        change_core=FeatureBlock(d_val, d_mask, change_names),
        optional_history=FeatureBlock(oh_val, oh_mask, optional_history_names),
        optional_history_recency=FeatureBlock(r_val, r_mask, optional_history_recency_names),
        optional_current=FeatureBlock(oc_val, oc_mask, optional_current_names),
        stream_availability=FeatureBlock(s_val, s_mask, stream_names),
    )


def engineered_control_matrix(result: RepresentationResult, include_optional: bool) -> tuple[np.ndarray, tuple[str, ...]]:
    """Transparent control representation with explicit masks concatenated."""
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for prefix, block in [
        ("history", result.history_core),
        ("utilization", result.utilization_core),
        ("current", result.current_core),
        ("change", result.change_core),
    ]:
        blocks.extend([block.values, block.masks])
        names.extend([f"{prefix}__{x}" for x in block.names])
        names.extend([f"{prefix}__mask__{x}" for x in block.names])
    if include_optional:
        for prefix, block in [
            ("optional_history", result.optional_history),
            ("optional_history_recency", result.optional_history_recency),
            ("optional_current", result.optional_current),
            ("stream", result.stream_availability),
        ]:
            blocks.extend([block.values, block.masks])
            names.extend([f"{prefix}__{x}" for x in block.names])
            names.extend([f"{prefix}__mask__{x}" for x in block.names])
    return np.concatenate(blocks, axis=1).astype(np.float32), tuple(names)


def representation_feature_schema(result: RepresentationResult) -> dict[str, Any]:
    return {
        "schema_version": "v2_01_1",
        "blocks": {
            "history_core": list(result.history_core.names),
            "utilization_core": list(result.utilization_core.names),
            "current_core": list(result.current_core.names),
            "change_core": list(result.change_core.names),
            "optional_history": list(result.optional_history.names),
            "optional_history_recency": list(result.optional_history_recency.names),
            "optional_current": list(result.optional_current.names),
            "stream_availability": list(result.stream_availability.names),
        },
        "pre_post_contract": {
            "pre_visible_blocks": ["history_core", "utilization_core", "optional_history", "optional_history_recency"],
            "post_additional_blocks": ["current_core", "change_core", "optional_current", "stream_availability"],
            "strict_history_cutoff": "prior landmark_day < current episode_start_day",
            "current_episode_source_components_in_pre": False,
        },
        "semantics": {
            "clinical_state": "authoritative CHORD three-state assessment only; NON_PROGRESSIVE is not split into stable/responding",
            "coverage_zero": "region not imaged in this authoritative episode; not a disease-negative assertion",
            "coverage_missing": "mask=0; distinct from coverage value zero with mask=1",
            "site_positive": "positive mention/support only; zero is not a disease-negative assertion",
            "first_observed_positive_mention": "first positive mention in the observed scan history; never interpreted as new metastasis",
            "utilization": "kept as a separate model-facing block from biological/content features",
            "optional_masking": "optional values under mask=0 are exact-zero neutralized before model input",
        },
    }
