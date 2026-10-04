from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCAN_WINDOWS = (0, 3, 7)
PRIMARY_SCAN_WINDOW = 3
DAY_SPLIT = 365
PROGRESSION_GRACE_DAYS = 28
RADIOLOGY_LOOKBACK_DAYS = 90
TRAIN_HORIZON_DAYS = 730

REGIONS = ("CHEST", "ABDOMEN", "PELVIS", "HEAD", "OTHER")

DIAGNOSIS_PATTERN = re.compile(
    r"^\s*(?P<histologic>[^|]+?)\s*\|\s*"
    r"(?P<site>[^,|()]+?)(?:\s*,\s*(?P<subset>.*?))?"
    r"\s*(?=\(\s*M\d{4}/)\(\s*"
    r"(?P<icdo_morph>M\d{4}/[012369](?:[0-4]|9)?)"
    r"\s*\|\s*(?P<icdo_topo>C\d{3})\s*\)\s*$",
    flags=re.IGNORECASE | re.VERBOSE,
)


def norm_col(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_one(root: Path, basename: str) -> Path:
    matches = [p for p in root.rglob(basename) if p.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {basename} under {root}; found {len(matches)}")
    return matches[0]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        comment="#",
        low_memory=False,
    )


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", pd.NA), errors="coerce")


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def boolish(x: Any) -> bool:
    return clean_text(x).upper() in {"1", "TRUE", "T", "Y", "YES"}


def json_safe_dict(row: pd.Series, exclude: set[str] | None = None) -> str:
    exclude = exclude or set()
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if key in exclude:
            continue
        text = clean_text(value)
        if text != "":
            payload[str(key)] = text
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"empty write: {path}")
    tmp.replace(path)


def atomic_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    json.loads(text)
    atomic_text(path, text + "\n")


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"empty parquet write: {path}")
    pd.read_parquet(tmp, columns=list(df.columns[: min(3, len(df.columns))]))
    tmp.replace(path)


def stable_split(patient_id: str) -> str:
    bucket = int(hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def parse_diagnosis(diag: pd.DataFrame) -> pd.DataFrame:
    out = diag.copy()
    if "DX_DESCRIPTION" not in out.columns:
        raise RuntimeError("diagnosis timeline lacks DX_DESCRIPTION")
    extracted = out["DX_DESCRIPTION"].astype(str).str.extract(DIAGNOSIS_PATTERN)
    out["PARSED_SITE"] = extracted["site"].fillna("").str.strip().str.upper()
    out["PARSED_HISTOLOGIC"] = extracted["histologic"].fillna("").str.strip().str.upper()
    out["PARSED_ICDO_TOPO"] = extracted["icdo_topo"].fillna("").str.strip().str.upper()
    out["START_DATE_NUM"] = to_num(out["START_DATE"])
    return out


def breast_cohorts(sample: pd.DataFrame, diagnosis: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    if "PATIENT_ID" not in sample.columns:
        raise RuntimeError("clinical sample table lacks PATIENT_ID")
    cancer_cols = [c for c in sample.columns if norm_col(c) in {"cancer_type", "cancer_type_detailed"}]
    if not cancer_cols:
        raise RuntimeError("clinical sample table lacks a cancer type field")
    sample_mask = pd.Series(False, index=sample.index)
    for col in cancer_cols:
        sample_mask |= sample[col].astype(str).str.contains("breast", case=False, na=False)
    sample_breast = set(sample.loc[sample_mask, "PATIENT_ID"].astype(str).str.strip())

    diagnosed_breast = set(diagnosis.loc[diagnosis["PARSED_SITE"] == "BREAST", "PATIENT_ID"])
    diagnosed_other = set(
        diagnosis.loc[(diagnosis["PARSED_SITE"] != "") & (diagnosis["PARSED_SITE"] != "BREAST"), "PATIENT_ID"]
    )
    breast_only = diagnosed_breast - diagnosed_other
    if not breast_only:
        raise RuntimeError("diagnosis parsing produced zero breast-only patients")

    stage_col = "STAGE_CDM_DERIVED" if "STAGE_CDM_DERIVED" in diagnosis.columns else None
    stage4: set[str] = set()
    if stage_col:
        stage_norm = diagnosis[stage_col].astype(str).str.upper()
        mask = stage_norm.str.contains(r"STAGE\s*(?:4|IV)\b", regex=True, na=False)
        stage4 = set(diagnosis.loc[mask & diagnosis["PATIENT_ID"].isin(breast_only), "PATIENT_ID"])
    return sample_breast, breast_only, stage4


def first_metastasis_dates(tumor_sites: pd.DataFrame, patient_ids: set[str]) -> dict[str, int]:
    work = tumor_sites[tumor_sites["PATIENT_ID"].isin(patient_ids)].copy()
    work["START_DATE_NUM"] = to_num(work["START_DATE"])
    work["TUMOR_SITE_NORM"] = work["TUMOR_SITE"].astype(str).str.strip().str.upper()
    work = work.dropna(subset=["START_DATE_NUM"])
    work = work[~work["TUMOR_SITE_NORM"].isin({"OTHER", "LYMPH NODES", ""})]
    first = work.groupby("PATIENT_ID", observed=True)["START_DATE_NUM"].min()
    return {str(pid): int(day) for pid, day in first.items()}


def death_days(clinical_patient: pd.DataFrame, patient_ids: set[str]) -> dict[str, int]:
    required = {"PATIENT_ID", "OS_MONTHS", "OS_STATUS"}
    if not required.issubset(clinical_patient.columns):
        return {}
    work = clinical_patient[clinical_patient["PATIENT_ID"].isin(patient_ids)].copy()
    work["OS_MONTHS_NUM"] = to_num(work["OS_MONTHS"])
    status = work["OS_STATUS"].astype(str).str.upper()
    dead = work[status.str.contains("DECEASED", na=False) & work["OS_MONTHS_NUM"].notna()].copy()
    dead["DEATH_DAY"] = np.floor(dead["OS_MONTHS_NUM"] * 30.437).astype(int)
    return dict(zip(dead["PATIENT_ID"].astype(str), dead["DEATH_DAY"].astype(int)))


def filter_mbc_rows(
    df: pd.DataFrame,
    breast_only: set[str],
    stage4: set[str],
    metastasis: dict[str, int],
) -> pd.DataFrame:
    work = df[df["PATIENT_ID"].isin(breast_only)].copy()
    work["START_DATE"] = to_num(work["START_DATE"])
    work = work.dropna(subset=["START_DATE"])
    work["START_DATE"] = work["START_DATE"].astype(int)
    keep = []
    for pid, day in zip(work["PATIENT_ID"], work["START_DATE"]):
        if pid in stage4:
            keep.append(True)
        else:
            onset = metastasis.get(str(pid))
            keep.append(onset is not None and int(day) >= onset)
    return work.loc[keep].reset_index(drop=True)


def radiology_measurement_map(
    progression: pd.DataFrame,
    cancer_presence: pd.DataFrame,
    tumor_sites: pd.DataFrame,
    patient_ids: set[str],
) -> dict[str, np.ndarray]:
    bucket: dict[str, list[int]] = defaultdict(list)
    for df in (progression, cancer_presence, tumor_sites):
        work = df[df["PATIENT_ID"].isin(patient_ids)].copy()
        days = to_num(work["START_DATE"])
        for pid, day in zip(work["PATIENT_ID"], days):
            if pd.notna(day):
                bucket[str(pid)].append(int(day))
    return {pid: np.sort(np.asarray(days, dtype=np.int64)) for pid, days in bucket.items() if days}


def next_treatment_index(
    start_dates: np.ndarray,
    current_idx: int,
    target_day: int,
    inclusive: bool,
    agents: np.ndarray,
    require_new_agent: bool,
) -> int:
    side = "left" if inclusive else "right"
    idx = int(np.searchsorted(start_dates, target_day, side=side))
    idx = max(idx, current_idx + 1)
    idx = min(idx, len(start_dates))
    if require_new_agent and idx < len(start_dates):
        last_idx = max(min(idx - 1, len(agents) - 1), current_idx)
        last_agent = agents[last_idx]
        while idx < len(start_dates) and agents[idx] == last_agent:
            idx += 1
    return idx


def has_prior_measurement(measurement_map: dict[str, np.ndarray], pid: str, line_start: int) -> bool:
    days = measurement_map.get(pid)
    if days is None or days.size == 0:
        return False
    left = int(np.searchsorted(days, line_start - RADIOLOGY_LOOKBACK_DAYS, side="left"))
    right = int(np.searchsorted(days, line_start, side="right"))
    return right > left


def assign_patient_lines(
    patient_treatments: pd.DataFrame,
    patient_progression: pd.DataFrame,
    measurement_map: dict[str, np.ndarray],
    death_day: int | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tx = patient_treatments.sort_values("START_DATE", kind="mergesort").reset_index(drop=True).copy()
    start_dates = tx["START_DATE"].to_numpy(dtype=int)
    n = len(tx)
    agents = tx.get("AGENT", pd.Series([""] * n)).fillna("").astype(str).to_numpy()
    events = patient_progression.sort_values("START_DATE", kind="mergesort").reset_index(drop=True)
    event_dates = events["START_DATE"].to_numpy(dtype=int) if not events.empty else np.array([], dtype=int)
    event_types = events["PROGRESSION"].astype(str).str.upper().to_numpy() if not events.empty else np.array([], dtype=str)
    lines = np.empty(n, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    pid = str(tx["PATIENT_ID"].iat[0])
    event_ptr = 0
    idx = 0
    line_no = 1

    while idx < n:
        start_idx = idx
        line_start = int(start_dates[start_idx])
        split_idx = next_treatment_index(
            start_dates, start_idx, line_start + DAY_SPLIT, True, agents, True
        )
        boundary_day = int(start_dates[split_idx]) if split_idx < n else None

        search = event_ptr
        while search < len(event_dates) and event_dates[search] <= line_start:
            search += 1
        ptr = search
        last_n: int | None = None
        pfs_event: int | None = None
        event_day: int | None = None
        line_source = "undefined"

        while ptr < len(event_dates) and (boundary_day is None or event_dates[ptr] <= boundary_day):
            day = int(event_dates[ptr])
            kind = event_types[ptr]
            if kind == "N":
                last_n = day
                ptr += 1
                continue
            if kind == "Y":
                if day >= line_start + PROGRESSION_GRACE_DAYS:
                    event_day = day
                    pfs_event = 1
                    line_source = "labelled_by_progression_event"
                    ptr += 1
                    break
                ptr += 1
                continue
            ptr += 1

        event_ptr = ptr

        if pfs_event == 1:
            assert event_day is not None
            next_idx = next_treatment_index(start_dates, start_idx, event_day, True, agents, True)
            if next_idx > start_idx:
                window_start = event_day - PROGRESSION_GRACE_DAYS
                mask = start_dates[start_idx:next_idx] > window_start
                if mask.any():
                    next_idx = int(np.argmax(mask)) + start_idx
            pfs_time = event_day - line_start
        elif last_n is not None:
            next_idx = split_idx
            event_day = int(last_n)
            pfs_event = 0
            pfs_time = event_day - line_start
            line_source = "labelled_by_non_progression"
        else:
            next_idx = split_idx
            event_day = line_start
            pfs_event = -1
            pfs_time = 0
            line_source = "no_radiology_follow_up"

        lines[start_idx:next_idx] = line_no

        if next_idx == n and pfs_event <= 0 and death_day is not None:
            if death_day >= line_start:
                event_day = int(death_day)
                pfs_event = 1
                pfs_time = event_day - line_start
                line_source = "labelled_by_death_event"

        prior_rad = has_prior_measurement(measurement_map, pid, line_start)
        original_event = None
        original_time = None
        if not prior_rad:
            original_event = int(pfs_event)
            original_time = int(max(0, pfs_time))
            pfs_event = -1
            line_source = f"no_radiology_within_{RADIOLOGY_LOOKBACK_DAYS}_prior"

        rows.append(
            {
                "patient_id": pid,
                "line": line_no,
                "line_start_day": line_start,
                "line_event_day": int(event_day),
                "pfs_time_days": int(max(0, pfs_time)),
                "pfs_event": int(pfs_event),
                "line_source": line_source,
                "has_radiology_within_90d_prior": bool(prior_rad),
                "original_pfs_time_days": original_time,
                "original_pfs_event": original_event,
            }
        )

        if next_idx >= n:
            break
        idx = next_idx
        line_no += 1

    tx["LINE"] = lines
    return tx, rows


def build_scan_day_aggregates(
    progression: pd.DataFrame,
    cancer_presence: pd.DataFrame,
    tumor_sites: pd.DataFrame,
    patient_ids: set[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    agg: dict[tuple[str, int], dict[str, Any]] = {}

    def get(pid: str, day: int) -> dict[str, Any]:
        key = (pid, day)
        if key not in agg:
            agg[key] = {
                "patient_id": pid,
                "day": day,
                "progression_values": [],
                "has_cancer_values": [],
                "modalities": set(),
                "tumor_sites": set(),
                "coverage": {r: False for r in REGIONS},
                "component_counts": Counter(),
            }
        return agg[key]

    for _, row in progression.iterrows():
        pid = clean_text(row.get("PATIENT_ID"))
        if pid not in patient_ids:
            continue
        day = pd.to_numeric(row.get("START_DATE"), errors="coerce")
        if pd.isna(day):
            continue
        a = get(pid, int(day))
        value = clean_text(row.get("PROGRESSION")).upper()
        if value:
            a["progression_values"].append(value)
        modality = clean_text(row.get("PROCEDURE_TYPE"))
        if modality:
            a["modalities"].add(modality)
        a["component_counts"]["progression"] += 1

    for _, row in cancer_presence.iterrows():
        pid = clean_text(row.get("PATIENT_ID"))
        if pid not in patient_ids:
            continue
        day = pd.to_numeric(row.get("START_DATE"), errors="coerce")
        if pd.isna(day):
            continue
        a = get(pid, int(day))
        value = clean_text(row.get("HAS_CANCER")).upper()
        if value:
            a["has_cancer_values"].append(value)
        modality = clean_text(row.get("PROCEDURE_TYPE"))
        if modality:
            a["modalities"].add(modality)
        for region in REGIONS:
            if region in row.index and boolish(row.get(region)):
                a["coverage"][region] = True
        a["component_counts"]["cancer_presence"] += 1

    for _, row in tumor_sites.iterrows():
        pid = clean_text(row.get("PATIENT_ID"))
        if pid not in patient_ids:
            continue
        day = pd.to_numeric(row.get("START_DATE"), errors="coerce")
        if pd.isna(day):
            continue
        a = get(pid, int(day))
        site = clean_text(row.get("TUMOR_SITE"))
        if site:
            a["tumor_sites"].add(site)
        modality = clean_text(row.get("SOURCE_SPECIFIC"))
        if modality:
            a["modalities"].add(modality)
        for region in REGIONS:
            if region in row.index and boolish(row.get(region)):
                a["coverage"][region] = True
        a["component_counts"]["tumor_sites"] += 1

    return agg


def build_scan_episodes(day_agg: dict[tuple[str, int], dict[str, Any]], window: int) -> pd.DataFrame:
    by_patient: dict[str, list[int]] = defaultdict(list)
    for pid, day in day_agg:
        by_patient[pid].append(day)

    rows: list[dict[str, Any]] = []
    for pid in sorted(by_patient):
        days = sorted(set(by_patient[pid]))
        clusters: list[list[int]] = []
        current: list[int] = []
        previous: int | None = None
        for day in days:
            if previous is None or day - previous <= window:
                current.append(day)
            else:
                clusters.append(current)
                current = [day]
            previous = day
        if current:
            clusters.append(current)

        for idx, cluster in enumerate(clusters, start=1):
            pvals: list[str] = []
            cancer_vals: list[str] = []
            modalities: set[str] = set()
            sites: set[str] = set()
            coverage = {r: False for r in REGIONS}
            counts: Counter[str] = Counter()
            for day in cluster:
                a = day_agg[(pid, day)]
                pvals.extend(a["progression_values"])
                cancer_vals.extend(a["has_cancer_values"])
                modalities.update(a["modalities"])
                sites.update(a["tumor_sites"])
                counts.update(a["component_counts"])
                for region in REGIONS:
                    coverage[region] = coverage[region] or bool(a["coverage"][region])

            pnorm = {x.upper() for x in pvals}
            if "Y" in pnorm:
                state = "PROGRESSIVE"
            elif "N" in pnorm:
                state = "NON_PROGRESSIVE"
            else:
                state = "INDETERMINATE"

            row: dict[str, Any] = {
                "patient_id": pid,
                "scan_episode_id": f"{pid}::SCAN::{idx:04d}::W{window}",
                "episode_window_days": window,
                "episode_start_day": min(cluster),
                "episode_end_day": max(cluster),
                "landmark_day": max(cluster),
                "episode_span_days": max(cluster) - min(cluster),
                "source_days_json": json.dumps(cluster),
                "progression_state_3": state,
                "progression_like_primary": True if state == "PROGRESSIVE" else (False if state == "NON_PROGRESSIVE" else None),
                "progression_discordant": "Y" in pnorm and "N" in pnorm,
                "progression_values_json": json.dumps(sorted(pnorm)),
                "has_cancer_values_json": json.dumps(sorted({x.upper() for x in cancer_vals})),
                "modalities_json": json.dumps(sorted(modalities)),
                "tumor_sites_json": json.dumps(sorted(sites)),
                "component_counts_json": json.dumps(dict(sorted(counts.items()))),
                "availability_quality": "CHORD_RELATIVE_RADIOLOGY_DAY_PROXY",
            }
            for region in REGIONS:
                row[f"coverage_{region.lower()}"] = bool(coverage[region])
                row[f"coverage_state_{region.lower()}"] = "IMAGED" if coverage[region] else "NOT_IMAGED"
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["patient_id", "landmark_day", "scan_episode_id"]).reset_index(drop=True)


def build_bpc_msk_scan_audit(bpc_root: Path) -> pd.DataFrame:
    path = find_one(bpc_root, "data_timeline_imaging.txt")
    df = read_table(path)
    if "INSTITUTION" not in df.columns:
        raise RuntimeError("BPC imaging timeline lacks INSTITUTION")
    work = df[df["INSTITUTION"].astype(str).str.upper() == "MSK"].copy()
    work["START_DATE_NUM"] = to_num(work["START_DATE"])
    work = work.dropna(subset=["START_DATE_NUM"])
    mapping = {
        "PROGRESSING/WORSENING/ENLARGING": "PROGRESSIVE",
        "MIXED": "MIXED",
        "STABLE/NO CHANGE": "STABLE",
        "IMPROVING/RESPONDING": "RESPONDING",
        "NOT STATED/INDETERMINATE": "INDETERMINATE",
    }
    state = work["CURATED_CANCER_STATUS"].astype(str).str.strip().str.upper().map(mapping).fillna("INDETERMINATE")
    out = pd.DataFrame(
        {
            "patient_id": work["PATIENT_ID"].astype(str),
            "scan_day": work["START_DATE_NUM"].astype(int),
            "scan_number": work.get("SCAN_NUMBER", "").astype(str),
            "response_state_5": state,
            "progression_like_primary": state.isin(["PROGRESSIVE", "MIXED"]),
            "scan_type": work.get("IMAGE_SCAN_TYPE", "").astype(str),
            "scan_sites": work.get("SCAN_SITES", "").astype(str),
            "cancer_status": work.get("CANCER_STATUS", "").astype(str),
            "source": "BPC_MSK",
        }
    )
    return out.sort_values(["patient_id", "scan_day", "scan_number"]).reset_index(drop=True)


def active_line_for_day(lines: pd.DataFrame, pid: str, day: int, progressive: bool = False) -> int | None:
    g = lines[lines["patient_id"] == pid].sort_values("line_start_day")
    if g.empty:
        return None
    if progressive:
        exact = g[(g["line_event_day"] == day) & (g["line_source"] == "labelled_by_progression_event")]
        if not exact.empty:
            return int(exact.iloc[0]["line"])
    starts = g["line_start_day"].to_numpy(dtype=int)
    idx = int(np.searchsorted(starts, day, side="right")) - 1
    if idx < 0:
        return None
    return int(g.iloc[idx]["line"])


def make_generic_events(
    chord_root: Path,
    table_name: str,
    event_type: str,
    breast_ids: set[str],
    availability_quality: str = "CHORD_RELATIVE_EVENT_DAY_PROXY",
) -> list[dict[str, Any]]:
    path = find_one(chord_root, table_name)
    df = read_table(path)
    if "PATIENT_ID" not in df.columns or "START_DATE" not in df.columns:
        return []
    work = df[df["PATIENT_ID"].isin(breast_ids)].copy()
    work["START_DATE_NUM"] = to_num(work["START_DATE"])
    work = work.dropna(subset=["START_DATE_NUM"])
    rows: list[dict[str, Any]] = []
    for source_idx, row in work.iterrows():
        pid = clean_text(row["PATIENT_ID"])
        day = int(row["START_DATE_NUM"])
        subtype = clean_text(row.get("SUBTYPE")) or clean_text(row.get("EVENT_TYPE"))
        rows.append(
            {
                "patient_id": pid,
                "event_day": day,
                "availability_day": day,
                "event_type": event_type,
                "event_subtype": subtype,
                "value_text": "",
                "value_num": np.nan,
                "unit": "",
                "treatment_line": np.nan,
                "source_table": table_name,
                "source_row_id": str(source_idx),
                "availability_quality": availability_quality,
                "coverage_json": "{}",
                "payload_json": json_safe_dict(row, {"PATIENT_ID", "START_DATE_NUM"}),
            }
        )
    return rows


def assign_lines_to_events(events: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    by_pid = {pid: g.sort_values("line_start_day") for pid, g in lines.groupby("patient_id", observed=True)}
    assigned: list[float] = []
    for pid, day, etype, payload in zip(
        events["patient_id"], events["event_day"], events["event_type"], events["payload_json"]
    ):
        g = by_pid.get(pid)
        if g is None or g.empty or pd.isna(day):
            assigned.append(np.nan)
            continue
        progressive = etype == "SCAN_EPISODE" and '"PROGRESSIVE"' in str(payload)
        if progressive:
            exact = g[(g["line_event_day"] == int(day)) & (g["line_source"] == "labelled_by_progression_event")]
            if not exact.empty:
                assigned.append(float(exact.iloc[0]["line"]))
                continue
        starts = g["line_start_day"].to_numpy(dtype=int)
        idx = int(np.searchsorted(starts, int(day), side="right")) - 1
        assigned.append(float(g.iloc[idx]["line"]) if idx >= 0 else np.nan)
    out = events.copy()
    existing = pd.to_numeric(out["treatment_line"], errors="coerce")
    computed = pd.Series(assigned, index=out.index)
    out["treatment_line"] = existing.fillna(computed)
    return out


def build_landmarks_and_targets(
    lines: pd.DataFrame,
    scans: pd.DataFrame,
    death_map: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    line_groups = {pid: g.sort_values("line_start_day").reset_index(drop=True) for pid, g in lines.groupby("patient_id", observed=True)}
    scan_groups = {pid: g.sort_values("landmark_day").reset_index(drop=True) for pid, g in scans.groupby("patient_id", observed=True)}
    landmarks: list[dict[str, Any]] = []
    next_scan_rows: list[dict[str, Any]] = []
    postprog_rows: list[dict[str, Any]] = []

    for pid, lg in line_groups.items():
        sg = scan_groups.get(pid, pd.DataFrame())
        for i, line in lg.iterrows():
            line_no = int(line["line"])
            line_start = int(line["line_start_day"])
            next_line_start = int(lg.iloc[i + 1]["line_start_day"]) if i + 1 < len(lg) else None

            def target_for(landmark_day: int) -> dict[str, Any]:
                candidates: list[tuple[int, int, str]] = []
                if not sg.empty:
                    prog = sg[(sg["progression_state_3"] == "PROGRESSIVE") & (sg["landmark_day"] > landmark_day)]
                    prog = prog[prog["treatment_line"] == line_no]
                    if not prog.empty:
                        candidates.append((int(prog.iloc[0]["landmark_day"]), 0, "PROGRESSION"))
                death = death_map.get(pid)
                if death is not None and death > landmark_day:
                    candidates.append((int(death), 1, "DEATH"))
                if next_line_start is not None and next_line_start > landmark_day:
                    candidates.append((next_line_start, 2, "SWITCH"))

                last_follow = None
                if not sg.empty:
                    follow = sg[(sg["landmark_day"] > landmark_day) & (sg["treatment_line"] == line_no)]
                    if not follow.empty:
                        last_follow = int(follow["landmark_day"].max())

                if candidates:
                    day, _, event_type = min(candidates)
                    raw_time = day - landmark_day
                elif last_follow is not None:
                    day = last_follow
                    event_type = "CENSOR"
                    raw_time = day - landmark_day
                else:
                    day = landmark_day
                    event_type = "NO_FOLLOWUP"
                    raw_time = 0

                if raw_time > TRAIN_HORIZON_DAYS:
                    train_time = TRAIN_HORIZON_DAYS
                    train_event = "ADMIN_CENSOR"
                else:
                    train_time = raw_time
                    train_event = event_type

                return {
                    "raw_event_day": int(day),
                    "raw_time_days": int(raw_time),
                    "raw_event_type": event_type,
                    "train_time_days": int(train_time),
                    "train_event_type": train_event,
                    "primary_pfs_event": int(train_event in {"PROGRESSION", "DEATH"}),
                    "competing_switch_event": int(train_event == "SWITCH"),
                    "has_followup": bool(raw_time > 0),
                }

            t = target_for(line_start)
            landmarks.append(
                {
                    "landmark_id": f"{pid}::LINE::{line_no:02d}",
                    "patient_id": pid,
                    "landmark_type": "LINE_START",
                    "landmark_day": line_start,
                    "treatment_line": line_no,
                    "scan_episode_id": "",
                    "scan_state": "",
                    "eligible_primary": bool(t["has_followup"]),
                    "eligible_extended": bool(t["has_followup"]),
                    **t,
                }
            )

            if sg.empty:
                continue
            current_scans = sg[sg["treatment_line"] == line_no].sort_values("landmark_day").reset_index(drop=True)
            if line["line_source"] in {"labelled_by_progression_event", "labelled_by_death_event"}:
                current_scans = current_scans[
                    current_scans["landmark_day"] <= int(line["line_event_day"]) + PRIMARY_SCAN_WINDOW
                ].reset_index(drop=True)
            if next_line_start is not None:
                current_scans = current_scans[current_scans["landmark_day"] < next_line_start].reset_index(drop=True)
            for j, scan in current_scans.iterrows():
                day = int(scan["landmark_day"])
                state = str(scan["progression_state_3"])
                if state == "PROGRESSIVE":
                    next_tx = next_line_start if next_line_start is not None and next_line_start > day else None
                    death = death_map.get(pid)
                    options: list[tuple[int, str]] = []
                    if next_tx is not None:
                        options.append((next_tx, "NEXT_TREATMENT"))
                    if death is not None and death > day:
                        options.append((int(death), "DEATH"))
                    if options:
                        trans_day, trans_type = min(options)
                        delta = trans_day - day
                    else:
                        trans_day, trans_type, delta = day, "CONTINUED_OR_CENSORED", 0
                    postprog_rows.append(
                        {
                            "patient_id": pid,
                            "scan_episode_id": scan["scan_episode_id"],
                            "progression_day": day,
                            "treatment_line": line_no,
                            "transition_type": trans_type,
                            "transition_day": int(trans_day),
                            "time_to_transition_days": int(delta),
                        }
                    )
                    continue

                t = target_for(day)
                landmarks.append(
                    {
                        "landmark_id": f"{pid}::SCAN::{scan['scan_episode_id']}",
                        "patient_id": pid,
                        "landmark_type": "SCAN",
                        "landmark_day": day,
                        "treatment_line": line_no,
                        "scan_episode_id": scan["scan_episode_id"],
                        "scan_state": state,
                        "eligible_primary": bool(state == "NON_PROGRESSIVE" and t["has_followup"]),
                        "eligible_extended": bool(state in {"NON_PROGRESSIVE", "INDETERMINATE"} and t["has_followup"]),
                        **t,
                    }
                )

                if j + 1 < len(current_scans):
                    nxt = current_scans.iloc[j + 1]
                    next_scan_rows.append(
                        {
                            "patient_id": pid,
                            "scan_episode_id": scan["scan_episode_id"],
                            "scan_day": day,
                            "treatment_line": line_no,
                            "next_scan_episode_id": nxt["scan_episode_id"],
                            "next_scan_day": int(nxt["landmark_day"]),
                            "time_to_next_scan_days": int(nxt["landmark_day"] - day),
                            "next_scan_state_3": str(nxt["progression_state_3"]),
                            "next_scan_response_state_5": "",
                        }
                    )

    landmark_df = pd.DataFrame(landmarks)
    if not landmark_df.empty:
        eligible_counts = (
            landmark_df[landmark_df["eligible_primary"]]
            .groupby("patient_id", observed=True)
            .size()
            .to_dict()
        )
        landmark_df["patient_landmark_count"] = (
            landmark_df["patient_id"].map(eligible_counts).fillna(0).astype(int)
        )
        landmark_df["patient_landmark_weight"] = np.where(
            landmark_df["eligible_primary"] & (landmark_df["patient_landmark_count"] > 0),
            1.0 / landmark_df["patient_landmark_count"].replace(0, np.nan),
            0.0,
        )
        landmark_df["patient_landmark_weight"] = landmark_df["patient_landmark_weight"].fillna(0.0)
        landmark_df = landmark_df.sort_values(["patient_id", "landmark_day", "landmark_type"]).reset_index(drop=True)
    return landmark_df, pd.DataFrame(next_scan_rows), pd.DataFrame(postprog_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="artifacts/checkpoint1")
    parser.add_argument("--config", default="configs/dynamic_scan/canonical_v1.json")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    global SCAN_WINDOWS, PRIMARY_SCAN_WINDOW, DAY_SPLIT, PROGRESSION_GRACE_DAYS
    global RADIOLOGY_LOOKBACK_DAYS, TRAIN_HORIZON_DAYS
    PRIMARY_SCAN_WINDOW = int(cfg["scan_episode"]["primary_window_days"])
    SCAN_WINDOWS = tuple(int(x) for x in cfg["scan_episode"]["sensitivity_windows_days"])
    DAY_SPLIT = int(cfg["line_construction"]["day_split_days"])
    PROGRESSION_GRACE_DAYS = int(cfg["line_construction"]["progression_grace_days"])
    RADIOLOGY_LOOKBACK_DAYS = int(cfg["line_construction"]["radiology_lookback_days"])
    TRAIN_HORIZON_DAYS = int(cfg["targets"]["administrative_horizon_days"])

    out = (repo / args.output_dir).resolve()
    data_out = out / "canonical"
    data_out.mkdir(parents=True, exist_ok=True)

    chord_root = repo / "data/external_sources/msk_chord_2024/raw"
    bpc_root = repo / "data/external_sources/bpc_brca_1_0_public"
    if not chord_root.exists() or not bpc_root.exists():
        raise SystemExit("[CKPT1_FATAL] required source roots are missing")

    print("[CKPT1] loading source tables", flush=True)
    sample = read_table(find_one(chord_root, "data_clinical_sample.txt"))
    clinical_patient = read_table(find_one(chord_root, "data_clinical_patient.txt"))
    diagnosis_raw = read_table(find_one(chord_root, "data_timeline_diagnosis.txt"))
    diagnosis = parse_diagnosis(diagnosis_raw)
    progression_raw = read_table(find_one(chord_root, "data_timeline_progression.txt"))
    cancer_presence_raw = read_table(find_one(chord_root, "data_timeline_cancer_presence.txt"))
    tumor_sites_raw = read_table(find_one(chord_root, "data_timeline_tumor_sites.txt"))
    treatment_raw = read_table(find_one(chord_root, "data_timeline_treatment.txt"))

    sample_breast, breast_only, stage4 = breast_cohorts(sample, diagnosis)
    metastasis = first_metastasis_dates(tumor_sites_raw, breast_only)

    print(
        f"[CKPT1] sample_breast={len(sample_breast)} breast_only={len(breast_only)} "
        f"stage4={len(stage4)} metastasis_dates={len(metastasis)}",
        flush=True,
    )

    tx_breast_all = treatment_raw[treatment_raw["PATIENT_ID"].isin(sample_breast)].copy()
    if "SUBTYPE" in tx_breast_all.columns:
        tx_breast_all = tx_breast_all[tx_breast_all["SUBTYPE"].astype(str) != "Bone Treatment"]
    if "AGENT" in tx_breast_all.columns:
        tx_breast_all = tx_breast_all[tx_breast_all["AGENT"].astype(str) != "INVESTIGATIVE"]
    tx_breast_all["START_DATE"] = to_num(tx_breast_all["START_DATE"])
    tx_breast_all = tx_breast_all.dropna(subset=["START_DATE"]).copy()
    tx_breast_all["START_DATE"] = tx_breast_all["START_DATE"].astype(int)
    if "STOP_DATE" in tx_breast_all.columns:
        tx_breast_all["STOP_DATE"] = to_num(tx_breast_all["STOP_DATE"])

    tx_for_lines = tx_breast_all[tx_breast_all["PATIENT_ID"].isin(breast_only)].copy()
    tx_mbc = filter_mbc_rows(tx_for_lines, breast_only, stage4, metastasis)
    if "STOP_DATE" in tx_mbc.columns:
        tx_mbc["STOP_DATE"] = to_num(tx_mbc["STOP_DATE"])

    progression_all = progression_raw[progression_raw["PATIENT_ID"].isin(breast_only)].copy()
    progression_all["START_DATE"] = to_num(progression_all["START_DATE"])
    progression_all = progression_all.dropna(subset=["START_DATE"])
    progression_all["START_DATE"] = progression_all["START_DATE"].astype(int)
    progression_all["PROGRESSION"] = progression_all["PROGRESSION"].astype(str).str.strip().str.upper()
    progression_for_lines = progression_all[progression_all["PROGRESSION"].isin(["Y", "N"])].copy()
    progression_mbc = filter_mbc_rows(progression_for_lines, breast_only, stage4, metastasis)

    model_ids = set(tx_mbc["PATIENT_ID"]) & set(progression_mbc["PATIENT_ID"])
    tx_mbc = tx_mbc[tx_mbc["PATIENT_ID"].isin(model_ids)].reset_index(drop=True)
    progression_mbc = progression_mbc[progression_mbc["PATIENT_ID"].isin(model_ids)].reset_index(drop=True)
    death_map = death_days(clinical_patient, model_ids)
    measurement_map = radiology_measurement_map(
        progression_mbc, cancer_presence_raw, tumor_sites_raw, model_ids
    )

    print(f"[CKPT1] metastatic modeling cohort patients={len(model_ids)}", flush=True)
    progression_groups = {
        pid: g[["START_DATE", "PROGRESSION"]].reset_index(drop=True)
        for pid, g in progression_mbc.groupby("PATIENT_ID", observed=True)
    }

    line_rows: list[dict[str, Any]] = []
    tx_line_parts: list[pd.DataFrame] = []
    for n, (pid, patient_tx) in enumerate(tx_mbc.groupby("PATIENT_ID", observed=True), start=1):
        if n == 1 or n % 500 == 0:
            print(f"[CKPT1] line assignment patient={n}/{len(model_ids)}", flush=True)
        patient_prog = progression_groups.get(pid, pd.DataFrame(columns=["START_DATE", "PROGRESSION"]))
        tx_labeled, rows = assign_patient_lines(
            patient_tx, patient_prog, measurement_map, death_map.get(str(pid))
        )
        tx_line_parts.append(tx_labeled)
        line_rows.extend(rows)

    tx_lines = pd.concat(tx_line_parts, ignore_index=True) if tx_line_parts else pd.DataFrame()
    lines = pd.DataFrame(line_rows).sort_values(["patient_id", "line"]).reset_index(drop=True)
    if not lines.empty:
        lines["next_line_start_day"] = lines.groupby("patient_id")["line_start_day"].shift(-1)
        lines["metastatic_onset_day"] = lines["patient_id"].map(metastasis)
        stage4_start = (
            diagnosis[diagnosis["PATIENT_ID"].isin(stage4)]
            .dropna(subset=["START_DATE_NUM"])
            .groupby("PATIENT_ID")["START_DATE_NUM"].min().to_dict()
        )
        for idx, row in lines.iterrows():
            if row["patient_id"] in stage4:
                lines.at[idx, "metastatic_onset_day"] = stage4_start.get(row["patient_id"], row["line_start_day"])

    print("[CKPT1] building scan episode sensitivity datasets", flush=True)
    scan_day_agg = build_scan_day_aggregates(
        progression_raw, cancer_presence_raw, tumor_sites_raw, sample_breast
    )
    scans_by_window: dict[int, pd.DataFrame] = {}
    for window in SCAN_WINDOWS:
        scans = build_scan_episodes(scan_day_agg, window)
        scans_by_window[window] = scans
        write_parquet_atomic(scans, data_out / f"chord_breast_scan_episodes_w{window}.parquet")
        print(f"[CKPT1] scan window={window} episodes={len(scans)}", flush=True)

    scans_primary = scans_by_window[PRIMARY_SCAN_WINDOW].copy()
    scans_mbc = scans_primary[scans_primary["patient_id"].isin(model_ids)].copy()
    if not scans_mbc.empty:
        lines_by_pid = {pid: g.sort_values("line_start_day").reset_index(drop=True) for pid, g in lines.groupby("patient_id", observed=True)}
        scan_lines: list[float] = []
        for _, scan in scans_mbc.iterrows():
            pid = scan["patient_id"]
            lg = lines_by_pid.get(pid)
            if lg is None or lg.empty:
                scan_lines.append(np.nan)
                continue
            day = int(scan["landmark_day"])
            if scan["progression_state_3"] == "PROGRESSIVE":
                episode_start = int(scan["episode_start_day"])
                episode_end = int(scan["episode_end_day"])
                exact = lg[
                    (lg["line_event_day"] >= episode_start)
                    & (lg["line_event_day"] <= episode_end)
                    & (lg["line_source"] == "labelled_by_progression_event")
                ]
                if not exact.empty:
                    scan_lines.append(float(exact.iloc[0]["line"]))
                    continue
            starts = lg["line_start_day"].to_numpy(dtype=int)
            idx = int(np.searchsorted(starts, day, side="right")) - 1
            scan_lines.append(float(lg.iloc[idx]["line"]) if idx >= 0 else np.nan)
        scans_mbc["treatment_line"] = scan_lines
        scans_mbc = scans_mbc.dropna(subset=["treatment_line"]).copy()
        scans_mbc["treatment_line"] = scans_mbc["treatment_line"].astype(int)

    print("[CKPT1] building canonical event stream", flush=True)
    event_rows: list[dict[str, Any]] = []

    # Treatment starts/stops for the full breast cohort. mBC lines are attached later where available.
    for source_idx, row in tx_breast_all.iterrows():
        pid = clean_text(row["PATIENT_ID"])
        day = int(row["START_DATE"])
        line = np.nan
        payload = json_safe_dict(row, {"PATIENT_ID"})
        event_rows.append(
            {
                "patient_id": pid,
                "event_day": day,
                "availability_day": day,
                "event_type": "TREATMENT_START",
                "event_subtype": clean_text(row.get("AGENT")) or clean_text(row.get("SUBTYPE")),
                "value_text": clean_text(row.get("AGENT")),
                "value_num": np.nan,
                "unit": "",
                "treatment_line": line,
                "source_table": "data_timeline_treatment.txt",
                "source_row_id": str(source_idx),
                "availability_quality": "CHORD_RELATIVE_EVENT_DAY_PROXY",
                "coverage_json": "{}",
                "payload_json": payload,
            }
        )
        stop = row.get("STOP_DATE")
        if pd.notna(stop) and float(stop) >= day:
            event_rows.append(
                {
                    "patient_id": pid,
                    "event_day": int(float(stop)),
                    "availability_day": int(float(stop)),
                    "event_type": "TREATMENT_END",
                    "event_subtype": clean_text(row.get("AGENT")) or clean_text(row.get("SUBTYPE")),
                    "value_text": clean_text(row.get("AGENT")),
                    "value_num": np.nan,
                    "unit": "",
                    "treatment_line": line,
                    "source_table": "data_timeline_treatment.txt",
                    "source_row_id": str(source_idx),
                    "availability_quality": "CHORD_RELATIVE_EVENT_DAY_PROXY",
                    "coverage_json": "{}",
                    "payload_json": payload,
                }
            )

    # Scan episode tokens cover the full breast cohort. The episode END is the availability cutoff.
    scan_line_map = (
        scans_mbc.set_index("scan_episode_id")["treatment_line"].to_dict()
        if not scans_mbc.empty
        else {}
    )
    for _, row in scans_primary.iterrows():
        coverage = {r.lower(): row[f"coverage_state_{r.lower()}"] for r in REGIONS}
        payload = {
            "progression_state_3": row["progression_state_3"],
            "progression_values": json.loads(row["progression_values_json"]),
            "has_cancer_values": json.loads(row["has_cancer_values_json"]),
            "modalities": json.loads(row["modalities_json"]),
            "tumor_sites": json.loads(row["tumor_sites_json"]),
            "episode_start_day": int(row["episode_start_day"]),
            "episode_end_day": int(row["episode_end_day"]),
            "episode_span_days": int(row["episode_span_days"]),
        }
        event_rows.append(
            {
                "patient_id": row["patient_id"],
                "event_day": int(row["landmark_day"]),
                "availability_day": int(row["landmark_day"]),
                "event_type": "SCAN_EPISODE",
                "event_subtype": row["progression_state_3"],
                "value_text": row["progression_state_3"],
                "value_num": np.nan,
                "unit": "",
                "treatment_line": scan_line_map.get(row["scan_episode_id"], np.nan),
                "source_table": "CHORD_RADIOLOGY_EPISODE_W3",
                "source_row_id": row["scan_episode_id"],
                "availability_quality": row["availability_quality"],
                "coverage_json": json.dumps(coverage, sort_keys=True),
                "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            }
        )

    generic_tables = [
        ("data_timeline_diagnosis.txt", "DIAGNOSIS"),
        ("data_timeline_performance_status.txt", "PERFORMANCE_STATUS"),
        ("data_timeline_pdl1.txt", "PDL1"),
        ("data_timeline_mmr.txt", "MMR"),
        ("data_timeline_prior_meds.txt", "PRIOR_MEDICATION"),
        ("data_timeline_radiation.txt", "RADIATION"),
        ("data_timeline_surgery.txt", "SURGERY"),
        ("data_timeline_specimen.txt", "SPECIMEN"),
        ("data_timeline_specimen_surgery.txt", "SPECIMEN_SURGERY"),
    ]
    for table, etype in generic_tables:
        try:
            event_rows.extend(make_generic_events(chord_root, table, etype, sample_breast))
        except RuntimeError:
            pass

    marker_tables = [
        ("data_timeline_ca_15-3_labs.txt", "CA15_3"),
        ("data_timeline_cea_labs.txt", "CEA"),
        ("data_timeline_ca_19-9_labs.txt", "CA19_9"),
        ("data_timeline_psa_labs.txt", "PSA"),
    ]
    for table, marker in marker_tables:
        path = find_one(chord_root, table)
        df = read_table(path)
        work = df[df["PATIENT_ID"].isin(sample_breast)].copy()
        work["START_DATE_NUM"] = to_num(work["START_DATE"])
        work = work.dropna(subset=["START_DATE_NUM"])
        for source_idx, row in work.iterrows():
            raw = clean_text(row.get("RESULT"))
            value_num = pd.to_numeric(raw, errors="coerce")
            event_rows.append(
                {
                    "patient_id": clean_text(row["PATIENT_ID"]),
                    "event_day": int(row["START_DATE_NUM"]),
                    "availability_day": int(row["START_DATE_NUM"]),
                    "event_type": "TUMOR_MARKER",
                    "event_subtype": marker,
                    "value_text": raw,
                    "value_num": float(value_num) if pd.notna(value_num) else np.nan,
                    "unit": clean_text(row.get("UNIT")),
                    "treatment_line": np.nan,
                    "source_table": table,
                    "source_row_id": str(source_idx),
                    "availability_quality": "CHORD_RELATIVE_EVENT_DAY_PROXY",
                    "coverage_json": "{}",
                    "payload_json": json_safe_dict(row, {"PATIENT_ID", "START_DATE_NUM"}),
                }
            )

    # Explicit genomic-availability proxy from SEQ_DATE, without pretending it is a report-release timestamp.
    specimen_surg = read_table(find_one(chord_root, "data_timeline_specimen_surgery.txt"))
    if "SEQ_DATE" in specimen_surg.columns:
        ss = specimen_surg[specimen_surg["PATIENT_ID"].isin(sample_breast)].copy()
        ss["SEQ_DAY"] = to_num(ss["SEQ_DATE"])
        ss = ss.dropna(subset=["SEQ_DAY"])
        for source_idx, row in ss.iterrows():
            event_rows.append(
                {
                    "patient_id": clean_text(row["PATIENT_ID"]),
                    "event_day": int(row["SEQ_DAY"]),
                    "availability_day": int(row["SEQ_DAY"]),
                    "event_type": "GENOMIC_RESULT_AVAILABLE",
                    "event_subtype": "SEQ_DATE_PROXY",
                    "value_text": clean_text(row.get("SAMPLE_ID")),
                    "value_num": np.nan,
                    "unit": "",
                    "treatment_line": np.nan,
                    "source_table": "data_timeline_specimen_surgery.txt",
                    "source_row_id": str(source_idx),
                    "availability_quality": "SEQ_DATE_PROXY_NOT_REPORT_TIME",
                    "coverage_json": "{}",
                    "payload_json": json_safe_dict(row, {"PATIENT_ID", "SEQ_DAY"}),
                }
            )

    events = pd.DataFrame(event_rows)
    events = events[events["patient_id"].isin(sample_breast)].copy()
    events["event_day"] = pd.to_numeric(events["event_day"], errors="coerce")
    events["availability_day"] = pd.to_numeric(events["availability_day"], errors="coerce")
    events = events.dropna(subset=["event_day", "availability_day"])
    events["event_day"] = events["event_day"].astype(int)
    events["availability_day"] = events["availability_day"].astype(int)
    events = assign_lines_to_events(events, lines)

    priorities = {
        "DIAGNOSIS": 10,
        "SPECIMEN": 20,
        "SPECIMEN_SURGERY": 21,
        "GENOMIC_RESULT_AVAILABLE": 25,
        "TREATMENT_START": 30,
        "TUMOR_MARKER": 40,
        "PERFORMANCE_STATUS": 45,
        "PDL1": 46,
        "MMR": 47,
        "SCAN_EPISODE": 50,
        "RADIATION": 60,
        "SURGERY": 61,
        "PRIOR_MEDICATION": 62,
        "TREATMENT_END": 70,
    }
    events["event_priority"] = events["event_type"].map(priorities).fillna(99).astype(int)
    events["event_id"] = [
        hashlib.sha1(
            f"{pid}|{etype}|{day}|{table}|{rowid}".encode("utf-8")
        ).hexdigest()[:20]
        for pid, etype, day, table, rowid in zip(
            events["patient_id"], events["event_type"], events["availability_day"], events["source_table"], events["source_row_id"]
        )
    ]
    events = events.sort_values(
        ["patient_id", "availability_day", "event_priority", "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    print("[CKPT1] building landmarks and targets", flush=True)
    landmarks, next_scan, postprog = build_landmarks_and_targets(lines, scans_mbc, death_map)
    if not landmarks.empty:
        history_counts = []
        grouped_days = {
            pid: np.sort(g["availability_day"].to_numpy(dtype=int))
            for pid, g in events.groupby("patient_id", observed=True)
        }
        for pid, day in zip(landmarks["patient_id"], landmarks["landmark_day"]):
            arr = grouped_days.get(pid, np.array([], dtype=int))
            history_counts.append(int(np.searchsorted(arr, int(day), side="right")))
        landmarks["history_event_count"] = history_counts
        landmarks["split"] = landmarks["patient_id"].map(stable_split)

    splits = pd.DataFrame(
        {"patient_id": sorted(model_ids), "split": [stable_split(pid) for pid in sorted(model_ids)]}
    )

    bpc_msk = build_bpc_msk_scan_audit(bpc_root)

    # Static source tables are preserved separately, with no timing assumptions injected.
    sample_static = sample[sample["PATIENT_ID"].isin(sample_breast)].copy()
    patient_static = clinical_patient[clinical_patient["PATIENT_ID"].isin(sample_breast)].copy()

    outputs = {
        "chord_breast_patient_static.parquet": patient_static,
        "chord_breast_sample_static.parquet": sample_static,
        "chord_mbc_treatment_rows.parquet": tx_lines,
        "chord_mbc_lines.parquet": lines,
        "chord_mbc_scan_episodes_w3.parquet": scans_mbc,
        "chord_canonical_events.parquet": events,
        "chord_landmarks.parquet": landmarks,
        "chord_next_scan_targets.parquet": next_scan,
        "chord_post_progression_targets.parquet": postprog,
        "chord_patient_splits.parquet": splits,
        "bpc_msk_scan_audit.parquet": bpc_msk,
    }
    for name, df in outputs.items():
        write_parquet_atomic(df, data_out / name)

    print("[CKPT1] running leakage and structural invariants", flush=True)
    errors: list[str] = []
    warnings: list[str] = []

    if len(sample_breast) != 5368:
        warnings.append(f"sample breast cohort is {len(sample_breast)}, expected release-scale 5368")
    if len(model_ids) == 0:
        errors.append("metastatic modeling cohort is empty")
    if lines.empty:
        errors.append("line table is empty")
    if scans_mbc.empty:
        errors.append("mBC scan episode table is empty")
    if events.empty:
        errors.append("canonical event stream is empty")
    if not events.empty and (events["availability_day"] < events["event_day"]).any():
        errors.append("availability_day precedes event_day")
    if not scans_mbc.empty and (scans_mbc["landmark_day"] != scans_mbc["episode_end_day"]).any():
        errors.append("scan landmark is not episode end day")
    if not landmarks.empty:
        bad_prog = landmarks[(landmarks["landmark_type"] == "SCAN") & (landmarks["scan_state"] == "PROGRESSIVE")]
        if not bad_prog.empty:
            errors.append("progressive scans leaked into residual-PFS landmarks")
        bad_time = landmarks[landmarks["eligible_primary"] & (landmarks["raw_time_days"] <= 0)]
        if not bad_time.empty:
            errors.append("eligible landmark has non-positive follow-up")
        if (landmarks["history_event_count"] <= 0).any():
            warnings.append("some landmarks have zero prior/current canonical events")
    if set(splits.groupby("split")["patient_id"].apply(set).get("train", set())) & set(
        splits.groupby("split")["patient_id"].apply(set).get("val", set())
    ):
        errors.append("train/val patient overlap")
    if set(splits.groupby("split")["patient_id"].apply(set).get("train", set())) & set(
        splits.groupby("split")["patient_id"].apply(set).get("test", set())
    ):
        errors.append("train/test patient overlap")
    if not bpc_msk.empty and set(bpc_msk["source"].unique()) != {"BPC_MSK"}:
        errors.append("non-MSK BPC data entered audit output")

    state_counts = scans_mbc["progression_state_3"].value_counts().to_dict() if not scans_mbc.empty else {}
    landmark_counts = landmarks["landmark_type"].value_counts().to_dict() if not landmarks.empty else {}
    target_counts = landmarks["train_event_type"].value_counts().to_dict() if not landmarks.empty else {}
    line_sources = lines["line_source"].value_counts().to_dict() if not lines.empty else {}
    split_counts = splits["split"].value_counts().to_dict()

    qc = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "sample_breast_patients": len(sample_breast),
            "diagnosis_breast_only_patients": len(breast_only),
            "stage4_patients": len(stage4),
            "patients_with_first_metastasis_date": len(metastasis),
            "mbc_modeling_patients": len(model_ids),
            "mbc_lines": len(lines),
            "mbc_treatment_rows": len(tx_lines),
            "mbc_scan_episodes_w3": len(scans_mbc),
            "canonical_events": len(events),
            "landmarks": len(landmarks),
            "eligible_primary_landmarks": int(landmarks["eligible_primary"].sum()) if not landmarks.empty else 0,
            "eligible_extended_landmarks": int(landmarks["eligible_extended"].sum()) if not landmarks.empty else 0,
            "next_scan_targets": len(next_scan),
            "post_progression_targets": len(postprog),
            "bpc_msk_scan_rows": len(bpc_msk),
            "bpc_msk_scan_patients": int(bpc_msk["patient_id"].nunique()) if not bpc_msk.empty else 0,
        },
        "scan_state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "landmark_type_counts": {str(k): int(v) for k, v in landmark_counts.items()},
        "training_target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "line_source_counts": {str(k): int(v) for k, v in line_sources.items()},
        "split_patient_counts": {str(k): int(v) for k, v in split_counts.items()},
        "soft_replication_reference": {
            "published_like_mbc_patients": 2881,
            "published_like_lines": 8791,
            "observed_patient_delta": len(model_ids) - 2881,
            "observed_line_delta": len(lines) - 8791,
        },
        "frozen_rules": {
            "scan_episode_primary_window_days": PRIMARY_SCAN_WINDOW,
            "scan_episode_landmark": "episode_end_day",
            "line_day_split": DAY_SPLIT,
            "progression_grace_days": PROGRESSION_GRACE_DAYS,
            "radiology_lookback_days": RADIOLOGY_LOOKBACK_DAYS,
            "training_horizon_days": TRAIN_HORIZON_DAYS,
            "chord_scan_state": "PROGRESSIVE/NON_PROGRESSIVE/INDETERMINATE; do not fabricate response-vs-stable",
            "chord_coverage": "source-derived region coverage flags; false means NOT_IMAGED, never disease-negative by itself",
            "genomic_timing": "SEQ_DATE proxy retained with explicit quality flag; not treated as report time",
            "external_policy": "only BPC-MSK scan values materialized; DFCI/VICC values are not materialized",
        },
    }
    atomic_json(out / "qc.json", qc)

    manifest = {
        "checkpoint": 1,
        "status": qc["status"],
        "canonical_dir": str(data_out.relative_to(repo)),
        "files": {
            p.name: {"bytes": p.stat().st_size}
            for p in sorted(data_out.glob("*.parquet"))
        },
        "qc": "artifacts/checkpoint1/qc.json",
    }
    atomic_json(out / "manifest.json", manifest)

    md: list[str] = []
    md.append("# Checkpoint 1 - Canonical Longitudinal Dataset")
    md.append("")
    md.append(f"Status: **{qc['status']}**")
    md.append("")
    md.append("## Cohort and line construction")
    md.append("")
    for k, v in qc["counts"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## Line sources")
    md.append("")
    for k, v in qc["line_source_counts"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## Scan states")
    md.append("")
    for k, v in qc["scan_state_counts"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## Dynamic targets")
    md.append("")
    for k, v in qc["training_target_counts"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## Important semantic decisions")
    md.append("")
    for k, v in qc["frozen_rules"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    if warnings:
        md.append("## Warnings")
        md.append("")
        for w in warnings:
            md.append(f"- {w}")
        md.append("")
    if errors:
        md.append("## Errors")
        md.append("")
        for e in errors:
            md.append(f"- {e}")
        md.append("")
    atomic_text(out / "audit.md", "\n".join(md) + "\n")

    handoff = {
        "checkpoint": 1,
        "name": "canonical_longitudinal_dataset_and_dynamic_landmarks",
        "status": qc["status"],
        "counts": qc["counts"],
        "frozen_rules": qc["frozen_rules"],
        "artifacts": {
            "manifest": "artifacts/checkpoint1/manifest.json",
            "qc": "artifacts/checkpoint1/qc.json",
            "audit": "artifacts/checkpoint1/audit.md",
            "canonical_dir": "artifacts/checkpoint1/canonical",
        },
        "next_required_action": (
            "Implement endpoint replication and strong line-start/scan-landmark baselines"
            if qc["status"] == "PASS"
            else "Resolve canonical dataset QC failures before modeling"
        ),
    }
    atomic_json(repo / "artifacts/handoff/checkpoint_01.json", handoff)

    print("")
    print("========== CKPT1 SUMMARY ==========")
    print(f"status={qc['status']}")
    for key, value in qc["counts"].items():
        print(f"{key}={value}")
    print(f"scan_state_counts={qc['scan_state_counts']}")
    print(f"line_source_counts={qc['line_source_counts']}")
    print(f"training_target_counts={qc['training_target_counts']}")
    print(f"split_patient_counts={qc['split_patient_counts']}")
    print(f"soft_replication_reference={qc['soft_replication_reference']}")
    print(f"warnings={qc['warnings']}")
    print(f"errors={qc['errors']}")
    print("audit_md=artifacts/checkpoint1/audit.md")
    print("handoff=artifacts/handoff/checkpoint_01.json")
    print("========== CKPT1 SUMMARY END ==========")

    print("")
    print("========== CKPT1 DECISION PACKET ==========")
    print("scan_windows:")
    for window in SCAN_WINDOWS:
        df = scans_by_window[window]
        print(
            f"  window_{window}d episodes={len(df)} patients={df['patient_id'].nunique()} "
            f"states={df['progression_state_3'].value_counts().to_dict()}"
        )
    print("line_sources:")
    for key, value in qc["line_source_counts"].items():
        print(f"  {key}={value}")
    print("targets:")
    for key, value in qc["training_target_counts"].items():
        print(f"  {key}={value}")
    print("availability_quality_counts:")
    print(f"  {events['availability_quality'].value_counts().to_dict()}")
    print("event_type_counts:")
    print(f"  {events['event_type'].value_counts().to_dict()}")
    print("BPC_MSK_response_state_counts:")
    print(f"  {bpc_msk['response_state_5'].value_counts().to_dict()}")
    print("========== CKPT1 DECISION PACKET END ==========")

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
