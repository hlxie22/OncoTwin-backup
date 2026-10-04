from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import traceback
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("ONCOTWIN_REPO", Path(__file__).resolve().parents[2])).resolve()
V2_DIR = REPO / "scripts/dynamic_scan_v2"
DS_DIR = REPO / "scripts/dynamic_scan"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(DS_DIR))

from v2_07_inference import (  # noqa: E402
    OncoTwinV2Predictor,
    PreparedLandmarkBatch,
    sha256_file,
)

DEPLOY = REPO / "artifacts/dynamic_scan_v2/v2_07/deploy_model.pt"
R1_ENCODER = REPO / "artifacts/checkpoint7r1_line_agnostic_temporal/temporal_encoder.pt"
SITE_VOCAB = REPO / "artifacts/checkpoint4/prepared/site_vocab_32.txt"
RELEASE_MANIFEST = REPO / "artifacts/dynamic_scan_v2/v2_07/release_manifest.json"
FIXTURE_REPORT = REPO / "artifacts/dynamic_scan_v2/v2_07/inference_fixture_report.json"

EXPECTED_DEPLOY_SHA256 = "a2de9909b2f7323dbafb1e9d9343946815bea160791b53f10eb31b85c4eaa4be"
EXPECTED_R1_SHA256 = "daa47e1d30d61a3c3a6c86a146c683a0a1ac0a635e0d8e6247f3ce47c4ae1411"
ADAPTER_VERSION = "app_cp2_r1_temporal_v1"


def _load_r1():
    path = REPO / "scripts/dynamic_scan/ckpt7r1_line_agnostic_temporal.py"
    spec = importlib.util.spec_from_file_location("oncotwin_r1_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load repaired temporal module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1 = _load_r1()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return hashlib.sha256(x.tobytes()).hexdigest()


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text[:10], text):
        try:
            return date.fromisoformat(candidate)
        except Exception:
            pass
    return None


def attrs(fact: dict[str, Any]) -> dict[str, Any]:
    raw = fact.get("attributes")
    if isinstance(raw, dict):
        return raw
    value_json = fact.get("value_json")
    if isinstance(value_json, dict):
        nested = value_json.get("attributes")
        if isinstance(nested, dict):
            return nested
    return {}


def fact_value(fact: dict[str, Any]) -> str:
    if fact.get("value") is not None:
        return str(fact.get("value") or "").strip()
    value_json = fact.get("value_json")
    if isinstance(value_json, dict):
        return str(value_json.get("value") or "").strip()
    return ""


def scan_state_known(value: str) -> bool:
    t = value.strip().lower().replace("_", " ")
    if not t:
        return False
    tokens = (
        "progress",
        "stable",
        "response",
        "responding",
        "non progressive",
        "nonprogressive",
        "indeterminate",
        "equivocal",
        "uncertain",
        "mixed",
    )
    return any(x in t for x in tokens)


def metastatic_anchor(facts: list[dict[str, Any]]) -> tuple[date | None, str | None]:
    explicit: list[tuple[date, str]] = []
    inferred: list[tuple[date, str]] = []
    for f in facts:
        d = parse_date(f.get("effective_date"))
        if d is None:
            continue
        a = attrs(f)
        if bool(a.get("model_clock_anchor")) or f.get("fact_type") in {"metastatic_anchor", "metastatic_diagnosis"}:
            explicit.append((d, str(f.get("id") or "")))
            continue
        if f.get("fact_type") != "diagnosis":
            continue
        text = (fact_value(f) + " " + canonical_json(a)).lower()
        if "metastatic" in text or "stage iv" in text or "stage 4" in text:
            inferred.append((d, str(f.get("id") or "")))
    if explicit:
        return min(explicit, key=lambda x: x[0])
    if inferred:
        return min(inferred, key=lambda x: x[0])
    return None, None


def same_day_facts(facts: list[dict[str, Any]], fact_type: str, d: date) -> list[dict[str, Any]]:
    return [f for f in facts if f.get("fact_type") == fact_type and parse_date(f.get("effective_date")) == d]


def build_app_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    patient_id = str(payload.get("patient_id") or "").strip()
    facts = list(payload.get("facts") or [])
    facts = [f for f in facts if str(f.get("status") or "active") == "active"]

    scans = []
    for f in facts:
        if f.get("fact_type") != "scan_assessment":
            continue
        d = parse_date(f.get("effective_date"))
        if d is None:
            continue
        scans.append((d, str(f.get("id") or ""), f))
    scans.sort(key=lambda x: (x[0], x[1]))

    support_reasons: list[str] = []
    support_explanations: list[str] = []

    if not scans:
        return {
            "supported": False,
            "support_status": "UNAVAILABLE",
            "support_reasons": ["NO_DATED_SCAN_ASSESSMENT"],
            "support_explanations": ["A dated scan assessment is required before the research forecast can be calculated."],
        }

    current_date, current_fact_id, current_fact = scans[-1]
    current_value = fact_value(current_fact)
    if not scan_state_known(current_value):
        return {
            "supported": False,
            "support_status": "UNAVAILABLE",
            "support_reasons": ["ABSTAIN_UNKNOWN_CURRENT_SCAN_STATE"],
            "support_explanations": ["The latest scan does not have a verified stable, progression, or indeterminate assessment."],
            "current_scan_fact_id": current_fact_id,
            "current_scan_date": current_date.isoformat(),
        }

    anchor, anchor_fact_id = metastatic_anchor(facts)
    if anchor is None:
        return {
            "supported": False,
            "support_status": "UNAVAILABLE",
            "support_reasons": ["ABSTAIN_UNKNOWN_REQUIRED_CLOCK"],
            "support_explanations": ["A dated metastatic-disease anchor is required to place the cancer history on the model's time axis."],
            "current_scan_fact_id": current_fact_id,
            "current_scan_date": current_date.isoformat(),
        }
    if current_date < anchor:
        return {
            "supported": False,
            "support_status": "UNAVAILABLE",
            "support_reasons": ["CURRENT_SCAN_PRECEDES_MODEL_CLOCK_ANCHOR"],
            "support_explanations": ["The latest scan date precedes the metastatic-disease anchor, so the model clock cannot be constructed safely."],
            "current_scan_fact_id": current_fact_id,
            "current_scan_date": current_date.isoformat(),
        }

    valid_scans = [(d, fid, f) for d, fid, f in scans if d >= anchor and d <= current_date and scan_state_known(fact_value(f))]
    if len(valid_scans) < 2:
        support_reasons.append("SPARSE_SCAN_HISTORY")
        support_explanations.append("Only one dated scan assessment is available on the model time axis; the forecast can run but has limited longitudinal context.")

    genomic_facts = [f for f in facts if f.get("fact_type") in {"genomic_assay", "genomic_alteration"}]
    if genomic_facts:
        support_reasons.append("GENOMICS_PRESENT_NOT_MODEL_ENCODED")
        support_explanations.append(
            "Genomic results are present in the patient record, but the app does not infer the frozen 468-gene model tensor from positive-only report facts. The model therefore uses its validated genomics-unavailable pathway."
        )

    def day_of(d: date) -> int:
        return int((d - anchor).days)

    # Only event classes that can be mapped without inventing line identity or unsupported numeric semantics.
    canonical_records: list[dict[str, Any]] = []
    for f in facts:
        d = parse_date(f.get("effective_date"))
        if d is None or d > current_date:
            continue
        ft = str(f.get("fact_type") or "")
        value = fact_value(f)
        event_type = None
        if ft in {"metastatic_anchor", "metastatic_diagnosis"}:
            event_type = "DIAGNOSIS"
        elif ft == "diagnosis":
            event_type = "DIAGNOSIS"
        elif ft == "treatment":
            event_type = "TREATMENT_END" if attrs(f).get("active") is False else "TREATMENT_START"
        elif ft == "lab":
            lab_text = (value + " " + canonical_json(attrs(f))).lower()
            if any(x in lab_text for x in ("ca 15-3", "ca15-3", "ca 27.29", "ca27.29", "cea")):
                event_type = "TUMOR_MARKER"
        if event_type is None:
            continue
        canonical_records.append(
            {
                "patient_id": patient_id,
                "availability_day": float(day_of(d)),
                "event_type": event_type,
                "treatment_line": 0,
                "source": "APP_COMMITTED_FACT",
                "payload_text": f"{ft}={value}",
                "value_numeric": np.nan,
            }
        )

    if not canonical_records:
        canonical_records.append(
            {
                "patient_id": patient_id,
                "availability_day": 0.0,
                "event_type": "DIAGNOSIS",
                "treatment_line": 0,
                "source": "APP_COMMITTED_FACT",
                "payload_text": "metastatic_disease_anchor",
                "value_numeric": np.nan,
            }
        )

    scan_records: list[dict[str, Any]] = []
    for index, (d, fid, f) in enumerate(valid_scans, start=1):
        state_text = fact_value(f).strip().lower()
        modalities = [fact_value(x) for x in same_day_facts(facts, "scan_modality", d) if fact_value(x)]
        sites = [fact_value(x) for x in same_day_facts(facts, "disease_site", d) if fact_value(x)]
        scan_id = f"APP_SCAN_{fid or index}"
        day = day_of(d)
        scan_records.append(
            {
                "patient_id": patient_id,
                "scan_episode_id": scan_id,
                "bpc_episode_id": scan_id,
                "episode_id": scan_id,
                "landmark_day": day,
                "day": day,
                "episode_start_day": day,
                "episode_end_day": day,
                "progression_state_3": state_text,
                "scan_state": state_text,
                "treatment_line": 0,
                "line": 0,
                "modalities_json": json.dumps(sorted(set(modalities))),
                "tumor_sites_json": json.dumps(sorted(set(sites))),
                "source": "APP_COMMITTED_FACT",
                "source_table": "app_committed_facts",
                "source_row_id": fid,
                "scan_number_patient": index,
            }
        )

    current_scan_id = scan_records[-1]["scan_episode_id"]
    return {
        "supported": True,
        "support_status": "LIMITED" if support_reasons else "STANDARD",
        "support_reasons": support_reasons,
        "support_explanations": support_explanations,
        "current_scan_fact_id": current_fact_id,
        "current_scan_id": current_scan_id,
        "current_scan_date": current_date.isoformat(),
        "current_scan_assessment": current_value,
        "anchor_date": anchor.isoformat(),
        "anchor_fact_id": anchor_fact_id,
        "canonical": canonical_records,
        "scan_table": scan_records,
        "scan_number_patient": len(scan_records),
    }


def build_temporal_prepost(app_inputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    canonical = pd.DataFrame(app_inputs["canonical"])
    scans = pd.DataFrame(app_inputs["scan_table"])
    patient_id = str(scans.iloc[-1]["patient_id"])
    current_scan_id = str(app_inputs["current_scan_id"])

    with tempfile.TemporaryDirectory(prefix="oncotwin_cp2_r1_") as td:
        out = Path(td)
        prep = out / "prepared"
        prep.mkdir(parents=True, exist_ok=True)
        canonical_path = prep / "app_canonical_events.parquet"
        scan_path = prep / "app_scans.parquet"
        canonical.to_parquet(canonical_path, index=False)
        scans.to_parquet(scan_path, index=False)
        splits = pd.DataFrame({"patient_id": [patient_id], "split": ["test"]})
        site_vocab = [x.strip() for x in SITE_VOCAB.read_text(encoding="utf-8").splitlines() if x.strip()]

        breast_base = R1.prepare_breast(canonical_path, scan_path, splits, site_vocab)
        tokens = R1.add_time_and_sequence_targets(breast_base)
        tokens.to_parquet(prep / "breast_tokens.parquet", index=False)

        scan_index = (
            breast_base[breast_base["type_id"] == R1.EVENT_TO_ID["SCAN_EPISODE"]][
                [
                    "patient_id",
                    "day",
                    "line",
                    "scan_state",
                    "scan_episode_id",
                    "site_mask",
                    "site_known",
                    "region_imaged_mask",
                    "region_known_mask",
                    "split",
                ]
            ]
            .copy()
            .sort_values(["patient_id", "day", "scan_episode_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        if scan_index.empty:
            raise RuntimeError("R1 preparation produced no scan tokens")
        scan_index.to_parquet(prep / "breast_scan_index.parquet", index=False)
        shutil.copy2(R1_ENCODER, out / "temporal_encoder.pt")
        cache_qc = R1.cache_scan_states(out, 32)

        emb_index = pd.read_parquet(out / "breast_scan_prepost_index.parquet").reset_index(drop=True)
        embeddings = np.asarray(np.load(out / "breast_scan_prepost_embeddings_f16.npy"), dtype=np.float32)
        if embeddings.shape != (len(emb_index), 2, 192):
            raise RuntimeError(f"Unexpected repaired temporal cache shape: {embeddings.shape}")

        matches = emb_index[
            (emb_index["patient_id"].astype(str) == patient_id)
            & (emb_index["scan_episode_id"].astype(str) == current_scan_id)
        ]
        if len(matches) != 1:
            # Defensive fallback to the unique latest scan day; do not use line identity.
            current_day = int(scans.iloc[-1]["landmark_day"])
            matches = emb_index[
                (emb_index["patient_id"].astype(str) == patient_id)
                & (pd.to_numeric(emb_index["day"], errors="coerce") == current_day)
            ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely align current scan to repaired R1 cache: scan_id={current_scan_id} matches={len(matches)}"
            )
        row = int(matches.index[0])
        pre = embeddings[row, 0].astype(np.float32)
        post = embeddings[row, 1].astype(np.float32)
        return pre, post, {
            "cache_qc": cache_qc,
            "temporal_row": row,
            "scan_rows": int(len(emb_index)),
            "current_scan_id": current_scan_id,
        }


def _curve_dict(pred: dict[str, Any], key: str) -> list[float]:
    return np.asarray(pred[key]["pfs_survival"][0], dtype=np.float32).tolist()


class Runtime:
    def __init__(self):
        if sha256_file(DEPLOY) != EXPECTED_DEPLOY_SHA256:
            raise RuntimeError("Frozen V2-07 deploy hash mismatch")
        if sha256_file(R1_ENCODER) != EXPECTED_R1_SHA256:
            raise RuntimeError("Frozen CKPT7R1 temporal encoder hash mismatch")
        self.predictor = OncoTwinV2Predictor(DEPLOY, device=os.environ.get("ONCOTWIN_MODEL_DEVICE", "cpu"))
        self.upstream_checks = self.predictor.verify_upstream_files(REPO)
        self.release_manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "checkpoint": "V2-07",
            "family": "r1_temporal_fixed",
            "adapter_version": ADAPTER_VERSION,
            "locked_alpha": 0.52,
            "deploy_sha256": EXPECTED_DEPLOY_SHA256,
            "temporal_encoder_sha256": EXPECTED_R1_SHA256,
            "upstream_hashes_verified": bool(all(self.upstream_checks.values())),
            "external_confirmation": "PENDING_NEW_INDEPENDENT_COHORT",
        }

    def predict_app(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_hash = sha256_json(payload)
        with self._lock:
            if request_hash in self._cache:
                return dict(self._cache[request_hash])

        app_inputs = build_app_inputs(payload)
        base = {
            "request_hash": request_hash,
            "checkpoint": "V2-07",
            "family": "r1_temporal_fixed",
            "adapter_version": ADAPTER_VERSION,
            "locked_alpha": 0.52,
            "deploy_sha256": EXPECTED_DEPLOY_SHA256,
            "temporal_encoder_sha256": EXPECTED_R1_SHA256,
            "external_confirmation": "PENDING_NEW_INDEPENDENT_COHORT",
            "current_scan_fact_id": app_inputs.get("current_scan_fact_id"),
            "current_scan_date": app_inputs.get("current_scan_date"),
            "current_scan_assessment": app_inputs.get("current_scan_assessment"),
        }
        if not app_inputs.get("supported"):
            result = {
                **base,
                "supported": False,
                "support_status": app_inputs["support_status"],
                "support_reasons": app_inputs["support_reasons"],
                "support_explanations": app_inputs["support_explanations"],
            }
            with self._lock:
                self._cache[request_hash] = dict(result)
            return result

        pre, post, temporal_meta = build_temporal_prepost(app_inputs)

        # Exact V2-07 portable context formulas from the repaired transport path.
        scan_number_patient = float(app_inputs["scan_number_patient"])
        scan_number_patient_log = np.asarray([np.log1p(scan_number_patient) / 5.0], dtype=np.float32)

        # CP1 stores positive genomic report facts but not the exact 468x2 frozen gene tensor / tested-negative coverage.
        # Never synthesize that tensor. Use the model's verified unavailable-genomics path instead.
        tumor = np.zeros((1, 128), dtype=np.float32)
        genomic_available = np.zeros(1, dtype=np.float32)
        genomic_age_scaled = np.zeros(1, dtype=np.float32)

        batch = PreparedLandmarkBatch(
            temporal_pre=pre.reshape(1, 192),
            temporal_post=post.reshape(1, 192),
            tumor_embedding=tumor,
            scan_number_patient_log=scan_number_patient_log,
            genomic_available=genomic_available,
            genomic_age_scaled=genomic_age_scaled,
            current_scan_state_known=np.ones(1, dtype=np.uint8),
            required_clock_known=np.ones(1, dtype=np.uint8),
        )
        prediction = self.predictor.predict(batch, batch_size=1)
        supported = bool(np.asarray(prediction["supported"])[0])
        if not supported:
            reason = str(np.asarray(prediction["status"], dtype=object)[0])
            return {
                **base,
                "supported": False,
                "support_status": "UNAVAILABLE",
                "support_reasons": [reason],
                "support_explanations": ["The frozen model abstained for this landmark."],
            }

        months = list(range(1, 25))
        pre_pfs = _curve_dict(prediction, "pre_curves")
        post_pfs = _curve_dict(prediction, "post_curves")
        selected_pfs = _curve_dict(prediction, "selected_curves")
        horizons = {}
        for month in (3, 6, 12, 18):
            idx = month - 1
            horizons[str(month)] = {
                "pre_pfs": pre_pfs[idx],
                "selected_pfs": selected_pfs[idx],
                "post_pfs": post_pfs[idx],
                "selected_progression_or_death_cif": float(prediction["selected_horizon_progression_or_death_cif"][0][(3, 6, 12, 18).index(month)]),
                "selected_switch_cif": float(prediction["selected_horizon_switch_cif"][0][(3, 6, 12, 18).index(month)]),
            }

        prepared_hashes = {
            "temporal_pre_sha256": sha256_array(pre),
            "temporal_post_sha256": sha256_array(post),
            "tumor_embedding_sha256": sha256_array(tumor),
            "portable_context_sha256": sha256_array(
                np.stack([scan_number_patient_log, genomic_available, genomic_age_scaled], axis=1)
            ),
        }
        result = {
            **base,
            "supported": True,
            "support_status": app_inputs["support_status"],
            "support_reasons": app_inputs["support_reasons"],
            "support_explanations": app_inputs["support_explanations"],
            "anchor_date": app_inputs["anchor_date"],
            "anchor_fact_id": app_inputs["anchor_fact_id"],
            "scan_number_patient": int(app_inputs["scan_number_patient"]),
            "prepared_hashes": prepared_hashes,
            "temporal_meta": temporal_meta,
            "portable_context": {
                "scan_number_patient_log": float(scan_number_patient_log[0]),
                "genomic_available": 0.0,
                "genomic_age_scaled": 0.0,
            },
            "curves": {
                "months": months,
                "pre_pfs": pre_pfs,
                "selected_pfs": selected_pfs,
                "post_pfs": post_pfs,
            },
            "horizons": horizons,
            "pre_logits": np.asarray(prediction["pre_logits"][0], dtype=np.float32).tolist(),
            "post_logits": np.asarray(prediction["post_logits"][0], dtype=np.float32).tolist(),
            "selected_logits": np.asarray(prediction["selected_logits"][0], dtype=np.float32).tolist(),
        }
        with self._lock:
            self._cache[request_hash] = dict(result)
        return result

    def release_fixture_test(self) -> dict[str, Any]:
        report = json.loads(FIXTURE_REPORT.read_text(encoding="utf-8"))
        input_sha = report["input_sha256"]
        expected_sha = report["expected_sha256"]
        files = [p for p in (REPO / "artifacts/dynamic_scan_v2/v2_07").iterdir() if p.is_file()]
        by_hash = {sha256_file(p): p for p in files}
        if input_sha not in by_hash:
            raise RuntimeError("Frozen inference fixture input hash not found")
        if expected_sha not in by_hash:
            raise RuntimeError("Frozen inference fixture expected-output hash not found")
        fixture = by_hash[input_sha]
        with np.load(fixture, allow_pickle=False) as z:
            batch = PreparedLandmarkBatch(
                temporal_pre=np.asarray(z["temporal_pre"]),
                temporal_post=np.asarray(z["temporal_post"]),
                tumor_embedding=np.asarray(z["tumor_embedding"]),
                scan_number_patient_log=np.asarray(z["scan_number_patient_log"]),
                genomic_available=np.asarray(z["genomic_available"]),
                genomic_age_scaled=np.asarray(z["genomic_age_scaled"]),
                current_scan_state_known=np.asarray(z["current_scan_state_known"]) if "current_scan_state_known" in z.files else None,
                required_clock_known=np.asarray(z["required_clock_known"]) if "required_clock_known" in z.files else None,
            )
        a = self.predictor.predict(batch, batch_size=23)
        b = self.predictor.predict(batch, batch_size=1)
        logit_delta = float(np.nanmax(np.abs(a["selected_logits"] - b["selected_logits"])))
        curve_delta = float(np.nanmax(np.abs(a["selected_curves"]["pfs_survival"] - b["selected_curves"]["pfs_survival"])))
        if logit_delta > 3e-6 or curve_delta > 5e-7:
            raise RuntimeError(f"Frozen fixture batch invariance failed: logit={logit_delta} curve={curve_delta}")

        modified_tumor = np.asarray(batch.tumor_embedding, dtype=np.float32).copy()
        unavailable = np.asarray(batch.genomic_available).reshape(-1) == 0
        modified_tumor[unavailable] += np.float32(123.456)
        modified_batch = PreparedLandmarkBatch(
            batch.temporal_pre,
            batch.temporal_post,
            modified_tumor,
            batch.scan_number_patient_log,
            batch.genomic_available,
            batch.genomic_age_scaled,
            batch.current_scan_state_known,
            batch.required_clock_known,
        )
        c = self.predictor.predict(modified_batch, batch_size=23)
        genomic_delta = float(np.nanmax(np.abs(a["selected_logits"] - c["selected_logits"])))
        if genomic_delta != 0.0:
            raise RuntimeError(f"Unavailable-genomics invariance failed: {genomic_delta}")
        return {
            "fixture_input": fixture.name,
            "fixture_expected": by_hash[expected_sha].name,
            "batch_logit_max_abs": logit_delta,
            "batch_pfs_curve_max_abs": curve_delta,
            "unavailable_genomics_max_abs": genomic_delta,
            "rows": int(batch.validate()),
        }

    def adapter_smoke_test(self) -> dict[str, Any]:
        payload = {
            "patient_id": "cp2-synthetic",
            "state_hash": "synthetic",
            "facts": [
                {"id": "dx", "fact_type": "diagnosis", "value": "metastatic breast cancer", "effective_date": "2025-01-01", "status": "active", "attributes": {"model_clock_anchor": True}},
                {"id": "tx", "fact_type": "treatment", "value": "capecitabine", "effective_date": "2025-01-15", "status": "active", "attributes": {"active": True}},
                {"id": "s1", "fact_type": "scan_assessment", "value": "stable", "effective_date": "2025-03-01", "status": "active", "attributes": {}},
                {"id": "m1", "fact_type": "scan_modality", "value": "CT", "effective_date": "2025-03-01", "status": "active", "attributes": {}},
                {"id": "site1", "fact_type": "disease_site", "value": "liver", "effective_date": "2025-03-01", "status": "active", "attributes": {}},
                {"id": "s2", "fact_type": "scan_assessment", "value": "progression", "effective_date": "2025-05-15", "status": "active", "attributes": {}},
                {"id": "m2", "fact_type": "scan_modality", "value": "CT", "effective_date": "2025-05-15", "status": "active", "attributes": {}},
                {"id": "site2", "fact_type": "disease_site", "value": "liver", "effective_date": "2025-05-15", "status": "active", "attributes": {}},
            ],
        }
        out = self.predict_app(payload)
        if not out.get("supported"):
            raise RuntimeError(f"Synthetic adapter smoke abstained: {out}")
        if len(out["curves"]["selected_pfs"]) != 24:
            raise RuntimeError("Synthetic adapter did not return 24-month PFS curve")
        return {
            "support_status": out["support_status"],
            "current_scan_date": out["current_scan_date"],
            "scan_number_patient": out["scan_number_patient"],
            "prepared_hashes": out["prepared_hashes"],
            "selected_pfs_6m": out["horizons"]["6"]["selected_pfs"],
        }


RUNTIME: Runtime | None = None


def get_runtime() -> Runtime:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = Runtime()
    return RUNTIME


class Handler(BaseHTTPRequestHandler):
    server_version = "OncoTwinFrozenModel/2"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[MODEL_HTTP] " + (fmt % args), flush=True)

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/healthz":
                self._send(200, get_runtime().health())
                return
            self._send(404, {"error": "not_found"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/predict":
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > 2_000_000:
                self._send(413, {"error": "invalid_request_size"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = get_runtime().predict_app(payload)
            self._send(200, result)
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": str(exc), "type": type(exc).__name__})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    runtime = get_runtime()
    if args.self_test:
        packet = {
            "health": runtime.health(),
            "release_fixture": runtime.release_fixture_test(),
            "adapter_smoke": runtime.adapter_smoke_test(),
        }
        print(json.dumps(packet, indent=2, sort_keys=True))
        return 0

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "model_service_ready", **runtime.health(), "host": args.host, "port": args.port}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
