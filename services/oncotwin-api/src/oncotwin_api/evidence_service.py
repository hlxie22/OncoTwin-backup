from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .demo_fixtures import fixture_mode_enabled, fixture_root
from .intelligence_schemas import EvidenceSummaryLLM, TrialRelevanceLLM
from .llm_router import RouterUnavailable, run_structured
from .models import GeneratedArtifact, PatientStateSnapshot
from .state import canonical_json

RETRIEVAL_VERSION = "cp4.evidence.v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _latest_state(db: Session, patient_id: str) -> PatientStateSnapshot:
    state = db.scalar(
        select(PatientStateSnapshot)
        .where(PatientStateSnapshot.patient_id == patient_id)
        .order_by(PatientStateSnapshot.created_at.desc())
        .limit(1)
    )
    if state is None:
        raise ValueError("Patient state is not available")
    return state


def build_evidence_queries(state: dict[str, Any]) -> dict[str, str]:
    diagnosis = str(state.get("diagnosis") or "metastatic breast cancer")
    subtype = str(state.get("subtype") or "")
    latest_scan = state.get("latest_scan") or {}
    genomics = [str(x.get("alteration")) for x in (state.get("genomics") or []) if x.get("alteration")]
    receptor_bits = []
    receptors = state.get("receptors") or {}
    if receptors.get("ER") == "positive":
        receptor_bits.append("ER-positive")
    if receptors.get("HER2") == "negative":
        receptor_bits.append("HER2-negative")
    core = " ".join([diagnosis, subtype, *receptor_bits]).strip()
    if genomics:
        core += " " + " ".join(genomics[-2:])
    if latest_scan.get("assessment") == "progression":
        core += " progression"
    core = re.sub(r"\s+", " ", core).strip()
    return {
        "literature": core,
        "trials": core,
    }


def _pubmed_search(query: str, limit: int) -> list[dict[str, Any]]:
    settings = get_settings()
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(limit),
        "sort": "pub date",
        "tool": "oncotwin",
    }
    if settings.ncbi_email:
        params["email"] = settings.ncbi_email
    with httpx.Client(timeout=settings.evidence_timeout_seconds, headers={"User-Agent": "OncoTwin/0.1 research-demo"}) as client:
        res = client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
        res.raise_for_status()
        ids = (res.json().get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return []
        efetch = client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={**{"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "oncotwin"}, **({"email": settings.ncbi_email} if settings.ncbi_email else {})},
        )
        efetch.raise_for_status()
    root = ET.fromstring(efetch.text)
    rows: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = "".join(article.findtext(".//PMID", default=""))
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else "Untitled PubMed record"
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            if text:
                label = node.attrib.get("Label")
                abstract_parts.append(f"{label}: {text}" if label else text)
        journal = article.findtext(".//Journal/Title") or article.findtext(".//MedlineTA") or ""
        year = article.findtext(".//PubDate/Year") or article.findtext(".//ArticleDate/Year") or ""
        rows.append({
            "source_id": f"PMID:{pmid}",
            "kind": "literature",
            "title": title,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "https://pubmed.ncbi.nlm.nih.gov/",
            "date": year,
            "journal": journal,
            "abstract": " ".join(abstract_parts)[:7000],
        })
    return rows[:limit]


def _ctgov_search(query: str, limit: int) -> list[dict[str, Any]]:
    settings = get_settings()
    params = {
        "format": "json",
        "pageSize": str(limit),
        "query.cond": "metastatic breast cancer",
        "query.term": query,
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING",
    }
    with httpx.Client(timeout=settings.evidence_timeout_seconds, headers={"User-Agent": "OncoTwin/0.1 research-demo"}) as client:
        res = client.get("https://clinicaltrials.gov/api/v2/studies", params=params)
        res.raise_for_status()
        studies = res.json().get("studies") or []
    out: list[dict[str, Any]] = []
    for study in studies[:limit]:
        p = study.get("protocolSection") or {}
        ident = p.get("identificationModule") or {}
        status = p.get("statusModule") or {}
        desc = p.get("descriptionModule") or {}
        elig = p.get("eligibilityModule") or {}
        design = p.get("designModule") or {}
        contacts = p.get("contactsLocationsModule") or {}
        nct = str(ident.get("nctId") or "")
        locations = []
        for loc in (contacts.get("locations") or [])[:8]:
            bits = [loc.get("facility"), loc.get("city"), loc.get("state"), loc.get("country")]
            label = ", ".join(str(x) for x in bits if x)
            if label:
                locations.append(label)
        out.append({
            "source_id": nct,
            "kind": "trial",
            "title": ident.get("briefTitle") or ident.get("officialTitle") or nct,
            "url": f"https://clinicaltrials.gov/study/{nct}" if nct else "https://clinicaltrials.gov/",
            "status": status.get("overallStatus"),
            "last_update": (status.get("studyFirstPostDateStruct") or {}).get("date"),
            "phase": design.get("phases") or [],
            "summary": desc.get("briefSummary") or "",
            "eligibility": str(elig.get("eligibilityCriteria") or "")[:7000],
            "sex": elig.get("sex"),
            "minimum_age": elig.get("minimumAge"),
            "maximum_age": elig.get("maximumAge"),
            "locations": locations,
        })
    return out


def _snapshot_path() -> Path:
    return fixture_root() / "evidence_snapshot_v1.json"


def load_snapshot() -> dict[str, Any]:
    path = _snapshot_path()
    if not path.is_file():
        return {
            "snapshot_version": "cp4.evidence.snapshot.v1",
            "status": "NOT_CAPTURED",
            "literature": [],
            "trials": [],
            "note": "No vetted evidence snapshot has been captured yet. Use live evidence mode during development, then freeze a vetted snapshot before the showcase.",
        }
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Evidence snapshot must be a JSON object")
    return data


def _source_grounding(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "key": str(source.get("source_id") or "source"),
        "kind": "trial" if source.get("kind") == "trial" else "literature",
        "label": str(source.get("title") or source.get("source_id") or "Source"),
        "source_id": source.get("source_id"),
        "url": source.get("url"),
    }]


def _summarize_literature(db: Session, patient_id: str, state_hash: str, state: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if fixture_mode_enabled():
        return {**row, "patient_summary": row.get("patient_summary"), "summary_status": "snapshot" if row.get("patient_summary") else "source_only"}
    payload = {"verified_patient_context": state, "retrieved_literature_source": row}
    try:
        result = run_structured(
            db,
            task="evidence_summary",
            payload=payload,
            response_model=EvidenceSummaryLLM,
            patient_id=patient_id,
            state_hash=state_hash,
            grounding=_source_grounding(row),
            cache_basis={"source_id": row.get("source_id"), "source_version": row.get("date"), "state_hash": state_hash},
        )
        return {**row, "patient_summary": result.value.model_dump(), "summary_status": "grounded_llm"}
    except RouterUnavailable as exc:
        return {**row, "patient_summary": None, "summary_status": "source_only", "summary_error": str(exc)[:500]}


def _summarize_trial(db: Session, patient_id: str, state_hash: str, state: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if fixture_mode_enabled():
        return {**row, "patient_relevance": row.get("patient_relevance"), "summary_status": "snapshot" if row.get("patient_relevance") else "source_only"}
    payload = {"verified_patient_context": state, "retrieved_trial": row}
    try:
        result = run_structured(
            db,
            task="trial_relevance",
            payload=payload,
            response_model=TrialRelevanceLLM,
            patient_id=patient_id,
            state_hash=state_hash,
            grounding=_source_grounding(row),
            cache_basis={"source_id": row.get("source_id"), "source_version": row.get("last_update"), "state_hash": state_hash},
        )
        return {**row, "patient_relevance": result.value.model_dump(), "summary_status": "grounded_llm"}
    except RouterUnavailable as exc:
        return {**row, "patient_relevance": None, "summary_status": "source_only", "summary_error": str(exc)[:500]}


def search_evidence(db: Session, patient_id: str, *, refresh: bool = False, max_literature: int = 4, max_trials: int = 4) -> dict[str, Any]:
    state_row = _latest_state(db, patient_id)
    state = state_row.state_json
    queries = build_evidence_queries(state)
    settings = get_settings()
    mode = "snapshot" if fixture_mode_enabled() else settings.evidence_mode
    cache_key = _sha({
        "artifact": "evidence_bundle",
        "state_hash": state_row.state_hash,
        "queries": queries,
        "mode": mode,
        "version": RETRIEVAL_VERSION,
        "limits": [max_literature, max_trials],
    })
    if not refresh:
        existing = db.scalar(select(GeneratedArtifact).where(GeneratedArtifact.cache_key == cache_key, GeneratedArtifact.status == "COMPLETED"))
        if existing is not None:
            return {"artifact_id": existing.id, "cache_hit": True, **existing.content_json}

    if mode == "snapshot":
        snapshot = load_snapshot()
        literature = list(snapshot.get("literature") or [])[:max_literature]
        trials = list(snapshot.get("trials") or [])[:max_trials]
        status = str(snapshot.get("status") or "READY")
        note = snapshot.get("note")
    else:
        literature = _pubmed_search(queries["literature"], max_literature)
        trials = _ctgov_search(queries["trials"], max_trials)
        status = "READY"
        note = None

    literature = [_summarize_literature(db, patient_id, state_row.state_hash, state, row) for row in literature]
    trials = [_summarize_trial(db, patient_id, state_row.state_hash, state, row) for row in trials]
    content = {
        "status": status,
        "mode": mode,
        "retrieval_version": RETRIEVAL_VERSION,
        "queries": queries,
        "literature": literature,
        "trials": trials,
        "note": note,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "trial_disclaimer": "Potential relevance is not an eligibility determination. Confirm eligibility and treatment implications with the oncology team and trial site.",
    }
    artifact = GeneratedArtifact(
        patient_id=patient_id,
        artifact_type="evidence_bundle",
        cache_key=cache_key,
        status="COMPLETED",
        provider="authoritative_retrieval",
        model_name="pubmed+clinicaltrials.gov",
        prompt_version=RETRIEVAL_VERSION,
        schema_version=RETRIEVAL_VERSION,
        input_hash=_sha(queries),
        state_hash=state_row.state_hash,
        source_hashes_json=[_sha(x) for x in literature + trials],
        content_json=content,
        grounding_json=[
            {"key": x.get("source_id"), "kind": x.get("kind"), "label": x.get("title"), "source_id": x.get("source_id"), "url": x.get("url")}
            for x in literature + trials
        ],
        metadata_json={"mode": mode},
    )
    db.add(artifact)
    db.flush()
    return {"artifact_id": artifact.id, "cache_hit": False, **content}
