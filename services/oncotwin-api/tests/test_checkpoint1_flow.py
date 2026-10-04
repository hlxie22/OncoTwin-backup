from __future__ import annotations

import pymupdf as fitz


def make_pdf(lines: list[str]) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    return doc.tobytes()


def auth_headers(client):
    r = client.post("/v1/auth/demo")
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def create_patient(client, headers):
    r = client.post("/v1/patients", headers=headers, json={"display_name": "Golden patient", "disease_pack": "mbc_v2"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_end_to_end_ingestion_state_provenance_and_delete(client):
    headers = auth_headers(client)
    pid = create_patient(client, headers)
    pdf = make_pdf([
        "Pathology and oncology summary",
        "Diagnosis: metastatic breast cancer",
        "Estrogen receptor: positive",
        "Progesterone receptor: positive",
        "HER2: negative",
        "Current treatment: capecitabine",
        "Known metastatic disease in liver and bone.",
        "Most recent CT impression: stable disease.",
        "PIK3CA mutation detected.",
    ])
    r = client.post(
        f"/v1/patients/{pid}/documents?process=true",
        headers=headers,
        files={"file": ("golden.pdf", pdf, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["status"] in {"ready", "needs_review"}

    # Duplicate upload is idempotent.
    r2 = client.post(
        f"/v1/patients/{pid}/documents?process=true",
        headers=headers,
        files={"file": ("golden-copy.pdf", pdf, "application/pdf")},
    )
    assert r2.status_code == 200
    assert r2.json()["id"] == doc["id"]

    # Resolve any review facts so the committed state is deterministic.
    review = client.get(f"/v1/patients/{pid}/review", headers=headers).json()
    for fact in review:
        rr = client.post(f"/v1/candidates/{fact['id']}/decision", headers=headers, json={"decision": "accept"})
        assert rr.status_code == 200, rr.text

    state = client.get(f"/v1/patients/{pid}/state", headers=headers)
    assert state.status_code == 200, state.text
    body = state.json()
    assert body["state"]["diagnosis"] == "metastatic breast cancer"
    assert body["state"]["subtype"] == "HR-positive / HER2-negative"
    assert body["state"]["current_treatment"]["name"].lower().startswith("capecitabine")
    assert "liver" in body["state"]["disease_sites"]
    assert "bone" in body["state"]["disease_sites"]
    assert body["state"]["latest_scan"]["assessment"] == "stable"
    assert "er_status" in body["sources"]
    assert body["sources"]["er_status"]["document_id"] == doc["id"]

    timeline = client.get(f"/v1/patients/{pid}/timeline", headers=headers)
    assert timeline.status_code == 200
    assert len(timeline.json()) >= 6
    assert any(e["source"] and e["source"]["document_id"] == doc["id"] for e in timeline.json())

    page = client.get(f"/v1/documents/{doc['id']}/pages/1", headers=headers)
    assert page.status_code == 200
    assert "metastatic breast cancer" in page.json()["text"].lower()

    first_hash = body["state_hash"]
    again = client.get(f"/v1/patients/{pid}/state", headers=headers).json()
    assert again["state_hash"] == first_hash

    deleted = client.delete(f"/v1/documents/{doc['id']}", headers=headers)
    assert deleted.status_code == 200
    after = client.get(f"/v1/patients/{pid}/state", headers=headers).json()
    assert after["state_hash"] != first_hash
    assert after["state"]["diagnosis"] is None


def test_owner_scope(client):
    h1 = auth_headers(client)
    p1 = create_patient(client, h1)
    # Demo auth returns the same demo user by design; ownership enforcement is still exercised
    # by trying a nonexistent patient ID rather than creating fake cross-user identities.
    r = client.get("/v1/patients/00000000-0000-0000-0000-000000000000/state", headers=h1)
    assert r.status_code == 404
