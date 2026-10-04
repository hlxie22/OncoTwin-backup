# OncoTwin deterministic demo-fixture runtime

The canonical synthetic patient lives at `demo/synthetic_patient_v1`.

Use fixture mode when repeatedly testing or presenting this exact patient:

```bash
bash scripts/app/run_demo_fixture_dev.sh
```

The mode is intentionally narrow:

- PDF upload, storage, page rendering/OCR and source viewing still run normally.
- Extraction results for the 12 canonical demo PDFs are loaded by exact SHA256.
- Unknown PDFs fail closed in fixture mode; the app never silently falls back to a paid LLM.
- Candidate facts, verification/conflicts, committed state, timeline and source provenance are real.
- The frozen Dynamic Scan V2 forecast is always real and is never hard-coded.
- Extraction provenance is stored as `provider=demo_fixture`, `model=fixture:maya_rowan_v1`.
- `demo_fixtures/workflows_v1.json` is a typed-content source for future CP4 fixture-backed explanations/questions/briefs. Numerical forecast values must always be injected from the real forecast run.

Normal development behavior is unchanged when `ONCOTWIN_DEMO_MODE` is unset.
