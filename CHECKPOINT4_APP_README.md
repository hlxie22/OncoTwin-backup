# OncoTwin Checkpoint 4 — patient intelligence

Checkpoint 4 adds one shared language/evidence platform layer on top of the accepted CP1–CP3 product.

## Boundaries
- Quantitative prognosis remains the frozen V2-07 model only.
- LLM output never enters the forecast adapter.
- Generated claims are source/state grounded and cached with provenance.
- Trial output is potential relevance, never an eligibility verdict.
- Deterministic Maya mode never falls through to paid LLM calls.

## Live development
Configure `ONCOTWIN_GROQ_API_KEY` and/or `ONCOTWIN_GOOGLE_API_KEY`, optionally override model IDs, then run `bash scripts/app/run_cp4_dev.sh`.

## Deterministic Maya demo
Continue to use `bash scripts/app/run_demo_fixture_dev.sh`. CP4 language workflows use `workflows_v1.json`; the forecast remains real. Evidence uses `evidence_snapshot_v1.json`, which must be replaced by a vetted captured snapshot before the final showcase.

## Acceptance
`bash scripts/app/safe_run.sh bash scripts/app/cp4_acceptance.sh`
