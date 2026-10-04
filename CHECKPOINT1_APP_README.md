# OncoTwin App Checkpoint 1

This tree is additive to the frozen Dynamic Scan V2 research release. It does not modify `scripts/dynamic_scan_v2` or `artifacts/dynamic_scan_v2`.

## Install

```bash
bash scripts/app/setup_cp1.sh
```

## Run acceptance

```bash
ROOT=$(pwd) PY=/home/henryxie/orcd/scratch/.conda/envs/oncotwin/bin/python3 \
  bash scripts/app/cp1_acceptance.sh
```

The wrapper intentionally exits 0 so a failed strict child does not kill an interactive Slurm allocation. Read `EXIT_CODE=` and `summary.json`.

## Preview on Slurm

```bash
bash scripts/app/run_cp1_dev.sh
```

The script prints the compute hostname and the exact local SSH-tunnel pattern. You only need to forward the Next.js port; the browser never talks directly to the API port.

## Real LLM extraction

Default development/acceptance uses the deterministic source-grounded extractor so tests are reproducible. For the real extraction path:

```bash
export ONCOTWIN_EXTRACTOR_PROVIDER=openai
export ONCOTWIN_OPENAI_API_KEY='...'
export ONCOTWIN_OPENAI_MODEL='YOUR_APPROVED_MODEL'
bash scripts/app/run_cp1_dev.sh
```

Structured output is validated again by Pydantic and deterministic commit rules before state mutation.

## Important safety boundary

Use synthetic or de-identified documents for the hosted CAC/development instance. The forecast model is not wired in until checkpoint 2; no LLM output makes a quantitative prognosis.
