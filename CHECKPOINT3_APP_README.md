# OncoTwin App Checkpoint 3

Checkpoint 3 completes the primary patient-facing experience without changing the frozen Dynamic Scan V2 model.

## What changed

- Overview is reorganized around **What OncoTwin knows right now**.
- New guided **Add newest scan** flow preserves the before-scan state, processes the real PDF, pauses for verification when necessary, then runs the frozen forecast.
- The completed update keeps three concepts separate:
  1. what the scan reported,
  2. how the research forecast changed,
  3. what new structured information entered the patient state.
- Trajectory page now uses the shared session/proxy API instead of bypassing the web proxy, uses the existing OncoTwin visual system, and progressively discloses technical model details.
- Records, Review, Timeline, source evidence, responsive behavior, keyboard focus and reduced-motion behavior were brought into the same final patient experience.

## Scientific boundaries preserved

- V2-07 remains frozen.
- Selected update remains `PRE + 0.52 * (POST - PRE)`.
- PRE remains history before the newest scan; selected update adds current scan evidence through the frozen pipeline.
- No treatment recommendation is produced.
- No lesion-level causal attribution is shown.
- Missing remains distinct from negative.
- External confirmation is still pending.

## Development

```bash
bash scripts/app/run_cp3_dev.sh
```

## Acceptance

```bash
bash scripts/app/safe_run.sh bash scripts/app/cp3_acceptance.sh
```

The CP3 acceptance run first reruns the complete CP2 scientific/platform regression, then checks the patient-experience contract and serves the production Next.js build to smoke-test all primary routes.
