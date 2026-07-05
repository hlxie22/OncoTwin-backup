# OncoTwin Implementation Roadmap

## Evidence-first roadmap update

The implementation roadmap is now supplemented by two modeling-first roadmaps:

- `roadmap/PROJECT_MODELING_EVIDENCE_ROADMAP.md`
- `roadmap/MECHANISTIC_MODEL_EVIDENCE_ROADMAP.md`

These roadmaps reframe the project around evidence gates, real-data baselines, calibration, identifiability, and claims discipline. Engineering completion is not treated as success unless the resulting model behavior improves evidence, uncertainty quality, interpretability, or safety compared with simpler alternatives.


## Strategy

Build OncoTwin risk-first, not screen-first.

The first implementation work should attack the parts most likely to change the shape of the app if they fail:

1. Public data availability and harmonization.
2. MRI segmentation and tumor-volume reliability.
3. Mechanistic simulation feasibility and identifiability.
4. Bayesian updating under sparse observations.
5. Patient-facing LLM usefulness and safety.
6. Whether tumor-response modeling and person-burden tracking can be coupled without making unsafe claims.

The polished app shell should come after these risks are tested with small harnesses and demo fixtures. Otherwise the team may build a beautiful interface around assumptions that later collapse.

## Roadmap Principles

- Prefer early experiments over early architecture.
- Treat every uncertain scientific or safety assumption as a gate.
- Build disposable harnesses first, then productize only the pieces that pass.
- Keep patient-facing claims conservative from the beginning.
- Use mock/demo data only when the roadmap explicitly marks the feature as a prototype placeholder.
- Do not let symptoms directly update tumor biology; symptoms affect person-burden summaries, patient planning, and treatment-context interpretation.

## Phase 0: Risk Register And Evaluation Harness

**Goal:** Create a way to measure whether the hard parts are working before building the full app.

### Build

- A top-level `experiments/` area for disposable notebooks/scripts.
- A `fixtures/` area with small synthetic/demo cases:
  - HR-positive / HER2-negative endocrine therapy case.
  - HER2-positive therapy case.
  - TNBC chemotherapy case.
  - Longitudinal tumor-measurement case with T0/T1/T2/T3 values.
  - Patient-reported symptom log with realistic daily check-ins.
- A risk register document listing:
  - Risk.
  - Why it could change the app.
  - Experiment to run.
  - Pass/fail criteria.
  - Fallback if it fails.
- A lightweight evaluation runner that can execute:
  - Simulation sanity checks.
  - LLM safety prompt checks.
  - Schema fixture validation.

### Exit Criteria

- Every major uncertain subsystem has a named experiment and fallback.
- Demo cases can be loaded by scripts without the full app.
- The project can distinguish "working prototype assumption" from "validated model claim."

## Phase 1: Public Data Feasibility

**Goal:** Confirm which public datasets can realistically support each part of the app.

This phase comes first because it can change the whole architecture. If longitudinal imaging, subtype labels, treatment timing, or outcome fields are not usable at the expected quality, the app must lean more heavily on demo mode, manual measurements, or simplified simulations.

### Build

- Dataset inventory for:
  - I-SPY2.
  - BreastDCEDL / BreastDCEDL_ISPY2.
  - MAMA-MIA.
  - Duke-Breast-Cancer-MRI.
  - ACRIN-6698.
  - QIN-BREAST-DCE-MRI.
  - RIDER Breast MRI.
  - TCGA-BRCA / METABRIC.
- For each dataset, record:
  - Access path.
  - Required permissions.
  - Available imaging format.
  - Available masks.
  - Longitudinal timepoints.
  - Treatment metadata.
  - Subtype/pathology fields.
  - Outcome fields.
  - Patient-reported outcome availability.
- A small harmonized case manifest format:

```text
case_id
dataset_source
timepoints
imaging_uri
mask_uri
tumor_volume_ml
longest_diameter_cm
er_status
pr_status
her2_status
subtype
grade
ki67
treatment_phase
treatment_dates
outcome_label
```

### Experiments

- Load at least 5 sample cases into the manifest shape.
- Confirm whether T0/T1/T2/T3 tumor measurements can be derived or imported.
- Confirm which datasets can support model development versus only validation.

### Exit Criteria

- Clear decision on the first real imaging dataset to use.
- Clear fallback if data access is slow or incomplete.
- Confirmed minimum viable data shape for the simulator and app demo.

### Fallbacks

- If MRI access is blocked: start with manual tumor measurements and public metadata.
- If longitudinal data are sparse: use volume-only simulation with synthetic trajectories for demo mode.
- If molecular overlap is weak: use curated molecular rules before any learned graph model.

## Phase 2: Patient-Facing LLM Co-Pilot Feasibility

**Goal:** Test whether the LLM daily co-pilot can be useful, safe, and controllable before building the patient app around it.

This is high-risk because a weak or unsafe LLM layer would change the patient-facing product. If it cannot reliably stay within guardrails, the app should use template-based summaries instead of free-form generation.

### Build

- Prompt/evaluation harness for these task types:
  - `select_daily_check_in`
  - `generate_daily_impact_card`
  - `explain_symptom_trends`
  - `generate_care_team_questions`
  - `summarize_for_visit`
  - `patient_scenario_planning`
- Approved check-in item library using PRO-CTCAE-style symptom language where appropriate.
- Deterministic trend flag generator:
  - New symptom.
  - Worsening symptom.
  - Persistently high symptom.
  - Treatment-day pattern.
  - Medication adherence gap.
  - Sleep/fatigue/activity relationship.
- Red-team cases for unsafe output:
  - Fever threshold requests.
  - "Should I stop medication?"
  - "Does this mean the cancer is growing?"
  - Supplement questions.
  - Severe symptom reports.
  - Treatment ranking requests.

### Experiments

- Generate daily impact cards for the demo cases.
- Generate doctor-ready summaries from 2-4 weeks of symptom logs.
- Run safety checks against prohibited claims:
  - No diagnosis.
  - No treatment changes.
  - No invented urgent thresholds.
  - No symptom-to-tumor-response claims.
  - No guaranteed outcomes.

### Exit Criteria

- LLM outputs are useful enough to justify a patient-facing co-pilot.
- Safety filters catch prohibited outputs.
- The app can fall back to templates if an LLM output fails validation.

### Fallbacks

- If LLM output is inconsistent: constrain generation to JSON fields and approved sentence templates.
- If safety checks are unreliable: use deterministic summaries plus LLM only for editable draft questions.
- If daily cards feel too generic: add richer patient priorities and treatment-day context before expanding UI.

## Phase 3: Mechanistic Simulation Feasibility

**Goal:** Prove that a simple simulator can produce plausible, explainable trajectories before building advanced imaging or AI personalization around it.

This phase comes before app polish because simulator behavior drives the product's core promise. If the model is unstable, unidentifiable, or not explainable, the app should reposition around tracking, education, and research visualization rather than personalized response simulation.

### Build

- Volume-only tumor response simulator.
- Simple treatment schedule representation.
- Parameter sampling for:
  - Growth rate.
  - Treatment sensitivity.
  - Resistant fraction.
  - Observation noise.
- Trajectory summarizer:
  - Median.
  - 80% interval.
  - Uncertainty score.
  - Driver summary.
- Plotting harness outside the app.

### Experiments

- Simulate trajectories for synthetic HR+/HER2-, HER2+, and TNBC-style cases.
- Fit or tune parameters against simple longitudinal tumor measurements.
- Stress-test parameter ranges for impossible or misleading behavior.
- Check whether different parameter sets produce indistinguishable trajectories.

### Current v0 Status

Implemented on July 5, 2026 as an experimental harness under:

```text
experiments/mechanistic_simulator/
fixtures/mechanistic_simulator/
schemas/mechanistic_simulator/
tests/mechanistic_simulator/
```

The volume-only simulator v0 now supports demo/manual tumor-volume trajectories,
ensemble uncertainty, synthetic particle reweighting, identifiability reports,
stress-test guards, and Bayesian-update-ready particle outputs. It is not
clinically validated and should be shown only as an exploratory research
simulation.

Latest local evaluation:

- 35 unit/integration tests passed with `python -m unittest discover -s tests`.
- TNBC demo ensemble wrote JSON and SVG outputs; final median volume was 9.480 mL with uncertainty score 1.214.
- Synthetic fitting improved trajectory RMSE from 11.811 to 1.058, but effective sample size was only 2.6.
- Identifiability found no strongly constrained parameters; anthracycline sensitivity, taxane sensitivity, growth rate, and resistant fraction were only weakly constrained.
- Carrying capacity, unused drug sensitivities, observation noise, and resistant sensitivity scale remained prior-dominated.

Follow-up recovery and identifiability sweep:

- Ran 96 synthetic recovery cases across 3 seeds, 2 particle counts, 4 assumed observation-noise levels, and 4 reduced-parameter variants.
- Particle reweighting improved median held-out prediction versus the prior ensemble.
- Effective sample size stayed below 5% of particles in the median run, so posterior narrowing remains fragile.
- Increasing full-model particles from 900 to 2000 and 5000 raised absolute ESS, but ESS fraction stayed near 2.7%.
- The best median held-out method was a simple exponential baseline: 1.220 mL median absolute error to truth, compared with 1.824 mL for the posterior particle mean.
- The best mechanistic variant was `shared_chemo_fixed_core`, with 0.902 mL median posterior held-out error, but it achieved that by reducing degrees of freedom and tying chemotherapy sensitivities.
- Current recommendation: do not expose parameter-level explanations or digital-twin claims yet. Keep v0 as an exploratory research simulator, compare against simple baselines, and prefer reduced-parameter variants.

### Exit Criteria

- Simulator produces plausible trajectories under known synthetic settings.
- Uncertainty bands widen when data are sparse.
- Model explanations can state which assumptions drive the result.
- Clear limitations are documented.

### Fallbacks

- If parameters are not identifiable: use fewer parameters and broader uncertainty.
- If volume-only dynamics are too weak: delay personalized claims and show educational scenario ranges.
- If treatment-specific sensitivity is too speculative: use regimen category templates instead of drug-specific effects.

## Phase 4: Bayesian Update Feasibility

**Goal:** Test whether new observations actually improve or appropriately shift the twin.

This phase is high-risk because the "digital twin" framing depends on updating. If updating is brittle, the product should frame itself as an exploratory simulation dashboard with observation overlays rather than a learning twin.

### Build

- Batch importance-sampling updater from original prior particles.
- Observation likelihoods for:
  - Tumor volume.
  - Longest diameter.
  - Biomarker result.
  - Treatment delay or dose modification.
- Effective sample size calculation.
- Update explanation:
  - What changed.
  - Why uncertainty changed.
  - Which simulated futures gained or lost weight.

### Experiments

- Synthetic recovery test: generate data from known parameters and verify posterior shifts toward them.
- Noisy measurement test: confirm uncertainty does not over-collapse.
- Contradictory observation test: confirm uncertainty rises or fallback triggers.

### Exit Criteria

- Updates behave sensibly for strong response, weak response, noisy measurement, and treatment delay.
- Symptoms do not directly update tumor-response parameters.
- The update explanation is understandable without overstating certainty.

### Fallbacks

- If posterior collapse is common: keep wider priors or reduce parameter dimensionality.
- If updating is hard to explain: show "observation changed the scenario" rather than "the twin learned."
- If longitudinal real data are unavailable: restrict updating to demo/synthetic cases until real validation exists.

## Phase 5: Imaging And Segmentation Feasibility

**Goal:** Determine whether MRI ingestion and segmentation can support reliable tumor-volume extraction.

This phase can heavily change the app. If automatic segmentation is unreliable, the MVP should support manual or clinician-reviewed measurements rather than pretending fully automated MRI analysis is ready.

### Build

- NIfTI-first ingestion path.
- DICOM support only after NIfTI is stable.
- Integration spike with MAMA-MIA or MONAI-based segmentation.
- Tumor-volume and longest-diameter extraction.
- QC flags:
  - Missing sequence.
  - Low segmentation confidence.
  - Implausible volume change.
  - Empty or tiny mask.
  - Out-of-distribution spacing/intensity.

### Experiments

- Run segmentation on a small representative sample.
- Compare extracted volume against known masks or metadata.
- Test failure cases with corrupted, missing, or misaligned data.

### Exit Criteria

- The app can safely distinguish usable from low-confidence imaging.
- MRI-derived tumor volume can feed the simulator.
- QC warnings are clear enough for a non-expert demo user.

### Fallbacks

- If segmentation is weak: use manual mask upload or manual tumor measurements.
- If DICOM complexity is high: restrict prototype to NIfTI.
- If QC is hard: block simulation from low-confidence imaging unless user explicitly switches to demo/research mode.

## Phase 6: AI Personalization Feasibility

**Goal:** Test whether baseline imaging/pathology/molecular context improves simulator priors over generic population priors.

This is a later high-risk phase because it requires data and fitted parameters from earlier phases. It should not block the first prototype.

### Build

- Simple pathology encoder.
- Baseline imaging feature encoder or frozen 3D MRI encoder spike.
- Prior predictor that outputs parameter distributions.
- Evaluation against generic priors.
- Modality dropout for missing data.

### Experiments

- Compare personalized priors versus generic priors on held-out longitudinal cases.
- Test missing biomarker behavior.
- Test subgroup calibration where data permit.

### Exit Criteria

- Personalized priors improve held-out trajectory likelihood or calibration.
- Missing data increases uncertainty rather than creating false precision.
- The explanation layer can describe which inputs affected the prior.

### Fallbacks

- If learned personalization does not beat generic priors: keep curated subtype/treatment priors.
- If imaging encoder is too expensive: use extracted radiomic/volume features first.
- If molecular data overlap is poor: keep molecular explanations rule-based.

## Phase 7: Product Shell From Proven Pieces

**Goal:** Only now build the durable app around the parts that survived experimentation.

### Build

- Frontend app with:
  - Build My Twin.
  - Daily Impact Co-Pilot.
  - Virtual Tumor State.
  - Mechanistic Simulation.
  - Twin Update Timeline.
  - Scenario Lab.
  - Doctor / Research Summary.
- Backend API with:
  - Case service.
  - Pathology service.
  - Molecular service.
  - Treatment service.
  - Imaging service.
  - Simulation service.
  - Bayesian update service.
  - Scenario Lab service.
  - Daily co-pilot service.
  - LLM orchestration service.
  - Summary service.
- PostgreSQL schema based on the stable subset from experiments.
- Async job queue for long-running imaging, simulation, LLM summary, and report tasks.

### Exit Criteria

- A demo user can create or load a case.
- The daily co-pilot works from structured logs.
- A tumor simulation runs with uncertainty.
- A follow-up observation updates the trajectory.
- A patient-safe scenario generates planning suggestions and care-team questions.
- A doctor-ready summary can be exported or displayed.

## Phase 8: Scenario Lab And Person-Burden Integration

**Goal:** Connect tumor response, uncertainty, daily symptom burden, and patient planning without unsafe causal claims.

### Build

- Current-plan scenario.
- Measurement-update scenario.
- Missing-biomarker scenario.
- Patient-planning scenario.
- Research-mode treatment comparison.
- Person-burden score.
- Symptom trajectory summary.
- Treatment-delay coupling.

### Safety Rules

- Patient-facing scenarios do not rank treatments.
- Treatment ranking is research/clinician-facing only.
- Symptom burden can affect scenario interpretation and treatment-delay assumptions.
- Symptom burden cannot claim tumor response or progression.

### Exit Criteria

- Scenario outputs show assumptions, uncertainty, and safety labels.
- Patient-facing scenarios produce preparation suggestions and questions.
- Research-mode comparisons include uncertainty and disclaimer.

## Phase 9: Validation, Model Cards, And Safety Audit

**Goal:** Make the prototype honest, testable, and ready for external review.

### Build

- Model cards for:
  - Segmentation.
  - Simulator.
  - Bayesian updater.
  - AI personalization.
  - Molecular explanation rules/model.
  - LLM co-pilot.
- Validation reports:
  - Segmentation quality.
  - Tumor trajectory calibration.
  - Uncertainty coverage.
  - LLM safety evaluation.
  - Subgroup/data coverage limitations.
- Automated safety tests:
  - No guaranteed outcome claims.
  - No treatment directives.
  - No missing disclaimers for ranking.
  - No invented urgent thresholds.
  - No symptom-to-tumor-response claims.

### Exit Criteria

- The app has an explicit limitations page.
- Every model/LLM output records a version and audit trail.
- Unsafe language checks run in CI.
- The team can clearly say what is prototype, what is validated, and what is research-only.

## Suggested Build Order

```text
0. Risk register and experiment harness
1. Public data feasibility
2. Patient-facing LLM co-pilot feasibility
3. Mechanistic simulation feasibility
4. Bayesian update feasibility
5. Imaging and segmentation feasibility
6. AI personalization feasibility
7. Product shell from proven pieces
8. Scenario Lab and person-burden integration
9. Validation, model cards, and safety audit
```

## Decision Gates

| Gate | Question | If yes | If no |
|---|---|---|---|
| Data gate | Can public data support longitudinal response modeling? | Build real-data simulator pipeline. | Use manual/demo measurements and label outputs as synthetic/demo. |
| LLM gate | Can daily cards be useful and safe? | Build patient-facing co-pilot. | Use deterministic templates and editable question drafts. |
| Simulator gate | Does volume-only simulation behave plausibly? | Productize simulation MVP. | Reframe as educational scenario explorer. |
| Update gate | Does Bayesian updating improve trajectories without false precision? | Use twin update timeline. | Show observation overlays and uncertainty only. |
| Imaging gate | Is segmentation reliable enough for demo use? | Feed MRI-derived volumes into simulator. | Require manual measurement/mask review. |
| Personalization gate | Do personalized priors beat generic priors? | Add AI personalization. | Keep curated subtype/treatment priors. |

## Minimum Prototype Definition

The minimum credible prototype is not the full digital twin. It is:

- Case creation with subtype, treatment context, and patient context.
- LLM daily co-pilot with safe daily check-ins, impact cards, and care-team summaries.
- Manual tumor measurement entry.
- Volume-only simulation with uncertainty bands.
- Bayesian update from a follow-up measurement.
- Patient-safe Scenario Lab.
- Clear disclaimers, limitations, and audit trails.

MRI segmentation, spatial residual-risk maps, learned AI personalization, and molecular graph attention are important but should not block the first credible prototype.

## Interfaces To Stabilize Early

### Case Context

```text
case_id
mode
pathology_profile
molecular_profile
treatment_context
patient_context
imaging_timepoints
patient_reported_outcomes
```

### Daily Co-Pilot

```text
POST /cases/{case_id}/daily-check-in/plan
POST /cases/{case_id}/daily-check-in/responses
GET  /cases/{case_id}/daily-impact/today
GET  /cases/{case_id}/daily-impact/trends
```

### Simulation And Update

```text
POST /cases/{case_id}/twin/initialize
POST /cases/{case_id}/twin/simulate
POST /cases/{case_id}/twin/update-observation
GET  /cases/{case_id}/uncertainty
```

### Scenario And Summary

```text
POST /cases/{case_id}/scenario-lab/run
GET  /cases/{case_id}/summary/doctor
```

## Testing Strategy

### Early Experiment Tests

- Fixture loading tests.
- Deterministic trend flag tests.
- LLM red-team output tests.
- Simulator sanity tests.
- Posterior update synthetic recovery tests.
- Imaging QC failure tests.

### Product Tests

- Case creation and demo-case loading.
- Daily check-in planning and response submission.
- Daily impact card generation.
- Tumor simulation run.
- Follow-up observation update.
- Scenario Lab run.
- Doctor summary generation.

### Safety Tests

- Recommendation or ranking without disclaimer fails.
- Guaranteed outcome language fails.
- Treatment-change instruction fails.
- Invented fever threshold fails.
- Symptom-to-tumor-response claim fails.
- Patient-facing treatment ranking fails.

## Assumptions

- The first build target is a high-quality prototype, not a clinically deployed product.
- Risk-first sequencing is more important than building the full UI early.
- FastAPI/Python is the default backend direction because imaging and modeling are Python-heavy.
- The first simulator is volume-only.
- The first patient-facing AI feature is the LLM co-pilot.
- The first imaging path is NIfTI, with DICOM deferred until core feasibility is proven.
- Learned AI personalization is deferred until public data and calibration feasibility are proven.
- Treatment ranking remains research/clinician-facing, not patient-facing.
