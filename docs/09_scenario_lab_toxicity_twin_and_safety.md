# Twin Scenario Lab, Toxicity Twin, and Safety

## Purpose

The Twin Scenario Lab is the main simulation feature. It allows the current posterior twin to be run under clearly labeled research scenarios, and it also supports patient-facing planning scenarios that turn treatment context and symptom logs into preparation suggestions and care-team questions.

The Scenario Lab can surface **exploratory, model-based treatment suggestions and rankings** in addition to tradeoffs and uncertainty. Every such output is exploratory and not guaranteed, and must carry the standard disclaimer (see *Safety language*) and explicit uncertainty.

For patient-facing mode, the default should be planning and communication support, not treatment ranking. Patient-facing scenario outputs should answer questions such as "what should I prepare for this week?" or "what should I ask my care team?" rather than "which treatment is best?"

## Scenario types

### Current-plan simulation

```text
What does the current twin predict under the current treatment context?
```

Outputs:

```text
median tumor trajectory
uncertainty band
residual-risk map
uncertainty drivers
```

### Alternative timing template

```text
What happens in the model if treatment timing follows a different research template?
```

Use only predefined, clinically plausible templates and label them as research simulations.

### Missing-biomarker scenario

```text
How would the twin change if BRCA/HRD were positive, negative, or still unknown?
```

This is useful because it teaches why missing data matters.

### Measurement-update scenario

```text
If an early MRI showed strong shrinkage, how would the twin update?
If it showed weak shrinkage, how would the twin update?
```

This is a powerful educational simulation of how new evidence reshapes the twin.

### Treatment-comparison scenario

```text
Across several clinically plausible candidate schedules, how do the model's
predicted response, residual-risk, and toxicity tradeoffs compare and rank?
```

This produces an **exploratory ranking** of candidate options. It is decision-support, not a directive: every option is shown with uncertainty bands and the standard not-guaranteed disclaimer, and the ranking may change as new data arrive.

### Toxicity-sensitive scenario

```text
How does the model's tradeoff change when patient-reported burden is high?
```

This should be framed as exploratory decision-support, with uncertainty made explicit.

### Patient-planning scenario

```text
What should the patient prepare for this treatment week?
Which logged symptoms tend to worsen around treatment days?
What care-team questions should be saved if a symptom keeps increasing?
What information would make the model or daily plan less uncertain?
```

This scenario uses the LLM API over structured logs, deterministic trend flags, treatment context, and care-team instructions. It does not rank treatment options for the patient.

## Scenario object schema

```typescript
type ScenarioRequest = {
  scenarioType:
    | "current_plan"
    | "timing_template"
    | "missing_biomarker"
    | "measurement_update"
    | "treatment_comparison"
    | "toxicity_sensitive"
    | "patient_planning";
  assumptions: Record<string, unknown>;
  treatmentSchedule?: TreatmentSchedule;
  // For treatment_comparison: the candidate options to rank.
  candidateSchedules?: TreatmentSchedule[];
  safetyLabelRequired: true;
};
```

## Scenario simulation loop

```text
current posterior particles
        ↓
scenario assumptions
        ↓
modify treatment schedule, biomarker assumption, or observation
        ↓
simulate each particle
        ↓
summarize trajectory distribution
        ↓
compute uncertainty and explanation
        ↓
render safe comparison
```

Pseudocode:

```python
def run_scenario_lab(case, scenario):
    posterior = load_current_posterior(case)
    modified_inputs = apply_scenario_assumptions(case, scenario)

    outputs = []
    for particle in posterior.particles:
        result = simulate_tumor(
            initial_state=case.twin_state,
            params=particle.params,
            treatment_schedule=modified_inputs.treatment_schedule,
            times=modified_inputs.times,
        )
        outputs.append((particle.weight, result))

    return summarize_scenario_outputs(outputs, scenario)
```

## Output format

```typescript
type ScenarioResult = {
  scenarioId: string;
  scenarioName: string;
  safetyLabel: string;
  tumorTrajectory: {
    times: number[];
    median: number[];
    lower80: number[];
    upper80: number[];
  };
  residualRisk?: {
    riskMapUri: string;
    summary: string;
  };
  toxicity?: ToxicitySummary;
  patientPlan?: PatientPlanningSummary;
  // Present for treatment_comparison: an exploratory, not-guaranteed ranking.
  rankedOptions?: RankedOption[];
  uncertaintyDrivers: string[];
  explanation: string;
  careTeamQuestions: string[];
  // Standard not-guaranteed / not-medical-advice disclaimer; always required
  // on outputs that include rankedOptions.
  disclaimer: string;
};

type PatientPlanningSummary = {
  focusAreas: string[];
  preparationSuggestions: string[];
  careTeamQuestions: string[];
  trendSummary: string;
  safetyNotes: string[];
  llmTraceId: string;
};

type RankedOption = {
  scheduleId: string;
  scheduleName: string;
  rank: number;
  predictedResponse: { median: number; lower80: number; upper80: number };
  predictedToxicity?: "low" | "moderate" | "high";
  tradeoffScore: number;
  uncertainty: "low" | "moderate" | "high";
  // Exploratory rationale, e.g. "lower median residual burden but wider
  // uncertainty than option B."
  note: string;
};
```

## Toxicity/person-burden twin

The app concept should model the person, not only the tumor. Add a second coupled model for treatment burden and a daily LLM co-pilot that helps patients track the right things and communicate patterns.

### Inputs

```text
age
baseline symptoms
comorbidities if entered
regimen category
patient priorities
fatigue
nausea
neuropathy
sleep
pain
appetite
activity level
temperature concerns per care-team instructions
treatment delays or dose changes
daily check-in responses
care-team instructions
```

### Output

```text
toxicity burden score
symptom trajectory
dominant burden category
trend warning
impact on scenario interpretation
LLM-generated care-team questions
daily impact card
```

### Schema

```typescript
type PatientReportedOutcome = {
  date: string;
  treatmentDay?: number;
  treatmentPhase?: string;
  fatigue: number;
  nausea: number;
  neuropathy: number;
  pain: number;
  sleepQuality: number;
  appetite: number;
  activityLevel: number;
  medicationTaken?: boolean;
  hotFlashes?: number;
  skinIrritation?: number;
  mouthSores?: number;
  shortnessOfBreath?: "none" | "mild" | "worsening";
  swelling?: boolean;
  temperature?: number;
  notes?: string;
};
```

```typescript
type ToxicitySummary = {
  burdenLevel: "low" | "moderate" | "high";
  worseningSymptoms: string[];
  trendExplanation: string;
  uncertainty: "low" | "moderate" | "high";
};
```

## Daily LLM co-pilot

The daily co-pilot is the patient-facing layer on top of the person-burden data. It uses an LLM API to generate adaptive check-ins, daily impact cards, trend explanations, and care-team questions.

For the MVP, do not train a separate time-series AI model. Use simple structured trend calculations over patient logs, then pass those trend flags to the LLM for explanation and prioritization.

LLM tasks:

```text
select daily check-in questions
generate daily impact card
explain symptom/adherence trends
generate care-team questions
summarize logs before a visit
generate patient-safe planning scenario output
```

LLM inputs:

```text
subtype and treatment context
treatment dates and phase
recent patient-reported outcomes
deterministic trend flags
care-team instructions
approved symptom/check-in item library
allowed suggestion templates
safety rules
```

LLM outputs:

```text
short check-in item list
today's focus areas
plain-language trend summary
preparation suggestions
care-team questions
safety notes
```

## Coupling tumor and person twins

Do not let symptoms directly imply tumor response. Couple them through treatment context and scenario interpretation.

Examples:

```text
High toxicity → may increase risk of treatment interruptions in scenario simulations.
Treatment delay → changes drug-exposure schedule in tumor simulator.
Patient priorities → change how tradeoffs are displayed, not the biological model.
Daily logs → change check-in focus, care-team questions, and burden summaries.
LLM patient cards → explain trends and planning steps, not tumor biology.
```

## Utility function for research display

A research scenario can display a tradeoff score and use it to rank candidate options:

```math
U(r) = w_1 E[response] - w_2 E[toxicity] - w_3 uncertainty
```

The UI may present this as an **exploratory ranking** of candidate options, as long as the uncertainty and the standard disclaimer are shown alongside it.

Suggested wording:

```text
This exploratory score summarizes the model's simulated tradeoff between tumor response, symptom burden, and uncertainty, and is used to rank the candidate options below. The ranking is exploratory and not guaranteed; discuss any treatment decision with a qualified oncology team.
```

## Safety language

The safety model is **lightweight disclaimers**, not a strict language filter. Recommendation-style and ranking outputs are allowed, provided every page that shows them carries the standard disclaimer:

```text
Exploratory research simulation. Predictions and rankings are uncertain and not guaranteed, and this is not a substitute for professional medical advice. Discuss all treatment decisions with a qualified oncology team.
```

## Output guidance

The app **may** suggest, compare, and rank candidate treatment options, as long as each such output is exploratory, shown with uncertainty, and accompanied by the disclaimer above.

Patient-facing mode should not present a treatment ranking as the answer. It may show planning scenarios, uncertainty explanations, and care-team questions. Treatment comparison and ranking should be reserved for research or clinician-facing views unless a deployment has explicit clinical governance for patient presentation.

The one hard rule is to avoid **false certainty** — do not claim a guaranteed, definite, or curative outcome. For example, avoid:

```text
This schedule will work.
Your cancer will respond.
This treatment is guaranteed to cure you.
```

Acceptable exploratory outputs:

```text
In this model simulation, Option A ranked highest, producing a lower median residual tumor burden than Option B across the current posterior ensemble — though its uncertainty band is wider.
Based on the model, Option A looks the most promising of the candidates considered, but this is exploratory and not guaranteed.
Uncertainty remains high because only baseline imaging is available, so this ranking may change after a follow-up scan.
Discuss these options with a qualified oncology team before making any decision.
```

Acceptable patient-facing daily outputs:

```text
Based on your recent logs, fatigue has been higher on the two days after infusion. Today's check-in will focus on fatigue, nausea, appetite, and new symptoms.
This may be worth asking your care team: "What symptoms should make me call before the next appointment?"
Follow the fever or urgent-call instructions your oncology team gave you.
```

Unacceptable patient-facing daily outputs:

```text
Your symptoms mean the tumor is progressing.
You should stop this medication.
This fever threshold applies to you.
This treatment is the best choice.
```

## Scenario UI sections

### 1. Scenario assumptions

```text
Current treatment context
What was changed for this scenario
Which assumptions are hypothetical
```

### 2. Tumor trajectory chart

```text
observed measurements
median simulated trajectory
uncertainty band
```

### 3. Residual-risk map

```text
3D tumor/residual-risk visualization if MRI is available
```

### 4. Person-burden panel

```text
symptom trend
toxicity burden
quality-of-life notes
daily impact card
care-team questions from recent logs
```

### 5. Explanation

```text
why the scenario changed the simulation
which parameters mattered
what data would reduce uncertainty
```

### 6. Care-team questions

```text
Questions to ask your oncology team based on this simulation
Questions generated from daily logs and trend flags
Questions should be editable by the patient
```

### 7. Treatment-option ranking (comparison scenarios only)

```text
exploratory ranking of candidate options
predicted response and toxicity per option, with uncertainty
tradeoff score and rationale per option
"exploratory / not guaranteed" disclaimer shown inline
```

## Implementation milestones

1. Current-plan simulation.
2. Missing-biomarker scenarios.
3. Measurement-update scenarios.
4. Safe scenario-comparison UI.
5. Patient-reported outcome tracker.
6. LLM daily check-in selection.
7. LLM daily impact card.
8. Patient-planning scenario mode.
9. Toxicity burden score.
10. Treatment-delay coupling.
11. Tradeoff visualization.
12. Care-team summary generation.
13. Treatment-comparison ranking view.
14. Disclaimer enforcement on recommendation/ranking outputs and LLM daily outputs.
