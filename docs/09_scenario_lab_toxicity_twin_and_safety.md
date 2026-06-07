# Twin Scenario Lab, Toxicity Twin, and Safety

## Purpose

The Twin Scenario Lab is the main user-facing simulation feature. It allows the current posterior twin to be run under clearly labeled research scenarios.

The Scenario Lab can surface **exploratory, model-based treatment suggestions and rankings** in addition to tradeoffs and uncertainty. Every such output is exploratory and not guaranteed, and must carry the standard disclaimer (see *Safety language*) and explicit uncertainty.

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

## Scenario object schema

```typescript
type ScenarioRequest = {
  scenarioType:
    | "current_plan"
    | "timing_template"
    | "missing_biomarker"
    | "measurement_update"
    | "treatment_comparison"
    | "toxicity_sensitive";
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
  // Present for treatment_comparison: an exploratory, not-guaranteed ranking.
  rankedOptions?: RankedOption[];
  uncertaintyDrivers: string[];
  explanation: string;
  careTeamQuestions: string[];
  // Standard not-guaranteed / not-medical-advice disclaimer; always required
  // on outputs that include rankedOptions.
  disclaimer: string;
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

The v4 app concept should model the person, not only the tumor. Add a second coupled model for treatment burden.

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
```

### Output

```text
toxicity burden score
symptom trajectory
dominant burden category
trend warning
impact on scenario interpretation
```

### Schema

```typescript
type PatientReportedOutcome = {
  date: string;
  fatigue: number;
  nausea: number;
  neuropathy: number;
  pain: number;
  sleepQuality: number;
  appetite: number;
  activityLevel: number;
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

## Coupling tumor and person twins

Do not let symptoms directly imply tumor response. Couple them through treatment context and scenario interpretation.

Examples:

```text
High toxicity → may increase risk of treatment interruptions in scenario simulations.
Treatment delay → changes drug-exposure schedule in tumor simulator.
Patient priorities → change how tradeoffs are displayed, not the biological model.
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
6. Toxicity burden score.
7. Treatment-delay coupling.
8. Tradeoff visualization.
9. Care-team summary generation.
10. Treatment-comparison ranking view.
11. Disclaimer enforcement on recommendation/ranking outputs.
