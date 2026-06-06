# Twin Scenario Lab, Toxicity Twin, and Safety

## Purpose

The Twin Scenario Lab is the main user-facing simulation feature. It allows the current posterior twin to be run under safe, clearly labeled research scenarios.

The Scenario Lab should not recommend treatment. It should show model-based tradeoffs and uncertainty.

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

This is a powerful educational simulation and avoids recommending treatment.

### Toxicity-sensitive scenario

```text
How does the model's tradeoff change when patient-reported burden is high?
```

This should be framed as discussion support, not treatment guidance.

## Scenario object schema

```typescript
type ScenarioRequest = {
  scenarioType:
    | "current_plan"
    | "timing_template"
    | "missing_biomarker"
    | "measurement_update"
    | "toxicity_sensitive";
  assumptions: Record<string, unknown>;
  treatmentSchedule?: TreatmentSchedule;
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
  uncertaintyDrivers: string[];
  explanation: string;
  careTeamQuestions: string[];
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

A research scenario can display a tradeoff score:

```math
U(r) = w_1 E[response] - w_2 E[toxicity] - w_3 uncertainty
```

But the UI must avoid presenting this as an optimization recommendation.

Better wording:

```text
This research score summarizes the model's simulated tradeoff between tumor response, symptom burden, and uncertainty. It is not a treatment recommendation.
```

## Safety language

Every scenario page should include:

```text
Research simulation only. This is not medical advice and not a treatment recommendation. Discuss treatment decisions with a qualified oncology team.
```

## Prohibited outputs

The app should not say:

```text
You should switch treatments.
You should delay treatment.
This treatment is best for you.
This schedule will work.
Your cancer will respond.
```

Allowed outputs:

```text
In this model simulation, this scenario produced a lower median residual tumor burden across the current posterior ensemble.
Uncertainty remains high because only baseline imaging is available.
This result should be discussed with an oncology team and should not be used to change treatment.
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
10. Safety filter and prohibited-output tests.
