# Patient-Facing LLM Co-Pilot

## Purpose

OncoTwin should not ask patients to memorize a static "life impact plan." The patient-facing layer should convert subtype, treatment context, care-team instructions, and daily logs into a small set of timely prompts, suggestions, and questions.

The core patient-facing product is an LLM-powered daily co-pilot:

```text
What should I pay attention to today?
What should I track today?
What pattern has changed?
What question should I bring to my care team?
What should I prepare for this treatment week?
```

This is not a diagnostic or treatment-prescribing assistant. It is a structured reminder, tracking, explanation, and care-team communication layer.

## Product features

### 1. Adaptive daily check-in

The app should generate a short check-in each day. It should usually ask only 3-6 questions, selected from the patient's current context.

Inputs:

```text
subtype and pathology
treatment category and treatment day
current phase: diagnosis, neoadjuvant therapy, radiation, endocrine therapy, survivorship
recent patient-reported outcomes
missed logs or medication entries
care-team instructions entered by the patient or clinician
patient priorities
```

Examples:

```text
HR-positive / HER2-negative on endocrine therapy:
- Did you take your endocrine medication today?
- Joint stiffness this morning: 0-10
- Hot flashes last night: 0-10
- Sleep quality: 0-10
- Planned activity: walk / strength / rest / unsure
```

```text
Chemotherapy context:
- Fatigue: 0-10
- Nausea: 0-10
- Appetite: normal / lower / very low
- Numbness or tingling: none / mild / worsening
- Any new symptom you want to remember for your care team?
```

```text
Radiation context:
- Skin irritation: none / mild / moderate / severe
- Fatigue: 0-10
- Breast or chest-wall discomfort: 0-10
- Shoulder tightness: none / mild / worsening
```

### 2. Daily impact card

The app should turn recent logs into a plain-language daily card.

Example:

```text
Today's focus: joint stiffness and sleep

Joint stiffness has been higher for four mornings, and sleep quality has been lower this week. If your care team has cleared activity, gentle movement may be worth planning today.

Question saved for your next visit:
"Could endocrine therapy be contributing to joint stiffness, and what safe options could help?"
```

Another example:

```text
Today's focus: infusion recovery

Your nausea has tended to peak two days after infusion. Today's check-in will focus on nausea, appetite, hydration, and fatigue.

If symptoms become severe or you cannot keep fluids down, follow the instructions from your oncology team.
```

### 3. Pattern detection and interpretation

For the MVP, do not train a separate time-series AI model. Use deterministic trend calculations over structured logs, then use the LLM API to interpret and explain those trends in context.

Structured trend calculations:

```text
new symptom appeared
symptom severity increased by threshold over baseline
symptom stayed high for several entries
symptom clusters changed together
symptom tends to peak after infusion or radiation sessions
medication adherence gap occurred
activity decreased while fatigue increased
sleep worsened alongside hot flashes or pain
```

The LLM should not infer tumor response from symptoms. It should translate trend outputs into:

```text
patient-friendly explanations
daily check-in focus
care-team questions
summary bullets
preparation suggestions
```

### 4. Doctor-ready summary

Before a visit, the LLM should generate a concise summary using only structured logs, patient notes, and scenario outputs.

Sections:

```text
time window summarized
new or worsening symptoms
highest-severity symptoms
patterns around treatment days
medication adherence notes if tracked
patient's top concerns
questions to ask care team
safety disclaimer
```

Example:

```text
Since the last visit:
- Fatigue averaged 6/10 and peaked at 8/10 on days 2-3 after infusion.
- Nausea was highest on the two days after infusion and improved by day 5.
- Tingling in fingers was first logged 5 days ago and is now marked "worsening."
- Appetite was low for 4 of the last 7 days.

Questions to ask:
- Could the tingling be early neuropathy?
- What nausea plan should I follow after the next infusion?
- What symptoms should make me call before the next appointment?
```

### 5. Patient-safe scenario lab

The patient-facing Scenario Lab should focus on planning, burden, and questions rather than ranking treatment choices.

Patient-facing scenarios:

```text
What should I prepare for this treatment week?
Which symptoms have usually changed after infusion?
How did my fatigue and activity change after the last cycle?
What questions should I ask if neuropathy keeps increasing?
What information would make the model less uncertain?
```

Treatment comparison and ranking can exist in research or clinician-facing mode, but patient-facing scenario outputs should avoid saying which treatment is best. They should produce care-team questions and preparation suggestions.

## LLM architecture

The LLM API should be used as an orchestration and language layer over structured inputs. It should not be the only source of medical logic.

```text
patient context
  subtype, treatment phase, care-team instructions, priorities
        ↓
structured evidence/rule context
  subtype-to-treatment-impact rules, PRO-CTCAE-style symptom vocabulary,
  safety constraints, allowed suggestion templates
        ↓
daily logs and deterministic trends
  symptom scores, adherence entries, treatment dates, trend flags
        ↓
LLM task call
  select check-in items, generate daily impact card,
  explain trends, draft care-team questions, summarize visit
        ↓
post-processing safety checks
  no diagnosis, no treatment directive, no guaranteed outcome,
  no symptom-to-tumor-response claim, include disclaimer when needed
```

Recommended LLM task types:

```typescript
type LlmTaskType =
  | "select_daily_check_in"
  | "generate_daily_impact_card"
  | "explain_symptom_trends"
  | "generate_care_team_questions"
  | "summarize_for_visit"
  | "patient_scenario_planning";
```

## Guardrails

The LLM may:

```text
summarize logged symptoms
notice and explain structured trend flags
choose relevant check-in questions from an approved library
suggest general preparation steps tied to care-team instructions
generate questions for the oncology team
explain uncertainty in plain language
```

The LLM must not:

```text
diagnose a new condition
claim symptoms prove tumor progression or response
recommend starting, stopping, or changing cancer treatment
replace urgent care or clinic instructions
invent fever thresholds, medication changes, supplement advice, or exercise restrictions
rank patient-facing treatment choices as the best option
claim a guaranteed, definite, or curative outcome
```

## Data sources and evidence

Use a hybrid evidence model:

```text
Structured app data:
  pathology, subtype, treatment phase, treatment dates, symptom logs,
  care-team instructions, medication/adherence entries, patient priorities

Symptom taxonomy:
  PRO-CTCAE-style patient-reported symptom items where appropriate

Evidence rules:
  curated subtype/treatment-context rules for daily-life focus areas

LLM outputs:
  language, prioritization, summaries, questions, and patient-friendly explanations
```

There is no expected public dataset that directly labels "daily-life priorities by breast-cancer subtype." The practical approach is to curate the clinical logic and use app-native logs to personalize the day-to-day experience.

## Example output object

```typescript
type DailyImpactCard = {
  caseId: string;
  date: string;
  focusAreas: string[];
  checkInItems: DailyCheckInItem[];
  trendSummary: string;
  suggestedActions: string[];
  careTeamQuestions: string[];
  safetyNotes: string[];
  llmTraceId: string;
  sourceLogIds: string[];
};

type DailyCheckInItem = {
  itemId: string;
  label: string;
  responseType: "scale_0_10" | "yes_no" | "choice" | "free_text";
  reason: string;
  source: "baseline" | "subtype_context" | "treatment_context" | "recent_trend" | "care_team_instruction";
};
```

## MVP scope

Build order:

```text
1. Structured symptom/adherence/check-in schema.
2. Approved symptom and check-in item library.
3. Deterministic trend flags over recent logs.
4. LLM daily check-in selection.
5. LLM daily impact card.
6. LLM doctor-ready summary.
7. Patient-safe scenario planning mode.
8. Safety tests for prohibited claims and missing disclaimers.
```
