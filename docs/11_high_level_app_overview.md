# High-Level App Overview

## Plain-language summary

OncoTwin is a proposed breast-cancer research app that creates a patient-specific "digital twin" of a tumor. A digital twin is a computer model that represents a real-world thing closely enough that researchers can run simulations on it. In this project, the real-world thing is a breast tumor during neoadjuvant treatment, meaning treatment given before surgery.

The app is not meant to diagnose cancer, guarantee outcomes, or replace an oncology team. Its intended role is to help researchers, clinicians, and eventually patients explore questions such as:

- How might this tumor change over time under treatment?
- Which parts of the prediction are uncertain?
- What new information would make the prediction more reliable?
- Which imaging, pathology, or molecular features are influencing the model?
- How do different research scenarios compare when tumor response, uncertainty, and treatment burden are considered together?
- What should the patient track today, and what questions should they bring to the care team?

The main idea is to combine two types of modeling:

1. **Mechanistic modeling**, which uses equations inspired by biology to simulate tumor growth, spread, and treatment response.
2. **Artificial intelligence**, which helps personalize those equations from MRI scans, pathology results, molecular markers, and patient context.

This combination is important. A simple AI model might only say, "This case has a 68% chance of responding." OncoTwin is designed to say something richer: "Here are many plausible tumor-response paths, here is why the model believes them, here is what remains uncertain, and here is what kind of new data would reduce that uncertainty."

## The problem the app is trying to address

Cancer treatment decisions are complex because every patient and every tumor is different. Even when two people have the same cancer subtype, their tumors may grow differently, respond to drugs differently, show different MRI patterns, and cause different treatment burdens.

Many prediction tools are static. They take a snapshot of the patient at one time and produce a risk score or response probability. That can be useful, but it does not fully capture the changing nature of treatment. During neoadjuvant therapy, the tumor can be observed over time through MRI and clinical measurements. A useful digital twin should therefore update itself as new evidence arrives.

OncoTwin is designed around that dynamic view. It starts with what is known at baseline, builds an initial model of the tumor, simulates possible futures, and then revises those futures as follow-up MRI scans, tumor measurements, biomarker results, treatment changes, or symptom reports become available.

## What the app does

At a high level, the app has eight major functions.

### 1. Build a patient case

The app begins by creating a case. A case can include:

- MRI scans or manually entered tumor measurements.
- Pathology details such as ER status, PR status, HER2 status, tumor grade, Ki-67, and node status.
- Molecular markers such as BRCA1, BRCA2, HRD, TP53, PIK3CA, ESR1, or ERBB2 when available.
- Treatment context, such as the current regimen, dose events, timing, delays, or modifications.
- Patient context and optional symptom information.

The app is designed to handle different levels of data. If only a tumor diameter is available, it can build a very rough model. If a baseline MRI is available, it can build a spatial tumor model. If multiple MRI timepoints are available, it can perform stronger updating because it can compare the simulated tumor against observed tumor change over time.

### 2. Turn MRI scans into useful tumor information

MRI scans are not directly usable by the simulator. They must first be processed into measurements and maps. The imaging pipeline is responsible for:

- Accepting DICOM or NIfTI MRI data.
- Standardizing image orientation, spacing, and intensity.
- Identifying dynamic contrast-enhanced MRI phases.
- Segmenting the tumor, meaning drawing a 3D boundary around it.
- Measuring tumor volume and longest diameter.
- Computing imaging features such as enhancement, shape, heterogeneity, and possible low-enhancement regions.
- Producing quality-control warnings when the image or segmentation may be unreliable.

This step matters because the simulator needs a starting tumor geometry. In simple terms, the app first asks, "Where is the tumor, how large is it, and what does it look like on imaging?"

### 3. Estimate patient-specific model settings

The tumor simulator has parameters that control how the simulated tumor behaves. These include things like:

- How quickly tumor cells proliferate.
- How much the tumor spreads or invades nearby tissue.
- How sensitive the tumor may be to different drug categories.
- How drug exposure changes over time.
- How much local MRI enhancement may act as a rough proxy for drug delivery.
- How much resistant tumor-cell population may remain.

The app uses an AI personalization layer to estimate a distribution over these parameters. A distribution means the app does not pretend there is one exact answer. Instead, it represents a range of plausible settings.

This is one of the app's central design choices: **the AI does not replace the tumor simulator. The AI estimates the simulator's patient-specific inputs.**

That keeps the system more interpretable. Instead of producing only a black-box response score, the app can explain which biological assumptions are driving the simulated result.

### 4. Simulate tumor response over time

The mechanistic simulator is the scientific core of the app. It is based on a reaction-diffusion style tumor model. In beginner terms, the simulator tries to represent three broad processes:

- **Growth**: tumor cells can multiply.
- **Spread or invasion**: tumor cells can move into nearby tissue.
- **Treatment effect**: drugs can kill or suppress tumor cells, depending on exposure and sensitivity.

The simulator can run at different levels of detail:

- A **volume-only model** when only tumor size is known.
- A **regional model** when the tumor can be divided into meaningful regions, such as enhancing core and boundary.
- A **voxelwise spatial model** when MRI data are good enough to simulate the tumor across a 3D grid.

The app does not simulate just one future. It samples many plausible parameter sets and simulates an ensemble of possible tumor futures. The result is a range of trajectories rather than a single line. This is how the app can show uncertainty bands around tumor-volume predictions.

### 5. Update the twin when new data arrive

The app becomes more useful when it updates. A first simulation is based on baseline information, but follow-up data can confirm, weaken, or change the model's assumptions.

For example:

- If an early-treatment MRI shows stronger shrinkage than expected, the app may shift weight toward simulated twins with higher drug sensitivity.
- If the tumor shrinks less than expected, the app may shift weight toward lower sensitivity, resistant disease, or uncertainty about the measurement.
- If BRCA or HRD status becomes available, the molecular assumptions may narrow or shift.
- If treatment is delayed or modified, the future drug-exposure schedule changes.
- If symptoms become severe, the tumor biology does not automatically change, but scenario interpretation may change because treatment burden matters.

The update method is Bayesian in spirit. That means the app starts with a prior belief about plausible tumor behavior, compares simulated predictions to new observations, and then gives more weight to the simulations that better match the evidence.

In plain language: **the twin learns from new measurements by keeping the simulated versions that fit the evidence better and downweighting the ones that fit poorly.**

### 6. Explain why the model behaves the way it does

OncoTwin is intended to be explainable. It should not simply show a graph and expect users to trust it.

The explanation system is designed to identify major drivers such as:

- Imaging features, such as tumor volume, enhancement pattern, or irregular boundary.
- Pathology features, such as subtype, grade, Ki-67, or hormone receptor status.
- Molecular features, such as BRCA/HRD status or HER2-related signals.
- Missing data, such as unknown biomarkers.
- Data-quality issues, such as low segmentation confidence.
- Model uncertainty, such as wide disagreement across simulated twins.

The molecular graph explanation engine adds a mechanism-aware layer. It represents genes, pathways, drug mechanisms, and phenotypes as connected concepts. For an early version, this can be rule-based. For a later version, it can become a graph attention model trained on molecular and clinical datasets.

The goal is not to overclaim that one mutation determines the outcome. The goal is to say, carefully, that certain biomarkers change the model's assumptions about pathways such as DNA repair, HER2 signaling, estrogen receptor signaling, proliferation, immune response, or drug resistance.

### 7. Support daily life with an LLM co-pilot

For patient-facing use, OncoTwin should not present a static "life impact plan" and expect the patient to remember it. Instead, it should use an LLM API over structured case data, treatment context, care-team instructions, and recent logs to generate daily support.

The daily co-pilot should:

- Select a short daily check-in, usually 3-6 relevant questions.
- Generate a daily impact card with timely suggestions and reminders.
- Explain symptom and adherence patterns in plain language.
- Draft questions for the oncology team.
- Prepare doctor-ready summaries before visits.
- Help patients plan around treatment burden without ranking treatment choices.

For example, a patient receiving chemotherapy might see a check-in focused on fatigue, nausea, appetite, neuropathy, and new symptoms. A patient on endocrine therapy might see medication adherence, joint stiffness, hot flashes, sleep, and activity. A patient receiving radiation might see skin irritation, fatigue, discomfort, and shoulder tightness.

This layer should use deterministic trend calculations over structured logs and an LLM API for interpretation, prioritization, and language. It should not use symptoms to infer tumor response directly, and it should not diagnose side effects or recommend changing treatment.

### 8. Run safe research scenarios

The Twin Scenario Lab is the app's interactive exploration area. It lets users run clearly labeled research simulations such as:

- What does the current twin predict under the current treatment context?
- How would the model change if an early MRI showed strong or weak shrinkage?
- How would uncertainty change if a missing biomarker were positive, negative, or still unknown?
- How do several clinically plausible candidate schedules compare in the model?
- How does high symptom burden change the interpretation of tradeoffs?

The Scenario Lab may compare and rank candidate options, but only as exploratory model outputs. Any ranking must be shown with uncertainty and a clear disclaimer. The app must not say that a treatment will work, will cure a patient, or should be followed as medical advice.

In patient-facing mode, the Scenario Lab should emphasize planning and care-team questions rather than treatment ranking. Example patient-facing scenarios include, "What should I prepare for this treatment week?", "Which symptoms usually worsened after infusion?", and "What information would make the model less uncertain?"

## How the methods fit together

The app can be understood as a pipeline.

```text
Patient data
  ↓
MRI processing and tumor segmentation
  ↓
Imaging, pathology, molecular, treatment, and context features
  ↓
AI personalization layer estimates plausible simulator parameters
  ↓
Mechanistic tumor simulator creates many possible response paths
  ↓
Bayesian update gives more weight to simulations that match new evidence
  ↓
Scenario Lab compares research scenarios
  ↓
LLM daily co-pilot creates check-ins, daily impact cards, and visit summaries
  ↓
Explanations, uncertainty summaries, residual-risk maps, patient planning cards, and care-team discussion summaries
```

Each part has a distinct role:

- **MRI processing** answers, "What does the tumor look like now?"
- **AI personalization** answers, "Which simulator settings seem plausible for this case?"
- **Mechanistic simulation** answers, "What could happen over time under these assumptions?"
- **Bayesian updating** answers, "Which simulated futures still fit the evidence?"
- **Molecular explanation** answers, "Which biological pathways may be affecting the assumptions?"
- **Scenario Lab** answers, "How do model-based possibilities compare under clearly labeled hypothetical conditions?"
- **LLM daily co-pilot** answers, "What should the patient track today, and what question should they ask next?"
- **Safety and uncertainty design** answers, "How do we avoid making the output sound more certain than it is?"

## Main user experience

A typical user flow would look like this:

1. **Create a case** by entering basic case details and selecting research, demo, or user-entered mode.
2. **Add clinical context** such as pathology, molecular markers, treatment plan, and available symptoms.
3. **Upload MRI data or enter tumor measurements** depending on what is available.
4. **Review the virtual tumor state**, including tumor size, segmentation confidence, imaging features, and missing-data warnings.
5. **Initialize the twin**, which creates an ensemble of possible patient-specific tumor models.
6. **View simulated response**, including a median trajectory, uncertainty bands, and possible residual-risk maps.
7. **Add new observations** such as a follow-up MRI, updated biomarker, or treatment delay.
8. **Review how the twin changed** and why some simulated futures became more or less likely.
9. **Use the daily co-pilot** to complete short check-ins, review daily impact cards, and save care-team questions.
10. **Run research or patient-planning scenarios** in the Scenario Lab.
11. **Generate a doctor or research summary** that lists observations, symptom patterns, uncertainty drivers, exploratory comparisons, and questions to discuss with a qualified oncology team.

## Important outputs

The app is designed to produce several kinds of output:

- **Tumor trajectory chart**: projected tumor volume over time, with observed measurements overlaid.
- **Uncertainty bands**: visual ranges showing that the model has more than one plausible answer.
- **Residual-risk map**: a spatial visualization of where simulated residual disease may be more likely, when MRI data support it.
- **Parameter summary**: a research-facing view of the assumptions controlling growth, invasion, drug sensitivity, and related biology.
- **Explanation panel**: plain-language reasons the model behaved as it did.
- **Missing-data ranking**: a list of additional measurements that may reduce uncertainty.
- **Scenario comparison**: exploratory comparison of current and hypothetical research scenarios.
- **Daily impact card**: LLM-generated daily check-in focus, suggestions, and care-team questions based on structured logs and safety rules.
- **Toxicity or person-burden summary**: a separate view of symptom trends and treatment burden.
- **Care-team summary**: a concise report intended to support discussion, not replace professional judgment.

## Why uncertainty is central

OncoTwin is built around uncertainty because the available data will often be incomplete. A baseline MRI, pathology report, and a few biomarkers cannot fully determine tumor biology. Even follow-up measurements have noise. Segmentations can be imperfect. Public datasets may not represent every patient group equally. Molecular markers may be missing. Treatments may change in real life.

Because of that, the app should avoid fake confidence. It should show:

- Uncertainty from limited data.
- Uncertainty from image quality or segmentation confidence.
- Uncertainty from missing biomarkers.
- Uncertainty from model assumptions.
- Uncertainty from disagreement across simulated twins.

This design makes the app more honest and more useful. A high-uncertainty output tells the user not only that the model is unsure, but also why it is unsure and what information might help.

## Intended impact

The intended impact of OncoTwin is to make breast-cancer treatment-response modeling more dynamic, transparent, and useful for research and decision support.

For researchers, the app could provide a structured environment for testing digital-twin methods, comparing mechanistic and AI components, studying uncertainty, and validating response simulations against public datasets.

For clinicians, a future validated version could help organize complex information into a clearer picture: what has been observed, what the model thinks is plausible, what remains uncertain, and which questions may be useful to discuss.

For patients, a carefully designed version could make the treatment journey more understandable and practical by showing that predictions are not certainties, explaining why more data can matter, choosing relevant daily check-ins, generating timely suggestions, and supporting better conversations with the care team.

For app development, the project creates a practical blueprint for combining medical imaging, mechanistic simulation, AI personalization, Bayesian updating, molecular explanation, scenario comparison, and safety-focused user interfaces.

The intended impact is not to automate treatment decisions. The intended impact is to provide an exploratory simulation system that makes tumor response, uncertainty, and evidence easier to reason about.

## Safety stance

OncoTwin must be presented as an exploratory research and decision-support simulation. It should not be presented as:

- A diagnostic tool.
- A guaranteed prediction engine.
- A treatment-prescribing system.
- A replacement for clinical judgment.
- A substitute for professional medical advice.

The patient-facing LLM co-pilot must also avoid inventing clinic-specific instructions. Fever thresholds, medication changes, supplement advice, exercise restrictions, and urgent-call rules should come from the oncology team or approved app content, not free-form model generation.

Any output that compares, suggests, or ranks treatment-related options must include a disclaimer similar to:

```text
Exploratory research simulation. Predictions and rankings are uncertain and not guaranteed, and this is not a substitute for professional medical advice. Discuss all treatment decisions with a qualified oncology team.
```

The hard rule is to avoid false certainty. The app may say that one option ranked higher in a model simulation under specific assumptions. It must not say that an option will work, will cure disease, or should be followed without professional review.

## Current documentation status

The repository currently describes a proposed system design and implementation plan. The documents define the app concept, architecture, data strategy, modeling approach, safety stance, and product surfaces. They should be read as a blueprint for building and validating OncoTwin, not as evidence that a clinically validated product already exists.

For deeper detail, see:

- `01_motivation_background.md` for the scientific motivation and clinical framing.
- `02_overall_system_architecture.md` for the full architecture and data flow.
- `03_data_pretraining_and_validation.md` for datasets, training stages, and validation.
- `04_mri_ingestion_segmentation_feature_extraction.md` for MRI processing.
- `05_mechanistic_tumor_simulator.md` for the tumor simulation model.
- `06_ai_personalization_parameter_amortizer.md` for AI-based parameter personalization.
- `07_bayesian_twin_update_uncertainty.md` for updating and uncertainty.
- `08_molecular_graph_explanation_engine.md` for biomarker and pathway explanations.
- `09_scenario_lab_toxicity_twin_and_safety.md` for scenario simulation, toxicity, and safety.
- `10_product_backend_frontend_api.md` for product surfaces, backend services, frontend screens, and APIs.
- `12_patient_facing_llm_copilot.md` for adaptive daily check-ins, daily impact cards, trend explanations, patient-safe planning scenarios, and doctor-ready summaries.
