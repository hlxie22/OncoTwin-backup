# OncoTwin Documentation Set

This folder contains a project write-up for **OncoTwin**, a hybrid mechanistic-AI breast-cancer digital twin simulation system designed around the app concept discussed in the v4 direction.

The core idea is not to build a black-box response predictor. The system is designed as a **reaction-diffusion tumor simulator whose patient-specific parameters are initialized by multimodal AI and updated through Bayesian inference** as new MRI, pathology, molecular, treatment, and symptom data arrive.

## Files

1. [`01_motivation_background.md`](01_motivation_background.md)  
   Scientific motivation, clinical context, digital-twin framing, safety boundaries, and how the project extends the 2025 MRI-based digital-twin literature.

2. [`02_overall_system_architecture.md`](02_overall_system_architecture.md)  
   End-to-end architecture: offline training loop, online app loop, services, data flow, APIs, and deployment components.

3. [`03_data_pretraining_and_validation.md`](03_data_pretraining_and_validation.md)  
   Public datasets, pretrained models, training stages, validation plan, labels, harmonization, and model evaluation.

4. [`04_mri_ingestion_segmentation_feature_extraction.md`](04_mri_ingestion_segmentation_feature_extraction.md)  
   Detailed implementation of MRI ingestion, preprocessing, tumor segmentation, tumor volume extraction, quality control, and imaging feature generation.

5. [`05_mechanistic_tumor_simulator.md`](05_mechanistic_tumor_simulator.md)  
   Detailed implementation of the mechanistic reaction-diffusion treatment-response simulator, including state variables, equations, numerical solver, drug scheduling, and calibration targets.

6. [`06_ai_personalization_parameter_amortizer.md`](06_ai_personalization_parameter_amortizer.md)  
   AI layer that maps baseline MRI, pathology, molecular features, and context into patient-specific parameter distributions for the mechanistic model.

7. [`07_bayesian_twin_update_uncertainty.md`](07_bayesian_twin_update_uncertainty.md)  
   Bayesian updating, particle/ensemble representation, observation likelihoods, uncertainty bands, value-of-information, and update explanations.

8. [`08_molecular_graph_explanation_engine.md`](08_molecular_graph_explanation_engine.md)  
   Molecular graph attention layer, pathway representation, missing biomarker handling, parameter modifiers, and explainability design.

9. [`09_scenario_lab_toxicity_twin_and_safety.md`](09_scenario_lab_toxicity_twin_and_safety.md)  
   Twin Scenario Lab, schedule simulation, toxicity/person-burden model, patient-facing explanations, and safety constraints.

10. [`10_product_backend_frontend_api.md`](10_product_backend_frontend_api.md)  
    Product and engineering implementation details: backend services, database schema, API endpoints, frontend screens, job queue, and milestones.

## Suggested build order

1. Start with the **MRI segmentation and feature extraction pipeline** using MAMA-MIA / BreastDCEDL-style DCE-MRI data.
2. Implement the **mechanistic simulator** in a simplified spatial or regional form.
3. Add **Bayesian calibration** from longitudinal tumor-volume measurements.
4. Train the **AI parameter amortizer** using fitted patient parameters from public longitudinal data.
5. Add the **molecular graph explanation layer**.
6. Build the **Twin Scenario Lab** and safe patient-facing UI.
7. Add the **toxicity/person-burden twin**.
8. Wrap the system with clinical-safety language and model-validation reports.

## Key safety stance

OncoTwin should be presented as a **research simulation and educational explanation system**, not as a treatment-selection or diagnosis tool. It can show plausible response trajectories, uncertainty, missing-data sensitivity, and questions to discuss with an oncology team. It should not tell a patient which treatment to take.
