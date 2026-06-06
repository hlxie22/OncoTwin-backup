# Motivation and Background

## Project thesis

OncoTwin is a hybrid mechanistic-AI digital twin system for breast-cancer treatment-response simulation. Its purpose is to create a patient-specific virtual tumor model that can simulate plausible response trajectories under neoadjuvant treatment, update itself when new data arrive, and explain the biological and imaging evidence driving the simulation.

The project is motivated by a gap between two common approaches:

1. **Black-box AI response predictors**, which may predict pathological complete response or residual disease but often do not explain the mechanism of tumor change.
2. **Mechanistic mathematical models**, which can represent tumor growth and drug response but can be hard to personalize from limited baseline data.

OncoTwin combines the strengths of both approaches:

- A mechanistic model handles tumor growth, invasion, and treatment-induced cell death.
- Multimodal AI estimates patient-specific parameter distributions from MRI, pathology, molecular data, and patient context.
- Bayesian updating revises the twin as new MRI, tumor measurements, biomarkers, or symptom data arrive.
- The app displays uncertainty, explanations, residual-risk maps, and research scenario comparisons.

## Clinical and research context

Neoadjuvant therapy creates a natural setting for a digital twin because the tumor is still present and can be measured over time. Longitudinal MRI can show whether the tumor is shrinking, where residual disease may remain, and whether the response trajectory is consistent with high or low treatment sensitivity.

A digital twin system is especially useful when it can answer questions like:

- What range of tumor-volume trajectories is plausible for this patient?
- Which model parameters are most uncertain?
- Which new measurement would reduce uncertainty the most?
- How does the tumor's spatial response pattern change after early-treatment imaging?
- Which biological mechanisms appear to be influencing the simulation?
- How does patient-reported tolerance change the interpretation of research scenarios?

## Relationship to the 2025 MRI-based digital-twin paper

The closest published inspiration is the 2025 npj Digital Medicine paper, **"MRI-based digital twins to improve treatment response of breast cancer by optimizing neoadjuvant chemotherapy regimens."** The paper developed MRI-calibrated digital twins for triple-negative breast cancer using a biology-based mathematical model with tumor-cell migration, proliferation, and treatment-induced death. It used 105 ARTEMIS trial patients and simulated clinically feasible Adriamycin/Cytoxan-Taxol schedules.

OncoTwin should start from the same general mechanistic family: a reaction-diffusion tumor response model. However, it should extend that approach in ways that are especially useful for an app:

| Component | 2025 paper inspiration | OncoTwin extension |
|---|---|---|
| Tumor model | MRI-calibrated biology-based tumor dynamics | Reaction-diffusion tumor simulator with AI-personalized priors |
| Data | ARTEMIS trial cohort | Public I-SPY2, BreastDCEDL, MAMA-MIA, I-SPY1, and molecular datasets |
| Personalization | Calibration from patient MRI timepoints | Baseline MRI + pathology + molecular graph + Bayesian updates |
| Outputs | pCR prediction and schedule simulations | Response trajectories, uncertainty, residual-risk maps, mechanism explanations, patient-facing scenario lab |
| Safety/product | Research study | Educational/research app with strict non-recommendation constraints |

## Why not just train a pCR classifier?

A pure pCR classifier can be useful, but it is not enough for this project because it gives a static endpoint prediction rather than a dynamic explanation. OncoTwin should model the **path** of response, not only the final label.

A static model might output:

```text
Predicted pCR probability: 0.68
```

The digital twin should output:

```text
The current ensemble of simulated tumors predicts rapid early shrinkage but persistent residual-risk signal near the posterior tumor margin. The most uncertain parameters are taxane sensitivity and initial cellularity. A mid-treatment MRI would most reduce uncertainty.
```

That is the difference between a prediction tool and a twin.

## Intended user-facing framing

The app should be framed carefully:

> This is a research simulation and educational explanation system. It does not diagnose disease, predict your personal outcome with certainty, or recommend treatment. It shows plausible model-based response trajectories and uncertainty factors to discuss with a qualified oncology team.

## Core scientific claim

OncoTwin is best described as:

> A multimodal mechanistic-AI breast-cancer digital twin that combines MRI-derived tumor geometry, pathology biomarkers, molecular graph attention, Bayesian parameter updating, and patient-reported tolerance tracking to simulate personalized treatment-response trajectories with uncertainty.

## Key references

- MRI-based digital twins for TNBC neoadjuvant chemotherapy response: https://www.nature.com/articles/s41746-025-01579-1
- I-SPY2 on TCIA: https://www.cancerimagingarchive.net/collection/ispy2/
- BreastDCEDL_ISPY2: https://www.cancerimagingarchive.net/analysis-result/breastdcedl_ispy2/
- MAMA-MIA dataset and pretrained nnU-Net: https://github.com/LidiaGarrucho/MAMA-MIA
- MAMA-MIA Scientific Data paper: https://www.nature.com/articles/s41597-025-04707-4
- TNBC_DigitalTwins public mechanistic codebase: https://github.com/cchristenson2/TNBC_DigitalTwins
