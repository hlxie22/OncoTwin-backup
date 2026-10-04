# OncoTwin V2 research model card

## Frozen release

- Checkpoint: V2-07
- Internal specification: `r1_temporal_fixed`
- Scan update: fixed logit interpolation alpha = 0.52
- Final supervised head: one all-development fit on 17,194 survival-eligible landmarks from 2,443 patients
- Temporal encoder: repaired CKPT7R1 line-agnostic encoder (frozen)
- Genomic encoder: CKPT3 genomic encoder (frozen)

## Evidence status

The deploy head is fitted after internal model selection and is **not** evaluated on its own training rows as validation evidence. The original CHORD historical test and completed DFCI/VICC external outcomes are not reopened in V2-07. Independent V2 external confirmation remains pending.

## Input boundary

The public inference interface consumes prepared PRE/POST temporal states, the frozen 128-D tumor embedding, patient-level scan count context, genomic availability, and genomic age. Raw-cohort predictor construction is upstream and must satisfy the frozen V2-06 protocol. Treatment-line identity is not accepted.

## Intended use

Research forecasting and prospective evaluation under the frozen protocol. This artifact is not a clinical device and is not validated for autonomous clinical decision-making.
