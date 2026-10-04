# Project State


## Checkpoint 1C - public reference reconciliation

- CKPT1B strict PFS reconstruction exactly matches the current public
  `mbc-pfs` pipeline at the patient-line level:
  - 3,216 patients
  - 11,696 usable PFS lines
- Current public-main downstream QC produces:
  - 2,947 patients
  - 9,553 lines
  - 247 final design-matrix columns
  - 2,143 QC-excluded lines
- Manuscript reports:
  - 2,881 patients
  - 8,791 lines
- Therefore the PFS reconstruction is considered validated. The remaining
  manuscript-count discrepancy is downstream of PFS construction and is
  recorded as a public-code/manuscript-version reproducibility discrepancy.
- Do not force canonical cohort rules merely to reproduce the manuscript
  count.
- Next: endpoint replication and line-start/scan-landmark baselines.

<!-- CKPT2_ENDPOINT_BASELINES -->
## Checkpoint 2 - Endpoint replication and baseline proof

Status: **PASS_SCAN_UPDATE_INFORMATION_GAIN**

Frozen endpoint:
- CKPT1B/current-public PFS: 3216 patients / 11696 lines.
- Canonical PFS rules are frozen.
- Do not force the manuscript 2,881 / 8,791 counts.
- Current-public downstream QC discrepancy remains a reproducibility/version finding.

Evaluation design:
- Patient-disjoint CKPT1 split reused before landmark expansion.
- Line-start reference baselines: penalized Cox, regularized discrete-time hazard, gradient-boosted discrete-time hazard.
- Primary scan proof uses NON_PROGRESSIVE scan landmarks only.
- Stale forecast: line-start survival curve conditionally carried forward to scan time.
- Pre-scan model: line-start features + elapsed time + prior scan history.
- Post-scan model: pre-scan features + current scan evidence.
- Paired information gain is bootstrapped by patient.

Primary held-out post-vs-pre NLL gain:
- mean: 3.3699710964888623
- patient-bootstrap 95% CI: [1.5738900034242818, 5.3251224863911535]

Artifacts:
- artifacts/checkpoint2/endpoint_replication.json
- artifacts/checkpoint2/persistent_leakage_checks.json
- artifacts/checkpoint2/line_start_baselines.json
- artifacts/checkpoint2/scan_updating_proof.json
- artifacts/checkpoint2/scan_policy_sensitivity.json
- artifacts/checkpoint2/scan_test_predictions.parquet
- artifacts/checkpoint2/paired_scan_nll_test.parquet
- artifacts/checkpoint2/qc.json
- artifacts/checkpoint2/audit.md
- artifacts/handoff/checkpoint_02.json

<!-- CKPT3_GENOMIC_PRETRAINING -->
## Checkpoint 3 - Coverage-aware GENIE genomic encoder

Status: **PASS_GENOMIC_PRETRAINING**

Frozen genomic interface:
- fixed 468-gene space
- mutation state is separate from assay coverage
- unassayed is never encoded as observed wild type
- two graph-attention blocks, hidden 128, 8 heads
- pooled tumor embedding: 128D
- EMA teacher receives full genuinely observed sample panel
- student receives synthetic reduced coverage derived from real training panels
- masked reconstruction loss is scored only on genuinely assayed genes
- positive alterations are prevalence-weighted

Leakage policy:
- exact DFCI/VICC overlapping GENIE patients excluded
- primary molecular pretraining is center-disjoint from DFCI and VICC; MSK remains in development/training
- held-out institution and held-out panel evaluations completed independently

Evaluation:
- validation overall AUPRC: 0.3023464138155686
- held-out panel overall AUPRC: 0.18944649170196037
- held-out institution overall AUPRC: 0.17020345751289492

Artifacts:
- artifacts/checkpoint3/genomic_encoder.pt
- artifacts/checkpoint3/evaluation.json
- artifacts/checkpoint3/genie_sample_embeddings_f16.npy
- artifacts/checkpoint3/genie_embedding_index.parquet
- artifacts/checkpoint3/audit.md
- artifacts/handoff/checkpoint_03.json

<!-- CKPT4_TEMPORAL_PRETRAINING -->
## Checkpoint 4 - CHORD temporal pretraining

Status: **PASS_TEMPORAL_PRETRAINING**

Frozen temporal architecture:
- causal event Transformer
- 4 layers / hidden 192 / 6 heads / FF 768 / dropout 0.15
- maximum causal context 512 structured tokens
- pan-cancer CHORD pretraining followed by breast-specific continued pretraining
- objectives: masked event reconstruction, next-event type, next-time bucket,
  next-scan state, next-site pattern, event-dropout consistency
- CKPT1 breast patient splits remain patient-disjoint
- progressive scans remain endpoint/state-update tokens and are not residual-PFS landmarks

Held-out breast temporal evaluation:
- next-event micro-AUPRC: 0.3674041383980083
- next-event prior baseline: 0.35002058627229427
- next-scan macro-AUPRC: 0.4754212670510281
- next-scan prior baseline: 0.3333333333333333

Cached scan states:
- rows: 49189
- representation: PRE-scan + POST-scan, each 192-D
- effective rank: 33.00719451904297

Artifacts:
- artifacts/checkpoint4/temporal_encoder.pt
- artifacts/checkpoint4/breast_scan_prepost_embeddings_f16.npy
- artifacts/checkpoint4/breast_scan_prepost_index.parquet
- artifacts/handoff/checkpoint_04.json

<!-- CKPT4_FINAL_QC_CORRECTION -->
### Checkpoint 4 final independent QC correction

Final independent QC: **PASS**

- Total CKPT1 mBC modeling patients: 3461
- Patients represented in authoritative W3 scans: 3351
- mBC patients without a W3 scan: 110
- Authoritative W3 scan episodes: 49189
- PRE/POST temporal state rows: 49189
- Exact scan-key reconciliation: patient_id+scan_episode_id_exact
- No preparation, training, evaluation, or scan-cache rerun was required.

The earlier QC failure was caused solely by incorrectly requiring all
3,461 modeling patients to occur in a scan-time embedding table.

<!-- CKPT6A_SCAN_UPDATE_DIAGNOSIS -->
## Checkpoint 6A — Scan-update mechanism diagnosis

Status: **DIAGNOSTIC_COMPLETE_RETRAIN_REQUIRED**

CKPT5 remains frozen as:
`TRAINED_EVALUATED_PRIMARY_SCAN_GAIN_NOT_ESTABLISHED`

No upstream cohort, endpoint, encoder, or external-validation rule changed.

CKPT6A now makes the patient-facing progression/death PFS estimand explicit:
- progression + death are target PFS events
- switch is a separately observed competing event
- censoring uses training-derived IPCW
- primary diagnostic aggregation is patient balanced
- 3/6/12/18-month Brier, discrimination, and calibration are reported

Update-shrinkage rule tested without retraining:

`PRE_logits + alpha * (POST_logits - PRE_logits)`

Alpha was selected from validation only.

Selected validation alpha by patient PFS IBS:
**0.75**

Selected validation alpha by patient NLL:
**0.5**

Validation signals:
`{'bounded_update_signal': True, 'selected_alpha_by_patient_pfs_ibs': 0.75, 'selected_alpha_by_patient_nll': 0.5, 'full_scan_update_improves_patient_pfs_ibs': True, 'pre_plus_scan_improves_patient_pfs_ibs': True, 'post_temporal_without_explicit_scan_improves_patient_pfs_ibs': False, 'post_without_explicit_scan_beats_full_post': False, 'validation_full_pre_minus_post_patient_pfs_ibs': 0.002725073596930694, 'validation_no_genomics_pre_minus_post_patient_pfs_ibs': 0.00031941130447016297}`

CKPT6B should now retrain only the targeted candidate architectures identified
in `artifacts/checkpoint6a/decision_packet.json`.

DFCI/VICC outcomes remain untouched.

Artifacts:
- artifacts/checkpoint6a/metrics_summary.json
- artifacts/checkpoint6a/alpha_shrinkage.json
- artifacts/checkpoint6a/mechanistic_ablations.json
- artifacts/checkpoint6a/stratified_metrics.json
- artifacts/checkpoint6a/stale_pre_post_comparison.json
- artifacts/checkpoint6a/val_diagnostic_predictions.parquet
- artifacts/checkpoint6a/test_diagnostic_predictions.parquet
- artifacts/checkpoint6a/decision_packet.json
- artifacts/checkpoint6a/audit.md
- artifacts/handoff/checkpoint_06A.json

<!-- CKPT6B_BOUNDED_SCAN_UPDATE -->
## Checkpoint 6B — Bounded scan-update candidate

Status: **PASS_BOUNDED_UPDATE_CANDIDATE_FREEZE**

CKPT6A diagnosed CKPT5 as an update-amplitude/calibration problem rather
than absence of current-scan signal.

Frozen candidate:

`POST_bounded = PRE + 0.75 * (POST_full - PRE)`

Alpha 0.75 was chosen using CKPT6A validation patient-balanced PFS IBS
only. It was not chosen using test outcomes.

Internal test:
- PRE patient IBS: 0.1673434533268599
- full POST patient IBS: 0.16462208637892925
- bounded POST patient IBS: 0.16352614872814603
- PRE patient NLL: 2.7634854849024917
- full POST patient NLL: 2.7515328816266393
- bounded POST patient NLL: 2.7314647641077596
- PRE minus bounded IBS bootstrap: {'mean': 0.0038173045987138776, 'ci_low': -0.00020630524466693707, 'ci_high': 0.008235945667347045, 'patients': 409, 'repetitions': 2000}
- PRE minus bounded NLL bootstrap: {'mean': 0.032020720794731734, 'ci_low': 0.006535086311358488, 'ci_high': 0.059797989947081916, 'patients': 437}

Decision:
Freeze the bounded candidate, but retain targeted survival-only/direct-scan retraining as a secondary development ablation because one or more bootstrap intervals still include zero.

DFCI/VICC outcomes remain untouched.

Artifacts:
- artifacts/checkpoint6b/bounded_dynamic_scan_candidate.pt
- artifacts/checkpoint6b/bounded_update_metrics.json
- artifacts/checkpoint6b/test_bounded_update_predictions.parquet
- artifacts/checkpoint6b/audit.md
- artifacts/handoff/checkpoint_06B.json

<!-- CKPT6C_UPDATE_SPECIFICITY -->
## Checkpoint 6C — Scan-update specificity

Status: **PASS_INTERNAL_DYNAMIC_CANDIDATE_LOCK**

Frozen candidate remains:

`PRE + 0.75 * (POST - PRE)`

No retraining or test-based tuning was performed.

Negative controls:
- validation global mean scan update
- validation observable-stratum mean scan update
- within-stratum permutation of individual test scan updates

Test PRE patient IBS:
0.1673434533268599

Test individualized bounded patient IBS:
0.16352614872814603

Test PRE patient NLL:
2.7634854849024917

Test individualized bounded patient NLL:
2.7314647641077596

Permutation control:
{'repetitions': 500, 'stratification': 'patient_scan_number x treatment_line x genomic_availability', 'observed': {'patient_nll_gain': 0.032020718811432314, 'patient_ibs_gain': 0.0038173057697393764}, 'permuted': {'nll_gain_mean': -0.05388962574051024, 'nll_gain_p95': -0.03839160731631565, 'nll_gain_p99': -0.031548466394257416, 'ibs_gain_mean': -0.0023349059957201677, 'ibs_gain_p95': -0.00020633950882816037, 'ibs_gain_p99': 0.0007546491717056355}, 'empirical_one_sided_p': {'nll': 0.001996007984031936, 'ibs': 0.001996007984031936}}

Specificity flags:
{'bounded_point_ibs_improves': True, 'bounded_nll_improves': True, 'beats_validation_stratum_mean_nll': True, 'beats_validation_stratum_mean_ibs': False, 'permutation_nll_specific': True, 'permutation_ibs_specific': True, 'individualized_specificity_nll': True, 'individualized_specificity_ibs': False}

Next:
Keep the CKPT6B bounded candidate frozen. Proceed to CKPT6D W0/W3/W7 grouping robustness using the frozen encoders, then open CKPT7 external validation only if window sensitivity is acceptable.

DFCI/VICC outcomes remain untouched.

Artifacts:
- artifacts/checkpoint6c/specificity_report.json
- artifacts/checkpoint6c/control_metrics.json
- artifacts/checkpoint6c/permutation_negative_control.json
- artifacts/checkpoint6c/val_specificity_predictions.parquet
- artifacts/checkpoint6c/test_specificity_predictions.parquet
- artifacts/checkpoint6c/audit.md
- artifacts/handoff/checkpoint_06C.json

# CHECKPOINT 6D — scan grouping robustness

Status: `PASS_WINDOW_GROUPING_ROBUSTNESS`

- frozen candidate: `PRE + 0.75 × (POST - PRE)`
- no retraining or parameter selection
- W0/W3/W7 temporal histories regenerated with the frozen CKPT4 encoder
- authoritative CKPT1 treatment-line mapping preserved exactly
- primary robustness cohort is exact patient/day/line matched and remains
  NON_PROGRESSIVE under all three grouping windows
- common test rows: 2912
- common test patients: 432
- external DFCI/VICC outcomes remain untouched

Window results:

- W0:
  - NLL gain: 0.01627492904663086
  - IBS gain: 0.0020783883781614054
- W3:
  - NLL gain: 0.03403925895690918
  - IBS gain: 0.0035415235649546295
- W7:
  - NLL gain: 0.020157337188720703
  - IBS gain: 0.0027321716906411886

Next action:

Freeze CKPT6E external-validation protocol and immutable candidate manifest before opening DFCI/VICC outcomes.

# CHECKPOINT 6E — external validation protocol freeze

Status: `PASS_EXTERNAL_VALIDATION_PROTOCOL_FREEZE`

External validation is now locked before DFCI/VICC outcome access.

Frozen primary specification:

- candidate: bounded dynamic scan model
- alpha: 0.75
- scan grouping: W3
- primary scan state: NON_PROGRESSIVE
- PRE: history strictly before current scan
- POST: PRE plus exactly current scan
- genomic availability: availability_day < landmark_day
- PFS events: progression/death
- switch: competing event
- administrative horizon: 730 days
- patient-balanced primary metrics
- Brier horizons: 3/6/12/18 months
- 2,000 patient bootstrap replicates
- DFCI and VICC reported separately
- pooled external analysis secondary
- no primary recalibration
- no model/alpha/metric changes between external centers

External outcome state at freeze:

- DFCI outcomes: untouched
- VICC outcomes: untouched

Lock digest:

`0a56b6067820226414e5a735a63dce42c1a6d404779b1a44cce74dab51791180`

Next:

CKPT7A outcome-blinded external feature/landmark adapter, followed by CKPT7B
one-shot DFCI/VICC primary evaluation.
