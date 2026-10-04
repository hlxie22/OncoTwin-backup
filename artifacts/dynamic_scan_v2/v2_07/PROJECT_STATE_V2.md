# OncoTwin project state after V2-07

Status: **V2 research release complete; new independent external confirmation pending.**

Locked internal specification: `r1_temporal_fixed` with alpha=0.52.

Final deploy head: `artifacts/dynamic_scan_v2/v2_07/deploy_model.pt`

The deploy head was fitted on all 17,194 survival-eligible V2 development landmarks (2,443 patients). No historical CHORD test rows or DFCI/VICC outcomes were used for this fit. The fit is deployment-only; its in-sample performance is not validation evidence.

For a genuinely new cohort, follow the frozen V2-06 stage order: outcome-blind intake/overlap -> immutable V2-07 prediction freeze -> endpoint reveal and one-shot primary evaluation -> optional separately labeled recalibration.
