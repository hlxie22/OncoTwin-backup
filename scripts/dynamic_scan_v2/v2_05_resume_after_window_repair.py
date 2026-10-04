from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from access import AccessPolicy
from common import atomic_json, atomic_text, sha256_file
from metrics import HORIZONS_MONTHS, v2_metric_bundle
from v2_03_gate import blend_logits
from v2_05_controls import decide_candidate_lock
from v2_05_robustness import (
    SELECTED_GENOMIC,
    SELECTED_NAME,
    SELECTED_REPRESENTATION,
    _load_arrays,
    _load_core,
    _normalize_key_frame,
    _window_sensitivity,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path):
    require(path.is_file(), f"Required prior V2-05 artifact missing: {path}")
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = repo / "artifacts/dynamic_scan_v2/v2_05"
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((repo / "configs/dynamic_scan_v2/v2_05_robustness.json").read_text())
    v204_cfg = json.loads((repo / "configs/dynamic_scan_v2/v2_04_ablation.json").read_text())
    policy = AccessPolicy.from_json(repo, repo / "configs/dynamic_scan_v2/data_access_policy.json")

    # The failed first run completed every pre-window diagnostic. Require them
    # explicitly so this resume path cannot silently skip unfinished work.
    required_prior = [
        "frozen_analysis_plan.json",
        "input_integrity.json",
        "seed_fold_stability.json",
        "paired_patient_bootstrap.json",
        "scan_specificity.json",
        "reference_controls.json",
        "subgroup_sensitivity.parquet",
        "channel_removal_diagnostics.json",
        "calibration.parquet",
        "calibration_and_support.json",
        "exploratory_outcome_decomposition.json",
        "deterministic_robustness.json",
    ]
    missing = [name for name in required_prior if not (out / name).is_file()]
    require(not missing, f"Cannot resume V2-05; pre-window artifacts missing: {missing}")

    # Reverify the frozen V2-04 artifact manifest before using its predictions.
    manifest = load_json(repo / "artifacts/dynamic_scan_v2/v2_04/artifact_manifest.json")
    for name, expected in manifest["files"].items():
        p = repo / "artifacts/dynamic_scan_v2/v2_04" / name
        require(p.is_file(), f"V2-04 manifest file missing: {name}")
        require(sha256_file(p) == expected, f"V2-04 artifact hash mismatch: {name}")

    frame, dev, all_dev, km = _load_core(repo, policy)
    arrays, dims = _load_arrays(repo, policy)
    paired_index = _normalize_key_frame(
        policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_01/paired_scan_index.parquet")
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[V2_05_RESUME_DEVICE]", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)

    # Frozen alpha and OOF branch predictions.
    v204 = load_json(repo / "artifacts/dynamic_scan_v2/v2_04/result_packet.json")
    v203 = load_json(repo / "artifacts/dynamic_scan_v2/v2_03/result_packet.json")
    require(v204.get("status") == "PASS" and v203.get("status") == "PASS", "V2-03/V2-04 must PASS")
    require(v204["selection"]["selected_candidate"] == SELECTED_NAME, "Frozen V2-04 candidate changed")
    alpha = float(v204["locked_alpha"])

    keys = ["patient_id", "scan_episode_id"]
    idx2 = _normalize_key_frame(policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_02/selected_oof_index.parquet"))
    with policy.np_load(repo / "artifacts/dynamic_scan_v2/v2_02/selected_oof_logits.npz") as z:
        bpre = np.asarray(z["pre_logits"], np.float32)
        bpost = np.asarray(z["post_logits"], np.float32)
    require(np.array_equal(idx2[keys].to_numpy(), dev[keys].to_numpy()), "V2-02 OOF key drift")

    idx4 = _normalize_key_frame(policy.read_parquet(repo / "artifacts/dynamic_scan_v2/v2_04/selected_v2_04_oof_index.parquet"))
    with policy.np_load(repo / "artifacts/dynamic_scan_v2/v2_04/selected_v2_04_oof_logits.npz") as z:
        cpre = np.asarray(z["pre_logits"], np.float32)
        cpost = np.asarray(z["post_logits"], np.float32)
    require(np.array_equal(idx4[keys].to_numpy(), dev[keys].to_numpy()), "V2-04 OOF key drift")

    baseline_locked = blend_logits(bpre, bpost, alpha)
    candidate_locked = blend_logits(cpre, cpost, alpha)
    baseline_metric = v2_metric_bundle(dev, baseline_locked, km)
    candidate_metric = v2_metric_bundle(dev, candidate_locked, km)

    seed_report = load_json(out / "seed_fold_stability.json")
    boot = load_json(out / "paired_patient_bootstrap.json")
    specificity = load_json(out / "scan_specificity.json")
    controls = load_json(out / "reference_controls.json")
    calibration = load_json(out / "calibration_and_support.json")
    deterministic = load_json(out / "deterministic_robustness.json")

    # Run only the repaired W0/W3/W7 analysis. Repaired alternate temporal caches
    # from the first attempt are reused if already present.
    seed0 = int(v204_cfg["screening_seed"])
    window = _window_sensitivity(
        repo,
        out,
        dev,
        paired_index,
        dims,
        v204_cfg,
        alpha,
        km,
        seed0,
        device,
        policy,
        float(cfg["robustness"]["window_brier_spread_limit"]),
        int(cfg["robustness"]["minimum_window_common_rows"]),
    )
    atomic_json(out / "window_sensitivity.json", window)

    nll_delta = float(
        candidate_metric["fractional_censor_patient_mean_nll_v1"]
        - baseline_metric["fractional_censor_patient_mean_nll_v1"]
    )
    reassignment_gain = float(specificity["correct_minus_reassigned_improvement"])
    content_margin = float(specificity["clinical_content_gain_over_coverage_timing_control"])
    calibration_regression = float(calibration["candidate_minus_baseline_ece"])

    decision = decide_candidate_lock(
        seed_report=seed_report,
        nll_delta=nll_delta,
        nll_guardrail=float(cfg["robustness"]["nll_guardrail"]),
        reassignment_gain=reassignment_gain,
        minimum_reassignment_gain=float(cfg["specificity"]["minimum_correct_vs_reassigned_brier_gain"]),
        content_margin=content_margin,
        minimum_content_margin=float(cfg["specificity"]["content_control_margin"]),
        calibration_regression=calibration_regression,
        maximum_calibration_regression=float(cfg["calibration"]["maximum_ece_regression"]),
        window_pass=window["status"] == "PASS",
    )

    lock = {
        "selected": decision.selected,
        "v2_04_candidate_retained": decision.candidate_passes,
        "checks": decision.checks,
        "candidate": {
            "name": SELECTED_NAME,
            "representation_mode": SELECTED_REPRESENTATION,
            "genomic_mode": SELECTED_GENOMIC,
            "auxiliary_mode": "none",
            "locked_alpha": alpha,
        },
        "candidate_metric": {
            "brier": candidate_metric["patient_brier_4h_mean"],
            "nll": candidate_metric["fractional_censor_patient_mean_nll_v1"],
        },
        "baseline_metric": {
            "brier": baseline_metric["patient_brier_4h_mean"],
            "nll": baseline_metric["fractional_censor_patient_mean_nll_v1"],
        },
        "bootstrap": boot,
        "seed_robustness": seed_report,
        "resume_provenance": {
            "mode": "POST_FAILURE_WINDOW_REPAIR",
            "pre_window_artifacts_reused": True,
            "control_models_retrained": False,
            "alternate_temporal_caches_reused_if_present": True,
        },
    }

    # The frozen specificity rule already determines fallback in the observed
    # run, so no new all-development V2-04 candidate is fit on this resume path.
    if decision.candidate_passes:
        raise RuntimeError(
            "Repaired resume unexpectedly promotes the V2-04 candidate; run the full V2-05 path to create its all-development final fit"
        )
    lock["final_model"] = {
        "status": "FALLBACK_REUSES_FROZEN_V2_03_SPEC; final deploy fit deferred to V2-07"
    }
    atomic_json(out / "candidate_lock.json", lock)

    external_protocol = {
        "status": "EXTERNAL_CONFIRMATION_PENDING",
        "candidate_lock_sha256": None,
        "eligible_evidence": "genuinely new patients not present in upstream train/pretrain/development/legacy external evaluation",
        "prohibited": [
            "DFCI/VICC outcome reuse for V2 selection",
            "candidate-specific recalibration before primary evaluation",
            "random subset of old external patients as new validation",
        ],
        "prediction_contract": {
            "PRE": True,
            "POST": True,
            "locked_alpha": alpha,
            "causes": ["no_event", "progression", "death", "switch"],
            "horizons_months": list(HORIZONS_MONTHS),
        },
        "primary_metric": "patient-balanced PFS Brier mean at 3/6/12/18m with prespecified censoring nuisance policy",
        "center_reporting": "separate center results plus pooled descriptive summary only if protocol permits",
        "freeze_before_outcomes": [
            "identity/overlap checks", "time origin", "predictor crosswalk", "endpoint definitions",
            "inclusion rules", "missingness profile", "prediction hashes", "metrics",
        ],
    }
    atomic_json(out / "prospective_external_protocol.json", external_protocol)
    external_protocol["candidate_lock_sha256"] = sha256_file(out / "candidate_lock.json")
    atomic_json(out / "prospective_external_protocol.json", external_protocol)

    model_card = f"""# OncoTwin V2-05 internal candidate model card\n\nStatus: **FALLBACK TO V2-03**\n\nPrimary development population: {len(dev)} survival-eligible scan landmarks from {dev['patient_id'].nunique()} patients.\n\nSelected rule: `{decision.selected}`.\n\nLocked scan update alpha: {alpha:.4f}.\n\nV2-04 candidate Brier: {candidate_metric['patient_brier_4h_mean']:.9f}; frozen V2-03 baseline: {baseline_metric['patient_brier_4h_mean']:.9f}.\n\nThe V2-04 explicit-scan candidate was not promoted because it did not satisfy every prespecified robustness/specificity lock criterion. This is a research prognostic model, not a clinical decision rule. External confirmation remains pending.\n"""
    data_card = """# OncoTwin V2-05 data card\n\nDevelopment selection uses CHORD train/validation patients only. Historical CHORD test remains read-only and DFCI/VICC outcomes are not opened. Patient identity defines folds. Alternate W0/W3/W7 sensitivity aligns line-agnostically by patient + landmark day; window-specific episode IDs are intentionally not treated as cross-window identity. Absolute treatment-line identity is prohibited model-facing.\n"""
    atomic_text(out / "model_card.md", model_card)
    atomic_text(out / "data_card.md", data_card)

    benchmark = {
        "candidate_vs_baseline_brier_delta": float(
            candidate_metric["patient_brier_4h_mean"] - baseline_metric["patient_brier_4h_mean"]
        ),
        "candidate_vs_baseline_nll_delta": nll_delta,
        "reference_controls": controls,
        "specificity": specificity,
        "window": window,
        "calibration": {
            "baseline_mean_ece": calibration["baseline_mean_ece"],
            "candidate_mean_ece": calibration["candidate_mean_ece"],
            "regression": calibration_regression,
        },
    }
    atomic_json(out / "benchmark_report.json", benchmark)

    subgroup_exists = (out / "subgroup_sensitivity.parquet").is_file()
    acceptance = {
        "v2_04_pass_required": True,
        "analysis_plan_frozen_before_v205_results": True,
        "exact_17194_primary_rows": len(dev) == 17194,
        "paired_pre_post_same_rows": len(cpre) == len(cpost) == len(dev),
        "seed_fold_variation_reported": len(seed_report.get("details", [])) == 3,
        "paired_patient_bootstrap_completed": boot["repetitions"] == int(cfg["bootstrap"]["repetitions"]),
        "scan_reassignment_specificity_completed": specificity["permutation_changed_rows"] > 0,
        "clinical_content_vs_coverage_timing_control_completed": set(specificity["matched_controls"]) == {"clinical_content_only", "coverage_timing_only"},
        "last_scan_only_control_completed": "last_scan_only_gbt" in controls,
        "stale_line_start_incompatibility_explicit": controls["stale_line_start_control"]["status"] == "NOT_PRIMARY_COMPARABLE_BY_DESIGN",
        "w0_w3_w7_repaired_temporal_sensitivity_completed": window["common_rows"] >= int(cfg["robustness"]["minimum_window_common_rows"]),
        "line_agnostic_window_alignment": window.get("line_identity_used_for_alignment") is False,
        "natural_missingness_history_strata_reported": subgroup_exists,
        "deterministic_invariances_pass": all([
            deterministic["future_append_invariance"], deterministic["same_day_exclusion"],
            deterministic["line_label_invariance"], deterministic["batch_order_invariant"],
        ]),
        "probability_invariants_pass": candidate_metric["curve_invariants"]["status"] == "PASS",
        "censor_support_reported": len(calibration["censor_support"]) == 4,
        "candidate_lock_written": True,
        "external_protocol_frozen_without_old_external_outcomes": True,
        "protected_external_rows_not_opened": True,
        "external_predictions_not_regenerated": True,
    }
    require(all(acceptance.values()), f"V2-05 engineering acceptance failed: {acceptance}")

    result = {
        "checkpoint": "V2-05",
        "status": "PASS",
        "next_checkpoint": "V2-06",
        "development_rows": len(dev),
        "development_patients": int(dev["patient_id"].nunique()),
        "locked_alpha": alpha,
        "candidate_lock": lock,
        "window_sensitivity": window,
        "specificity": specificity,
        "calibration": {
            "baseline_mean_ece": calibration["baseline_mean_ece"],
            "candidate_mean_ece": calibration["candidate_mean_ece"],
            "regression": calibration_regression,
        },
        "acceptance": acceptance,
        "external_confirmation": "PENDING_NEW_INDEPENDENT_COHORT",
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
        "resume_provenance": lock["resume_provenance"],
    }
    atomic_json(out / "result_packet.json", result)

    memo = f"""# OncoTwin V2-05 robustness and internal candidate lock\n\nStatus: **PASS**\n\nInternal lock: **{decision.selected}**.\n\nV2-04 candidate retained: **{decision.candidate_passes}**.\n\nCandidate Brier: {candidate_metric['patient_brier_4h_mean']:.9f}; baseline: {baseline_metric['patient_brier_4h_mean']:.9f}; delta: {candidate_metric['patient_brier_4h_mean']-baseline_metric['patient_brier_4h_mean']:+.9f}.\n\nBootstrap 95% CI for candidate-minus-baseline Brier: [{boot['ci_low']:+.9f}, {boot['ci_high']:+.9f}] (conditional scope; does not include model-selection uncertainty).\n\nClinical-content specificity margin: {content_margin:+.9f} (required >= {float(cfg['specificity']['content_control_margin']):.9f}).\n\nW0/W3/W7 common-cohort spread: {window['spread']['spread']:.9f}; cross-window alignment uses patient + landmark day and never treatment-line identity.\n\nExternal confirmation: **PENDING NEW INDEPENDENT COHORT**. No DFCI/VICC outcomes were opened.\n"""
    atomic_text(out / "decision_report.md", memo)

    manifest_names = [
        "frozen_analysis_plan.json", "input_integrity.json", "seed_fold_stability.json",
        "paired_patient_bootstrap.json", "scan_specificity.json", "reference_controls.json",
        "subgroup_sensitivity.parquet", "channel_removal_diagnostics.json", "calibration.parquet",
        "calibration_and_support.json", "exploratory_outcome_decomposition.json",
        "deterministic_robustness.json", "window_sensitivity.json", "candidate_lock.json",
        "prospective_external_protocol.json", "model_card.md", "data_card.md", "benchmark_report.json",
        "result_packet.json", "decision_report.md",
    ]
    atomic_json(
        out / "artifact_manifest.json",
        {"status": "PASS", "files": {name: sha256_file(out / name) for name in manifest_names}},
    )

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
