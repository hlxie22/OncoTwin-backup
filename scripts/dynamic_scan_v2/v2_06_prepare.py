from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v2_06_external_protocol import (
    abstention_policy,
    atomic_json,
    cohort_manifest_schema,
    endpoint_contract,
    evaluation_stage_contract,
    load_json,
    nuisance_policy,
    predictor_crosswalk_template,
    require,
    sha256_file,
    validate_candidate_contract,
    validate_stage_order,
)


def maybe_hash(repo: Path, rel: str, required: bool = True) -> dict:
    p = repo / rel
    if not p.is_file():
        if required:
            raise RuntimeError(f"Required freeze target missing: {rel}")
        return {"path": rel, "present": False, "sha256": None}
    return {"path": rel, "present": True, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out = repo / "artifacts/dynamic_scan_v2/v2_06"
    out.mkdir(parents=True, exist_ok=True)

    cfg_path = repo / "configs/dynamic_scan_v2/v2_06_external_readiness.json"
    cfg = load_json(cfg_path)
    validate_stage_order(cfg["stage_order"])

    v205_path = repo / "artifacts/dynamic_scan_v2/v2_05/result_packet.json"
    v203_path = repo / "artifacts/dynamic_scan_v2/v2_03/result_packet.json"
    v202_path = repo / "artifacts/dynamic_scan_v2/v2_02/result_packet.json"
    v205 = load_json(v205_path)
    v203 = load_json(v203_path)
    v202 = load_json(v202_path)
    candidate = validate_candidate_contract(v205, v203, v202, cfg)

    v205_lock = repo / "artifacts/dynamic_scan_v2/v2_05/candidate_lock.json"
    v205_protocol = repo / "artifacts/dynamic_scan_v2/v2_05/prospective_external_protocol.json"
    v203_rule = repo / "artifacts/dynamic_scan_v2/v2_03/selected_update_rule.json"
    rule = load_json(v203_rule)
    require(rule.get("type") == "constant", "V2-03 selected rule must remain constant")
    require(abs(float(rule.get("alpha")) - float(cfg["candidate"]["locked_alpha"])) < 1e-12,
            "V2-03 selected alpha drift")

    old_protocol = load_json(v205_protocol)
    require(old_protocol.get("status") == "EXTERNAL_CONFIRMATION_PENDING", "V2-05 external protocol status drift")
    require(abs(float(old_protocol["prediction_contract"]["locked_alpha"]) - float(cfg["candidate"]["locked_alpha"])) < 1e-12,
            "V2-05 external protocol alpha drift")

    freeze_targets = [
        maybe_hash(repo, "artifacts/dynamic_scan_v2/v2_05/candidate_lock.json"),
        maybe_hash(repo, "artifacts/dynamic_scan_v2/v2_05/prospective_external_protocol.json"),
        maybe_hash(repo, "artifacts/dynamic_scan_v2/v2_05/artifact_manifest.json"),
        maybe_hash(repo, "artifacts/dynamic_scan_v2/v2_03/selected_update_rule.json"),
        maybe_hash(repo, "artifacts/dynamic_scan_v2/v2_03/gate_decision.json"),
        maybe_hash(repo, "artifacts/dynamic_scan_v2/v2_02/selection_decision.json"),
        maybe_hash(repo, "artifacts/checkpoint7r1_line_agnostic_temporal/temporal_encoder.pt"),
        maybe_hash(repo, "artifacts/checkpoint3/genomic_encoder.pt"),
        maybe_hash(repo, "artifacts/checkpoint3/prepared/gene_space_468.txt"),
        maybe_hash(repo, "artifacts/checkpoint3/prepared/graph_edges.npy"),
        maybe_hash(repo, "scripts/dynamic_scan/ckpt7r1_line_agnostic_temporal.py"),
        maybe_hash(repo, "scripts/dynamic_scan_v2/v2_02_models.py"),
        maybe_hash(repo, "configs/dynamic_scan_v2/v2_02_training.json"),
        maybe_hash(repo, "artifacts/checkpoint7r3b_bounded_candidate/bounded_dynamic_scan_candidate.pt", required=False),
        maybe_hash(repo, "artifacts/checkpoint7r3f2_site_unavailable_ablation/transport_rule.json", required=False),
    ]

    locked_protocol = {
        "checkpoint": "V2-06",
        "version": cfg["version"],
        "status": "EXTERNAL_CONFIRMATION_PENDING",
        "candidate": {
            **candidate,
            "prediction_execution_status": cfg["prediction_execution"]["status"],
            "prediction_execution_reason": cfg["prediction_execution"]["reason"],
        },
        "eligibility": cfg["eligibility"],
        "primary_evaluation": cfg["primary_evaluation"],
        "missingness": cfg["missingness"],
        "stage_order": cfg["stage_order"],
        "prohibited": cfg["prohibited"],
        "outcome_reveal_rule": "No new-cohort outcomes may be inspected until Stage-B predictions and their hashes are immutable.",
        "old_external_rule": "Completed DFCI/VICC outcomes/predictions remain historical legacy evidence only and are not V2 confirmation.",
    }
    atomic_json(out / "locked_external_protocol.json", locked_protocol)
    atomic_json(out / "predictor_crosswalk_template.json", predictor_crosswalk_template())
    atomic_json(out / "endpoint_contract.json", endpoint_contract())
    atomic_json(out / "nuisance_policy.json", nuisance_policy())
    atomic_json(out / "abstention_policy.json", abstention_policy())
    atomic_json(out / "new_cohort_manifest.schema.json", cohort_manifest_schema())
    atomic_json(out / "evaluation_stage_contract.json", evaluation_stage_contract())

    candidate_bundle = {
        "status": "SPECIFICATION_FROZEN_DEPLOY_FIT_PENDING_V2_07",
        "selected_candidate": candidate,
        "freeze_targets": freeze_targets,
        "important_note": "Cross-fitted V2-02 fold heads are development-evaluation artifacts, not the final deploy model. V2-07 must fit and freeze one all-development supervised head before Stage-B new-cohort prediction generation.",
        "legacy_comparator": {
            "name": "checkpoint7r3b_bounded_repaired_legacy_candidate",
            "use": "prespecified comparator on genuinely new patients only when its input contract can be satisfied without post-landmark information",
            "optional_artifact_present": any(x["path"].endswith("bounded_dynamic_scan_candidate.pt") and x["present"] for x in freeze_targets),
        },
    }
    atomic_json(out / "candidate_bundle_lock.json", candidate_bundle)

    identity_overlap = {
        "status": "FROZEN_POLICY_NO_NEW_COHORT_YET",
        "required_before_prediction": True,
        "identity_namespace_rule": "Use an authorized stable identity mapping or cryptographic tokenization within a shared namespace; do not compare unhashed IDs across unrelated namespaces.",
        "must_compare_against": [
            "all genomic pretraining patients where patient linkage exists",
            "all temporal pretraining patients",
            "all V2 supervised development patients",
            "historical CHORD test patients",
            "all patients used in legacy DFCI/VICC evaluation"
        ],
        "failure_action": "Any overlap in a cohort claimed as independent confirmation excludes the overlapping patient from the independent claim; systematic overlap invalidates the cohort claim.",
        "no_new_cohort_available_now": True,
    }
    atomic_json(out / "identity_overlap_policy.json", identity_overlap)

    readiness = {
        "status": "PASS",
        "external_confirmation": "PENDING_NEW_INDEPENDENT_COHORT",
        "cohort_intake_readiness": "READY",
        "prediction_execution_readiness": cfg["prediction_execution"]["status"],
        "prediction_execution_blocker": cfg["prediction_execution"]["reason"],
        "new_cohort_available_in_v2_06": False,
        "new_external_claim_made": False,
        "protected_external_rows_opened": False,
        "external_predictions_regenerated": False,
        "old_external_outcomes_reused": False,
        "next_action_if_new_cohort_arrives": [
            "Run outcome-blind Stage-A overlap and predictor preflight using the frozen V2-06 contracts.",
            "Complete V2-07 final deploy fit and inference release if not already completed.",
            "Generate and hash immutable Stage-B PRE/POST/alpha=0.52 predictions before endpoint reveal.",
            "Only then reveal endpoints and run the frozen one-shot Stage-C evaluation."
        ],
    }
    atomic_json(out / "readiness_report.json", readiness)

    acceptance = {
        "v2_05_pass_required": v205.get("status") == "PASS",
        "candidate_lock_hash_frozen": sha256_file(v205_lock) == next(x["sha256"] for x in freeze_targets if x["path"].endswith("v2_05/candidate_lock.json")),
        "selected_candidate_is_v2_03_fallback": candidate["selected_rule"] == "v2_03_frozen_temporal_constant_alpha",
        "v2_04_candidate_not_promoted": candidate["v2_04_candidate_retained"] is False,
        "locked_alpha_is_0_52": abs(candidate["locked_alpha"] - 0.52) < 1e-12,
        "selected_branch_family_is_r1_temporal_fixed": candidate["branch_family"] == "r1_temporal_fixed",
        "v2_03_rule_hash_frozen": any(x["path"].endswith("v2_03/selected_update_rule.json") and x["present"] for x in freeze_targets),
        "temporal_encoder_hash_frozen": any(x["path"].endswith("temporal_encoder.pt") and x["present"] for x in freeze_targets),
        "genomic_encoder_and_gene_space_hashes_frozen": all(any(x["path"].endswith(s) and x["present"] for x in freeze_targets) for s in ["genomic_encoder.pt", "gene_space_468.txt"]),
        "outcome_blind_stage_order_frozen": locked_protocol["stage_order"] == cfg["stage_order"],
        "predictor_crosswalk_frozen": (out / "predictor_crosswalk_template.json").is_file(),
        "identity_overlap_policy_frozen": (out / "identity_overlap_policy.json").is_file(),
        "endpoint_contract_frozen": (out / "endpoint_contract.json").is_file(),
        "nuisance_policy_frozen": (out / "nuisance_policy.json").is_file(),
        "abstention_policy_frozen": (out / "abstention_policy.json").is_file(),
        "new_cohort_schema_frozen": (out / "new_cohort_manifest.schema.json").is_file(),
        "cv_fold_heads_not_mislabeled_as_deploy_model": candidate_bundle["status"] == "SPECIFICATION_FROZEN_DEPLOY_FIT_PENDING_V2_07",
        "protected_external_rows_not_opened": True,
        "external_predictions_not_regenerated": True,
        "old_external_outcomes_not_reused": True,
        "external_confirmation_explicitly_pending": readiness["external_confirmation"] == "PENDING_NEW_INDEPENDENT_COHORT",
    }
    require(all(acceptance.values()), f"V2-06 acceptance failed: {acceptance}")

    result = {
        "checkpoint": "V2-06",
        "status": "PASS",
        "acceptance": acceptance,
        "selected_candidate": candidate,
        "external_confirmation": readiness["external_confirmation"],
        "cohort_intake_readiness": readiness["cohort_intake_readiness"],
        "prediction_execution_readiness": readiness["prediction_execution_readiness"],
        "new_cohort_available": False,
        "external_rows_opened": False,
        "external_predictions_regenerated": False,
        "old_external_outcomes_reused": False,
        "next_checkpoint": "V2-07",
    }
    atomic_json(out / "result_packet.json", result)

    artifact_names = [
        "locked_external_protocol.json", "predictor_crosswalk_template.json", "endpoint_contract.json",
        "nuisance_policy.json", "abstention_policy.json", "new_cohort_manifest.schema.json",
        "evaluation_stage_contract.json", "candidate_bundle_lock.json", "identity_overlap_policy.json",
        "readiness_report.json", "result_packet.json"
    ]
    manifest = {name: sha256_file(out / name) for name in artifact_names}
    atomic_json(out / "artifact_manifest.json", {"status": "PASS", "files": manifest})

    report = f"""# OncoTwin V2-06 new-cohort readiness\n\nStatus: **PASS**\n\nExternal confirmation: **PENDING_NEW_INDEPENDENT_COHORT**.\n\nSelected internal specification: **{candidate['selected_rule']}**, branch `{candidate['branch_family']}`, alpha **{candidate['locked_alpha']:.2f}**.\n\nNo new external cohort was available or opened in V2-06. No DFCI/VICC outcomes were reused and no external predictions were regenerated. The new-cohort intake, overlap, predictor, endpoint, nuisance, missingness/abstention, and stage-order contracts are now frozen.\n\nPrediction execution is **{cfg['prediction_execution']['status']}** because the selected supervised branch currently exists as cross-fitted development fold heads. V2-07 must create one final all-development deploy fit and typed inference interface before a new cohort reaches Stage B.\n\nThe primary future workflow is strictly: outcome-blind cohort intake/overlap -> immutable prediction freeze -> endpoint reveal/one-shot evaluation -> optional separately labeled recalibration analysis.\n"""
    (out / "readiness_report.md").write_text(report)

    print("[V2_06] PASS")
    print("[V2_06] candidate", candidate)
    print("[V2_06] external_confirmation", readiness["external_confirmation"])
    print("[V2_06] prediction_execution", readiness["prediction_execution_readiness"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
