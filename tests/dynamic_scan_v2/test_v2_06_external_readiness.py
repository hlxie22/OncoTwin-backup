import pytest

from v2_06_external_protocol import (
    abstention_policy,
    canonical_patient_token,
    endpoint_contract,
    evaluation_stage_contract,
    nuisance_policy,
    overlap_report,
    predictor_crosswalk_template,
    validate_candidate_contract,
    validate_new_cohort_label,
    validate_outcome_blind_manifest,
    validate_predictor_inventory,
    validate_stage_order,
)


def test_stage_order_is_strict_prediction_before_outcomes():
    stages = [x["stage"] for x in evaluation_stage_contract()["stages"]]
    validate_stage_order(stages)
    assert stages.index("B_OUTCOME_BLIND_PREDICTION_FREEZE") < stages.index("C_ENDPOINT_REVEAL_AND_ONE_SHOT_PRIMARY_EVALUATION")


def test_outcome_like_predictor_fields_rejected():
    with pytest.raises(RuntimeError):
        validate_predictor_inventory(["patient_id", "landmark_day", "progression_day"])
    assert validate_predictor_inventory(["landmark_day", "coverage_chest"])["status"] == "PASS"


def test_old_external_cannot_be_relabelled_new():
    for label in ["DFCI", "VICC", "dfci-vicc"]:
        with pytest.raises(RuntimeError):
            validate_new_cohort_label(label)
    validate_new_cohort_label("prospective_center_c_2027")


def test_overlap_tokens_require_shared_namespace_and_detect_overlap():
    a = canonical_patient_token("health-system-x", "P001")
    assert a == canonical_patient_token("health-system-x", "P001")
    assert a != canonical_patient_token("health-system-y", "P001")
    assert overlap_report({a}, {a})["status"] == "FAIL"
    assert overlap_report({a}, set())["status"] == "PASS"


def test_candidate_contract_requires_frozen_fallback():
    cfg = {"candidate": {"selected_rule": "v2_03_frozen_temporal_constant_alpha", "branch_family": "r1_temporal_fixed", "locked_alpha": 0.52}}
    v205 = {"status": "PASS", "locked_alpha": 0.52, "candidate_lock": {"selected": "v2_03_frozen_temporal_constant_alpha", "v2_04_candidate_retained": False}}
    v203 = {"status": "PASS", "branch_family": "r1_temporal_fixed"}
    v202 = {"status": "PASS", "selection": {"best_screening_family": "r1_temporal_fixed"}}
    out = validate_candidate_contract(v205, v203, v202, cfg)
    assert out["locked_alpha"] == 0.52
    v205["candidate_lock"]["v2_04_candidate_retained"] = True
    with pytest.raises(RuntimeError):
        validate_candidate_contract(v205, v203, v202, cfg)


def test_missing_genomics_is_explicit_not_negative_disease_inference():
    p = abstention_policy()
    assert p["genomics_missing_policy"]["genomic_available"] == 0
    assert p["genomics_missing_policy"]["embedding_contribution"] == "zero"
    cross = predictor_crosswalk_template()
    assert "outcomes" in cross["prohibited_model_facing"]


def test_endpoint_contract_preserves_switch_competing_event():
    e = endpoint_contract()
    assert e["causes"]["3"] == "treatment switch as distinct competing event"
    assert "do not silently convert" in e["switch_policy"]


def test_primary_recalibration_is_forbidden():
    n = nuisance_policy()
    assert n["recalibration"]["primary_evaluation"] == "none"
    assert n["recalibration"]["may_replace_primary"] is False


def test_manifest_requires_outcomes_unavailable():
    manifest = {
        "cohort_id": "new1",
        "cohort_label": "prospective_2027",
        "centers": ["C"],
        "identity_namespace": "ns",
        "patient_identity_file": "ids.csv",
        "predictor_inventory_file": "predictors.csv",
        "outcomes_unavailable_to_prediction_team": True,
    }
    assert validate_outcome_blind_manifest(manifest, ["landmark_day", "coverage_chest"])["status"] == "PASS"
    manifest["outcomes_unavailable_to_prediction_team"] = False
    with pytest.raises(RuntimeError):
        validate_outcome_blind_manifest(manifest, ["landmark_day"])
