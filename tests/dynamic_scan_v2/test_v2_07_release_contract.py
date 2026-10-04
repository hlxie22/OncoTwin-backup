import json
from pathlib import Path


def test_release_config_is_frozen():
    repo = Path(__file__).resolve().parents[2]
    cfg = json.loads((repo / "configs/dynamic_scan_v2/v2_07_release.json").read_text())
    assert cfg["selected_rule"] == "v2_03_frozen_temporal_constant_alpha"
    assert cfg["branch_family"] == "r1_temporal_fixed"
    assert cfg["locked_alpha"] == 0.52
    assert cfg["deploy_seed"] == 20261001
    assert cfg["torch_epochs"] == 10
    assert cfg["portable_context_names"] == ["scan_number_patient_log","genomic_available","genomic_age_scaled"]
    assert "line" not in " ".join(cfg["portable_context_names"]).lower()


def test_final_fit_is_not_claimed_as_validation():
    repo = Path(__file__).resolve().parents[2]
    cfg = json.loads((repo / "configs/dynamic_scan_v2/v2_07_release.json").read_text())
    text = cfg["performance_claim_policy"].lower()
    assert "not" in text and "validation evidence" in text
    assert "train+val" in cfg["training_population"]


def test_public_inference_boundary_is_prepared_not_raw_claim():
    repo = Path(__file__).resolve().parents[2]
    cfg = json.loads((repo / "configs/dynamic_scan_v2/v2_07_release.json").read_text())
    assert "prepared" in cfg["inference_boundary"].lower()
    assert "v2-06" in cfg["inference_boundary"].lower()
