from oncotwin_api.extractor import rule_extract


def test_missing_is_not_negative():
    payload = rule_extract("Pathology report. Estrogen receptor: positive.", "pathology")
    by_type = {f.fact_type: f.value for f in payload.facts}
    assert by_type["er_status"] == "positive"
    assert "pr_status" not in by_type
    assert "her2_status" not in by_type


def test_scan_coverage_not_inferred_as_positive_site():
    payload = rule_extract("CT chest and abdomen. No statement of metastatic site. Impression: stable disease.", "radiology")
    # Rule extractor is intentionally conservative for site positivity in checkpoint 1.
    # Generic body-region words alone are not converted into a negative finding.
    assert not any(f.fact_type == "disease_site" and f.value == "abdomen" for f in payload.facts)
    assert any(f.fact_type == "scan_assessment" and f.value == "stable" for f in payload.facts)
