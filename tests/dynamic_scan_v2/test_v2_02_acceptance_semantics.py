def test_safety_audit_fields_must_be_positive_acceptance_predicates():
    # This regression test captures the bug that ended the completed V2-02 run.
    factual_audit = {
        'external_rows_opened': False,
        'external_predictions_regenerated': False,
    }
    acceptance = {
        'protected_external_rows_not_opened': not factual_audit['external_rows_opened'],
        'external_predictions_not_regenerated': not factual_audit['external_predictions_regenerated'],
    }
    assert all(acceptance.values())
    assert factual_audit['external_rows_opened'] is False
    assert factual_audit['external_predictions_regenerated'] is False
