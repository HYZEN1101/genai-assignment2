from trialmatch.pipeline.aggregate import DISCLAIMER, ELIGIBLE, INELIGIBLE, MORE_INFO, NONE_APPLICABLE, aggregate


def r(t, label, valid=True):
    return {"criterion_type": t, "pred_label": label if valid else None, "valid": valid}


def test_ineligible_beats_everything():
    out = aggregate([r("inclusion", "not included"), r("inclusion", "not enough information"), r("exclusion", "not excluded")])
    assert out["outcome"] == INELIGIBLE and out["blocking_criteria"] == [0]
    assert aggregate([r("exclusion", "excluded"), r("inclusion", "included")])["outcome"] == INELIGIBLE


def test_label_polarity_depends_on_type():
    # 'excluded' on an INCLUSION row is not a valid label; polarity rule only fires for the right type
    assert aggregate([r("inclusion", "excluded")])["outcome"] == ELIGIBLE
    assert aggregate([r("exclusion", "not included")])["outcome"] == ELIGIBLE


def test_needs_more_info_on_nei_or_invalid():
    assert aggregate([r("inclusion", "included"), r("exclusion", "not enough information")])["outcome"] == MORE_INFO
    assert aggregate([r("inclusion", "included"), r("exclusion", None, valid=False)])["outcome"] == MORE_INFO


def test_potentially_eligible_requires_a_decided_criterion():
    assert aggregate([r("inclusion", "included"), r("exclusion", "not excluded"), r("exclusion", "not applicable")])["outcome"] == ELIGIBLE
    assert aggregate([r("inclusion", "not applicable")])["outcome"] == NONE_APPLICABLE
    assert aggregate([])["outcome"] == NONE_APPLICABLE


def test_audit_trail_and_disclaimer():
    out = aggregate([r("inclusion", "included"), r("exclusion", "not enough information"), r("inclusion", "not applicable")])
    assert out["undecided_criteria"] == [1] and out["counts"] == {"n": 3, "decided": 1, "not_applicable": 1, "blocking": 0, "undecided": 1}
    assert out["disclaimer"] == DISCLAIMER and "not a clinical enrollment decision" in DISCLAIMER
