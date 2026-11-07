from trialmatch.eval.errors import is_label_error, pick_candidates, tag_row
from trialmatch.eval.explain import build_explanation, cited_evidence, explain_row, is_missing_info

SENTS = [{"id": 0, "text": "Age 58."}, {"id": 1, "text": "No diabetes."}, {"id": 2, "text": "Takes aspirin."}]


def row(g, p, ge=(), pe=(), valid=True, unk=(), aid=0, pid="p1"):
    return {"annotation_id": aid, "patient_id": pid, "gold_label": g, "pred_label": p if valid else None, "valid": valid,
            "gold_evidence_ids": list(ge), "pred_evidence_ids": list(pe), "unknown_ids": list(unk), "raw_output": "{}"}


def test_tags():
    assert tag_row(row("included", "included", [0], [0])) == []
    assert "polarity_flip" in tag_row(row("included", "not included", [0], [1]))
    t = tag_row(row("excluded", "not excluded", [1], [1]))
    assert {"polarity_flip", "contradictory_evidence", "rare_class_missed"} <= set(t)
    assert "false_nei" in tag_row(row("included", "not enough information", [0], []))
    assert "false_confident" in tag_row(row("not enough information", "included", [], [0]))
    assert {"na_mistake", "evidence_spurious"} <= set(tag_row(row("not applicable", "included", [], [0])))
    assert "invalid_output" in tag_row(row("included", None, [0], [], valid=False))
    assert "hallucinated_id" in tag_row(row("included", "included", [0], [0, 9], unk=[9]))
    assert "evidence_wrong_ids" in tag_row(row("included", "included", [0], [1]))
    assert "evidence_partial" in tag_row(row("included", "included", [0, 1], [0]))
    assert "evidence_missing" in tag_row(row("included", "included", [0], []))
    assert "truncation_related" in tag_row(row("included", "included", [2], [2]), {"truncated_ids": [2]})


def test_correct_label_with_bad_evidence_is_not_a_label_error():
    r = row("included", "included", [0], [1])
    assert not is_label_error(r) and "evidence_wrong_ids" in tag_row(r)


def test_pick_candidates_deterministic_and_unique():
    rows = [row("included", "not included", [0], [0], aid=i, pid=f"p{i % 3}") for i in range(12)]
    c1, c2 = pick_candidates(rows, per_tag=2), pick_candidates(rows, per_tag=2)
    assert c1 == c2 and len({c["annotation_id"] for c in c1}) == len(c1)


def test_explanations_use_only_cited_text():
    ev, bad = cited_evidence(SENTS, [1, 7])
    assert ev == [{"id": 1, "text": "No diabetes."}] and bad == [7]
    assert "[1]" in build_explanation("not excluded", True, ev, bad) and "7" in build_explanation("not excluded", True, ev, bad)
    nei = build_explanation("not enough information", True, [])
    assert "not treated as eligibility or ineligibility" in nei
    assert "absence of a mention is not proof" in build_explanation("not excluded", True, [])
    assert "could not be parsed" in build_explanation(None, False, [])
    assert is_missing_info("not enough information", True) and is_missing_info(None, False) and not is_missing_info("included", True)


def test_explain_row_invalid_has_no_evidence():
    x = explain_row(row("included", None, [0], [0], valid=False), SENTS)
    assert x["evidence"] == [] and x["missing_info"] is True
