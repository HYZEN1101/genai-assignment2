import pytest

from trialmatch.eval.quick_metrics import evidence_scores, example_evidence_prf, macro_f1, summarize


def row(g, p, ge, pe, valid=True, t="inclusion"):
    return {"gold_label": g, "pred_label": p, "valid": valid, "gold_evidence_ids": ge, "pred_evidence_ids": pe,
            "criterion_type": t, "unknown_ids": [], "hit_max": False}


def test_macro_f1_hand_case():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", None]            # None = invalid => wrong, no false positive
    f, per = macro_f1(gold, pred)
    assert per["a"][:3] == pytest.approx((1.0, 0.5, 2 / 3))
    assert per["b"][:3] == pytest.approx((0.5, 0.5, 0.5))
    assert f == pytest.approx((2 / 3 + 0.5) / 2)


def test_empty_evidence_policy():
    assert example_evidence_prf(set(), set()) == (1.0, 1.0, 1.0)
    assert example_evidence_prf(set(), {1}) == (0.0, 1.0, 0.0)
    assert example_evidence_prf({1}, set()) == (0.0, 0.0, 0.0)
    assert example_evidence_prf({1, 2}, {2, 3}) == pytest.approx((0.5, 0.5, 0.5))


def test_evidence_micro_macro():
    s = evidence_scores([{1, 2}, set()], [{2}, set()])
    assert s["micro_p"] == 1.0 and s["micro_r"] == 0.5
    assert s["macro_f1"] == pytest.approx((2 / 3 + 1.0) / 2)


def test_summarize_invalid_counts_wrong_and_nonempty_subset():
    rows = [row("included", "included", [0], [0]),
            row("not included", None, [1], [], valid=False),
            row("not excluded", "not excluded", [], [], t="exclusion")]
    s = summarize(rows)
    assert s["accuracy"] == pytest.approx(2 / 3) and s["valid_rate"] == pytest.approx(2 / 3)
    assert s["evidence_nonempty_gold"]["n"] == 2 and s["evidence_all"]["n"] == 3
    assert s["exclusion_accuracy"] == 1.0
