import pytest

from trialmatch.eval.evidence import evidence_report, example_prf, scores


def row(ge, pe, valid=True, t="inclusion", unk=()):
    return {"gold_evidence_ids": ge, "pred_evidence_ids": pe, "valid": valid, "criterion_type": t, "unknown_ids": list(unk)}


def test_empty_set_policy():
    assert example_prf(set(), set()) == (1.0, 1.0, 1.0)
    assert example_prf(set(), {1}) == (0.0, 1.0, 0.0)
    assert example_prf({1}, set()) == (0.0, 0.0, 0.0)
    assert example_prf({1, 2}, {2, 3}) == pytest.approx((0.5, 0.5, 0.5))
    assert example_prf({1}, {1}) == (1.0, 1.0, 1.0)


def test_micro_vs_macro_hand_case():
    s = scores([{1, 2}, {3}], [{2}, {3, 4}])
    # pooled: tp=2 fp=1 fn=1 -> P=2/3 R=2/3 ; per-example F1: ex1 P=1 R=.5 F1=2/3 ; ex2 P=.5 R=1 F1=2/3
    assert s["micro_p"] == pytest.approx(2 / 3) and s["micro_r"] == pytest.approx(2 / 3)
    assert s["macro_f1"] == pytest.approx(2 / 3)


def test_report_scopes_and_diagnostics():
    rows = [row([1], [1]), row([], []), row([], [2]), row([3], [], valid=False), row([4], [9], unk=[9], t="exclusion")]
    r = evidence_report(rows)
    assert r["overall"]["all"]["n"] == 5 and r["overall"]["nonempty_gold"]["n"] == 3
    d = r["diagnostics"]
    assert d["n_gold_empty"] == 2 and d["gold_empty_pred_nonempty_rate"] == 0.5
    assert d["gold_nonempty_pred_empty_rate"] == pytest.approx(1 / 3)  # the invalid row has an empty predicted set
    assert d["hallucinated_id_rate"] == pytest.approx(1 / 5)
    assert r["exclusion"]["all"]["n"] == 1 and r["inclusion"]["all"]["n"] == 4


def test_invalid_row_counts_as_empty_prediction():
    r = evidence_report([row([1, 2], [1, 2], valid=False)])
    assert r["overall"]["nonempty_gold"]["micro_r"] == 0.0


def test_no_examples():
    assert scores([], [])["n"] == 0
