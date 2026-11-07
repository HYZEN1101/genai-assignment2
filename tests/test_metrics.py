"""Phase 3 metrics: hand-computed cases, sklearn cross-check, invalid handling, per-type reports, confusion matrix."""
import random

import pytest
from sklearn.metrics import accuracy_score, f1_score

from trialmatch.eval.metrics import INVALID, classification_report, evaluate_rows


def row(g, p, ge=(), pe=(), valid=True, t="inclusion", unk=()):
    return {"gold_label": g, "pred_label": p if valid else None, "valid": valid, "gold_evidence_ids": list(ge),
            "pred_evidence_ids": list(pe), "criterion_type": t, "unknown_ids": list(unk), "hit_max": False}


def test_hand_case():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", None]  # None = invalid -> wrong, no false positive
    r = classification_report(gold, pred)
    assert r["accuracy"] == 0.5
    assert (r["per_class"]["a"]["precision"], r["per_class"]["a"]["recall"]) == (1.0, 0.5)
    assert (r["per_class"]["b"]["precision"], r["per_class"]["b"]["recall"]) == (0.5, 0.5)
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert r["invalid_rate"] == 0.25
    assert r["confusion"]["cols_pred"] == ["a", "b", INVALID]
    assert r["confusion"]["matrix"] == [[1, 1, 0], [0, 1, 1]]


def test_matches_sklearn_on_random_data():
    rng = random.Random(0)
    labs = ["included", "not included", "not enough information", "not applicable"]
    gold = [rng.choice(labs) for _ in range(300)]
    pred = [rng.choice(labs) for _ in range(300)]
    r = classification_report(gold, pred)
    assert r["accuracy"] == pytest.approx(accuracy_score(gold, pred))
    assert r["macro_f1"] == pytest.approx(f1_score(gold, pred, average="macro", zero_division=0))  # labels = gold U pred


def test_pred_only_class_counts_in_macro_but_not_in_gold_support_variant():
    gold = ["a", "a", "b"]
    pred = ["a", "a", "c"]            # 'c' never in gold
    r = classification_report(gold, pred)
    assert r["n_classes_macro"] == 3 and r["n_classes_gold_support"] == 2
    assert r["macro_f1"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert r["macro_f1_gold_support"] == pytest.approx((1.0 + 0.0) / 2)


def test_invalid_is_wrong_and_adds_no_false_positive():
    r = classification_report(["a", "a"], ["a", None])
    assert r["per_class"]["a"]["precision"] == 1.0 and r["per_class"]["a"]["recall"] == 0.5


def test_evaluate_rows_splits_by_type_and_counts_invalid():
    rows = [row("included", "included"), row("not included", "included"),
            row("not excluded", "not excluded", t="exclusion"), row("excluded", None, valid=False, t="exclusion")]
    e = evaluate_rows(rows)
    assert e["overall"]["n"] == 4 and e["inclusion"]["n"] == 2 and e["exclusion"]["n"] == 2
    assert e["inclusion"]["accuracy"] == 0.5 and e["exclusion"]["accuracy"] == 0.5
    assert e["generation"]["valid_rate"] == 0.75
    assert e["overall"]["invalid_rate"] == 0.25


def test_empty_rows_do_not_crash():
    assert classification_report([], [])["accuracy"] == 0.0
