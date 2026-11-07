"""Authoritative classification metrics for Phase 3 (own implementation, cross-checked against sklearn in tests).

Conventions (state them in the report):
- Invalid / unparseable output = WRONG prediction. It lowers recall of the gold class, adds no false positive to any class,
  and gets its own confusion-matrix column "<invalid>". Reported as `invalid_rate`.
- macro_f1 (HEADLINE; same convention as the P2 selection and sklearn's default): mean F1 over classes present in gold OR in
  the valid predictions. A class predicted but absent from gold therefore contributes F1 = 0.
- macro_f1_gold_support (secondary): mean F1 over classes with gold support > 0 only. The test split has NO gold "excluded",
  so the two can differ; both are reported.
- Everything is computed overall, inclusion-only and exclusion-only.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .evidence import evidence_report, prf

INVALID = "<invalid>"
LABEL_ORDER = ("included", "not included", "excluded", "not excluded", "not enough information", "not applicable")


def _order(classes) -> list[str]:
    known = [c for c in LABEL_ORDER if c in classes]
    return known + sorted(c for c in classes if c not in LABEL_ORDER)


def classification_report(gold: Sequence[str], pred: Sequence[str | None]) -> dict:
    """pred None = invalid output. Accuracy, macro-F1 (two conventions), per-class P/R/F1/support, confusion matrix."""
    n = len(gold)
    classes = _order(set(gold) | {p for p in pred if p is not None})
    per = {}
    for c in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        pr, rc, f1 = prf(tp, fp, fn)
        per[c] = {"precision": pr, "recall": rc, "f1": f1, "support": sum(1 for g in gold if g == c),
                  "n_pred": sum(1 for p in pred if p == c)}
    supported = [c for c in classes if per[c]["support"] > 0]
    cols = classes + [INVALID]
    idx = {c: i for i, c in enumerate(cols)}
    mat = [[0] * len(cols) for _ in classes]
    for g, p in zip(gold, pred):
        mat[classes.index(g)][idx[p if p is not None else INVALID]] += 1
    return {
        "n": n,
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0,
        "macro_f1": sum(per[c]["f1"] for c in classes) / len(classes) if classes else 0.0,
        "macro_f1_gold_support": sum(per[c]["f1"] for c in supported) / len(supported) if supported else 0.0,
        "n_classes_macro": len(classes), "n_classes_gold_support": len(supported),
        "invalid_rate": sum(1 for p in pred if p is None) / n if n else 0.0,
        "per_class": per,
        "confusion": {"rows_gold": classes, "cols_pred": cols, "matrix": mat},
        "gold_counts": dict(Counter(gold)),
        "pred_counts": dict(Counter(p if p is not None else INVALID for p in pred)),
    }


def effective_pred(row: dict) -> str | None:
    return row["pred_label"] if row["valid"] else None


def evaluate_rows(rows: Sequence[dict]) -> dict:
    """Full evaluation of prediction rows (Predictor.predict_records schema)."""
    out: dict = {"n": len(rows)}
    out["overall"] = classification_report([r["gold_label"] for r in rows], [effective_pred(r) for r in rows])
    for t in ("inclusion", "exclusion"):
        sub = [r for r in rows if r["criterion_type"] == t]
        if sub:
            out[t] = classification_report([r["gold_label"] for r in sub], [effective_pred(r) for r in sub])
    out["evidence"] = evidence_report(rows)
    out["generation"] = {
        "valid_rate": sum(1 for r in rows if r["valid"]) / len(rows) if rows else 0.0,
        "hit_max_rate": sum(1 for r in rows if r.get("hit_max")) / len(rows) if rows else 0.0,
    }
    return out
