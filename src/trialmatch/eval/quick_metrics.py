"""QUICK validation metrics for choosing a checkpoint in Phase 2.

This is a selection tool only. Phase 3 builds the authoritative metrics suite (metrics.py / evidence.py) and the
final report numbers come from there. Conventions here follow the policies frozen in Phase 1:
- an invalid / unparseable output is a WRONG prediction (counts against recall of the gold class, adds no false positive
  to any class) and has an empty evidence set;
- macro-F1 is averaged over classes present in gold OR in valid predictions (stated explicitly because small splits
  may miss classes);
- evidence per-example scores: both empty -> P=R=F1=1; gold empty & pred non-empty -> P=0 (R=1, F1=0);
  gold non-empty & pred empty -> P=R=F1=0. Reported on (a) all examples and (b) non-empty-gold examples (HEADLINE).
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def macro_f1(gold: Sequence[str], pred: Sequence[str | None]) -> tuple[float, dict]:
    """pred None = invalid output. Returns (macro_f1, per_class {label: (P, R, F1, support)})."""
    classes = sorted(set(gold) | {p for p in pred if p is not None})
    per = {}
    for c in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        per[c] = (*_prf(tp, fp, fn), sum(1 for g in gold if g == c))
    return (sum(v[2] for v in per.values()) / len(per) if per else 0.0), per


def example_evidence_prf(gold: set[int], pred: set[int]) -> tuple[float, float, float]:
    """Per-example P/R/F1 under the frozen empty-evidence policy."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 1.0, 0.0
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(gold & pred)
    return _prf(tp, len(pred - gold), len(gold - pred))


def evidence_scores(gold_sets: Sequence[set[int]], pred_sets: Sequence[set[int]]) -> dict:
    """micro (pooled counts) and macro (mean of per-example) P/R/F1 over the given examples."""
    if not gold_sets:
        return {"n": 0, "micro_p": 0.0, "micro_r": 0.0, "micro_f1": 0.0, "macro_p": 0.0, "macro_r": 0.0, "macro_f1": 0.0}
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    mp, mr, mf = _prf(tp, fp, fn)
    per = [example_evidence_prf(g, p) for g, p in zip(gold_sets, pred_sets)]
    n = len(per)
    return {"n": n, "micro_p": mp, "micro_r": mr, "micro_f1": mf,
            "macro_p": sum(x[0] for x in per) / n, "macro_r": sum(x[1] for x in per) / n,
            "macro_f1": sum(x[2] for x in per) / n}


def summarize(rows: Sequence[dict]) -> dict:
    """rows = prediction rows from Predictor.predict_records (gold_label, pred_label, valid, evidence lists...)."""
    n = len(rows)
    gold = [r["gold_label"] for r in rows]
    pred = [r["pred_label"] if r["valid"] else None for r in rows]
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    mf1, per = macro_f1(gold, pred)
    out = {
        "n": n, "valid_rate": sum(1 for r in rows if r["valid"]) / n if n else 0.0,
        "hit_max_rate": sum(1 for r in rows if r.get("hit_max")) / n if n else 0.0,
        "hallucinated_id_rate": sum(1 for r in rows if r.get("unknown_ids")) / n if n else 0.0,
        "accuracy": acc, "macro_f1": mf1,
        "per_class": {c: {"precision": v[0], "recall": v[1], "f1": v[2], "support": v[3]} for c, v in per.items()},
        "pred_label_counts": dict(Counter(p if p is not None else "<invalid>" for p in pred)),
    }
    for t in ("inclusion", "exclusion"):
        sub = [r for r in rows if r["criterion_type"] == t]
        if sub:
            g = [r["gold_label"] for r in sub]
            p = [r["pred_label"] if r["valid"] else None for r in sub]
            out[f"{t}_n"] = len(sub)
            out[f"{t}_accuracy"] = sum(a == b for a, b in zip(g, p)) / len(sub)
            out[f"{t}_macro_f1"] = macro_f1(g, p)[0]
    gs = [set(r["gold_evidence_ids"]) for r in rows]
    ps = [set(r["pred_evidence_ids"]) if r["valid"] else set() for r in rows]
    out["evidence_all"] = evidence_scores(gs, ps)
    ne = [i for i, g in enumerate(gs) if g]
    out["evidence_nonempty_gold"] = evidence_scores([gs[i] for i in ne], [ps[i] for i in ne])  # HEADLINE
    return out
