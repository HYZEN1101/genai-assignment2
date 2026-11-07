"""Evidence-sentence metrics (Phase 3, authoritative).

Policy frozen in Phase 1 (docs/DATA_NOTES.md), unit-tested in tests/test_evidence.py. Per example, sets of sentence IDs:
  both empty                 -> P = R = F1 = 1   (the model correctly cites nothing)
  gold empty, pred non-empty -> P = 0, R = 1 (nothing to recall), F1 = 0   (spurious evidence)
  gold non-empty, pred empty -> P = R = F1 = 0   (missed evidence)
  otherwise                  -> ordinary set P/R/F1
Aggregation: MICRO = pooled TP/FP/FN over examples; MACRO = mean of per-example P/R/F1.
Two scopes: (a) all examples, (b) examples with NON-EMPTY gold evidence (HEADLINE: ~70% of test records have empty gold,
so scope (a) is dominated by the "both empty" shortcut). Invalid outputs have an empty predicted set.
Hallucinated IDs (outside the note) stay in the predicted set, so they count as false positives.
"""
from __future__ import annotations

from typing import Sequence


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def example_prf(gold: set[int], pred: set[int]) -> tuple[float, float, float]:
    """Per-example P/R/F1 under the frozen empty-set policy (see module docstring)."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 1.0, 0.0
    if not pred:
        return 0.0, 0.0, 0.0
    return prf(len(gold & pred), len(pred - gold), len(gold - pred))


def scores(gold_sets: Sequence[set[int]], pred_sets: Sequence[set[int]]) -> dict:
    """Micro (pooled) and macro (per-example mean) P/R/F1 over the given examples."""
    n = len(gold_sets)
    keys = ("micro_p", "micro_r", "micro_f1", "macro_p", "macro_r", "macro_f1")
    if n == 0:
        return {"n": 0, **{k: 0.0 for k in keys}}
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    mp, mr, mf = prf(tp, fp, fn)
    per = [example_prf(g, p) for g, p in zip(gold_sets, pred_sets)]
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "micro_p": mp, "micro_r": mr, "micro_f1": mf,
            "macro_p": sum(x[0] for x in per) / n, "macro_r": sum(x[1] for x in per) / n,
            "macro_f1": sum(x[2] for x in per) / n}


def row_sets(rows: Sequence[dict]) -> tuple[list[set[int]], list[set[int]]]:
    gold = [set(r["gold_evidence_ids"]) for r in rows]
    pred = [set(r["pred_evidence_ids"]) if r["valid"] else set() for r in rows]
    return gold, pred


def _scope(rows: Sequence[dict]) -> dict:
    g, p = row_sets(rows)
    ne = [i for i, x in enumerate(g) if x]
    return {"all": scores(g, p), "nonempty_gold": scores([g[i] for i in ne], [p[i] for i in ne])}


def evidence_report(rows: Sequence[dict]) -> dict:
    """Overall + per criterion type, both scopes, plus empty-evidence behaviour diagnostics."""
    g, p = row_sets(rows)
    empty_g = [i for i, x in enumerate(g) if not x]
    n_ne = len(g) - len(empty_g)
    out = {"overall": _scope(rows)}
    for t in ("inclusion", "exclusion"):
        sub = [r for r in rows if r["criterion_type"] == t]
        if sub:
            out[t] = _scope(sub)
    out["diagnostics"] = {
        "n_gold_empty": len(empty_g),
        "gold_empty_pred_nonempty_rate": (sum(1 for i in empty_g if p[i]) / len(empty_g)) if empty_g else 0.0,
        "n_gold_nonempty": n_ne,
        "gold_nonempty_pred_empty_rate": (sum(1 for i, x in enumerate(g) if x and not p[i]) / n_ne) if n_ne else 0.0,
        "hallucinated_id_rate": (sum(1 for r in rows if r.get("unknown_ids")) / len(rows)) if rows else 0.0,
    }
    return out
