"""Explicit rule-based aggregation of criterion results into a PRELIMINARY patient-trial outcome.

A transparent rule, NOT a ground-truth label (the dataset only has criterion-level annotations) and NOT a clinical enrollment
decision. Precedence, first match wins:
  1. any inclusion 'not included' OR any exclusion 'excluded'                       -> "Likely ineligible"
  2. else any 'not enough information' or unparseable (invalid) result              -> "Needs more information"
  3. else at least one criterion decided (included / not excluded)                  -> "Potentially eligible (preliminary)"
  4. no criteria, or every criterion 'not applicable'                               -> "No applicable criteria evaluated"
Limits: one wrong criterion flips the outcome; the model's recall of 'not included'/'excluded' is poor (see error analysis), so
'Potentially eligible' can be wrong; absence of evidence is never treated as eligibility.
"""
from __future__ import annotations

from typing import Iterable

DISCLAIMER = "Educational screening prototype, not a clinical enrollment decision."
INELIGIBLE = "Likely ineligible"
MORE_INFO = "Needs more information"
ELIGIBLE = "Potentially eligible (preliminary)"
NONE_APPLICABLE = "No applicable criteria evaluated"
RULE_TEXT = (
    "Rule (first match wins): (1) any inclusion criterion 'not included' or exclusion criterion 'excluded' -> Likely ineligible; "
    "(2) otherwise any criterion 'not enough information' or unparseable -> Needs more information; "
    "(3) otherwise, if at least one criterion is decided -> Potentially eligible (preliminary); "
    "(4) otherwise -> No applicable criteria evaluated."
)


def _blocking(ctype: str, label: str | None) -> bool:
    return (ctype == "inclusion" and label == "not included") or (ctype == "exclusion" and label == "excluded")


def aggregate(results: Iterable[dict]) -> dict:
    """results: dicts with criterion_type, pred_label (None if invalid), valid. Returns outcome + audit trail (indices)."""
    res = list(results)
    blockers, unknowns, decided, n_na = [], [], 0, 0
    for i, r in enumerate(res):
        label = r["pred_label"] if r.get("valid", True) else None
        if label is None or label == "not enough information":
            unknowns.append(i)
        elif label == "not applicable":
            n_na += 1
        else:
            decided += 1
            if _blocking(r["criterion_type"], label):
                blockers.append(i)
    if blockers:
        outcome = INELIGIBLE
    elif unknowns:
        outcome = MORE_INFO
    elif decided > 0:
        outcome = ELIGIBLE
    else:
        outcome = NONE_APPLICABLE
    return {"outcome": outcome, "blocking_criteria": blockers, "undecided_criteria": unknowns,
            "counts": {"n": len(res), "decided": decided, "not_applicable": n_na,
                       "blocking": len(blockers), "undecided": len(unknowns)},
            "rule": RULE_TEXT, "disclaimer": DISCLAIMER}
