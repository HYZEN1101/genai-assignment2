"""Traceable, TEMPLATED explanations (no generated text, no GPT-4 text).

Built ONLY from (label, cited sentence IDs, the note's own sentences). Evidence text is copied verbatim from the patient
note by ID, never generated. Missing information is stated as missing, never turned into eligibility or ineligibility.
"""
from __future__ import annotations

from typing import Sequence

_DECIDED = {
    "included": "indicate that the patient meets this inclusion criterion",
    "not included": "indicate that the patient does not meet this inclusion criterion",
    "excluded": "indicate that the patient meets this exclusion criterion",
    "not excluded": "indicate that the patient does not meet this exclusion criterion",
}


def cited_evidence(sentences: Sequence[dict], ids: Sequence[int]) -> tuple[list[dict], list[int]]:
    """([{id, text}] for ids present in the note, [ids not in the note = hallucinated])."""
    by_id = {s["id"]: s["text"] for s in sentences}
    ok = [{"id": i, "text": by_id[i]} for i in sorted(set(ids)) if i in by_id]
    bad = [i for i in sorted(set(ids)) if i not in by_id]
    return ok, bad


def is_missing_info(label: str | None, valid: bool) -> bool:
    """True when the result must be shown as 'information missing / undecided'."""
    return (not valid) or label == "not enough information"


def build_explanation(label: str | None, valid: bool, evidence: Sequence[dict], unknown_ids: Sequence[int] = ()) -> str:
    if not valid or label is None:
        return "The model output could not be parsed, so no decision was made for this criterion. Treat it as unresolved."
    ids = ", ".join(str(e["id"]) for e in evidence)
    note = f" (Ignored non-existent sentence id(s): {', '.join(map(str, unknown_ids))}.)" if unknown_ids else ""
    if label in _DECIDED:
        if evidence:
            return f"The cited sentence(s) [{ids}] {_DECIDED[label]}.{note}"
        return (f"The model labelled this criterion '{label}' but cited no supporting sentence. The absence of a mention is not "
                f"proof either way; verify against the full record.{note}")
    if label == "not enough information":
        base = ("The note does not contain enough information to decide this criterion; missing information is not treated "
                "as eligibility or ineligibility.")
        return base + (f" Related, non-conclusive sentence(s): [{ids}]." if evidence else " No relevant sentence was cited.") + note
    if label == "not applicable":
        return "The model judged this criterion not applicable to this patient." + (f" Cited sentence(s): [{ids}]." if evidence else "") + note
    return f"Unrecognised label '{label}'."


def explain_row(row: dict, sentences: Sequence[dict]) -> dict:
    """row = prediction row; sentences = record['patient_sentences']."""
    ev, bad = cited_evidence(sentences, row["pred_evidence_ids"] if row["valid"] else [])
    return {"evidence": ev, "unknown_ids": bad,
            "explanation": build_explanation(row["pred_label"], row["valid"], ev, bad),
            "missing_info": is_missing_info(row["pred_label"], row["valid"])}
