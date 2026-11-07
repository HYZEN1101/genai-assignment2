"""Rule-based error tagging + deterministic candidate selection for the error analysis (rubric C4).

Tags are DIAGNOSTIC labels computed from gold vs prediction; the causal hypothesis for each chosen error is written by the
author in docs/ERROR_ANALYSIS.md after reading the patient excerpt. One row can carry several tags.
"""
from __future__ import annotations

from typing import Sequence

from .evidence import example_prf

DECIDED = {"included", "not included", "excluded", "not excluded"}
POLARITY_PAIRS = ({"included", "not included"}, {"excluded", "not excluded"})
TAG_DESCRIPTIONS = {
    "invalid_output": "output not parseable / label outside the allowed set",
    "polarity_flip": "gold and predicted labels are the two opposite decisions (included<->not included, excluded<->not excluded)",
    "contradictory_evidence": "polarity flip while citing at least one gold evidence sentence (same sentence read the opposite way)",
    "false_nei": "gold is a decision but the model says 'not enough information' (over-cautious)",
    "false_confident": "gold is 'not enough information' but the model commits to a decision (unsupported / missing-information case)",
    "na_mistake": "gold or prediction is 'not applicable' and the label is wrong",
    "rare_class_missed": "gold is 'not included' or 'excluded' (rare classes) and the label is wrong",
    "evidence_missing": "gold evidence non-empty, model cites nothing",
    "evidence_spurious": "gold evidence empty, model cites sentences",
    "evidence_wrong_ids": "both non-empty but no overlap",
    "evidence_partial": "both non-empty, partial overlap",
    "hallucinated_id": "cites a sentence id that does not exist in the note",
    "truncation_related": "a gold evidence sentence was removed by prompt truncation",
}
LABEL_TAGS = ("invalid_output", "polarity_flip", "contradictory_evidence", "false_nei", "false_confident", "na_mistake", "rare_class_missed")


def tag_row(row: dict, record: dict | None = None) -> list[str]:
    """Return the diagnostic tags of one prediction row (empty list = fully correct label AND evidence)."""
    tags: list[str] = []
    g, p = row["gold_label"], row["pred_label"]
    gold_ev = set(row["gold_evidence_ids"])
    pred_ev = set(row["pred_evidence_ids"]) if row["valid"] else set()
    if not row["valid"]:
        tags.append("invalid_output")
    elif p != g:
        if {g, p} in POLARITY_PAIRS:
            tags.append("polarity_flip")
            if gold_ev & pred_ev:
                tags.append("contradictory_evidence")
        if g in DECIDED and p == "not enough information":
            tags.append("false_nei")
        if g == "not enough information" and p in DECIDED:
            tags.append("false_confident")
        if "not applicable" in (g, p):
            tags.append("na_mistake")
        if g in ("not included", "excluded"):
            tags.append("rare_class_missed")
    if row["valid"] or gold_ev:
        if gold_ev and not pred_ev:
            tags.append("evidence_missing")
        elif pred_ev and not gold_ev:
            tags.append("evidence_spurious")
        elif gold_ev and pred_ev and not (gold_ev & pred_ev):
            tags.append("evidence_wrong_ids")
        elif gold_ev and pred_ev and gold_ev != pred_ev:
            tags.append("evidence_partial")
    if row.get("unknown_ids"):
        tags.append("hallucinated_id")
    if record is not None and set(record.get("truncated_ids") or []) & gold_ev:
        tags.append("truncation_related")
    return tags


def is_label_error(row: dict) -> bool:
    return (not row["valid"]) or row["pred_label"] != row["gold_label"]


def tag_counts(rows: Sequence[dict], records_by_id: dict | None = None) -> dict:
    """{tag: count} over rows, plus counts restricted to label errors."""
    all_c: dict[str, int] = {}
    for r in rows:
        rec = records_by_id.get(r["annotation_id"]) if records_by_id else None
        for t in tag_row(r, rec):
            all_c[t] = all_c.get(t, 0) + 1
    return all_c


def pick_candidates(rows: Sequence[dict], records_by_id: dict | None = None, per_tag: int = 3) -> list[dict]:
    """Deterministic: for every tag (LABEL_TAGS first), the first `per_tag` rows by annotation_id not yet picked, preferring
    different patients. Each candidate = {annotation_id, patient_id, tags, primary_tag}. The author chooses >= 8 distinct ones."""
    ordered = sorted(rows, key=lambda r: r["annotation_id"])
    picked: dict[int, dict] = {}
    for tag in list(LABEL_TAGS) + [t for t in TAG_DESCRIPTIONS if t not in LABEL_TAGS]:
        seen_patients: set[str] = set()
        taken = 0
        # pass 1: new patients first; pass 2: fill up
        for relax in (False, True):
            for r in ordered:
                if taken >= per_tag:
                    break
                rec = records_by_id.get(r["annotation_id"]) if records_by_id else None
                tags = tag_row(r, rec)
                if tag not in tags or r["annotation_id"] in picked:
                    continue
                if not relax and r["patient_id"] in seen_patients:
                    continue
                picked[r["annotation_id"]] = {"annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
                                              "tags": tags, "primary_tag": tag}
                seen_patients.add(r["patient_id"])
                taken += 1
    return sorted(picked.values(), key=lambda c: c["annotation_id"])
