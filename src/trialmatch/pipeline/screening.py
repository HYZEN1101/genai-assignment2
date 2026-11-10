"""App-side screening logic that needs NO model: note numbering, criteria parsing, input validation, result assembly.

Everything here is pure Python so it can be unit-tested without weights (tests/test_app_logic.py).

Design rules (same as training/eval):
  * Sentence IDs are 0-based and are the ONLY thing the model cites; evidence text is copied from the note by ID.
  * Nothing is cut silently: over-long notes are reduced by prompts.fit_note_to_budget and the omitted IDs are reported.
  * Unevaluated / failed criteria are UNRESOLVED (valid=False), never "eligible": the Phase-3 aggregation rule then
    yields "Needs more information" (or "Likely ineligible" if a decided criterion already blocks).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

from trialmatch.data.prompts import split_numbered_note
from trialmatch.eval.explain import explain_row
from trialmatch.pipeline.aggregate import aggregate

MAX_NOTE_CHARS = 12_000          # hard input guard (CPU latency); the token budget below is the real limit
MAX_CRITERION_CHARS = 1_000
MAX_CRITERIA = 20
PROMPT_TOKEN_BUDGET = 1024       # same max_len as training (Phase 1); longer notes are reduced WITH a visible warning
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u25cf\u25aa]+|\(?\d{1,3}[.)])\s+")
_HEADER = re.compile(r"^\s*(inclusion|exclusion)\s+criteria\s*:?\s*$", re.I)


class InputError(ValueError):
    """A user-fixable problem; the message is shown in the UI as-is."""


# --------------------------------------------------------------------------------------------------------------------
# patient note -> numbered sentences
# --------------------------------------------------------------------------------------------------------------------
def note_to_sentences(note: str) -> tuple[list[dict], str]:
    """Return ([{id, text}], how) with 0-based ids. how: 'already numbered' | 'numbered from lines' | 'auto-split'.

    1) If the note already uses the dataset format "0. ... 1. ..." it is split exactly like the training data.
    2) Else, if most non-empty lines look numbered/bulleted, each line is one sentence (markers stripped, renumbered from 0).
    3) Else a simple rule-based sentence splitter (. ! ? followed by whitespace + capital/digit/quote). Abbreviations such as
       "Dr." can cause an extra split; the numbering is shown to the user so this is visible.
    """
    if not note or not note.strip():
        raise InputError("The patient note is empty. Paste a clinical note (one or more sentences).")
    note = note.replace("\r\n", "\n").strip()
    try:
        sents = split_numbered_note(" ".join(note.split()))
        if len(sents) >= 2 and all(s["text"] for s in sents):
            return sents, "already numbered (dataset format '0. ... 1. ...')"
    except ValueError:
        pass
    lines = [ln.strip() for ln in note.split("\n") if ln.strip()]
    if len(lines) >= 2 and sum(bool(_BULLET.match(ln)) for ln in lines) >= 0.6 * len(lines):
        texts = [_BULLET.sub("", ln).strip() for ln in lines]
        return [{"id": i, "text": t} for i, t in enumerate(t for t in texts if t)], "numbered from list lines"
    flat = " ".join(note.split())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", flat)
    parts = [p.strip() for p in parts if p.strip()]
    return [{"id": i, "text": t} for i, t in enumerate(parts)], "auto-split into sentences"


def format_numbered(sentences: Sequence[dict]) -> str:
    return "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)


# --------------------------------------------------------------------------------------------------------------------
# criteria parsing
# --------------------------------------------------------------------------------------------------------------------
def parse_criteria_block(text: str) -> list[str]:
    """One criterion per bullet/number/line. If any line is bulleted, un-bulleted lines are wrapped continuations."""
    lines = [ln.rstrip() for ln in (text or "").replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    if any(_BULLET.match(ln) for ln in lines):
        out: list[str] = []
        for ln in lines:
            if _BULLET.match(ln) or not out:
                out.append(_BULLET.sub("", ln).strip())
            else:
                out[-1] += " " + ln.strip()
        return [c for c in out if c]
    return [ln.strip() for ln in lines]


def split_trial_text(text: str) -> tuple[str, str]:
    """Split a pasted ClinicalTrials.gov-style block ('Inclusion Criteria: ... Exclusion Criteria: ...') into two blocks.
    Text before any header counts as inclusion. Returns (inclusion_block, exclusion_block)."""
    cur, buf = "inclusion", {"inclusion": [], "exclusion": []}
    for ln in (text or "").replace("\r\n", "\n").split("\n"):
        m = _HEADER.match(ln)
        if m:
            cur = m.group(1).lower()
            continue
        buf[cur].append(ln)
    return "\n".join(buf["inclusion"]).strip(), "\n".join(buf["exclusion"]).strip()


def build_criteria(inclusion_block: str, exclusion_block: str) -> list[dict]:
    crit = [{"type": "inclusion", "text": t} for t in parse_criteria_block(inclusion_block)]
    crit += [{"type": "exclusion", "text": t} for t in parse_criteria_block(exclusion_block)]
    return crit


# --------------------------------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------------------------------
def validate_inputs(note: str, criteria: Sequence[dict]) -> list[str]:
    """Raise InputError for blocking problems; return a list of non-blocking warnings."""
    warnings: list[str] = []
    if not note or not note.strip():
        raise InputError("The patient note is empty. Paste a clinical note or choose a sample case.")
    if len(note) > MAX_NOTE_CHARS:
        raise InputError(f"The patient note has {len(note):,} characters; the limit is {MAX_NOTE_CHARS:,}. "
                         "Shorten it (the model was trained on notes of roughly 150-600 tokens).")
    if not criteria:
        raise InputError("No criteria found. Enter at least one inclusion or exclusion criterion (one per line).")
    if len(criteria) > MAX_CRITERIA:
        raise InputError(f"{len(criteria)} criteria entered; the limit is {MAX_CRITERIA} per run (CPU latency). Split the trial.")
    for i, c in enumerate(criteria, 1):
        if c["type"] not in ("inclusion", "exclusion"):
            raise InputError(f"Criterion {i} has an invalid type {c['type']!r}.")
        if len(c["text"]) > MAX_CRITERION_CHARS:
            raise InputError(f"Criterion {i} is {len(c['text'])} characters long (limit {MAX_CRITERION_CHARS}). Split it into shorter criteria.")
    letters = [ch for ch in note if ch.isalpha()]
    if letters and sum(ord(ch) > 127 for ch in letters) / len(letters) > 0.3:
        warnings.append("The note looks mostly non-English. The model was fine-tuned on English notes; results are unreliable.")
    if len(note.split()) < 8:
        warnings.append("The note is very short; most criteria will probably be 'not enough information'.")
    return warnings


def input_hash(note: str, criteria: Sequence[dict]) -> str:
    payload = json.dumps({"note": note.strip(), "criteria": [[c["type"], c["text"]] for c in criteria]}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------------------------------
# result assembly
# --------------------------------------------------------------------------------------------------------------------
def unresolved_row(ctype: str, text: str, reason: str) -> dict:
    """Prediction row for a criterion that could not be evaluated (error, timeout, skipped): treated as unresolved."""
    return {"criterion_type": ctype, "criterion_text": text, "raw_output": "", "pred_label": None, "pred_evidence_ids": [],
            "valid": False, "error": reason, "unknown_ids": [], "n_prompt_tokens": 0, "n_new_tokens": 0,
            "hit_max": False, "omitted_ids": [], "stats": None}


def finalize(rows: Sequence[dict], sentences: Sequence[dict]) -> dict:
    """Attach evidence/explanation (templated, copied from the note by ID) and the rule-based overall outcome."""
    out = []
    for i, r in enumerate(rows, 1):
        ex = explain_row(r, sentences)
        out.append({**r, "index": i, **ex})
    agg = aggregate([{"criterion_type": r["criterion_type"], "pred_label": r["pred_label"], "valid": r["valid"]} for r in rows])
    return {"rows": out, "aggregate": agg}


def highlight_note_html(sentences: Sequence[dict], cited: Sequence[int], omitted: Sequence[int] = ()) -> str:
    from html import escape
    cited, omitted = set(cited), set(omitted)
    parts = []
    for s in sentences:
        style = ("background:#fde68a;color:#111;" if s["id"] in cited else
                 "opacity:.45;text-decoration:line-through;" if s["id"] in omitted else "")
        tag = " (omitted: note too long)" if s["id"] in omitted else ""
        parts.append(f'<div style="padding:2px 6px;margin:2px 0;border-radius:4px;{style}"><b>[{s["id"]}]</b> {escape(s["text"])}{tag}</div>')
    return "".join(parts)
