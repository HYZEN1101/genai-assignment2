"""Patient-note sentence splitting, prompt construction, and length-budget truncation.

The raw `note` field numbers sentences inline: "0. A 58-year-old ... 1. The pain started ...". Gold evidence ids index
these numbers (0-based). We keep the ORIGINAL ids everywhere.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

from .schema import allowed_labels

SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility screening assistant for an educational prototype. "
    "You judge ONE criterion at a time using only the patient note. Absence of information is never proof."
)

LABEL_HELP = {
    "inclusion": (
        "included = the note shows the patient meets the criterion; not included = the note shows the patient does not meet it; "
        "not enough information = the note does not say enough to decide; not applicable = the criterion does not apply to this patient."
    ),
    "exclusion": (
        "excluded = the note shows the patient meets the exclusion criterion; not excluded = the note shows the patient does not meet it; "
        "not enough information = the note does not say enough to decide; not applicable = the criterion does not apply to this patient."
    ),
}


class PromptTooLongError(ValueError):
    pass


def split_numbered_note(note: str) -> list[dict]:
    """Split "0. text 1. text ..." into [{"id": 0, "text": ...}, ...]. Markers must appear in order 0,1,2,...

    Sequential search (look for marker i only after marker i-1) so numbers inside sentences ("58. ") are not mistaken
    for markers unless they equal the expected next id. Raises ValueError if the note does not start with "0.".
    """
    note = note.strip()
    pat = lambda i: re.compile(rf"(?:^|(?<=\s)){i}\.\s+")
    m0 = pat(0).match(note)
    if not m0:
        raise ValueError(f"note does not start with '0. ': {note[:60]!r}")
    marks = [(0, m0.start(), m0.end())]
    pos = m0.end()
    i = 1
    while True:
        m = pat(i).search(note, pos)
        if not m:
            break
        marks.append((i, m.start(), m.end()))
        pos = m.end()
        i += 1
    out = []
    for k, (sid, _, end) in enumerate(marks):
        stop = marks[k + 1][1] if k + 1 < len(marks) else len(note)
        out.append({"id": sid, "text": note[end:stop].strip()})
    return out


def format_patient(sentences: Sequence[dict]) -> str:
    return "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)


def build_user_message(sentences: Sequence[dict], criterion_type: str, criterion_text: str,
                       omitted_ids: Sequence[int] = ()) -> str:
    labels = " | ".join(allowed_labels(criterion_type))
    omitted = ""
    if omitted_ids:
        omitted = f"(Note: sentences {list(omitted_ids)} were omitted because the note was too long.)\n"
    return (
        "Patient note (each sentence has an ID in square brackets):\n"
        f"{format_patient(sentences)}\n{omitted}\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text.strip()}\n\n"
        f"Label meanings: {LABEL_HELP[criterion_type]}\n"
        f"Allowed labels: {labels}\n"
        'Answer with JSON only: {"label": <one allowed label>, "evidence_ids": [<IDs of the sentences that support the label>]}. '
        "Use an empty list if no sentence supports it."
    )


def build_prompt(tokenizer, sentences: Sequence[dict], criterion_type: str, criterion_text: str,
                 omitted_ids: Sequence[int] = ()) -> str:
    """Chat-template-rendered prompt ending with the assistant generation prompt. tokenize=False (transformers 5 API)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sentences, criterion_type, criterion_text, omitted_ids)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def fit_note_to_budget(sentences: Sequence[dict], count_prompt_tokens: Callable[[Sequence[dict]], int], budget: int,
                       protect_ids: Sequence[int] = ()) -> tuple[list[dict], list[int]]:
    """Truncation policy (never silent): if the prompt exceeds `budget` tokens, drop non-protected sentences from the END
    of the note first (binary search on how many to drop). Gold evidence ids (training only) are protected.
    Returns (kept_sentences, dropped_ids). Raises PromptTooLongError if even the protected sentences do not fit."""
    sentences = list(sentences)
    if count_prompt_tokens(sentences) <= budget:
        return sentences, []
    protect = set(protect_ids)
    order = [s["id"] for s in reversed(sentences) if s["id"] not in protect]

    def kept(d: int) -> list[dict]:
        drop = set(order[:d])
        return [s for s in sentences if s["id"] not in drop]

    if count_prompt_tokens(kept(len(order))) > budget:
        raise PromptTooLongError("prompt exceeds budget even after dropping every non-protected sentence")
    lo, hi = 0, len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_prompt_tokens(kept(mid)) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return kept(lo), sorted(order[:lo])
