"""Build leakage-free prompt/target records from the raw table for one split."""
from __future__ import annotations

import pandas as pd

from .prompts import build_prompt, count_tokens, fit_note_to_budget, split_numbered_note
from .schema import (ALL_LABELS, CRITERION_TYPES, USED_COLUMNS, LABELS_BY_TYPE, assert_no_gpt4_keys, make_target,
                     parse_evidence_ids)


def build_records(df: pd.DataFrame, tokenizer, max_len: int, reserve_tokens: int = 96) -> tuple[list[dict], list[dict]]:
    """df: rows of ONE split. Only USED_COLUMNS are read. Returns (records, dropped_rows).

    budget for the prompt = max_len - reserve_tokens (reserve covers the JSON target, EOS and the 'omitted' note).
    Rows with empty/missing criterion_text are dropped and reported (cannot be judged). Bad labels / out-of-range gold
    evidence ids raise (the data inspection step should have caught them).
    """
    df = df[list(USED_COLUMNS)].sort_values("annotation_id")
    note_cache: dict[str, list[dict]] = {}
    records, dropped = [], []
    for row in df.itertuples(index=False):
        ctext = row.criterion_text
        if ctext is None or (isinstance(ctext, float) and pd.isna(ctext)) or not str(ctext).strip():
            dropped.append({"annotation_id": int(row.annotation_id), "patient_id": row.patient_id, "reason": "empty_criterion_text"})
            continue
        ctype, label = row.criterion_type, row.expert_eligibility
        if ctype not in CRITERION_TYPES:
            raise ValueError(f"unknown criterion_type {ctype!r} (annotation {row.annotation_id})")
        if label not in LABELS_BY_TYPE[ctype]:
            raise ValueError(f"label {label!r} not valid for {ctype} (annotation {row.annotation_id})")
        if row.note not in note_cache:
            note_cache[row.note] = split_numbered_note(row.note)
        sentences = note_cache[row.note]
        gold = parse_evidence_ids(row.expert_sentences)
        valid_ids = {s["id"] for s in sentences}
        bad = [g for g in gold if g not in valid_ids]
        if bad:
            raise ValueError(f"gold evidence ids {bad} not in note (annotation {row.annotation_id}, {len(sentences)} sentences)")
        ctext = str(ctext).strip()
        full_prompt = build_prompt(tokenizer, sentences, ctype, ctext)
        n_full = count_tokens(tokenizer, full_prompt)
        kept, dropped_ids = fit_note_to_budget(
            sentences, lambda ss: count_tokens(tokenizer, build_prompt(tokenizer, ss, ctype, ctext)),
            max_len - reserve_tokens, protect_ids=gold)
        prompt = full_prompt if not dropped_ids else build_prompt(tokenizer, kept, ctype, ctext, omitted_ids=dropped_ids)
        target = make_target(label, gold)
        rec = {
            "annotation_id": int(row.annotation_id), "patient_id": str(row.patient_id), "trial_id": str(row.trial_id),
            "criterion_type": ctype, "criterion_text": ctext,
            "patient_sentences": sentences,  # ALL original sentences with original ids (even if omitted from prompt)
            "gold_label": label, "gold_evidence_ids": gold, "prompt": prompt, "target": target,
            "truncated_ids": dropped_ids, "n_prompt_tokens_full": n_full,
            "n_prompt_tokens": count_tokens(tokenizer, prompt), "n_target_tokens": count_tokens(tokenizer, target),
        }
        assert_no_gpt4_keys(rec)
        records.append(rec)
    return records, dropped
