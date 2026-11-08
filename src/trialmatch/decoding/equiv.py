"""Cached-vs-uncached equivalence checks and length-controlled prompts (rubric D2).

compare_paths() runs BOTH explicit decoders on the same prompt (greedy, fixed number of new tokens) and returns
token equality plus, if they diverge, the first divergent step, the top-2 logit gap of each path at that step and
the largest logit difference before it. A divergence with a tiny gap relative to fp16 resolution is a near-tie
flipped by rounding (different matmul shapes for 1 token vs the full sequence), not a logic bug; a re-run in fp32
removing it is the confirming evidence.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .greedy_kv import generate_greedy_kv
from .greedy_nocache import generate_greedy_nocache


def first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """Index of the first differing token, or None if equal (length mismatch counts as divergence at min length)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def compare_paths(model, input_ids: torch.Tensor, n_new: int, eos_id=None) -> dict:
    ref, st_ref = generate_greedy_nocache(model, input_ids, n_new, eos_id=eos_id, track=True, collect_logits=True)
    kv, st_kv = generate_greedy_kv(model, input_ids, n_new, eos_id=eos_id, track=True, collect_logits=True,
                                   check_cache=True)
    div = first_divergence(ref, kv)
    # logit agreement on the steps whose inputs were identical (everything before the first divergence)
    upto = len(st_ref.logits) if div is None else div + 1
    upto = min(upto, len(st_kv.logits))
    diffs = [float((a - b).abs().max()) for a, b in zip(st_ref.logits[:upto], st_kv.logits[:upto])]
    out = {
        "prompt_len": int(input_ids.shape[1]), "new_tokens": n_new, "equal": div is None, "first_div": div,
        "n_ref": len(ref), "n_kv": len(kv),
        "max_logit_absdiff": max(diffs) if diffs else 0.0,
        "logit_absdiff_step0": diffs[0] if diffs else 0.0,
        "gap_ref_at_div": st_ref.top2_gap[div] if div is not None and div < len(st_ref.top2_gap) else None,
        "gap_kv_at_div": st_kv.top2_gap[div] if div is not None and div < len(st_kv.top2_gap) else None,
        "min_gap_ref": min(st_ref.top2_gap) if st_ref.top2_gap else None,
        "tokens_ref": ref, "tokens_kv": kv,
    }
    return out


def length_prompt(tok, texts: Sequence[str], length: int, header: str = "<|im_start|>assistant\n") -> torch.Tensor:
    """Length-controlled REAL-token prompt of exactly `length` tokens: val prompts are concatenated and truncated, then the
    assistant header is appended so the model starts answering. NOT a valid single task instance: it only fixes L."""
    head = tok(header, add_special_tokens=False).input_ids
    assert length > len(head)
    ids: list[int] = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False).input_ids)
        if len(ids) >= length - len(head):
            break
    assert len(ids) >= length - len(head), f"need more source text to reach {length} tokens (have {len(ids)})"
    return torch.tensor([ids[: length - len(head)] + head], dtype=torch.long)
