"""Reference decoder WITHOUT cache reuse (rubric D1/D2 baseline).

Every step re-runs the model over the ENTIRE sequence (prompt + tokens generated so far) with use_cache=False,
takes the argmax of the last position and appends it. The whole prefix is recomputed each step, so total matmul work
for N new tokens is O(N*T) (projections/MLP) and O(N*T^2) (attention scores), versus O(N*T) / O(N*T) with a cache.
"""
from __future__ import annotations

import torch

from .common import GenStats, eos_set, model_device, now_ms


@torch.no_grad()
def generate_greedy_nocache(model, input_ids: torch.Tensor, max_new_tokens: int, eos_id=None,
                            track: bool = False, collect_logits: bool = False) -> tuple[list[int], GenStats]:
    """input_ids: [1, L]. Returns (new_token_ids, stats). With eos_id the EOS token is emitted, then generation stops.

    logits_to_keep=1: only the last position's logits are computed (the other L-1 rows are never used by greedy
    decoding; both decoders use it, so the comparison is fair and no [L, vocab] logits tensor is materialised).
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1
    dev = model_device(model)
    input_ids = input_ids.to(dev)
    stops = eos_set(eos_id)
    st = GenStats(path="nocache", prompt_len=input_ids.shape[1])
    seq = input_ids
    gen: list[torch.Tensor] = []

    t0 = now_ms(dev)
    t_first = None
    for step in range(max_new_tokens):
        logits = model(input_ids=seq, use_cache=False, logits_to_keep=1).logits      # full recompute, no cache
        last = logits[:, -1, :]
        nxt = last.argmax(dim=-1)                                                      # [1]
        if track:
            top2 = last.float().topk(2, dim=-1).values[0]
            st.top2_gap.append(float(top2[0] - top2[1]))
        if collect_logits:
            st.logits.append(last[0].float().cpu())
        gen.append(nxt)
        seq = torch.cat([seq, nxt.unsqueeze(0)], dim=1)
        if step == 0:
            t_first = now_ms(dev)                                                      # prefill = first full forward
        if stops and int(nxt) in stops:                                                # host sync only when EOS is used
            st.stopped_on_eos = True
            break
    t_end = now_ms(dev)

    toks = [int(t) for t in torch.cat(gen)] if gen else []
    st.new_tokens = len(toks)
    st.prefill_ms = (t_first - t0) if t_first is not None else 0.0
    st.decode_ms = (t_end - t_first) if t_first is not None else 0.0
    return toks, st
