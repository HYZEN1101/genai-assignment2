"""Explicit KV-cache greedy decoder (rubric D1).

Prefill: ONE forward over the prompt that fills the cache (K/V of every layer for positions 0..L-1).
Decode : each step feeds ONLY the newest token. We (not the library) supply
         position_ids   = [[cur_len]]           (RoPE absolute position of the new token),
         attention_mask = ones(1, cur_len + 1)  (cached keys + the new key, grown by one per step),
         cache slot     = cur_len               (new K/V are appended at this slot; asserted == cache length),
and the model's layers append the new K/V to the cache object we own and attend with the single query over all cached
keys (bottom-right aligned causality: a length-1 query sees every key). After every step the cache length must equal
prompt_len + tokens fed so far; `check_cache=True` asserts it (used in tests).

Step cost: projections for 1 token + attention of 1 query over T keys = O(T) per step, O(N*T) for N tokens
(vs a full recompute per step without cache).
"""
from __future__ import annotations

import torch

from .common import GenStats, cache_len, eos_set, forward_with_cache, greedy_next, model_device, new_cache, now_ms


@torch.no_grad()
def generate_greedy_kv(model, input_ids: torch.Tensor, max_new_tokens: int, eos_id=None, cache=None,
                       track: bool = False, collect_logits: bool = False, check_cache: bool = False
                       ) -> tuple[list[int], GenStats]:
    """input_ids: [1, L]. Returns (new_token_ids, stats). The cache (passed in or created) is left holding the prompt
    plus all generated tokens except the last one (the last token was chosen but never fed back).

    cache: optional DynamicCache already holding the first `c` prompt tokens (prefix reuse). Only input_ids[:, c:] is
    prefilled; c must be < L because the logits of the last prompt token are needed.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1
    dev = model_device(model)
    input_ids = input_ids.to(dev)
    L = input_ids.shape[1]
    stops = eos_set(eos_id)
    cache = new_cache(model) if cache is None else cache
    start = cache_len(cache)
    assert start < L, "cache already holds the whole prompt; leave at least one token to prefill"
    st = GenStats(path="kv_cache", prompt_len=L)
    gen: list[torch.Tensor] = []

    def record(logits):
        last = logits[:, -1, :]
        if track:
            top2 = last.float().topk(2, dim=-1).values[0]
            st.top2_gap.append(float(top2[0] - top2[1]))
        if collect_logits:
            st.logits.append(last[0].float().cpu())

    # ---- prefill: whole (remaining) prompt in one forward, positions start..L-1 ----------------------------------
    t0 = now_ms(dev)
    logits = forward_with_cache(model, input_ids[:, start:], cache, start=start)
    record(logits)
    nxt = greedy_next(logits)                                                          # [1]
    gen.append(nxt)
    cur = L                                                                            # positions held in cache == L
    if check_cache:
        assert cache_len(cache) == cur, (cache_len(cache), cur)
    t_first = now_ms(dev)

    # ---- decode: one token per forward ---------------------------------------------------------------------------
    if stops and int(nxt) in stops:
        st.stopped_on_eos = True
    else:
        for _ in range(max_new_tokens - 1):
            logits = forward_with_cache(model, nxt.unsqueeze(0), cache, start=cur)     # feed ONLY the newest token
            cur += 1
            if check_cache:
                assert cache_len(cache) == cur, (cache_len(cache), cur)
            record(logits)
            nxt = greedy_next(logits)
            gen.append(nxt)
            if stops and int(nxt) in stops:                                            # host sync only when EOS is used
                st.stopped_on_eos = True
                break
    t_end = now_ms(dev)

    toks = [int(t) for t in torch.cat(gen)]
    st.new_tokens = len(toks)
    st.prefill_ms = t_first - t0
    st.decode_ms = t_end - t_first
    st.cache_len_final = cache_len(cache)
    return toks, st
