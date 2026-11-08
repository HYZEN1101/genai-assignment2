"""Phase 5 tests (D1/D2): explicit KV-cache decoder vs uncached reference. CPU, tiny random Qwen2, fp32, no network.

Real-model evidence (0.5B / fine-tuned 1.5B, >=3 length combos, fp16 + fp32 recheck) comes from
`scripts/check_kv_equiv.py`; these tests prove the mechanism and guard the invariants cheaply.
"""
import pytest
import torch

from tiny_lm import make_model, make_records, make_tok
from trialmatch.attention import causal_mask, reference_attention
from trialmatch.decoding import (cache_len, cache_nbytes, crop_cache, forward_with_cache, generate_greedy_kv,
                                 generate_greedy_nocache, kv_bytes_per_token, new_cache)
from trialmatch.decoding.common import cache_shapes
from trialmatch.decoding.cached_attention_demo import CachedGQAAttention, apply_rope
from trialmatch.decoding.equiv import compare_paths, first_divergence

VOCAB = 60


@pytest.fixture(scope="module")
def model():
    m = make_model(VOCAB)
    m.eval()
    return m


def prompt(L: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, VOCAB, (1, L), generator=g)


# ---------------------------------------------------------------- D2: greedy equivalence on several length combos
@pytest.mark.parametrize("L,N", [(8, 4), (16, 16), (64, 24), (128, 32), (200, 40)])
def test_cached_equals_uncached_greedy(model, L, N):
    ref, _ = generate_greedy_nocache(model, prompt(L), N)
    kv, st = generate_greedy_kv(model, prompt(L), N, check_cache=True)
    assert kv == ref and len(kv) == N
    assert st.cache_len_final == L + N - 1            # last chosen token is never fed back


@pytest.mark.parametrize("seed", range(5))
def test_equivalence_many_prompts(model, seed):
    p = prompt(40 + seed * 7, seed=seed)
    assert generate_greedy_kv(model, p, 20)[0] == generate_greedy_nocache(model, p, 20)[0]


def test_compare_paths_reports_equal_and_tiny_logit_diff(model):
    r = compare_paths(model, prompt(48), 16)
    assert r["equal"] and r["first_div"] is None
    assert r["logit_absdiff_step0"] < 1e-5            # prefill is (almost) the SAME computation in both paths
    assert r["max_logit_absdiff"] < 1e-4              # later steps differ only by fp32 summation order


def test_eos_stops_both_paths_identically(model):
    p = prompt(20)
    free, _ = generate_greedy_nocache(model, p, 12)
    eos = free[4]                                      # force an EOS in the middle of the free-running output
    a, sa = generate_greedy_nocache(model, p, 12, eos_id=eos)
    b, sb = generate_greedy_kv(model, p, 12, eos_id=eos)
    assert a == b and a[-1] == eos and len(a) <= 5 and sa.stopped_on_eos and sb.stopped_on_eos


def test_logits_from_both_paths_close_at_every_step(model):
    p = prompt(32)
    _, sa = generate_greedy_nocache(model, p, 10, collect_logits=True)
    _, sb = generate_greedy_kv(model, p, 10, collect_logits=True)
    for a, b in zip(sa.logits, sb.logits):
        assert torch.allclose(a, b, atol=1e-4)


# ---------------------------------------------------------------- D1: the cache really grows / is really used
def test_cache_growth_shapes_and_bytes(model):
    cfg = model.config
    cache = new_cache(model)
    assert cache_len(cache) == 0
    L = 24
    forward_with_cache(model, prompt(L), cache, start=0)
    assert cache_len(cache) == L
    hd = cfg.hidden_size // cfg.num_attention_heads
    assert cache_nbytes(cache) == kv_bytes_per_token(cfg, 4) * L          # fp32: formula == measured, exactly
    for step in range(5):
        forward_with_cache(model, prompt(1, seed=step), cache, start=L + step)
        assert cache_len(cache) == L + step + 1
    assert cache_nbytes(cache) == kv_bytes_per_token(cfg, 4) * (L + 5)
    shapes = cache_shapes(cache)
    assert len(shapes) == cfg.num_hidden_layers
    assert all(s[0] == 1 and s[1] == cfg.num_key_value_heads and s[2] == L + 5 and s[3] == hd for s in shapes)


def test_gqa_cache_is_smaller_than_mha_formula(model):
    cfg = model.config
    assert cfg.num_key_value_heads < cfg.num_attention_heads
    mha = 2 * cfg.num_hidden_layers * cfg.num_attention_heads * (cfg.hidden_size // cfg.num_attention_heads) * 4
    assert kv_bytes_per_token(cfg, 4) * cfg.num_attention_heads == mha * cfg.num_key_value_heads


def test_wrong_start_is_rejected(model):
    cache = new_cache(model)
    forward_with_cache(model, prompt(10), cache, start=0)
    with pytest.raises(AssertionError):
        forward_with_cache(model, prompt(1), cache, start=3)               # caller lies about the position


def test_wrong_position_ids_break_the_output(model):
    """Negative control: if the new token is given position 0 instead of its true position, logits change.
    Proves that explicit position handling matters (RoPE offset), i.e. that the equality tests are not vacuous."""
    p = prompt(30)
    cache = new_cache(model)
    forward_with_cache(model, p, cache, start=0)
    nxt = prompt(1, seed=9)
    good = forward_with_cache(model, nxt, cache, start=30)[0, -1].clone()
    crop_cache(cache, 30)
    out = model(input_ids=nxt, attention_mask=torch.ones(1, 31, dtype=torch.long),
                position_ids=torch.zeros(1, 1, dtype=torch.long),            # WRONG on purpose: true position is 30
                past_key_values=cache, use_cache=True, logits_to_keep=1)
    assert (out.logits[0, -1] - good).abs().max() > 1e-3


def test_prefix_cache_reuse_matches_fresh(model):
    p = prompt(60)
    fresh, _ = generate_greedy_kv(model, p, 12)
    cache = new_cache(model)
    forward_with_cache(model, p[:, :40], cache, start=0)                   # pre-fill a prefix, e.g. a shared system prompt
    reused, st = generate_greedy_kv(model, p, 12, cache=cache)
    assert reused == fresh and st.cache_len_final == 60 + 12 - 1


def test_crop_then_continue_equals_fresh(model):
    """Rollback primitive for speculative decoding: after crop(n) the cache is exactly the first n positions."""
    p = prompt(50)
    cache = new_cache(model)
    forward_with_cache(model, p, cache, start=0)
    forward_with_cache(model, prompt(6, seed=3), cache, start=50)          # 6 speculative tokens ...
    crop_cache(cache, 50)                                                   # ... all rejected
    assert cache_len(cache) == 50
    crop_cache(cache, 50)                                                   # no-op crop is allowed
    nxt = prompt(1, seed=4)
    a = forward_with_cache(model, nxt, cache, start=50)
    c2 = new_cache(model)
    forward_with_cache(model, p, c2, start=0)
    b = forward_with_cache(model, nxt, c2, start=50)
    assert torch.allclose(a, b, atol=1e-5)


def test_multi_token_chunk_equals_one_by_one(model):
    """Verifying k+1 tokens in one cached forward (speculative verification) == feeding them one at a time."""
    p, chunk = prompt(30), prompt(5, seed=2)
    c1, c2 = new_cache(model), new_cache(model)
    forward_with_cache(model, p, c1, start=0)
    forward_with_cache(model, p, c2, start=0)
    big = forward_with_cache(model, chunk, c1, start=30, logits_to_keep=5)[0]
    singles = torch.stack([forward_with_cache(model, chunk[:, i:i + 1], c2, start=30 + i)[0, -1] for i in range(5)])
    assert torch.allclose(big, singles, atol=1e-4)


def test_first_divergence_helper():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_tiny_tokenizer_prompts_work_end_to_end():
    recs = make_records(2)
    tok = make_tok([r["prompt"] + r["target"] for r in recs])
    m = make_model(len(tok))
    ids = torch.tensor([tok(recs[0]["prompt"], add_special_tokens=False).input_ids])
    assert generate_greedy_kv(m, ids, 8)[0] == generate_greedy_nocache(m, ids, 8)[0]


# ---------------------------------------------------------------- Part B: manual attention layer with explicit cache
@pytest.fixture(scope="module")
def layer():
    return CachedGQAAttention(hidden=32, n_heads=4, n_kv_heads=2, seed=1)


def _x(T, hidden=32, seed=0):
    return torch.randn(1, T, hidden, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("use_tiled", [False, True])
def test_manual_cache_decode_equals_full_recompute(layer, use_tiled):
    T, P = 40, 25
    x = _x(T)
    full = layer.forward_full(x, use_tiled=use_tiled)
    cache = layer.new_cache()
    outs = [layer.forward_step(x[:, :P], cache, use_tiled=use_tiled)]          # prefill P tokens
    for t in range(P, T):                                                       # then one token at a time
        outs.append(layer.forward_step(x[:, t:t + 1], cache, use_tiled=use_tiled))
    inc = torch.cat(outs, dim=1)
    assert cache["len"] == T and cache["k"].shape == (1, 2, T, 8)               # GQA: 2 KV heads cached, not 4
    assert torch.allclose(full, inc, atol=1e-5)


def test_manual_cache_wrong_rope_offset_breaks(layer):
    x = _x(20)
    full = layer.forward_full(x)
    cache = layer.new_cache()
    layer.forward_step(x[:, :10], cache)
    bad = layer.forward_step(x[:, 10:11], cache, start=0)                       # new token rotated as if it were position 0
    assert (full[:, 10] - bad[:, 0]).abs().max() > 1e-3


def test_rope_depends_only_on_absolute_position():
    q = torch.randn(1, 2, 1, 8)
    a = apply_rope(q, torch.tensor([7]))
    b = apply_rope(q.expand(1, 2, 3, 8).contiguous(), torch.tensor([5, 6, 7]))[:, :, 2:3]
    assert torch.allclose(a, b, atol=1e-6)


def test_single_query_over_cache_matches_last_row_of_full_causal():
    q = torch.randn(1, 2, 12, 8)
    k, v = torch.randn(1, 2, 12, 8), torch.randn(1, 2, 12, 8)
    full = reference_attention(q, k, v, causal=True)
    last = reference_attention(q[:, :, -1:], k, v, attn_mask=causal_mask(1, 12))   # Lq=1 sees all 12 keys
    assert torch.allclose(full[:, :, -1:], last, atol=1e-6)
