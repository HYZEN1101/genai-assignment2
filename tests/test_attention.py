"""Phase 4 tests (E1-E3): reference vs tiled online-softmax vs torch SDPA. CPU only, no model needed."""
import math

import pytest
import torch
import torch.nn.functional as F

from trialmatch.attention import causal_mask, reference_attention, tiled_attention


def rnd(B, H, Lq, Lk, D, dtype=torch.float32, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (torch.randn(*s, generator=g) * scale).to(dtype)
    return mk(B, H, Lq, D), mk(B, H, Lk, D), mk(B, H, Lk, D)


def sdpa_ref(q, k, v, causal=False, mask=None):
    """torch SDPA with an EXPLICIT bool mask (never is_causal when Lq != Lk: upper-left alignment, DECISIONS D7)."""
    m = mask
    if causal:
        c = causal_mask(q.shape[-2], k.shape[-2])
        m = c if m is None else (c & m)
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=m)


# ---------- E1: reference ----------
def test_hand_computed_example():
    # one query, keys with scores 0 and ln2 (D=1, scale 1): weights 1/3 and 2/3, values 3 and 6 -> 1 + 4 = 5
    q = torch.tensor([[[[1.0]]]])
    k = torch.tensor([[[[0.0], [math.log(2)]]]])
    v = torch.tensor([[[[3.0], [6.0]]]])
    for fn in (reference_attention, lambda *a, **kw: tiled_attention(*a, block_k=1, **kw)):
        assert fn(q, k, v, scale=1.0).item() == pytest.approx(5.0, abs=1e-6)


def test_causal_mask_alignment():
    assert causal_mask(3, 3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    assert causal_mask(1, 4).tolist() == [[True] * 4]                    # decode step sees all cached keys
    assert causal_mask(2, 4).tolist() == [[True, True, True, False], [True] * 4]   # bottom-right aligned
    assert causal_mask(3, 2).tolist() == [[False, False], [True, False], [True, True]]  # first row fully masked


def test_reference_matches_sdpa_and_sums_to_one():
    q, k, v = rnd(2, 3, 17, 17, 16)
    assert torch.allclose(reference_attention(q, k, v, causal=True), sdpa_ref(q, k, v, causal=True), atol=1e-5)
    ones = torch.ones_like(v)                                           # softmax rows sum to 1 -> output == 1
    assert torch.allclose(reference_attention(q, k, ones, causal=True), torch.ones(2, 3, 17, 16), atol=1e-6)


def test_causality_future_keys_do_not_change_output():
    q, k, v = rnd(1, 2, 10, 10, 8)
    base = reference_attention(q, k, v, causal=True)
    k2, v2 = k.clone(), v.clone()
    k2[..., 6:, :] += 5.0
    v2[..., 6:, :] -= 3.0
    pert = reference_attention(q, k2, v2, causal=True)
    assert torch.allclose(base[..., :6, :], pert[..., :6, :], atol=1e-6)   # rows 0..5 never see keys >= 6


# ---------- E2/E3: tiled vs reference vs SDPA over many shapes ----------
SHAPES = [  # (B, H, Lq, Lk, D)
    (1, 1, 1, 1, 4), (1, 2, 1, 37, 16), (2, 3, 1, 128, 32),      # Lq=1 decode steps
    (1, 2, 16, 16, 8), (2, 2, 33, 33, 16), (1, 4, 64, 64, 64),   # square, odd, power of two
    (1, 2, 5, 40, 16), (1, 2, 17, 130, 32),                       # Lq < Lk (chunked prefill with a cache)
    (1, 2, 40, 5, 16),                                            # Lq > Lk (some rows fully masked when causal)
    (1, 1, 129, 129, 128),                                        # Qwen2.5-1.5B head_dim
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("blocks", [(1, 1), (7, 5), (16, 16), (64, 64), (1000, 1000)])   # (1000,1000) = single block
def test_tiled_equals_reference(shape, causal, blocks):
    q, k, v = rnd(*shape, seed=1)
    ref = reference_attention(q, k, v, causal=causal)
    out = tiled_attention(q, k, v, causal=causal, block_q=blocks[0], block_k=blocks[1])
    assert out.shape == ref.shape and torch.isfinite(out).all()
    assert (out - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_reference_and_tiled_match_torch_sdpa(shape, causal):
    q, k, v = rnd(*shape, seed=2)
    Lq, Lk = shape[2], shape[3]
    if causal and Lq > Lk:
        pytest.skip("SDPA returns NaN for fully masked rows; ours returns zeros (tested separately)")
    want = sdpa_ref(q, k, v, causal=causal)
    assert torch.allclose(reference_attention(q, k, v, causal=causal), want, atol=1e-5)
    assert torch.allclose(tiled_attention(q, k, v, causal=causal, block_q=16, block_k=8), want, atol=1e-5)


def test_fully_masked_rows_are_zero_not_nan():
    q, k, v = rnd(1, 2, 6, 3, 8)                                      # rows 0..2 have no visible key
    for out in (reference_attention(q, k, v, causal=True), tiled_attention(q, k, v, causal=True, block_q=2, block_k=2)):
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out[..., :3, :]) == 0
        assert torch.count_nonzero(out[..., 3:, :]) > 0


def test_padding_mask_bool_and_additive():
    q, k, v = rnd(2, 2, 9, 12, 16, seed=3)
    pad = torch.ones(2, 1, 1, 12, dtype=torch.bool)
    pad[0, ..., 9:] = False                                           # batch 0 has 3 padded keys
    pad[1, ..., 4:] = False
    want = sdpa_ref(q, k, v, mask=pad.expand(2, 2, 9, 12))
    assert torch.allclose(reference_attention(q, k, v, attn_mask=pad), want, atol=1e-5)
    assert torch.allclose(tiled_attention(q, k, v, attn_mask=pad, block_q=4, block_k=5), want, atol=1e-5)
    add = torch.zeros(2, 1, 1, 12).masked_fill(~pad, float("-inf"))   # same mask, additive form
    assert torch.allclose(tiled_attention(q, k, v, attn_mask=add, block_q=4, block_k=5), want, atol=1e-5)
    both = tiled_attention(q, k, v, causal=True, attn_mask=pad, block_q=4, block_k=5)   # causal AND padding
    assert torch.allclose(both, reference_attention(q, k, v, causal=True, attn_mask=pad), atol=1e-5)


def test_extreme_logits_are_stable():
    # logits of order 1e3: a naive exp(s) overflows to inf; max-subtraction / online rescaling must not.
    q, k, v = rnd(1, 2, 24, 24, 16, seed=4, scale=25.0)
    ref = reference_attention(q, k, v, causal=True)
    out = tiled_attention(q, k, v, causal=True, block_q=8, block_k=4)
    assert torch.isfinite(ref).all() and torch.isfinite(out).all()
    assert (out - ref).abs().max().item() < 1e-3
    naive = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(16), -1)  # torch's softmax is stable too
    assert torch.isfinite(naive).all()
    s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(16)
    assert torch.isinf(torch.exp(s)).any(), "test is meaningless unless naive exp() would overflow"


def test_running_max_rescaling_is_exercised():
    # later blocks carry much larger scores than earlier ones, forcing alpha = exp(m_old - m_new) << 1
    q = torch.ones(1, 1, 1, 4)
    k = torch.zeros(1, 1, 12, 4)
    k[..., 8:, :] = 3.0                                               # block 3 dominates
    v = torch.arange(12.0).view(1, 1, 12, 1).expand(1, 1, 12, 4).contiguous()
    assert torch.allclose(tiled_attention(q, k, v, block_k=4), reference_attention(q, k, v), atol=1e-6)


@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 3e-2), (torch.float16, 5e-3)])
def test_low_precision_inputs(dtype, tol):
    q, k, v = rnd(1, 2, 33, 33, 32, dtype=dtype, seed=5)
    try:
        ref = reference_attention(q, k, v, causal=True)
    except RuntimeError:
        pytest.skip(f"{dtype} matmul unsupported on this CPU build")
    out = tiled_attention(q, k, v, causal=True, block_q=16, block_k=8)
    assert out.dtype == dtype
    assert (out.float() - ref.float()).abs().max().item() < tol       # both accumulate in fp32; only the final cast rounds
    exact = reference_attention(*(t.float() for t in (q, k, v)), causal=True)
    assert (ref.float() - exact).abs().max().item() < tol


def test_block_skipping_matches_and_decode_equals_last_row():
    q, k, v = rnd(1, 2, 50, 50, 16, seed=6)
    full = reference_attention(q, k, v, causal=True)
    step = tiled_attention(q[..., -1:, :], k, v, causal=True, block_k=7)   # decode: Lq=1 over the whole "cache"
    assert torch.allclose(step, full[..., -1:, :], atol=1e-5)
    chunk = tiled_attention(q[..., 30:, :], k, v, causal=True, block_q=6, block_k=7)  # chunked prefill, offset 30
    assert torch.allclose(chunk, full[..., 30:, :], atol=1e-5)
