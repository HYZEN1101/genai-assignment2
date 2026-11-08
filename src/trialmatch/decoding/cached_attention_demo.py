"""Part B (viva demo): ONE grouped-query attention layer with a hand-written KV cache.

Everything a decoder layer does at inference, written out so each invariant is visible:
  q,k,v projections -> RoPE with an ABSOLUTE position offset -> append K/V to the cache (torch.cat) ->
  causal mask for the new queries over ALL cached keys (bottom-right aligned, Phase 4 kernels) -> attention ->
  output projection.

`forward_full(x)`            recomputes the whole sequence (no cache).
`forward_step(x_new, cache)` processes only new tokens and extends `cache` (a dict with "k", "v", "len").
Equality of the two (tests/test_kv_equiv.py) is the mechanism-level proof of the cache; passing `use_tiled=True`
swaps in the Phase 4 tiled online-softmax kernel, so cache + tiled attention are shown to compose.

Conventions follow Qwen2 (rotate-half RoPE, GQA with kv head = q head // n_rep). No GQA repetition happens inside the
attention kernels (decision D20): K/V are expanded with repeat_interleave right before the kernel call.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from trialmatch.attention import reference_attention, tiled_attention


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float = 10000.0, dtype=torch.float32):
    """cos/sin tables for the GIVEN absolute positions (1-D long tensor). Shape (len(positions), head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)          # (n, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                  # (n, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """x: (B, H, n, D); positions: (n,) ABSOLUTE positions of those n tokens. Rotation depends only on position."""
    cos, sin = rope_cos_sin(positions, x.shape[-1], theta, x.dtype)
    return x * cos + rotate_half(x) * sin


class CachedGQAAttention(nn.Module):
    def __init__(self, hidden: int = 64, n_heads: int = 8, n_kv_heads: int = 2, head_dim: Optional[int] = None,
                 theta: float = 10000.0, seed: int = 0):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.hkv, self.d = n_heads, n_kv_heads, head_dim or hidden // n_heads
        self.n_rep, self.theta = n_heads // n_kv_heads, theta
        g = torch.Generator().manual_seed(seed)
        init = lambda o, i: nn.Parameter(torch.randn(o, i, generator=g) / math.sqrt(i))
        self.wq, self.wk = init(self.h * self.d, hidden), init(self.hkv * self.d, hidden)
        self.wv, self.wo = init(self.hkv * self.d, hidden), init(hidden, self.h * self.d)

    @staticmethod
    def new_cache() -> dict:
        return {"k": None, "v": None, "len": 0}

    def _qkv(self, x: torch.Tensor, start: int):
        B, n, _ = x.shape
        pos = torch.arange(start, start + n, device=x.device)                # ABSOLUTE positions of the new tokens
        q = (x @ self.wq.T).view(B, n, self.h, self.d).transpose(1, 2)
        k = (x @ self.wk.T).view(B, n, self.hkv, self.d).transpose(1, 2)
        v = (x @ self.wv.T).view(B, n, self.hkv, self.d).transpose(1, 2)
        return apply_rope(q, pos, self.theta), apply_rope(k, pos, self.theta), v   # RoPE on q and k, never on v

    def _attend(self, q, k, v, use_tiled: bool, block: int):
        k = k.repeat_interleave(self.n_rep, dim=1)                           # GQA: share each KV head across n_rep q heads
        v = v.repeat_interleave(self.n_rep, dim=1)
        # causal=True is BOTTOM-RIGHT aligned (query i sees keys j <= i + Lk - Lq): n new queries over T = start+n cached keys.
        # For n=1 the single query sees every key; for prefill (n=T) it is the usual lower triangle.
        if use_tiled:
            return tiled_attention(q, k, v, causal=True, block_q=block, block_k=block)
        return reference_attention(q, k, v, causal=True)

    def forward_full(self, x: torch.Tensor, use_tiled: bool = False, block: int = 16) -> torch.Tensor:
        q, k, v = self._qkv(x, start=0)
        o = self._attend(q, k, v, use_tiled, block)
        return o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1) @ self.wo.T

    def forward_step(self, x_new: torch.Tensor, cache: dict, use_tiled: bool = False, block: int = 16,
                     start: Optional[int] = None) -> torch.Tensor:
        """Process n new tokens (n=1 in decoding, n=L in prefill) and extend the cache in place.
        `start` overrides the RoPE offset ONLY for the negative test (wrong positions must break equality)."""
        pos0 = cache["len"] if start is None else start
        q, k_new, v_new = self._qkv(x_new, start=pos0)
        cache["k"] = k_new if cache["k"] is None else torch.cat([cache["k"], k_new], dim=2)   # cache growth: +n along T
        cache["v"] = v_new if cache["v"] is None else torch.cat([cache["v"], v_new], dim=2)
        cache["len"] += x_new.shape[1]
        o = self._attend(q, cache["k"], cache["v"], use_tiled, block)        # new queries over ALL cached keys
        return o.transpose(1, 2).reshape(x_new.shape[0], x_new.shape[1], -1) @ self.wo.T
