"""Reference scaled dot-product attention (Phase 4, rubric E1).

    out = softmax(Q K^T * scale + mask) V          Q:(B,H,Lq,D)  K,V:(B,H,Lk,D)

Design points (each one is a likely viva question):
* Numerically stable softmax: subtract the row max before exp (exp never sees a large positive number).
* Scores and softmax are computed in float32 even for fp16/bf16 inputs; the result is cast back to the input dtype.
* Causal mask is BOTTOM-RIGHT aligned: query i may attend to key j iff j <= i + (Lk - Lq).
  With Lq == Lk this is the usual lower triangle; with Lq == 1 (cached decoding) the single query sees every key.
  (F.scaled_dot_product_attention(is_causal=True) is upper-left aligned when Lq != Lk: see docs/DECISIONS.md D7.)
* Fully masked rows (possible when Lq > Lk, or a padding mask hides everything) return zeros, never NaN.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

NEG_INF = float("-inf")


def causal_mask(lq: int, lk: int, device=None) -> torch.Tensor:
    """Bool (Lq, Lk), True = may attend. Bottom-right aligned: j <= i + (lk - lq)."""
    i = torch.arange(lq, device=device).unsqueeze(1)
    j = torch.arange(lk, device=device).unsqueeze(0)
    return j <= i + (lk - lq)


def _apply_mask(scores: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """attn_mask: bool (True = keep) or additive float, broadcastable to scores."""
    if attn_mask.dtype == torch.bool:
        return scores.masked_fill(~attn_mask, NEG_INF)
    return scores + attn_mask.to(scores.dtype)


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
                        attn_mask: Optional[torch.Tensor] = None, scale: Optional[float] = None) -> torch.Tensor:
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "expected (B,H,L,D)"
    assert k.shape[:3] == v.shape[:3] and q.shape[-1] == k.shape[-1], "q/k head_dim mismatch or k/v length mismatch"
    lq, lk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    scale = (1.0 / math.sqrt(d)) if scale is None else scale

    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale          # (B,H,Lq,Lk) fp32: the O(L^2) matrix
    if causal:
        s = _apply_mask(s, causal_mask(lq, lk, q.device))
    if attn_mask is not None:
        s = _apply_mask(s, attn_mask)

    m = s.amax(dim=-1, keepdim=True)                                            # row max
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)                     # all -inf row: avoid (-inf) - (-inf) = NaN
    p = torch.exp(s - m)                                                        # masked entries -> exp(-inf) = 0
    denom = p.sum(dim=-1, keepdim=True)
    p = p / torch.where(denom == 0, torch.ones_like(denom), denom)              # fully masked row -> all zeros
    return torch.matmul(p, v.float()).to(q.dtype)
