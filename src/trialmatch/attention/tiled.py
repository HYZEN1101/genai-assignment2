"""Tiled attention with ONLINE softmax (Phase 4, rubric E2). Never materialises the full Lq x Lk score matrix.

For one query block we stream over key/value blocks keeping, per query row:
    m   running max of the scores seen so far
    l   running softmax denominator  sum_j exp(s_j - m)
    acc running numerator            sum_j exp(s_j - m) v_j
When a new block with scores s arrives:
    m_new = max(m, rowmax(s))
    p     = exp(s - m_new)
    alpha = exp(m - m_new)                 # rescales everything accumulated under the OLD max
    l     = l * alpha + rowsum(p)
    acc   = acc * alpha + p @ v
    m     = m_new
and at the end  out = acc / l.   Result is mathematically identical to softmax(QK^T*scale + mask) V.
Accumulators m, l, acc are float32 whatever the input dtype. No F.scaled_dot_product_attention is used here.
Memory: O(block_q * block_k) scores per step instead of O(Lq * Lk).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

NEG_INF = float("-inf")


def tiled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
                    attn_mask: Optional[torch.Tensor] = None, scale: Optional[float] = None,
                    block_q: int = 64, block_k: int = 64) -> torch.Tensor:
    """Same contract as reference_attention (bottom-right causal alignment, bool or additive attn_mask,
    fully masked rows -> zeros). block sizes need not divide the lengths."""
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "expected (B,H,L,D)"
    assert block_q > 0 and block_k > 0
    B, H, lq, d = q.shape
    lk = k.shape[-2]
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    off = lk - lq                                           # query i may see keys j <= i + off
    if attn_mask is not None:
        attn_mask = attn_mask.expand(B, H, lq, lk) if attn_mask.dim() == 4 else attn_mask.expand(lq, lk)

    out = torch.empty(B, H, lq, v.shape[-1], dtype=q.dtype, device=q.device)
    for qs in range(0, lq, block_q):
        qe = min(qs + block_q, lq)
        qb = q[..., qs:qe, :].float()
        bq = qe - qs
        m = torch.full((B, H, bq, 1), NEG_INF, device=q.device)
        l = torch.zeros((B, H, bq, 1), device=q.device)
        acc = torch.zeros((B, H, bq, v.shape[-1]), device=q.device)

        for ks in range(0, lk, block_k):
            ke = min(ks + block_k, lk)
            if causal and ks > (qe - 1) + off:
                break                                       # this and all later key blocks are above the diagonal: skip
            s = torch.matmul(qb, k[..., ks:ke, :].float().transpose(-1, -2)) * scale   # (B,H,bq,bk)
            if causal and (ke - 1) > qs + off:              # block straddles the diagonal -> needs masking
                s = s.masked_fill(~_causal_block(qs, qe, ks, ke, off, q.device), NEG_INF)
            if attn_mask is not None:
                mb = attn_mask[..., qs:qe, ks:ke]
                s = s.masked_fill(~mb, NEG_INF) if mb.dtype == torch.bool else s + mb.to(s.dtype)

            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)   # row still fully masked: use 0
            p = torch.exp(s - m_safe)                       # masked -> 0
            alpha = torch.exp(m - m_safe)                   # m = -inf (first block) -> 0, correct
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p, v[..., ks:ke, :].float())
            m = m_new

        out[..., qs:qe, :] = (acc / torch.where(l == 0, torch.ones_like(l), l)).to(q.dtype)   # l==0 -> zeros
    return out


def _causal_block(qs: int, qe: int, ks: int, ke: int, off: int, device) -> torch.Tensor:
    i = torch.arange(qs, qe, device=device).unsqueeze(1)
    j = torch.arange(ks, ke, device=device).unsqueeze(0)
    return j <= i + off
