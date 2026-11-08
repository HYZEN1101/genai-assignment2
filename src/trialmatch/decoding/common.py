"""Shared pieces for the explicit decoding loops (Phase 5 KV cache; reused by Phase 6 speculative decoding).

The one function everything else is built on is `forward_with_cache`: ONE model forward over `n` new tokens
that sit at absolute positions [start, start+n), with the cache already holding positions [0, start).
The caller (not the library) decides the three things that make cached decoding correct:

  1. position_ids  = start .. start+n-1   -> RoPE rotates the NEW keys/queries by their absolute position.
                                              If you restart at 0 the new token "thinks" it is the first token.
  2. attention_mask = ones(1, start+n)    -> covers cached keys + new keys (length must equal the keys the layer sees).
  3. cache slot    = start .. start+n-1   -> where in the cache the new K/V are written. In transformers 5.x the slot is
                                              derived from the cache's own length (we assert cache_len == start first);
                                              4.x models additionally take it as `cache_position`, passed when accepted.
The causal structure inside the layer is then bottom-right aligned: new query i sees keys j <= start + i.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch


# ----------------------------------------------------------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class GenStats:
    """Timing/bookkeeping of ONE generation. Definitions (identical for the cached and uncached loops):
    prefill_ms = forward over the whole prompt + argmax of the first new token (time to first token)
    decode_ms  = every remaining step (new_tokens - 1 steps)
    """
    path: str = ""
    prompt_len: int = 0
    new_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    stopped_on_eos: bool = False
    cache_len_final: Optional[int] = None          # cached path only: positions held in the cache at the end
    top2_gap: list = field(default_factory=list)   # (top1 - top2) logit gap per step, only if track=True
    logits: list = field(default_factory=list)     # last-position logits (cpu fp32) per step, only if collect_logits=True

    @property
    def total_ms(self) -> float:
        return self.prefill_ms + self.decode_ms

    @property
    def decode_tok_s(self) -> float:
        n = max(self.new_tokens - 1, 0)
        return n / (self.decode_ms / 1000.0) if self.decode_ms > 0 and n > 0 else float("nan")

    @property
    def e2e_tok_s(self) -> float:
        return self.new_tokens / (self.total_ms / 1000.0) if self.total_ms > 0 else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("logits"), d.pop("top2_gap")
        d.update(total_ms=self.total_ms, decode_tok_s=self.decode_tok_s, e2e_tok_s=self.e2e_tok_s)
        return d


def sync(device) -> None:
    """Wait for queued GPU work so perf_counter measures finished work, not kernel launches."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def now_ms(device) -> float:
    sync(device)
    return time.perf_counter() * 1000.0


def eos_set(eos_id) -> frozenset:
    if eos_id is None:
        return frozenset()
    if isinstance(eos_id, int):
        return frozenset([eos_id])
    return frozenset(int(e) for e in eos_id)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


# ----------------------------------------------------------------------------------------------------------------------
# cache helpers (transformers >= 5: DynamicCache holds one layer object per decoder layer with .keys / .values
# of shape [B, kv_heads, T, head_dim]). Helpers fall back to older attribute names; the unit tests exercise them.
# ----------------------------------------------------------------------------------------------------------------------
def new_cache(model):
    from transformers import DynamicCache
    try:
        return DynamicCache(config=model.config)
    except TypeError:                      # older signature without config
        return DynamicCache()


def cache_len(cache) -> int:
    """Number of positions currently stored (same for every layer)."""
    return int(cache.get_seq_length())


def _layer_tensors(cache):
    if hasattr(cache, "layers"):
        for lyr in cache.layers:
            k, v = getattr(lyr, "keys", None), getattr(lyr, "values", None)
            if k is not None and v is not None and k.numel() > 0:
                yield k, v
    else:                                  # transformers 4.x
        for k, v in zip(cache.key_cache, cache.value_cache):
            if k is not None and k.numel() > 0:
                yield k, v


def cache_nbytes(cache) -> int:
    """Bytes actually held by the cache tensors (measured; compare with kv_bytes_per_token * length)."""
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in _layer_tensors(cache))


def cache_shapes(cache) -> list:
    return [tuple(k.shape) for k, _ in _layer_tensors(cache)]


def crop_cache(cache, n: int) -> None:
    """Roll the cache back to its first n positions (used by speculative decoding after a rejection).

    transformers 5.x: Cache.crop(tokens_to_remove) wants a NEGATIVE number of tokens to remove (a positive value raises
    ValueError; the 4.x meaning "keep max_length tokens" is gone). Negative values are also accepted by 4.x, so we always
    pass -(tokens to drop). Checked against transformers 5.22.1 cache_utils.py.
    """
    have = cache_len(cache)
    assert 0 <= n <= have, f"cannot crop a cache of {have} positions to {n}"
    if n < have:
        cache.crop(-(have - n))
    assert cache_len(cache) == n


def kv_bytes_per_token(config, dtype_bytes: int = 2) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x bytes.
    Qwen2.5-1.5B fp16: 2*28*2*128*2 = 28,672 B (GQA: 2 KV heads, not 12)."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * dtype_bytes


_CP_CACHE: dict = {}


def _takes_cache_position(model) -> bool:
    """transformers 4.x models take cache_position explicitly; 5.x models derive the cache slot from the cache object
    (passing it would only travel through **kwargs). Pass it only where it is a real parameter."""
    import inspect
    cls = type(model)
    if cls not in _CP_CACHE:
        _CP_CACHE[cls] = "cache_position" in inspect.signature(model.forward).parameters
    return _CP_CACHE[cls]


# ----------------------------------------------------------------------------------------------------------------------
# the explicit cache-aware forward
# ----------------------------------------------------------------------------------------------------------------------
def forward_with_cache(model, ids: torch.Tensor, cache, start: int, logits_to_keep: int = 1) -> torch.Tensor:
    """Run `ids` (shape [1, n]) at absolute positions [start, start+n) and APPEND their K/V to `cache`.

    Precondition (asserted): cache currently holds exactly `start` positions.
    Returns logits for the last `logits_to_keep` positions: [1, logits_to_keep, vocab].
    """
    assert ids.dim() == 2 and ids.shape[0] == 1, "batch size 1 only (docs/DECISIONS.md D21)"
    n = ids.shape[1]
    have = cache_len(cache)
    assert have == start, f"cache holds {have} positions but caller says the new tokens start at {start}"
    dev = ids.device
    position_ids = torch.arange(start, start + n, device=dev).unsqueeze(0)      # absolute positions -> RoPE
    cache_position = torch.arange(start, start + n, device=dev)                  # slots to write in the cache
    attention_mask = torch.ones(1, start + n, dtype=torch.long, device=dev)      # cached keys + new keys
    extra = {"cache_position": cache_position} if _takes_cache_position(model) else {}
    out = model(input_ids=ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=cache, use_cache=True, logits_to_keep=logits_to_keep, **extra)
    return out.logits


@torch.no_grad()
def greedy_next(logits_last: torch.Tensor) -> torch.Tensor:
    """argmax over vocab of the LAST position -> tensor [1] (stays on device; no host sync)."""
    return logits_last[:, -1, :].argmax(dim=-1)
