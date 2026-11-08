"""Benchmark helpers for the decoders (rubric D3): same model, same input ids, warm-up, synchronized repeated timing,
peak memory with reset before every run.

Peak memory = torch.cuda.max_memory_allocated() during the run (stats reset right before) and also "above weights"
= peak - memory allocated before the run started (what the generation itself needed: KV cache + activations).
On CPU peak memory is not measured (NaN).
"""
from __future__ import annotations

import statistics
from typing import Callable

import torch

from .common import cache_nbytes, kv_bytes_per_token, model_device, new_cache
from .equiv import first_divergence
from .greedy_kv import generate_greedy_kv
from .greedy_nocache import generate_greedy_nocache

MB = 1024 * 1024


def _run_with_memory(fn: Callable, dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    else:
        base = 0
    out = fn()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        return out, peak / MB, (peak - base) / MB
    return out, float("nan"), float("nan")


def _summ(xs: list[float]) -> dict:
    xs = [x for x in xs if x == x]
    if not xs:
        return {"median": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {"median": statistics.median(xs), "mean": statistics.fmean(xs), "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0}


def bench_cell(model, input_ids: torch.Tensor, n_new: int, warmup: int = 2, repeats: int = 5) -> dict:
    """Benchmark both decoders on ONE (prompt length, new tokens) cell. EOS stopping is disabled so both paths generate
    exactly n_new tokens (fixed work). Returns a flat dict ready for a CSV row."""
    dev = model_device(model)
    L = int(input_ids.shape[1])
    runs = {"nocache": [], "kv_cache": []}
    toks = {}
    kv_bytes = None
    for rep in range(warmup + repeats):
        for path in ("nocache", "kv_cache"):
            if path == "nocache":
                (t, st), peak, above = _run_with_memory(lambda: generate_greedy_nocache(model, input_ids, n_new), dev)
            else:
                cache = new_cache(model)
                (t, st), peak, above = _run_with_memory(lambda: generate_greedy_kv(model, input_ids, n_new, cache=cache), dev)
                kv_bytes = cache_nbytes(cache)
                del cache
            toks[path] = t
            if rep >= warmup:                                           # warm-up runs are executed but never recorded
                d = st.as_dict()
                d.update(peak_mb=peak, peak_above_weights_mb=above)
                runs[path].append(d)
    div = first_divergence(toks["nocache"], toks["kv_cache"])
    dtype_bytes = next(model.parameters()).element_size()
    row = {"prompt_len": L, "new_tokens": n_new, "warmup": warmup, "repeats": repeats, "device": str(dev),
           "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
           "tokens_equal": div is None, "first_div": -1 if div is None else div,
           "kv_bytes_measured": kv_bytes, "kv_bytes_formula": kv_bytes_per_token(model.config, dtype_bytes) * (L + n_new - 1)}
    for path, rs in runs.items():
        for key in ("prefill_ms", "decode_ms", "total_ms", "decode_tok_s", "e2e_tok_s", "peak_mb", "peak_above_weights_mb"):
            s = _summ([r[key] for r in rs])
            row[f"{path}_{key}_median"] = s["median"]
            row[f"{path}_{key}_std"] = s["std"]
    row["speedup_total"] = row["nocache_total_ms_median"] / row["kv_cache_total_ms_median"]
    row["speedup_decode"] = (row["nocache_decode_ms_median"] / row["kv_cache_decode_ms_median"]
                             if row["kv_cache_decode_ms_median"] > 0 else float("nan"))
    return row
