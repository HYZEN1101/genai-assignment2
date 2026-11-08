"""Phase 4 (E3): numerical error table + speed/memory benchmark for reference / tiled / torch SDPA backends.

  python scripts/bench_attn.py --mode error            # CPU ok, ~1 min  -> results/benchmarks/attn_error.csv
  python scripts/bench_attn.py --mode speed            # GPU (T4), ~5 min -> results/benchmarks/attn_bench.csv + figures
  python scripts/bench_attn.py --mode all

Timing: warm-up runs, then N timed repeats with CUDA events (perf_counter on CPU), torch.cuda.synchronize() around each.
Peak memory = max_memory_allocated during the call minus memory already allocated for the inputs (reset before each run).
Rows that fail (OOM / backend unsupported on this GPU) are KEPT in the CSV with status text: they are results too.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.attention import causal_mask, reference_attention, tiled_attention  # noqa: E402

OUT = ROOT / "results" / "benchmarks"
FIG = ROOT / "results" / "figures"
LOG = ROOT / "results" / "logs" / "bench_attn.log"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def sdpa_backend(backend_name: str):
    """Returns fn(q,k,v,causal) forcing ONE torch SDPA backend (no fallback to others)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    b = {"sdpa_math": SDPBackend.MATH, "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
         "sdpa_flash": SDPBackend.FLASH_ATTENTION}[backend_name]

    def fn(q, k, v, causal):
        with sdpa_kernel(b):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)   # Lq == Lk in the benchmark, so upper-left == bottom-right
    return fn


def methods(block: int):
    return {
        "reference": lambda q, k, v, c: reference_attention(q, k, v, causal=c),
        f"tiled_b{block}": lambda q, k, v, c: tiled_attention(q, k, v, causal=c, block_q=block, block_k=block),
        "sdpa_math": sdpa_backend("sdpa_math"),
        "sdpa_efficient": sdpa_backend("sdpa_efficient"),
        "sdpa_flash": sdpa_backend("sdpa_flash"),
    }


# ---------------------------------------------------------------- error table
def error_table(device: str) -> list[dict]:
    rows = []
    shapes = [(1, 2, 1, 128, 64), (1, 4, 64, 64, 64), (2, 4, 127, 127, 64), (1, 12, 256, 256, 128), (1, 2, 17, 130, 64), (1, 2, 512, 512, 64)]
    for dtype in (torch.float32, torch.float16):
        for (B, H, Lq, Lk, D) in shapes:
            for causal in (False, True):
                for block in (16, 64):
                    g = torch.Generator().manual_seed(0)
                    q, k, v = (torch.randn(B, H, L, D, generator=g).to(device=device, dtype=dtype) for L in (Lq, Lk, Lk))
                    gold = reference_attention(q.float(), k.float(), v.float(), causal=causal)    # fp32 reference = ground truth
                    cand = {"reference": reference_attention(q, k, v, causal=causal),
                            f"tiled_b{block}": tiled_attention(q, k, v, causal=causal, block_q=block, block_k=block)}
                    mask = causal_mask(Lq, Lk, q.device) if causal else None        # explicit mask: SDPA is_causal is upper-left for Lq != Lk
                    cand["torch_sdpa_fp32"] = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask)   # fp32 SDPA (math/CPU)
                    for name, o in cand.items():
                        d = (o.float() - gold).abs()
                        rows.append(dict(method=name, dtype=str(dtype).split(".")[-1], B=B, H=H, Lq=Lq, Lk=Lk, D=D, causal=causal,
                                         max_abs_err=d.max().item(), mean_abs_err=d.mean().item()))
    return rows


# ---------------------------------------------------------------- speed / memory
def time_call(fn, args, device, warmup, repeats):
    for _ in range(warmup):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if device == "cuda":
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(*args); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t = time.perf_counter(); fn(*args); times.append((time.perf_counter() - t) * 1e3)
    return times


def peak_mem_mb(fn, args, device):
    if device != "cuda":
        return float("nan")
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(*args)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def speed_table(device, dtype, lens, H, D, block, causal, warmup, repeats) -> list[dict]:
    rows = []
    for L in lens:
        g = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(1, H, L, D, generator=g).to(device=device, dtype=dtype) for _ in range(3))
        for name, fn in methods(block).items():
            row = dict(method=name, L=L, H=H, D=D, dtype=str(dtype).split(".")[-1], causal=causal, device=device, status="ok",
                       ms_median=float("nan"), ms_mean=float("nan"), ms_std=float("nan"), peak_mb=float("nan"))
            try:
                if name == "sdpa_flash" and device == "cpu":
                    raise RuntimeError("flash backend is CUDA-only")
                if name == "sdpa_efficient" and device == "cpu":
                    raise RuntimeError("mem-efficient backend is CUDA-only")
                args = (q, k, v, causal)
                ts = time_call(fn, args, device, warmup, repeats)
                row.update(ms_median=statistics.median(ts), ms_mean=statistics.mean(ts), ms_std=statistics.pstdev(ts),
                           peak_mb=peak_mem_mb(fn, args, device))
            except Exception as ex:                                   # OOM / unsupported backend: record it
                row["status"] = f"{type(ex).__name__}: {str(ex).splitlines()[0][:90]}"
                if device == "cuda":
                    torch.cuda.empty_cache()
            log(json.dumps(row))
            rows.append(row)
    return rows


def plots(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    ok = df[df.status == "ok"]
    for col, ylabel, fname in (("ms_median", "median time per call (ms)", "attn_time.png"), ("peak_mb", "peak extra memory (MiB)", "attn_mem.png")):
        if ok[col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, g in ok.groupby("method"):
            ax.plot(g.L, g[col], marker="o", label=name)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length L (Lq = Lk)"); ax.set_ylabel(ylabel)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title(f"{df.device.iloc[0]} {df.dtype.iloc[0]} H={df.H.iloc[0]} D={df.D.iloc[0]} causal={df.causal.iloc[0]}")
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)
        log("wrote", FIG / fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["error", "speed", "all"], default="all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--heads", type=int, default=12, help="12 = query heads of Qwen2.5-1.5B")
    ap.add_argument("--head-dim", type=int, default=128, help="128 = head_dim of Qwen2.5-1.5B")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--no-causal", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(torch=torch.__version__, device=a.device, gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
               capability=torch.cuda.get_device_capability(0) if a.device == "cuda" else None, args=vars(a))
    log("ENV", json.dumps(env))
    (OUT / "attn_env.json").write_text(json.dumps(env, indent=2))
    if a.mode in ("error", "all"):
        df = pd.DataFrame(error_table("cpu"))                       # error table on CPU: deterministic and needs no GPU
        df.to_csv(OUT / "attn_error.csv", index=False)
        log(df.groupby(["method", "dtype"])[["max_abs_err", "mean_abs_err"]].max().to_string())
    if a.mode in ("speed", "all"):
        df = pd.DataFrame(speed_table(a.device, getattr(torch, a.dtype), a.lens, a.heads, a.head_dim, a.block, not a.no_causal, a.warmup, a.repeats))
        df.to_csv(OUT / f"attn_bench_{a.device}.csv", index=False)
        plots(df)


if __name__ == "__main__":
    main()
