"""Phase 5 (D3): benchmark explicit KV-cache decoding vs the no-cache reference.

  python scripts/bench_kv.py --adapter weights/adapter                      # T4, full grid (~15-25 min), RESUMABLE
  python scripts/bench_kv.py --adapter weights/adapter --quick              # 1 warm-up, 3 repeats, no 1024 (~6-8 min)
  python scripts/bench_kv.py --mode plot                                    # CPU only: CSV -> figures

Grid: prompt lengths x new tokens (EOS disabled, so both paths do exactly N new tokens). Prompts are length-controlled
REAL-token streams (concatenated validation prompts truncated to L + assistant header), see decoding/equiv.length_prompt.
Per cell: warm-up runs (executed, not recorded) then repeats; CUDA synchronize around every timing; peak memory reset
before each run. One CSV row per cell is appended as soon as the cell finishes, so a Colab disconnect loses at most one cell.
Output: results/benchmarks/kv_bench.csv, kv_env.json, results/logs/bench_kv.log, results/figures/kv_*.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.decoding.bench import bench_cell  # noqa: E402
from trialmatch.decoding.equiv import length_prompt  # noqa: E402
from trialmatch.pipeline.predictor import load_model  # noqa: E402

OUT = ROOT / "results" / "benchmarks"
FIG = ROOT / "results" / "figures"
LOG = ROOT / "results" / "logs" / "bench_kv.log"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def read_rows(p: Path) -> list[dict]:
    return [dict(r) for r in csv.DictReader(open(p))] if p.exists() else []


def write_rows(p: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(a) -> None:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    val = [json.loads(l) for l in open(ROOT / "data" / "processed" / "val.jsonl") if l.strip()]
    texts = [r["prompt"] for r in val]
    model, tok = load_model(a.model, adapter=a.adapter, dtype=a.dtype)
    dev = next(model.parameters()).device
    tag = a.tag or ((Path(a.adapter).name + "+" if a.adapter else "") + a.model.split("/")[-1])
    env = {"tag": tag, "model": a.model, "adapter": a.adapter, "torch": torch.__version__, "device": str(dev),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "dtype": str(next(model.parameters()).dtype), "warmup": a.warmup, "repeats": a.repeats,
           "weights_mb": torch.cuda.memory_allocated(dev) / 2**20 if dev.type == "cuda" else None}
    json.dump(env, open(OUT / "kv_env.json", "w"), indent=1)
    log("env", env)

    done = {(r["tag"], int(r["prompt_len"]), int(r["new_tokens"])) for r in read_rows(out)}
    for L in a.lens:
        ids = length_prompt(tok, texts, L).to(dev)
        for N in a.news:
            if (tag, L, N) in done:
                log(f"skip L={L} N={N} (already in {out.name})")
                continue
            log(f"--- cell L={L} N={N} ...")
            try:
                row = bench_cell(model, ids, N, warmup=a.warmup, repeats=a.repeats)
            except torch.cuda.OutOfMemoryError as e:                       # OOM is a result too: keep it in the table
                torch.cuda.empty_cache()
                row = {"prompt_len": L, "new_tokens": N, "status": f"OOM: {str(e)[:80]}"}
            row.update(tag=tag, status=row.get("status", "ok"))
            rows = read_rows(out) + [row]
            write_rows(out, rows)                                          # append-by-rewrite: resumable
            if row["status"] == "ok":
                log(f"    equal={row['tokens_equal']} | prefill ms nocache/kv {row['nocache_prefill_ms_median']:.1f}/"
                    f"{row['kv_cache_prefill_ms_median']:.1f} | decode ms nocache/kv {row['nocache_decode_ms_median']:.0f}/"
                    f"{row['kv_cache_decode_ms_median']:.0f} | tok/s e2e nocache/kv {row['nocache_e2e_tok_s_median']:.1f}/"
                    f"{row['kv_cache_e2e_tok_s_median']:.1f} | speedup {row['speedup_total']:.1f}x | "
                    f"peak above weights MB nocache/kv {row['nocache_peak_above_weights_mb_median']:.0f}/"
                    f"{row['kv_cache_peak_above_weights_mb_median']:.0f}")
            else:
                log("   ", row["status"])
    log(f"wrote {out}")


def plot(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(a.out)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        log("no ok rows to plot")
        return
    if a.tag:
        df = df[df["tag"] == a.tag]
    df = df.sort_values(["new_tokens", "prompt_len"])
    FIG.mkdir(parents=True, exist_ok=True)
    news = sorted(df["new_tokens"].unique())
    sty = {"nocache": ("tab:red", "--", "o"), "kv_cache": ("tab:blue", "-", "s")}
    lab = {"nocache": "no cache (full recompute)", "kv_cache": "KV cache"}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    d0 = df[df["new_tokens"] == news[0]]
    for p, (c, ls, m) in sty.items():
        ax[0].errorbar(d0["prompt_len"], d0[f"{p}_prefill_ms_median"], yerr=d0[f"{p}_prefill_ms_std"], color=c, ls=ls, marker=m, label=lab[p])
    ax[0].set(xlabel="prompt length (tokens)", ylabel="prefill ms (median)", title="Prefill (time to first token)")
    for p, (c, ls, m) in sty.items():
        for N in news:
            d = df[df["new_tokens"] == N]
            ax[1].plot(d["prompt_len"], d[f"{p}_decode_ms_median"] / (N - 1), color=c, ls=ls, marker=m, alpha=0.4 + 0.6 * news.index(N) / max(len(news) - 1, 1),
                       label=f"{lab[p]}, N={N}")
    ax[1].set(xlabel="prompt length (tokens)", ylabel="decode ms per generated token", yscale="log", title="Decode cost per token")
    ax[1].legend(fontsize=6)
    for N in news:
        d = df[df["new_tokens"] == N]
        ax[2].plot(d["prompt_len"], d["speedup_total"], marker="o", label=f"N={N}")
    ax[2].axhline(1.0, color="k", lw=0.5)
    ax[2].set(xlabel="prompt length (tokens)", ylabel="end-to-end speedup (no cache / KV)", title="Speedup of KV cache")
    ax[2].legend()
    ax[0].legend(fontsize=7)
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "kv_time.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    dN = df[df["new_tokens"] == news[-1]]
    for p, (c, ls, m) in sty.items():
        ax[0].plot(dN["prompt_len"], dN[f"{p}_peak_above_weights_mb_median"], color=c, ls=ls, marker=m, label=lab[p])
    ax[0].set(xlabel="prompt length (tokens)", ylabel="peak GPU MB above weights", title=f"Peak memory (N={news[-1]})")
    ax[0].legend()
    d = df.drop_duplicates(["prompt_len", "new_tokens"])
    ax[1].scatter(d["kv_bytes_formula"] / 2**20, d["kv_bytes_measured"] / 2**20, color="tab:blue")
    mx = max(d["kv_bytes_formula"].max(), d["kv_bytes_measured"].max()) / 2**20
    ax[1].plot([0, mx], [0, mx], "k--", lw=0.8, label="y = x")
    ax[1].set(xlabel="formula: 2*layers*kv_heads*head_dim*2B*T (MiB)", ylabel="measured cache tensors (MiB)", title="KV cache size: formula vs measured")
    ax[1].legend()
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "kv_memory.png", dpi=150)
    log(f"wrote {FIG / 'kv_time.png'}, {FIG / 'kv_memory.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "plot", "all"], default="run")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lens", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--news", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="warm-up 1, repeats 3, lengths 128 512, new tokens 16 64")
    ap.add_argument("--out", default=str(OUT / "kv_bench.csv"))
    a = ap.parse_args()
    if a.quick:
        a.warmup, a.repeats, a.lens, a.news = 1, 3, [128, 512], [16, 64]
    if a.mode in ("run", "all"):
        run(a)
    if a.mode in ("plot", "all"):
        plot(a)


if __name__ == "__main__":
    main()
