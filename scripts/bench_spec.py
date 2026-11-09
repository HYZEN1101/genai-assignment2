"""Phase 6 (F1-F3): check + benchmark OUR greedy speculative decoder against OUR KV-cache greedy decoder.

  python scripts/bench_spec.py --mode check --adapter weights/adapter          # equality on real val task records (T4, ~3-6 min)
  python scripts/bench_spec.py --mode bench --adapter weights/adapter          # latency grid, RESUMABLE (T4, ~15-25 min)
  python scripts/bench_spec.py --mode plot                                     # CPU only: CSV -> summary table + figures
  add --draft-adapter DIR to use a task-tuned draft; --tag NAME labels the rows (default: auto)

Target = Qwen2.5-1.5B-Instruct + merged LoRA (same loading procedure as every phase); draft = Qwen2.5-0.5B-Instruct
(base unless --draft-adapter). Both fp16 on the same GPU. Baseline = generate_greedy_kv (P5), never HF generate.
Workloads: task   = real validation prompts, EOS enabled, max 64 new tokens (outputs ~20 tokens)
           free   = 3 short chat prompts asking for explanatory prose, EOS enabled, max 128 new tokens
           forced = length-controlled 512-token real-text prompt, EOS disabled, exactly 128 new tokens (not task-valid)
Outputs: results/benchmarks/spec_check.csv, spec_bench.csv, spec_summary.csv, spec_env.json, results/logs/bench_spec.log,
         results/figures/spec_*.png
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialmatch.decoding import (check_compatible, generate_greedy_kv, generate_speculative,  # noqa: E402
                                 tokenizers_identical)
from trialmatch.decoding.bench import MB, _run_with_memory  # noqa: E402
from trialmatch.decoding.equiv import first_divergence, length_prompt  # noqa: E402
from trialmatch.pipeline.predictor import load_model  # noqa: E402

OUT = ROOT / "results" / "benchmarks"
FIG = ROOT / "results" / "figures"
LOG = ROOT / "results" / "logs" / "bench_spec.log"
FREE_PROMPTS = [
    "Explain in a few sentences what inclusion and exclusion criteria are in a clinical trial and why they matter.",
    "Describe how a doctor decides whether a patient with type 2 diabetes might be eligible for a drug trial.",
    "Write a short paragraph on why missing information in a patient record should not be treated as evidence of eligibility.",
]


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


def med(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


def workloads(tok, dev, a):
    """-> list of (workload, item_id, input_ids[1,L], max_new, eos_ids or None)."""
    val = [json.loads(l) for l in open(ROOT / "data" / "processed" / "val.jsonl") if l.strip()]
    eos = [tok.convert_tokens_to_ids("<|im_end|>")]
    items = []
    for i, r in enumerate(val[: a.n_task]):
        ids = torch.tensor([tok(r["prompt"], add_special_tokens=False).input_ids], dtype=torch.long)
        items.append(("task", f"val{i}", ids, 64, eos))
    for i, q in enumerate(FREE_PROMPTS[: a.n_free]):
        txt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tok(txt, add_special_tokens=False).input_ids], dtype=torch.long)
        items.append(("free", f"free{i}", ids, 128, eos))
    if a.forced:
        items.append(("forced", "L512", length_prompt(tok, [r["prompt"] for r in val], 512), 128, None))
    return [(w, n, ids.to(dev), mx, e) for w, n, ids, mx, e in items]


def load_pair(a):
    tgt, tok = load_model(a.model, adapter=a.adapter, dtype=a.dtype)
    drf, tok_d = load_model(a.draft, adapter=a.draft_adapter, dtype=a.dtype)
    comp = check_compatible(tgt, drf)
    same_tok = tokenizers_identical(tok, tok_d)
    dev = next(tgt.parameters()).device
    tag = a.tag or (("ft1.5B" if a.adapter else "base1.5B") + "+" + ("ft0.5B" if a.draft_adapter else "base0.5B"))
    env = {"tag": tag, "target": a.model, "target_adapter": a.adapter, "draft": a.draft, "draft_adapter": a.draft_adapter,
           "torch": torch.__version__, "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "dtype": str(next(tgt.parameters()).dtype), "vocab_check": comp, "tokenizers_identical": same_tok,
           "target_params": sum(p.numel() for p in tgt.parameters()), "draft_params": sum(p.numel() for p in drf.parameters()),
           "weights_mb_both": torch.cuda.memory_allocated(dev) / MB if dev.type == "cuda" else None,
           "warmup": a.warmup, "repeats": a.repeats, "ks": a.ks}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(env, open(OUT / "spec_env.json", "w"), indent=1)
    log("env", env)
    assert same_tok and comp["equal"], "draft/target are NOT token-compatible"
    return tgt, drf, tok, dev, tag


# ----------------------------------------------------------------------------------------------------------------------
def check(a):
    """F3 equality: spec(k) == greedy_kv on every item, plus a logit-gap diagnosis of any mismatch."""
    tgt, drf, tok, dev, tag = load_pair(a)
    out = OUT / "spec_check.csv"
    rows = read_rows(out)
    done = {(r["tag"], r["item"], int(r["k"])) for r in rows}
    n_bad = 0
    for w, name, ids, mx, eos in workloads(tok, dev, a):
        ref, _ = generate_greedy_kv(tgt, ids, mx, eos_id=eos)
        for k in a.ks:
            if (tag, name, k) in done:
                continue
            got, st = generate_speculative(tgt, drf, ids, mx, k=k, eos_id=eos, check_cache=True)
            div = first_divergence(ref, got)
            row = {"tag": tag, "workload": w, "item": name, "k": k, "prompt_len": ids.shape[1], "new_tokens": len(got),
                   "equal": div is None, "first_div": -1 if div is None else div,
                   "acceptance_rate": st.acceptance_rate, "rounds": st.rounds, "dtype": a.dtype or "auto"}
            if div is not None:
                n_bad += 1
                # diagnosis: top-2 logit gap of the target at the divergent step (near-tie => fp16 shape effect)
                import torch as T
                from trialmatch.decoding.common import forward_with_cache, new_cache
                pre = ids if div == 0 else T.cat([ids, T.tensor([ref[:div]], device=dev)], 1)
                c = new_cache(tgt)
                lg = forward_with_cache(tgt, pre, c, start=0)[0, -1].float().topk(2)
                row["gap_at_div"] = float(lg.values[0] - lg.values[1])
                row["ref_tok"], row["spec_tok"] = ref[div] if div < len(ref) else -1, got[div] if div < len(got) else -1
            rows.append(row)
            write_rows(out, rows)
            log(f"[check] {name:7s} k={k} equal={row['equal']} acc={st.acceptance_rate:.2f} rounds={st.rounds}"
                + ("" if div is None else f"  MISMATCH at {div}, gap {row['gap_at_div']:.4f}"))
    log(f"[check] done, {n_bad} mismatching (item,k) pairs this session -> {out}")


# ----------------------------------------------------------------------------------------------------------------------
def bench_item(tgt, drf, ids, mx, eos, k, warmup, repeats, dev):
    rec = {"greedy": [], "spec": []}
    toks = {}
    for rep in range(warmup + repeats):
        (g, gst), gpk, gab = _run_with_memory(lambda: generate_greedy_kv(tgt, ids, mx, eos_id=eos), dev)
        (s, sst), spk, sab = _run_with_memory(lambda: generate_speculative(tgt, drf, ids, mx, k=k, eos_id=eos), dev)
        toks["greedy"], toks["spec"] = g, s
        if rep >= warmup:                                           # warm-up executed, never recorded
            rec["greedy"].append((gst.as_dict(), gpk, gab))
            rec["spec"].append((sst.as_dict(), spk, sab))
    div = first_divergence(toks["greedy"], toks["spec"])
    row = {"prompt_len": int(ids.shape[1]), "k": k, "new_tokens": len(toks["greedy"]), "tokens_equal": div is None,
           "first_div": -1 if div is None else div, "warmup": warmup, "repeats": repeats}
    for p, rs in rec.items():
        for key in ("prefill_ms", "decode_ms", "total_ms", "e2e_tok_s"):
            row[f"{p}_{key}_median"] = med([r[0][key] for r in rs])
        row[f"{p}_peak_mb_median"] = med([r[1] for r in rs])
        row[f"{p}_peak_above_weights_mb_median"] = med([r[2] for r in rs])
    sd = rec["spec"][-1][0]
    row.update(acceptance_rate=sd["acceptance_rate"], rounds=sd["rounds"], proposed=sd["proposed"], accepted=sd["accepted"],
               target_forwards=sd["target_forwards"], draft_forwards=sd["draft_forwards"],
               mean_accepted_per_round=sd["mean_accepted_per_round"], tokens_per_target_forward=sd["tokens_per_target_forward"])
    row["speedup_total"] = row["greedy_total_ms_median"] / row["spec_total_ms_median"]
    return row


def bench(a):
    tgt, drf, tok, dev, tag = load_pair(a)
    out = OUT / "spec_bench.csv"
    done = {(r["tag"], r["item"], int(r["k"])) for r in read_rows(out)}
    for w, name, ids, mx, eos in workloads(tok, dev, a):
        for k in a.ks:
            if (tag, name, k) in done:
                log(f"skip {name} k={k}")
                continue
            try:
                row = bench_item(tgt, drf, ids, mx, eos, k, a.warmup, a.repeats, dev)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                row = {"prompt_len": int(ids.shape[1]), "k": k, "status": f"OOM: {str(e)[:80]}"}
            row.update(tag=tag, workload=w, item=name, status=row.get("status", "ok"))
            write_rows(out, read_rows(out) + [row])
            if row["status"] == "ok":
                log(f"{name:7s} k={k} eq={row['tokens_equal']} acc={row['acceptance_rate']:.2f} tok/tgt-fwd={row['tokens_per_target_forward']:.2f} "
                    f"| ms greedy/spec {row['greedy_total_ms_median']:.0f}/{row['spec_total_ms_median']:.0f} | speedup {row['speedup_total']:.2f}x "
                    f"| peak above weights MB {row['greedy_peak_above_weights_mb_median']:.0f}/{row['spec_peak_above_weights_mb_median']:.0f}")
            else:
                log(name, k, row["status"])
    log(f"wrote {out}")


# ----------------------------------------------------------------------------------------------------------------------
def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(OUT / "spec_bench.csv")
    df = df[df["status"] == "ok"]
    if a.tag:
        df = df[df["tag"] == a.tag]
    rows = []
    for (tag, w, k), g in df.groupby(["tag", "workload", "k"]):
        gt, st_ = g["greedy_total_ms_median"].sum(), g["spec_total_ms_median"].sum()     # pooled latency over items
        ntok = g["new_tokens"].sum()
        rows.append({"tag": tag, "workload": w, "k": k, "items": len(g), "tokens": ntok,
                     "all_tokens_equal": bool(g["tokens_equal"].all()),
                     "acceptance_rate": g["accepted"].sum() / max(g["proposed"].sum(), 1),
                     "tokens_per_target_forward": ntok / g["target_forwards"].sum(),
                     "greedy_ms_per_item": gt / len(g), "spec_ms_per_item": st_ / len(g),
                     "greedy_tok_s": ntok / (gt / 1000), "spec_tok_s": ntok / (st_ / 1000), "speedup": gt / st_,
                     "greedy_peak_above_weights_mb": g["greedy_peak_above_weights_mb_median"].max(),
                     "spec_peak_above_weights_mb": g["spec_peak_above_weights_mb_median"].max()})
    sm = pd.DataFrame(rows).sort_values(["tag", "workload", "k"])
    sm.to_csv(OUT / "spec_summary.csv", index=False)
    log(sm.round(3).to_string(index=False))
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for (tag, w), g in sm.groupby(["tag", "workload"]):
        lab = f"{w} ({tag})"
        ax[0].plot(g["k"], g["acceptance_rate"], marker="o", label=lab)
        ax[1].plot(g["k"], g["tokens_per_target_forward"], marker="o", label=lab)
        ax[2].plot(g["k"], g["speedup"], marker="o", label=lab)
    ax[2].axhline(1.0, color="k", lw=0.6)
    for x, t in zip(ax, ["draft acceptance rate", "tokens per target forward", "speedup vs KV greedy (>1 = faster)"]):
        x.set(xlabel="draft length k", title=t)
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "spec_overview.png", dpi=150)
    log("wrote", FIG / "spec_overview.png", OUT / "spec_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "bench", "plot", "all"], default="all")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--draft-adapter", dest="draft_adapter", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-task", type=int, default=12, help="number of validation task records")
    ap.add_argument("--n-free", type=int, default=3)
    ap.add_argument("--no-forced", dest="forced", action="store_false")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    if a.mode in ("check", "all"):
        check(a)
    if a.mode in ("bench", "all"):
        bench(a)
    if a.mode in ("plot", "all"):
        plot(a)


if __name__ == "__main__":
    main()
