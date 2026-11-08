"""Phase 5 (D2): greedy-token equivalence of the explicit KV-cache decoder vs the no-cache reference on a REAL model.

  python scripts/check_kv_equiv.py --model Qwen/Qwen2.5-0.5B-Instruct                         # CPU ok (slow), fp32
  python scripts/check_kv_equiv.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter weights/adapter --recheck-fp32   # T4

Part 1: length-controlled prompts (prompt_len:new_tokens combos), EOS disabled so exactly N tokens are compared.
Part 2: the actual task (first --task-records validation records, EOS stopping, max_new_tokens 64): cached == uncached == HF generate.
On a mismatch it prints the first divergent step and the top-2 logit gap of both paths; with --recheck-fp32 the model is
reloaded in fp32 and the mismatching combos are rerun (a numerical near-tie flip disappears; a logic bug would not).
Writes results/benchmarks/kv_equiv.csv, results/benchmarks/kv_equiv_detail.json, results/logs/kv_equiv.log.
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
from trialmatch.decoding.equiv import compare_paths, first_divergence, length_prompt  # noqa: E402
from trialmatch.decoding.greedy_kv import generate_greedy_kv  # noqa: E402
from trialmatch.decoding.greedy_nocache import generate_greedy_nocache  # noqa: E402
from trialmatch.pipeline.predictor import END_TOKEN, load_model  # noqa: E402

OUT = ROOT / "results" / "benchmarks"
LOG = ROOT / "results" / "logs" / "kv_equiv.log"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def hf_reference(model, tok, ids: torch.Tensor, n_new: int, eos_ids) -> list[int]:
    """HF generate(), greedy -- REFERENCE ONLY (never the deliverable).

    Qwen's shipped generation_config has repetition_penalty=1.05 (+ sampling flags). generate() treats a user value equal to the
    library default (1.0) as "unset" and falls back to the MODEL's config, so passing repetition_penalty=1.0 is silently ignored.
    We therefore neutralise the model's own generation_config for the duration of the call and restore it afterwards.
    """
    gc = model.generation_config
    keep = {k: getattr(gc, k, None) for k in ("repetition_penalty", "eos_token_id", "pad_token_id", "do_sample", "temperature", "top_p", "top_k")}
    try:
        gc.repetition_penalty, gc.do_sample = 1.0, False
        gc.temperature = gc.top_p = gc.top_k = None
        gc.eos_token_id = list(eos_ids) if eos_ids else None
        gc.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=n_new,
                                 min_new_tokens=0 if eos_ids else n_new)
    finally:
        for k, v in keep.items():
            setattr(gc, k, v)
    new = out[0, ids.shape[1]:].tolist()
    if eos_ids:
        for i, t in enumerate(new):
            if t in eos_ids:
                return new[: i + 1]
    return new


def run_combos(model, tok, texts, combos, dev, tag, dtype):
    rows = []
    for L, N in combos:
        ids = length_prompt(tok, texts, L).to(dev)
        r = compare_paths(model, ids, N)
        try:
            hf = hf_reference(model, tok, ids, N, None)
            hf_equal = hf == r["tokens_kv"]
        except Exception as e:  # reference is optional evidence, never a reason to crash the check
            hf_equal = f"error: {type(e).__name__}"
        row = {"part": "length_combo", "tag": tag, "dtype": dtype, "prompt_len": L, "new_tokens": N, "equal": r["equal"],
               "first_div": -1 if r["first_div"] is None else r["first_div"], "hf_generate_equal": hf_equal,
               "max_logit_absdiff_before_div": r["max_logit_absdiff"], "logit_absdiff_step0": r["logit_absdiff_step0"],
               "gap_ref_at_div": r["gap_ref_at_div"], "gap_kv_at_div": r["gap_kv_at_div"], "min_gap_ref": r["min_gap_ref"]}
        rows.append(row)
        log(f"[{tag}/{dtype}] L={L:5d} N={N:4d} equal={r['equal']} first_div={row['first_div']} hf_equal={hf_equal} "
            f"max|dlogit|={r['max_logit_absdiff']:.4g} gap@div ref/kv={r['gap_ref_at_div']}/{r['gap_kv_at_div']}")
    return rows


def run_task(model, tok, records, dev, tag, dtype, max_new=64):
    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    n_eq = n_hf = 0
    mism = []
    for r in records:
        ids = torch.tensor([tok(r["prompt"], add_special_tokens=False).input_ids], device=dev)
        a, _ = generate_greedy_nocache(model, ids, max_new, eos_id=end_id)
        b, _ = generate_greedy_kv(model, ids, max_new, eos_id=end_id, check_cache=True)
        try:
            h = hf_reference(model, tok, ids, max_new, [end_id])
        except Exception:
            h = None
        n_eq += a == b
        n_hf += (h == b) if h is not None else 0
        if a != b:
            mism.append({"annotation_id": r["annotation_id"], "first_div": first_divergence(a, b)})
    row = {"part": "task_records", "tag": tag, "dtype": dtype, "n_records": len(records), "n_equal_nocache_vs_kv": n_eq,
           "n_equal_hf_vs_kv": n_hf, "mismatches": json.dumps(mism)}
    log(f"[{tag}/{dtype}] task records: kv==nocache {n_eq}/{len(records)}, kv==HF generate {n_hf}/{len(records)}, mismatches {mism}")
    return [row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default=None, help="label for the CSV (default: derived from model/adapter)")
    ap.add_argument("--combos", default="64:16,256:32,512:64,1024:128", help="prompt_len:new_tokens,...")
    ap.add_argument("--task-records", type=int, default=20)
    ap.add_argument("--recheck-fp32", action="store_true", help="reload in fp32 and rerun fp16 mismatches (and all combos if cheap)")
    ap.add_argument("--dtype", default=None, help="float16|float32 (default: fp16 on cuda, fp32 on cpu)")
    ap.add_argument("--out", default=str(OUT / "kv_equiv.csv"))
    a = ap.parse_args()

    combos = [tuple(int(x) for x in c.split(":")) for c in a.combos.split(",")]
    tag = a.tag or (Path(a.adapter).name + "+" if a.adapter else "") + a.model.split("/")[-1]
    val = read_jsonl(ROOT / "data" / "processed" / "val.jsonl")
    texts = [r["prompt"] for r in val]
    dev_name = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok = load_model(a.model, adapter=a.adapter, dtype=a.dtype)
    dev = next(model.parameters()).device
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    log(f"== {tag} on {dev_name}, dtype {dtype}, torch {torch.__version__}")
    rows = run_combos(model, tok, texts, combos, dev, tag, dtype)
    rows += run_task(model, tok, val[: a.task_records], dev, tag, dtype)

    bad = [(r["prompt_len"], r["new_tokens"]) for r in rows if r["part"] == "length_combo" and r["equal"] is not True]
    if a.recheck_fp32 and dtype != "float32":
        log(f"fp16 mismatching combos: {bad or 'none'}; reloading in fp32 for the recheck")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        model, tok = load_model(a.model, adapter=a.adapter, dtype="float32")
        dev = next(model.parameters()).device
        rows += run_combos(model, tok, texts, combos, dev, tag, "float32")
        rows += run_task(model, tok, val[: a.task_records], dev, tag, "float32")

    prev = []
    if Path(a.out).exists():                      # merge with earlier runs (e.g. 0.5B then fine-tuned 1.5B) under the union of columns
        prev = [dict(r) for r in csv.DictReader(open(a.out))]
    allrows = prev + rows
    keys = sorted({k for r in allrows for k in r})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)
    json.dump(rows, open(Path(a.out).with_name("kv_equiv_detail.json"), "w"), indent=1, default=str)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
