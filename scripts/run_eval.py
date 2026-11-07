"""Run greedy predictions on a split with the base model or base+adapter, save predictions + metadata + quick metrics.

Examples (COLAB, from the repo root, T4 runtime):
  python scripts/run_eval.py --split val --model base --out results/predictions/val_base.jsonl
  python scripts/run_eval.py --split val --model ft --adapter /content/drive/MyDrive/genai_a2/runs/A/best \
         --out results/predictions/val_A_best.jsonl

SAFETY: --split test is refused unless --allow-test is given AND the output file does not exist yet
(the test set is evaluated once, in Phase 3, after the freeze).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trialmatch.eval.quick_metrics import summarize  # noqa: E402
from trialmatch.pipeline.predictor import Predictor, load_model  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_freeze(split: str, data_path: Path, adapter: str | None) -> dict | None:
    """Test split only: FREEZE.json (scripts/freeze.py) must exist and match the data and adapter weights."""
    if split != "test":
        return None
    fp = ROOT / "results" / "logs" / "FREEZE.json"
    if not fp.exists():
        sys.exit("REFUSED: results/logs/FREEZE.json missing. Run scripts/freeze.py (after committing) before the test run.")
    fr = json.loads(fp.read_text())
    if fr["test_data_sha256"] != sha256_file(data_path):
        sys.exit("REFUSED: test.jsonl changed since the freeze.")
    if adapter:
        ad = Path(adapter) if Path(adapter).is_absolute() else ROOT / adapter
        w = ad / "adapter_model.safetensors"
        if not w.exists() or sha256_file(w) != fr["adapter_weights_sha256"]:
            sys.exit("REFUSED: adapter weights differ from the frozen adapter (use the Hub-downloaded weights/adapter).")
    return fr


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "eval.yaml"))
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--model", choices=["base", "ft"], required=True)
    ap.add_argument("--adapter", default=None, help="local dir or HF repo id (required for --model ft)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="name for results/tables/quick_<split>_<tag>.json (default: out stem)")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N records only")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    bs = a.batch_size or int(cfg.get("batch_size", 8))
    mnt = a.max_new_tokens or int(cfg.get("max_new_tokens", 64))
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.split == "test":
        if not a.allow_test:
            sys.exit("REFUSED: test split requires --allow-test (final run only, Phase 3).")
        if out.exists():
            sys.exit(f"REFUSED: {out} already exists; the test set is evaluated once.")
    if a.model == "ft" and not a.adapter:
        sys.exit("--model ft needs --adapter")
    if a.model == "base":
        a.adapter = None

    data_path = ROOT / "data" / "processed" / f"{a.split}.jsonl"
    records = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    if a.limit:
        records = records[: a.limit]
    freeze = verify_freeze(a.split, data_path, a.adapter)
    print(f"[eval] split={a.split} n={len(records)} model={a.model} adapter={a.adapter} bs={bs} max_new_tokens={mnt}")

    t0 = time.time()
    model, tok = load_model(cfg["model_id"], adapter=a.adapter)
    print(f"[eval] model loaded in {time.time() - t0:.0f}s")
    pred = Predictor(model, tok, max_new_tokens=mnt, batch_size=bs)
    rows, timing = pred.predict_records(records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows)
    tag = a.tag or out.stem
    meta = {
        "tag": tag, "split": a.split, "n": len(rows), "model_id": cfg["model_id"], "mode": a.model, "adapter": a.adapter,
        "decoding": {"strategy": "greedy", "max_new_tokens": mnt, "repetition_penalty": 1.0, "batch_size": bs,
                     "padding": "left", "eos": "<|im_end|>"},
        "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
        "adapter_merged": bool(a.adapter), "data_file": str(data_path.relative_to(ROOT)), "data_sha256": sha256_file(data_path),
        "git": git_hash(), "torch": torch.__version__, "transformers": transformers.__version__,
        "freeze_commit": freeze["git_commit"] if freeze else None, "timing": timing, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    tab = ROOT / "results" / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    (tab / f"quick_{a.split}_{tag}.json").write_text(json.dumps({"meta": meta, "metrics": summary}, indent=2))

    print("\n=== QUICK METRICS (selection tool; Phase 3 is authoritative) ===")
    keys = ["n", "valid_rate", "hit_max_rate", "hallucinated_id_rate", "accuracy", "macro_f1",
            "inclusion_accuracy", "inclusion_macro_f1", "exclusion_accuracy", "exclusion_macro_f1"]
    for k in keys:
        if k in summary:
            v = summary[k]
            print(f"{k:24s} {v:.4f}" if isinstance(v, float) else f"{k:24s} {v}")
    e = summary["evidence_nonempty_gold"]
    print(f"evidence (non-empty gold, n={e['n']}) micro P/R/F1 {e['micro_p']:.3f}/{e['micro_r']:.3f}/{e['micro_f1']:.3f}")
    print("pred label counts:", summary["pred_label_counts"])
    print(f"timing: {timing['seconds']:.0f}s for {timing['n']} records, {timing['new_tokens']} new tokens")
    print("first 3 raw outputs:")
    for r in rows[:3]:
        print("  ", repr(r["raw_output"][:160]), "| gold:", r["gold_label"], r["gold_evidence_ids"])


if __name__ == "__main__":
    main()
