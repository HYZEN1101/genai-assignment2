"""Write results/logs/FREEZE.json + FREEZE.txt BEFORE the single test run (Phase 3).

Records: git commit, test-data sha256, adapter weight sha256, decoding config, library versions, UTC time.
Refuses to freeze if tracked files have uncommitted changes (commit + push first), unless --allow-dirty.
run_eval.py --split test then refuses to run unless FREEZE.json exists and still matches (data + adapter hashes).
  COLAB:  python scripts/freeze.py --adapter weights/adapter
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="weights/adapter")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing FREEZE.json (documents a re-freeze)")
    a = ap.parse_args()
    out = ROOT / "results" / "logs" / "FREEZE.json"
    if out.exists() and not a.force:
        sys.exit(f"REFUSED: {out} exists (already frozen). Use --force only if you must re-freeze, and say so in the report.")
    dirty = git("status", "--porcelain", "-uno")
    if dirty and not a.allow_dirty:
        sys.exit("REFUSED: uncommitted changes in tracked files; commit + push first:\n" + dirty)
    ad = ROOT / a.adapter if not Path(a.adapter).is_absolute() else Path(a.adapter)
    weights = ad / "adapter_model.safetensors"
    if not weights.exists():
        sys.exit(f"adapter weights not found: {weights} (run: python scripts/download_weights.py)")
    cfg = yaml.safe_load(open(ROOT / "configs" / "eval.yaml"))
    test = ROOT / "data" / "processed" / "test.jsonl"
    info = {
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty_tracked": bool(dirty),
        "test_data_sha256": sha256_file(test), "n_test_records": sum(1 for l in open(test) if l.strip()),
        "adapter_dir": a.adapter, "adapter_weights_sha256": sha256_file(weights),
        "adapter_config_sha256": sha256_file(ad / "adapter_config.json"),
        "eval_config": cfg, "decoding": {"strategy": "greedy", "max_new_tokens": cfg["max_new_tokens"], "repetition_penalty": 1.0,
                                         "batch_size": cfg["batch_size"], "padding": "left"},
        "split_manifest_sha256": sha256_file(ROOT / "data" / "split_manifest.json"),
        "selected_checkpoint": "run C, epoch_5 (chosen on VALIDATION macro-F1 in Phase 2)",
    }
    try:
        import torch, transformers
        info["torch"], info["transformers"] = torch.__version__, transformers.__version__
    except Exception:
        pass
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    (out.with_suffix(".txt")).write_text(
        "FREEZE (written before the single test run; nothing was tuned on test)\n"
        + "\n".join(f"{k}: {v}" for k, v in info.items()) + "\n")
    print(json.dumps(info, indent=2))
    print(f"\nFrozen. Next: python scripts/run_eval.py --split test --model base|ft ... --allow-test")


if __name__ == "__main__":
    main()
