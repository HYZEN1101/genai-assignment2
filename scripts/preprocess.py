"""Phase 1 step 3: write data/processed/{train,val,test}.jsonl + label tables. Run: python scripts/preprocess.py"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import yaml
from transformers import AutoTokenizer

from trialmatch.data.load import REPO_ROOT, load_raw
from trialmatch.data.preprocess import build_records
from trialmatch.data.split import SPLITS, assign_split, verify_manifest

cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
df = load_raw()
manifest = json.loads((REPO_ROOT / "data" / "split_manifest.json").read_text())
verify_manifest(manifest, df)
df["split"] = assign_split(df, manifest)
tok = AutoTokenizer.from_pretrained(cfg["model_id"])
out_dir = REPO_ROOT / "data" / "processed"; out_dir.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)

rows, all_dropped = [], {}
for sp in SPLITS:
    recs, dropped = build_records(df[df.split == sp], tok, cfg["max_len"], cfg["reserve_tokens"])
    with open(out_dir / f"{sp}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    all_dropped[sp] = dropped
    trunc = sum(1 for r in recs if r["truncated_ids"])
    print(f"{sp:5s}: {len(recs):4d} records written, {len(dropped)} dropped, {trunc} truncated, "
          f"max prompt tokens {max(r['n_prompt_tokens'] for r in recs)}")
    for (t, l), n in Counter((r["criterion_type"], r["gold_label"]) for r in recs).items():
        rows.append({"split": sp, "criterion_type": t, "label": l, "n": n})
    for t in ("inclusion", "exclusion"):
        sub = [r for r in recs if r["criterion_type"] == t]
        rows.append({"split": sp, "criterion_type": t, "label": "__empty_gold_evidence__",
                     "n": sum(1 for r in sub if not r["gold_evidence_ids"])})
tab = pd.DataFrame(rows).sort_values(["split", "criterion_type", "label"])
tab.to_csv(REPO_ROOT / "results" / "tables" / "label_distribution.csv", index=False)
(REPO_ROOT / "results" / "logs" / "preprocess_dropped.json").write_text(json.dumps(all_dropped, indent=2))
print("\n", tab.pivot_table(index=["criterion_type", "label"], columns="split", values="n", fill_value=0).to_string())
print("\nexample prompt (first train record):\n" + "-" * 60)
first = json.loads((out_dir / "train.jsonl").read_text().splitlines()[0])
print(first["prompt"]); print("-" * 60); print("TARGET:", first["target"])
