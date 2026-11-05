"""Phase 1 step 4: token-length statistics + truncation analysis from data/processed. Run: python scripts/token_stats.py"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

from trialmatch.data.load import REPO_ROOT
from trialmatch.data.prompts import build_prompt, count_tokens, fit_note_to_budget

cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
tok = AutoTokenizer.from_pretrained(cfg["model_id"])
recs = {sp: [json.loads(l) for l in open(REPO_ROOT / "data" / "processed" / f"{sp}.jsonl")] for sp in ("train", "val", "test")}
CANDS = [512, 768, 1024, 1536, 2048]
reserve = cfg["reserve_tokens"]

rows = []
for sp, rs in recs.items():
    full = np.array([r["n_prompt_tokens_full"] for r in rs]); tgt = np.array([r["n_target_tokens"] for r in rs])
    row = {"split": sp, "n": len(rs), "prompt_p50": np.percentile(full, 50), "prompt_p95": np.percentile(full, 95),
           "prompt_p99": np.percentile(full, 99), "prompt_max": full.max(), "target_max": tgt.max(), "target_mean": tgt.mean()}
    for L in CANDS:  # NAIVE policy (cut from the end, ignoring gold) to show the risk
        n_over = n_gold_cut = 0
        for r in rs:
            if r["n_prompt_tokens_full"] <= L - reserve:
                continue
            n_over += 1
            ctype, ctext = r["criterion_type"], r["criterion_text"]
            kept, dropped = fit_note_to_budget(
                r["patient_sentences"], lambda ss: count_tokens(tok, build_prompt(tok, ss, ctype, ctext)), L - reserve, protect_ids=())
            if set(dropped) & set(r["gold_evidence_ids"]):
                n_gold_cut += 1
        row[f"over_{L}"] = n_over; row[f"gold_cut_naive_{L}"] = n_gold_cut
    rows.append(row)
tab = pd.DataFrame(rows).round(1)
(REPO_ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
tab.to_csv(REPO_ROOT / "results" / "tables" / "token_stats.csv", index=False)
print(tab.T.to_string())
print(f"\nCHOSEN max_len={cfg['max_len']}: records actually truncated in data/processed:",
      {sp: sum(1 for r in rs if r["truncated_ids"]) for sp, rs in recs.items()})
print("longest target (tokens):", max(r["n_target_tokens"] for rs in recs.values() for r in rs),
      "| most evidence ids on one record:", max(len(r["gold_evidence_ids"]) for rs in recs.values() for r in rs))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
for sp, rs in recs.items():
    ax.hist([r["n_prompt_tokens_full"] for r in rs], bins=40, alpha=0.5, label=sp)
ax.axvline(cfg["max_len"] - reserve, color="k", ls="--", label=f"prompt budget ({cfg['max_len']}-{reserve})")
ax.set_xlabel("prompt tokens (untruncated)"); ax.set_ylabel("records"); ax.legend(); ax.set_title("Prompt length by split")
(REPO_ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(REPO_ROOT / "results" / "figures" / "token_lengths.png", dpi=150)
print("figure -> results/figures/token_lengths.png")
