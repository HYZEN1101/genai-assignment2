"""Print per-class P/R/F1 + predicted-label counts for every results/tables/quick_val_*.json (selection aid, Phase 2).
   python scripts/show_quick.py [tag_substring]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
flt = sys.argv[1] if len(sys.argv) > 1 else ""
for p in sorted((ROOT / "results" / "tables").glob("quick_val_*.json")):
    d = json.loads(p.read_text())
    m, meta = d["metrics"], d["meta"]
    if flt not in meta["tag"] or meta["tag"].startswith("hubcheck"):
        continue
    print(f"\n=== {meta['tag']}  acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  valid {m['valid_rate']:.3f}  "
          f"ev-F1(non-empty gold) {m['evidence_nonempty_gold']['micro_f1']:.3f}")
    print(f"{'class':26s}{'P':>7s}{'R':>7s}{'F1':>7s}{'gold n':>8s}")
    for c, v in m["per_class"].items():
        print(f"{c:26s}{v['precision']:7.2f}{v['recall']:7.2f}{v['f1']:7.2f}{v['support']:8d}")
    print("predicted:", m["pred_label_counts"])
