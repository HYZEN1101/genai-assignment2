"""Error-analysis candidates from saved predictions (Phase 3 step 5). Tags are diagnostic; YOU write the causal analysis.

  python scripts/error_cases.py --split test --pred results/predictions/test_ft.jsonl --per-tag 3
Writes results/tables/<split>_error_tags.csv and docs/ERROR_CANDIDATES_<split>.md (full note with gold/predicted ids marked,
criterion, gold vs predicted, tags, blank 'cause hypothesis'). Pick >= 8 DISTINCT errors (different patient/criterion/tag mix).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trialmatch.eval.errors import TAG_DESCRIPTIONS, is_label_error, pick_candidates, tag_counts, tag_row  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--per-tag", type=int, default=3)
    a = ap.parse_args()
    recs = {r["annotation_id"]: r for r in map(json.loads, open(ROOT / "data" / "processed" / f"{a.split}.jsonl", encoding="utf-8"))}
    rows = [json.loads(l) for l in open(ROOT / a.pred if not Path(a.pred).is_absolute() else a.pred, encoding="utf-8") if l.strip()]
    counts = tag_counts(rows, recs)
    n_err = sum(is_label_error(r) for r in rows)
    out_csv = ROOT / "results" / "tables" / f"{a.split}_error_tags.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag", "count", "description"])
        for t, d in TAG_DESCRIPTIONS.items():
            w.writerow([t, counts.get(t, 0), d])
    cands = pick_candidates(rows, recs, a.per_tag)
    by_id = {r["annotation_id"]: r for r in rows}
    md = [f"# Error-analysis candidates ({a.split}; {n_err} label errors of {len(rows)}; tags auto-computed)\n",
          "Choose >= 8 DISTINCT errors and write the cause hypothesis yourself. Gold ids marked G, predicted ids marked P.\n"]
    for c in cands:
        r, rec = by_id[c["annotation_id"]], recs[c["annotation_id"]]
        g, p = set(r["gold_evidence_ids"]), set(r["pred_evidence_ids"]) if r["valid"] else set()
        md.append(f"\n## annotation {r['annotation_id']} | {r['patient_id']} | {r['trial_id']} | {r['criterion_type']} | picked for `{c['primary_tag']}`")
        md.append(f"- Criterion: {rec['criterion_text']}")
        md.append(f"- Gold: **{r['gold_label']}** evidence {sorted(g)} | Predicted: **{r['pred_label']}** evidence {sorted(p)} | valid={r['valid']}")
        md.append(f"- Tags: {', '.join(tag_row(r, rec)) or '(none)'}")
        md.append(f"- Raw output: `{r['raw_output'][:200]}`")
        md.append("- Note:")
        for s in rec["patient_sentences"]:
            mark = ("G" if s["id"] in g else "-") + ("P" if s["id"] in p else "-")
            md.append(f"  - [{s['id']}] ({mark}) {s['text']}")
        md.append("- Cause hypothesis: TODO")
    out_md = ROOT / "docs" / f"ERROR_CANDIDATES_{a.split}.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"label errors: {n_err}/{len(rows)}; candidates: {len(cands)}")
    for t, d in TAG_DESCRIPTIONS.items():
        print(f"  {t:24s} {counts.get(t, 0):4d}")
    print(f"wrote {out_csv.relative_to(ROOT)} and {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
