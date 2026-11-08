"""Run the fine-tuned model on hand-written (unseen) cases. Needs the model: run it in COLAB on a T4 (or CPU, slower).

  python scripts/run_cases.py --cases cases/case1.txt --adapter weights/adapter [--model base]
Case file format (sections in order; numbered note starts at 0):
  NOTE:
  0. first sentence
  1. second sentence
  INCLUSION:
  one criterion per line
  EXCLUSION:
  one criterion per line
Output: console table + results/app_tests/<case>_<model>.json (label, cited ids, evidence text, explanation, overall outcome).
Same prompt, decoding (greedy, 64 tokens, batch 8) and parser as the evaluation; explanation text is templated from cited sentences.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_case_file(text: str) -> dict:
    """-> {note: str, criteria: [(type, text)]}. Raises ValueError with a readable message on malformed input."""
    sec, buf = None, {"NOTE": [], "INCLUSION": [], "EXCLUSION": []}
    for line in text.splitlines():
        key = line.strip().rstrip(":").upper()
        if key in buf and line.strip().endswith(":"):
            sec = key
            continue
        if sec and line.strip():
            buf[sec].append(line.strip())
    if not buf["NOTE"]:
        raise ValueError("missing NOTE: section")
    crit = [("inclusion", c) for c in buf["INCLUSION"]] + [("exclusion", c) for c in buf["EXCLUSION"]]
    if not crit:
        raise ValueError("no criteria (INCLUSION:/EXCLUSION: sections are empty)")
    return {"note": " ".join(buf["NOTE"]), "criteria": crit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", choices=["ft", "base"], default="ft")
    ap.add_argument("--adapter", default="weights/adapter")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()

    from trialmatch.data.prompts import build_prompt, split_numbered_note
    from trialmatch.data.schema import parse_output
    from trialmatch.eval.explain import explain_row
    from trialmatch.pipeline.aggregate import aggregate
    from trialmatch.pipeline.predictor import Predictor, load_model

    case = parse_case_file(Path(a.cases).read_text(encoding="utf-8"))
    sents = split_numbered_note(case["note"])
    model, tok = load_model(a.model_id, adapter=a.adapter if a.model == "ft" else None)
    pred = Predictor(model, tok, max_new_tokens=64, batch_size=8)
    prompts = [build_prompt(tok, sents, t, c) for t, c in case["criteria"]]
    gens = pred.generate_batch(prompts)
    rows = []
    for (t, c), g in zip(case["criteria"], gens):
        p = parse_output(g["text"], t, len(sents))
        row = {"criterion_type": t, "criterion": c, "pred_label": p["label"], "valid": p["valid"],
               "pred_evidence_ids": p["evidence_ids"], "unknown_ids": p["unknown_ids"], "raw_output": g["text"]}
        row.update(explain_row(row, sents))
        rows.append(row)
    agg = aggregate(rows)
    print(f"\nNOTE ({len(sents)} sentences):")
    for s in sents:
        print(f"  [{s['id']}] {s['text']}")
    for r in rows:
        ids = r["pred_evidence_ids"]
        print(f"\n[{r['criterion_type']}] {r['criterion']}\n  label: {r['pred_label']} (valid={r['valid']})  cited: {ids}")
        for e in r["evidence"]:
            print(f"    [{e['id']}] {e['text']}")
        print(f"  explanation: {r['explanation']}")
    print(f"\nOVERALL (preliminary, rule-based): {agg['outcome']}\n  {agg['disclaimer']}")
    out = ROOT / "results" / "app_tests" / f"{Path(a.cases).stem}_{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "adapter": a.adapter if a.model == "ft" else None, "sentences": sents,
                               "results": rows, "overall": agg}, indent=2))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
