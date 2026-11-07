"""Phase 3 evaluation report from SAVED predictions only (no GPU, no model): tables (CSV + LaTeX), confusion-matrix figures,
before/after comparison, evidence metrics, explained predictions, patient-cluster bootstrap CIs.

  python scripts/make_eval_report.py --split val  --ft results/predictions/val_C_ep5.jsonl --base results/predictions/val_base.jsonl \
         --check-quick results/tables/quick_val_C_ep5.json
  python scripts/make_eval_report.py --split test --ft results/predictions/test_ft.jsonl --base results/predictions/test_base.jsonl

Outputs (prefix = split): results/tables/eval_<split>_{ft,base}.json, <split>_summary|per_class|evidence|before_after|bootstrap.csv (+ .tex),
<split>_confusion_<model>_<scope>.csv, results/figures/confusion_<split>_<model>_<scope>.png, results/predictions/<split>_ft_explained.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trialmatch.eval.explain import explain_row  # noqa: E402
from trialmatch.eval.metrics import effective_pred, classification_report, evaluate_rows  # noqa: E402

TAB, FIG, PRED = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "results" / "predictions"
SCOPES = ("overall", "inclusion", "exclusion")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    write_tex(path.with_suffix(".tex"), header, rows)


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("<", "$<$").replace(">", "$>$")


def write_tex(path: Path, header: list[str], rows: list[list]) -> None:
    """Minimal booktabs table body (no jinja dependency). \\input{} it inside a table environment."""
    lines = [r"\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}", r"\toprule",
             " & ".join(_esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(_esc(_fmt(c)) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def confusion_png(cm: dict, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = np.array(cm["matrix"])
    fig, ax = plt.subplots(figsize=(1.2 * len(cm["cols_pred"]) + 2.2, 1.0 * len(cm["rows_gold"]) + 2.2))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(cm["cols_pred"])), cm["cols_pred"], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm["rows_gold"])), cm["rows_gold"], fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=9, color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def bootstrap(rows_by_model: dict[str, list[dict]], n_boot: int, seed: int = 42) -> list[list]:
    """Patient-cluster bootstrap (resample patients with replacement, all their records together). Paired across models."""
    if n_boot <= 0:
        return []
    rng = np.random.default_rng(seed)
    names = list(rows_by_model)
    pats = sorted({r["patient_id"] for r in rows_by_model[names[0]]})
    by_pat = {m: {p: [r for r in rows if r["patient_id"] == p] for p in pats} for m, rows in rows_by_model.items()}
    draws = {m: {"accuracy": [], "macro_f1": []} for m in names}
    for _ in range(n_boot):
        sample = rng.choice(len(pats), size=len(pats), replace=True)
        for m in names:
            rr = [r for k in sample for r in by_pat[m][pats[k]]]
            rep = classification_report([r["gold_label"] for r in rr], [effective_pred(r) for r in rr])
            draws[m]["accuracy"].append(rep["accuracy"]); draws[m]["macro_f1"].append(rep["macro_f1"])
    out = []
    for m in names:
        for k in ("accuracy", "macro_f1"):
            d = np.array(draws[m][k])
            out.append([m, k, float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), n_boot, len(pats)])
    if "ft" in names and "base" in names:
        for k in ("accuracy", "macro_f1"):
            d = np.array(draws["ft"][k]) - np.array(draws["base"][k])
            out.append(["ft-base", k, float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), n_boot, len(pats)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--check-quick", default=None, help="quick_<split>_*.json to cross-check accuracy/macro-F1 against (P2 numbers)")
    a = ap.parse_args()
    sp = a.split
    records = {r["annotation_id"]: r for r in load_jsonl(ROOT / "data" / "processed" / f"{sp}.jsonl")}
    models: dict[str, list[dict]] = {}
    for name, p in (("ft", a.ft), ("base", a.base)):
        if p:
            rows = sorted(load_jsonl(ROOT / p if not Path(p).is_absolute() else Path(p)), key=lambda r: r["annotation_id"])
            assert {r["annotation_id"] for r in rows} == set(records), f"{name}: predictions do not cover exactly the {sp} records"
            models[name] = rows
    if "base" in models:  # identical examples, same order
        assert [r["annotation_id"] for r in models["ft"]] == [r["annotation_id"] for r in models["base"]]

    ev = {m: evaluate_rows(rows) for m, rows in models.items()}
    TAB.mkdir(parents=True, exist_ok=True)
    for m in ev:
        (TAB / f"eval_{sp}_{m}.json").write_text(json.dumps(ev[m], indent=2))

    # ---- tables
    summ, per, evid, conf = [], [], [], []
    for m, e in ev.items():
        for s in SCOPES:
            if s not in e:
                continue
            r = e[s]
            summ.append([m, s, r["n"], r["accuracy"], r["macro_f1"], r["macro_f1_gold_support"], r["n_classes_macro"], r["invalid_rate"]])
            for lab, v in r["per_class"].items():
                per.append([m, s, lab, v["precision"], v["recall"], v["f1"], v["support"], v["n_pred"]])
            cm = r["confusion"]
            write_csv(TAB / f"{sp}_confusion_{m}_{s}.csv", ["gold \\ pred"] + cm["cols_pred"],
                      [[g] + row for g, row in zip(cm["rows_gold"], cm["matrix"])])
            confusion_png(cm, f"{m} | {s} | {sp} (n={r['n']})", FIG / f"confusion_{sp}_{m}_{s}.png")
        for s, key in (("overall", "overall"), ("inclusion", "inclusion"), ("exclusion", "exclusion")):
            if key in e["evidence"]:
                for sub in ("nonempty_gold", "all"):
                    x = e["evidence"][key][sub]
                    evid.append([m, s, sub, x["n"], x["micro_p"], x["micro_r"], x["micro_f1"], x["macro_p"], x["macro_r"], x["macro_f1"]])
    write_csv(TAB / f"{sp}_summary.csv", ["model", "scope", "n", "accuracy", "macro_f1", "macro_f1_gold_support", "n_classes", "invalid_rate"], summ)
    write_csv(TAB / f"{sp}_per_class.csv", ["model", "scope", "label", "precision", "recall", "f1", "support", "n_pred"], per)
    write_csv(TAB / f"{sp}_evidence.csv", ["model", "scope", "examples", "n", "micro_P", "micro_R", "micro_F1", "macro_P", "macro_R", "macro_F1"], evid)
    if "base" in ev:
        ba = []
        for key, lab in (("accuracy", "accuracy"), ("macro_f1", "macro_f1"), ("macro_f1_gold_support", "macro_f1_gold_support")):
            for s in SCOPES:
                if s in ev["ft"]:
                    b, f = ev["base"][s][key], ev["ft"][s][key]
                    ba.append([f"{lab} ({s})", b, f, f - b])
        for sub in ("nonempty_gold",):
            b, f = ev["base"]["evidence"]["overall"][sub]["micro_f1"], ev["ft"]["evidence"]["overall"][sub]["micro_f1"]
            ba.append(["evidence micro_F1 (non-empty gold)", b, f, f - b])
        ba.append(["valid_json_rate", 1 - ev["base"]["overall"]["invalid_rate"], 1 - ev["ft"]["overall"]["invalid_rate"],
                   ev["base"]["overall"]["invalid_rate"] - ev["ft"]["overall"]["invalid_rate"]])
        write_csv(TAB / f"{sp}_before_after.csv", ["metric", "base (before)", "fine-tuned (after)", "delta"], ba)
    bs = bootstrap(models, a.bootstrap)
    if bs:
        write_csv(TAB / f"{sp}_bootstrap.csv", ["model", "metric", "mean", "ci_low_2.5", "ci_high_97.5", "n_boot", "n_patients"], bs)

    # ---- explained predictions (criterion, label, cited ids, evidence text, explanation)
    with open(PRED / f"{sp}_ft_explained.jsonl", "w", encoding="utf-8") as f:
        for r in models["ft"]:
            rec = records[r["annotation_id"]]
            x = explain_row(r, rec["patient_sentences"])
            f.write(json.dumps({"annotation_id": r["annotation_id"], "patient_id": r["patient_id"], "trial_id": r["trial_id"],
                                "criterion_type": r["criterion_type"], "criterion": rec["criterion_text"],
                                "pred_label": r["pred_label"], "valid": r["valid"], "cited_ids": r["pred_evidence_ids"],
                                "evidence": x["evidence"], "unknown_ids": x["unknown_ids"], "explanation": x["explanation"],
                                "missing_info": x["missing_info"], "gold_label": r["gold_label"],
                                "gold_evidence_ids": r["gold_evidence_ids"]}) + "\n")

    # ---- console summary + optional P2 cross-check
    print(f"\n=== {sp.upper()} (n={ev['ft']['n']}) ===")
    for row in summ:
        print(f"{row[0]:5s} {row[1]:10s} n={row[2]:4d} acc={row[3]:.4f} macroF1={row[4]:.4f} macroF1(gold-support)={row[5]:.4f} invalid={row[7]:.3f}")
    for row in evid:
        if row[2] == "nonempty_gold" and row[1] == "overall":
            print(f"evidence[{row[0]}] non-empty gold n={row[3]} micro P/R/F1 {row[4]:.3f}/{row[5]:.3f}/{row[6]:.3f} | macro F1 {row[9]:.3f}")
    for row in bs:
        print("bootstrap", row[0], row[1], f"mean {row[2]:.3f} 95% CI [{row[3]:.3f}, {row[4]:.3f}]")
    if a.check_quick:
        q = json.loads(Path(a.check_quick).read_text())["metrics"]
        ok = (abs(q["accuracy"] - ev["ft"]["overall"]["accuracy"]) < 1e-12 and abs(q["macro_f1"] - ev["ft"]["overall"]["macro_f1"]) < 1e-12
              and abs(q["evidence_nonempty_gold"]["micro_f1"] - ev["ft"]["evidence"]["overall"]["nonempty_gold"]["micro_f1"]) < 1e-12)
        print("CROSS-CHECK vs P2 quick metrics:", "MATCH" if ok else "MISMATCH")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
