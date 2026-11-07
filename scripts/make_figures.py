"""Figures/tables from saved logs (no GPU).  `python scripts/make_figures.py --what training`

Reads   results/logs/train_<run>.csv, results/logs/summary_<run>.json, results/tables/quick_val_*.json
Writes  results/figures/train_curve_<run>.png, results/figures/train_curves_all.png,
        results/tables/hparams.csv, results/tables/val_selection.csv
Later phases add more `--what` options (eval, bench, ...).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS, TABLES, FIGS = ROOT / "results" / "logs", ROOT / "results" / "tables", ROOT / "results" / "figures"


def read_log(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs = [float(r["step"]) for r in rows if r[key] not in ("", None)]
    ys = [float(r[key]) for r in rows if r[key] not in ("", None)]
    return xs, ys


def log_stats(rows: list[dict]) -> dict:
    """Authoritative per-run numbers derived from the CSV log (robust to resumed / re-run jobs)."""
    vals = [(float(r["val_loss"]), int(r["step"])) for r in rows if r["val_loss"] not in ("", None)]
    out = {"train_seconds": float(rows[-1]["elapsed_s"]), "peak_mem_gb": max(float(r["peak_mem_gb"]) for r in rows)}
    if vals:
        out["best_val_loss"], out["best_step"] = min(vals)
        out["last_val_loss"] = vals[-1][0]
    return out


def training() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    logs = sorted(LOGS.glob("train_*.csv"))
    logs = [p for p in logs if p.stem != "train_smoke"]
    if not logs:
        print("no results/logs/train_<run>.csv found")
        return
    fig_all, ax_all = plt.subplots(figsize=(7, 4))
    stats: dict[str, dict] = {}
    for p in logs:
        run = p.stem[len("train_"):]
        rows = read_log(p)
        stats[run] = log_stats(rows)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        tx, ty = _series(rows, "train_loss")
        vx, vy = _series(rows, "val_loss")
        ax.plot(tx, ty, alpha=0.6, label="train loss (per optimizer step)")
        ax.plot(vx, vy, "o-", label="validation loss")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("token cross-entropy")
        ax.set_title(f"Run {run}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / f"train_curve_{run}.png", dpi=200)
        plt.close(fig)
        ax_all.plot(vx, vy, "o-", label=f"{run} val")
    ax_all.set_xlabel("optimizer step")
    ax_all.set_ylabel("validation loss")
    ax_all.set_title("Validation loss, all configurations")
    ax_all.legend()
    ax_all.grid(alpha=0.3)
    fig_all.tight_layout()
    fig_all.savefig(FIGS / "train_curves_all.png", dpi=200)
    plt.close(fig_all)

    # hyper-parameter table
    sums = []
    for p in sorted(LOGS.glob("summary_*.json")):
        if p.stem == "summary_smoke":
            continue
        sm = json.loads(p.read_text())
        sm.update(stats.get(sm["run"], {}))          # numbers from the CSV win over a possibly overwritten summary
        p.write_text(json.dumps(sm, indent=2))
        sums.append(sm)
    cols = ["run", "lora_r", "lora_alpha", "lr", "epochs", "effective_batch", "steps", "trainable_params",
            "best_val_loss", "best_step", "last_val_loss", "train_seconds", "peak_mem_gb"]
    with open(TABLES / "hparams.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in sums:
            w.writerow([s.get(c, "") for c in cols])
    print("wrote", TABLES / "hparams.csv", f"({len(sums)} runs)")

    # validation selection table (base + every evaluated adapter)
    sel_cols = ["tag", "adapter", "n", "valid_rate", "accuracy", "macro_f1", "inclusion_macro_f1", "exclusion_macro_f1",
                "evidence_micro_f1_nonempty_gold", "hit_max_rate"]
    with open(TABLES / "val_selection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(sel_cols)
        for p in sorted(TABLES.glob("quick_val_*.json")):
            d = json.loads(p.read_text())
            m, meta = d["metrics"], d["meta"]
            if meta["tag"].startswith("hubcheck"):
                continue
            w.writerow([meta["tag"], meta.get("adapter") or "base", m["n"], f"{m['valid_rate']:.4f}", f"{m['accuracy']:.4f}",
                        f"{m['macro_f1']:.4f}", f"{m.get('inclusion_macro_f1', float('nan')):.4f}",
                        f"{m.get('exclusion_macro_f1', float('nan')):.4f}",
                        f"{m['evidence_nonempty_gold']['micro_f1']:.4f}", f"{m['hit_max_rate']:.4f}"])
    print("wrote", TABLES / "val_selection.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["training"], required=True)
    a = ap.parse_args()
    if a.what == "training":
        training()


if __name__ == "__main__":
    main()
