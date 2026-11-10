"""TrialMatch browser app (Gradio). Educational screening prototype, NOT a clinical enrollment decision.

Start:  python app/main.py            ->  http://127.0.0.1:7860
Flags:  --no-draft (skip 0.5B draft, disables speculative mode)  --device cpu|cuda  --port N  --no-adapter (base model)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from trialmatch.pipeline.aggregate import DISCLAIMER, INELIGIBLE, MORE_INFO, ELIGIBLE  # noqa: E402
from trialmatch.pipeline.report import CAVEAT, make_provenance, to_html, to_markdown  # noqa: E402
from trialmatch.pipeline.screening import (InputError, PROMPT_TOKEN_BUDGET, build_criteria, finalize, format_numbered,  # noqa: E402
                                           highlight_note_html, input_hash, note_to_sentences, split_trial_text,
                                           unresolved_row, validate_inputs)

MODES = ("nocache", "kv", "speculative")
MODE_LABELS = {"nocache": "No cache (full recompute)", "kv": "KV cache", "speculative": "Speculative (draft + KV cache)"}
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}
OUT_DIR = ROOT / "app_outputs"
SAMPLES = json.loads((ROOT / "app" / "sample_trials" / "samples.json").read_text(encoding="utf-8"))
LOCK = threading.Lock()          # one model, one inference at a time
OUTCOME_COLOR = {INELIGIBLE: "#dc2626", MORE_INFO: "#d97706", ELIGIBLE: "#16a34a"}


# ------------------------------------------------------------------------------------------------ helpers
def is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def banner_html() -> str:
    return (f'<div style="background:#fef3c7;color:#92400e;border:2px solid #d97706;padding:10px 14px;border-radius:8px;'
            f'font-weight:700;text-align:center">&#9888; {escape(DISCLAIMER)} Do not use it for patient care or enrollment.</div>')


def status_html(engine) -> str:
    d = engine.draft_id or "not loaded (speculative mode disabled)"
    sha = (engine.adapter_sha256 or "n/a")[:12]
    notes = "".join(f"<div>&#9888; {escape(n)}</div>" for n in engine.notes)
    return (f"<div style='font-size:13px;line-height:1.5'><b>Model ready.</b> target: <code>{escape(engine.model_id)}</code> + adapter "
            f"<code>{escape(engine.adapter or 'none (base model)')}</code> (sha256 {sha}...) | draft: <code>{escape(d)}</code> | "
            f"device: <code>{engine.device}</code> | dtype: <code>{engine.dtype_name}</code> | decoding: greedy | "
            f"prompt budget: {PROMPT_TOKEN_BUDGET} tokens{notes}</div>")


def outcome_html(res: dict, partial: str = "") -> str:
    a = res["aggregate"]
    c = OUTCOME_COLOR.get(a["outcome"], "#6b7280")
    rows = res["rows"]
    blk = ", ".join(f"#{rows[i]['index']}" for i in a["blocking_criteria"]) or "none"
    und = ", ".join(f"#{rows[i]['index']}" for i in a["undecided_criteria"]) or "none"
    cnt = a["counts"]
    return (f"<div style='border-left:8px solid {c};padding:10px 14px;border-radius:6px;border:1px solid {c}'>"
            f"<div style='font-size:13px;opacity:.7'>Overall PRELIMINARY outcome (rule-based aggregation, not ground truth)</div>"
            f"<div style='font-size:24px;font-weight:700;color:{c}'>{escape(a['outcome'])}</div>{partial}"
            f"<div style='font-size:13px;margin-top:6px'>criteria: {cnt['n']} | decided: {cnt['decided']} | not applicable: {cnt['not_applicable']} | "
            f"unresolved: {cnt['undecided']} | blocking: {blk} | unresolved list: {und}</div>"
            f"<div style='font-size:12px;opacity:.8;margin-top:6px'>{escape(CAVEAT)}</div>"
            f"<div style='font-size:12px;opacity:.8;margin-top:4px'>{escape(a['rule'])}</div></div>")


def results_table(res: dict) -> pd.DataFrame:
    recs = []
    for r in res["rows"]:
        recs.append({"#": r["index"], "type": r["criterion_type"], "criterion": r["criterion_text"],
                     "label": r["pred_label"] or "UNRESOLVED",
                     "cited IDs": ", ".join(map(str, r["pred_evidence_ids"])) if r["valid"] else "",
                     "evidence (verbatim from note)": " | ".join(f"[{e['id']}] {e['text']}" for e in r["evidence"]) or "(none cited)",
                     "explanation (templated)": r["explanation"] + (f" [{r['error']}]" if r.get("error") else "") +
                                                (f" Sentences {r['omitted_ids']} omitted (note too long)." if r.get("omitted_ids") else ""),
                     "missing info": "YES" if r["missing_info"] else "no",
                     "seconds": round((r["stats"] or {}).get("wall_s", 0.0), 1)})
    return pd.DataFrame(recs)


def focus_choices(res: dict) -> list[str]:
    return ["All cited sentences"] + [f"{r['index']}. [{r['criterion_type']}] {r['criterion_text'][:70]}" for r in res["rows"]]


def write_report(sentences, res, prov, warnings) -> list[str]:
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    h, m = OUT_DIR / f"screening_report_{stamp}.html", OUT_DIR / f"screening_report_{stamp}.md"
    h.write_text(to_html(sentences, res, prov, warnings), encoding="utf-8")
    m.write_text(to_markdown(sentences, res, prov, warnings), encoding="utf-8")
    return [str(h), str(m)]


# ------------------------------------------------------------------------------------------------ UI
def build_ui(engine, default_max_new: int = 64) -> gr.Blocks:
    on_gpu = engine.device.type == "cuda"
    mode_choices = [MODE_LABELS[m] for m in engine.available_modes()]

    # ---------- inputs / samples ----------
    def load_sample(title):
        s = next((x for x in SAMPLES if x["title"] == title), None)
        if s is None:
            return gr.update(), gr.update(), gr.update()
        return s["note"], "\n".join(s["inclusion"]), "\n".join(s["exclusion"])

    def do_split(text):
        if not text or not text.strip():
            gr.Warning("Paste a trial eligibility text first.")
            return gr.update(), gr.update()
        inc, exc = split_trial_text(text)
        return inc, exc

    def preview(note):
        try:
            s, how = note_to_sentences(note)
        except InputError as e:
            return f"**{e}**"
        return f"*{how}; {len(s)} sentences (IDs are what the model cites)*\n\n```\n{format_numbered(s)}\n```"

    # ---------- screening ----------
    def screen(note, inc, exc, mode_label, k, max_new, budget_s, progress=gr.Progress()):
        empty = (gr.update(),) * 7
        try:
            sentences, how = note_to_sentences(note)
            criteria = build_criteria(inc, exc)
            warnings = validate_inputs(note, criteria)
        except InputError as e:
            gr.Warning(str(e))
            yield (f"**Input problem:** {e}",) + empty
            return
        mode = LABEL_TO_MODE.get(mode_label, "kv")
        if mode not in engine.available_modes():
            yield (f"**Mode '{mode_label}' is unavailable** (draft model not loaded). Choose another mode.",) + empty
            return
        if not LOCK.acquire(blocking=False):
            yield ("**The model is busy with another request.** Wait for it to finish and try again.",) + empty
            return
        try:
            rows, t0, partial = [], time.time(), ""
            for i, c in enumerate(criteria):
                progress((i, len(criteria)), desc=f"criterion {i + 1}/{len(criteria)}")
                if time.time() - t0 > budget_s:
                    rows += [unresolved_row(x["type"], x["text"], f"Not evaluated: the {budget_s:.0f}s time budget was reached.")
                             for x in criteria[i:]]
                    partial = (f"<div style='color:#b45309;font-size:13px'>Time budget reached: {len(criteria) - i} criteria were NOT "
                               f"evaluated and count as unresolved.</div>")
                    break
                yield (f"Running criterion {i + 1}/{len(criteria)} ({MODE_LABELS[mode]}) ... elapsed {time.time() - t0:.0f}s",) + empty
                try:
                    rows.append(engine.run_criterion(sentences, c["type"], c["text"], mode, int(k), int(max_new)))
                except Exception as e:  # noqa: BLE001 - keep the session alive, report per criterion
                    msg = ("Out of memory while running this criterion (try --device cpu, a shorter note or no-cache off)." if is_oom(e)
                           else f"Inference error: {type(e).__name__}: {e}")
                    rows.append(unresolved_row(c["type"], c["text"], msg))
            res = finalize(rows, sentences)
            truncated = any(r.get("omitted_ids") for r in rows)
            prov = make_provenance(model_id=engine.model_id, adapter=engine.adapter, adapter_sha256=engine.adapter_sha256,
                                   device=str(engine.device), dtype=engine.dtype_name, mode=MODE_LABELS[mode], k=int(k),
                                   max_new_tokens=int(max_new), input_sha256=input_hash(note, criteria),
                                   versions=engine.versions(), note_how=how, prompt_budget=PROMPT_TOKEN_BUDGET, truncated=truncated)
            files = write_report(sentences, res, prov, warnings)
            cited = [i for r in res["rows"] if r["valid"] for i in r["pred_evidence_ids"]]
            wtxt = ("  \n".join(f"Warning: {w}" for w in warnings))
            done = (f"Done in {time.time() - t0:.0f}s with **{MODE_LABELS[mode]}**; note numbering: {how}." +
                    (f"  \n{wtxt}" if wtxt else "") + ("  \n**Truncation:** some sentences were omitted for long notes (shown struck through)." if truncated else ""))
            yield (done, outcome_html(res, partial), results_table(res), highlight_note_html(sentences, cited),
                   {"res": res, "sentences": sentences}, gr.update(choices=focus_choices(res), value="All cited sentences"),
                   files, preview(note))
        finally:
            LOCK.release()

    def focus(choice, state):
        if not state:
            return gr.update()
        res, sents = state["res"], state["sentences"]
        omitted = sorted({i for r in res["rows"] for i in r.get("omitted_ids", [])})
        if choice and choice != "All cited sentences":
            idx = int(choice.split(".", 1)[0]) - 1
            r = res["rows"][idx]
            cited = r["pred_evidence_ids"] if r["valid"] else []
            return highlight_note_html(sents, cited, r.get("omitted_ids", []))
        cited = [i for r in res["rows"] if r["valid"] for i in r["pred_evidence_ids"]]
        return highlight_note_html(sents, cited, omitted)

    # ---------- live comparison ----------
    def compare(note, inc, exc, crit_no, mode_labels, k, max_new, warm, budget_s, progress=gr.Progress()):
        none = (gr.update(), gr.update())
        try:
            sentences, _ = note_to_sentences(note)
            criteria = build_criteria(inc, exc)
            validate_inputs(note, criteria)
            n = int(crit_no)
            if not 1 <= n <= len(criteria):
                raise InputError(f"Criterion number must be between 1 and {len(criteria)} (inclusion criteria come first).")
            if not mode_labels:
                raise InputError("Select at least one decoding mode.")
        except (InputError, TypeError, ValueError) as e:
            gr.Warning(str(e))
            yield (f"**Input problem:** {e}", *none)
            return
        modes = [LABEL_TO_MODE[m] for m in mode_labels if LABEL_TO_MODE.get(m) in engine.available_modes()]
        if not LOCK.acquire(blocking=False):
            yield ("**The model is busy.** Try again in a moment.", *none)
            return
        try:
            c = criteria[n - 1]
            try:
                prompt, dropped = engine.prepare(sentences, c["type"], c["text"])
            except Exception as e:  # noqa: BLE001
                yield (f"**Cannot build the prompt:** {e}", *none)
                return
            if warm:
                yield ("Warm-up run (2 tokens, untimed) ...", *none)
                engine.run_prompt(prompt, "kv", max_new_tokens=2, measure_memory=False)
            order = [m for m in ("kv", "speculative", "nocache") if m in modes]      # cheapest first; no-cache last
            results, notes, t0 = {}, [], time.time()
            for j, m in enumerate(order):
                if m == "nocache" and "kv" in results:
                    est = results["kv"]["stats"]["prefill_ms"] / 1000 * int(max_new)       # n_new forwards of ~prompt size
                    if est > budget_s:
                        notes.append(f"No-cache run SKIPPED: estimated ~{est:.0f}s (one full-prompt forward per new token) exceeds the {budget_s:.0f}s budget. "
                                     "Lower 'max new tokens' or raise the budget.")
                        continue
                progress((j, len(order)), desc=MODE_LABELS[m])
                yield (f"Running {MODE_LABELS[m]} ... ({time.time() - t0:.0f}s elapsed)", *none)
                try:
                    results[m] = engine.run_prompt(prompt, m, k=int(k), max_new_tokens=int(max_new))
                except Exception as e:  # noqa: BLE001
                    notes.append(f"{MODE_LABELS[m]} failed: " + ("out of memory" if is_oom(e) else f"{type(e).__name__}: {e}"))
            if not results:
                yield ("**No mode finished.** " + " ".join(notes), *none)
                return
            ref_mode = "kv" if "kv" in results else next(iter(results))
            ref = results[ref_mode]["tokens"]
            tab, plot_rows = [], []
            base_total = results.get("nocache", {}).get("stats", {}).get("total_ms")
            kv_total = results.get("kv", {}).get("stats", {}).get("total_ms")
            for m in modes:
                if m not in results:
                    continue
                s, toks = results[m]["stats"], results[m]["tokens"]
                eq = toks == ref
                div = next((i for i, (a, b) in enumerate(zip(toks, ref)) if a != b), min(len(toks), len(ref)) if len(toks) != len(ref) else -1)
                tab.append({"mode": MODE_LABELS[m], f"tokens == {ref_mode}": "yes" if eq else f"NO (first diff at {div})",
                            "prompt tok": results[m]["prompt_len"], "new tok": s["new_tokens"], "prefill ms": round(s["prefill_ms"]),
                            "decode ms": round(s["decode_ms"]), "total ms": round(s["total_ms"]), "decode tok/s": round(s["decode_tok_s"], 2),
                            "e2e tok/s": round(s["e2e_tok_s"], 2), f"peak MB ({s['peak_kind']})": round(s["peak_mb"]),
                            "peak above start MB": round(s["peak_above_base_mb"]), "KV cache MB (computed)": round(s["kv_cache_mb_computed"], 1),
                            "accept rate": round(s["acceptance_rate"], 3) if m == "speculative" else "",
                            "mean accepted/round": round(s["mean_accepted_per_round"], 2) if m == "speculative" else "",
                            "target fwd": s.get("target_forwards", ""), "draft fwd": s.get("draft_forwards", ""),
                            "speedup vs no-cache": round(base_total / s["total_ms"], 2) if base_total else "",
                            "speedup vs KV": round(kv_total / s["total_ms"], 2) if kv_total else ""})
                plot_rows.append((MODE_LABELS[m].split(" (")[0], s["prefill_ms"], s["decode_ms"], s["e2e_tok_s"], s["peak_above_base_mb"]))
            fig = comparison_figure(plot_rows)
            mism = [r for r in tab if str(r[f"tokens == {ref_mode}"]).startswith("NO")]
            msg = (f"Compared on criterion {n} ({c['type']}): *{c['text'][:90]}*. Same prompt tokens, greedy, warm-up={'yes' if warm else 'no'}, "
                   f"single run (no error bars). Output text of the reference run: `{results[ref_mode]['text']}`  \n" +
                   ("**Token mismatch between modes** (fp rounding can flip near-tied argmaxes; see P5/P6 notes). " if mism else "All modes produced identical tokens.  \n") +
                   (f"  \n{' '.join(notes)}" if notes else "") + (f"  \nSentences {dropped} omitted from the prompt (note too long)." if dropped else ""))
            yield (msg, pd.DataFrame(tab), fig)
        finally:
            LOCK.release()

    # ---------- reference benchmarks (from results/benchmarks/*.csv) ----------
    def load_reference():
        b = ROOT / "results" / "benchmarks"
        out, figs = [], []
        try:
            kv = pd.read_csv(b / "kv_bench.csv")
            kv = kv[kv.get("status", "ok") == "ok"]
            cols = ["prompt_len", "new_tokens", "tokens_equal", "nocache_total_ms_median", "kv_cache_total_ms_median", "speedup_total",
                    "speedup_decode", "nocache_peak_above_weights_mb_median", "kv_cache_peak_above_weights_mb_median"]
            out.append(kv[[c for c in cols if c in kv.columns]].round(2))
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            for nt, g in kv.groupby("new_tokens"):
                g = g.sort_values("prompt_len")
                ax.plot(g["prompt_len"], g["speedup_total"], marker="o", label=f"{nt} new tokens")
            ax.set_xlabel("prompt length (tokens)"); ax.set_ylabel("KV speedup vs no-cache (total)"); ax.legend(); ax.grid(alpha=.3)
            ax.set_title("Stored benchmark: KV-cache speedup", fontsize=10); fig.tight_layout(); figs.append(fig)
        except Exception as e:  # noqa: BLE001
            out.append(pd.DataFrame({"error": [f"kv_bench.csv not readable: {e}"]})); figs.append(None)
        try:
            sp = pd.read_csv(b / "spec_bench.csv")
            sp = sp[sp.get("status", "ok") == "ok"]
            cols = ["tag", "item", "prompt_len", "new_tokens", "k", "tokens_equal", "greedy_total_ms_median", "spec_total_ms_median",
                    "speedup_total", "acceptance_rate", "mean_accepted_per_round", "tokens_per_target_forward"]
            out.append(sp[[c for c in cols if c in sp.columns]].round(3))
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            lab = [f"{r.item} k={r.k}" for r in sp.itertuples()]
            ax.bar(lab, sp["speedup_total"]); ax.axhline(1.0, color="k", ls="--", lw=1)
            ax.set_ylabel("speculative speedup vs greedy-KV"); ax.set_title("Stored benchmark: speculative (CPU)", fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout(); figs.append(fig)
        except Exception as e:  # noqa: BLE001
            out.append(pd.DataFrame({"error": [f"spec_bench.csv not readable: {e}"]})); figs.append(None)
        return out[0], figs[0], out[1], figs[1]

    # ---------- layout ----------
    with gr.Blocks(title="TrialMatch - educational screening prototype") as demo:
        gr.HTML(banner_html())
        gr.Markdown("# TrialMatch: criterion-level clinical-trial screening")
        gr.HTML(status_html(engine))
        with gr.Tabs():
            with gr.Tab("1. Screening"):
                with gr.Row():
                    sample = gr.Dropdown([s["title"] for s in SAMPLES], label="Bundled sample case (synthetic)", value=None)
                with gr.Row():
                    note = gr.Textbox(label="Patient note (plain text, or numbered '0. ... 1. ...')", lines=9, scale=3)
                    with gr.Column(scale=2):
                        inc = gr.Textbox(label="Inclusion criteria (one per line; editable)", lines=4)
                        exc = gr.Textbox(label="Exclusion criteria (one per line; editable)", lines=4)
                with gr.Accordion("Paste a full trial eligibility text and split it automatically", open=False):
                    trial_txt = gr.Textbox(label="Eligibility text with 'Inclusion Criteria:' / 'Exclusion Criteria:' headers", lines=6)
                    split_btn = gr.Button("Split into inclusion / exclusion boxes")
                with gr.Accordion("Show how the note is numbered for the model", open=False):
                    prev_btn = gr.Button("Show numbering")
                    prev_md = gr.Markdown()
                with gr.Row():
                    mode = gr.Radio(mode_choices, value=MODE_LABELS["kv"], label="Decoding mode")
                    kk = gr.Slider(1, 8, value=4, step=1, label="Speculative draft length k")
                    mx = gr.Slider(8, 64, value=min(64, default_max_new), step=1, label="max new tokens")
                    bud = gr.Slider(60, 1800, value=600, step=30, label="Time budget for the whole run (s)")
                run_btn = gr.Button("Run screening", variant="primary")
                status = gr.Markdown()
                outcome = gr.HTML()
                table = gr.Dataframe(label="Criterion-level results", wrap=True, interactive=False)
                focus_dd = gr.Dropdown(label="Highlight the evidence of one criterion in the note", choices=["All cited sentences"],
                                       value="All cited sentences")
                note_html = gr.HTML()
                files = gr.File(label="Download report (HTML and Markdown, with provenance)", file_count="multiple", interactive=False)
                state = gr.State()
                sample.change(load_sample, sample, [note, inc, exc])
                split_btn.click(do_split, trial_txt, [inc, exc])
                prev_btn.click(preview, note, prev_md)
                run_btn.click(screen, [note, inc, exc, mode, kk, mx, bud], [status, outcome, table, note_html, state, focus_dd, files, prev_md])
                focus_dd.change(focus, [focus_dd, state], note_html)

            with gr.Tab("2. Live decoding comparison"):
                gr.Markdown("Runs the **same prompt** of one criterion (from the boxes in tab 1) through the selected decoders and compares "
                            "tokens, latency, throughput and memory. Results come from the run you start now. "
                            + ("" if on_gpu else "**CPU detected:** no-cache decoding recomputes the whole prompt for every new token and can take minutes; "
                               "keep 'max new tokens' small."))
                with gr.Row():
                    cno = gr.Number(value=1, precision=0, label="Criterion number (inclusion first, then exclusion)")
                    cmodes = gr.CheckboxGroup(mode_choices, value=mode_choices, label="Modes to compare")
                with gr.Row():
                    ck = gr.Slider(1, 8, value=4, step=1, label="Speculative k")
                    cmx = gr.Slider(2, 64, value=32 if on_gpu else 10, step=1, label="max new tokens")
                    cwarm = gr.Checkbox(value=on_gpu, label="Untimed warm-up first")
                    cbud = gr.Slider(60, 3600, value=900, step=30, label="Skip no-cache if estimated longer than (s)")
                cmp_btn = gr.Button("Run comparison", variant="primary")
                cmp_msg = gr.Markdown()
                cmp_tab = gr.Dataframe(label="Result of this run", wrap=True, interactive=False)
                cmp_plot = gr.Plot(label="Latency / throughput / memory")
                cmp_btn.click(compare, [note, inc, exc, cno, cmodes, ck, cmx, cwarm, cbud], [cmp_msg, cmp_tab, cmp_plot])

            with gr.Tab("3. Stored benchmarks"):
                gr.Markdown("Reference numbers produced earlier by `scripts/bench_kv.py` and `scripts/bench_spec.py` (read from `results/benchmarks/*.csv`; "
                            "hardware and repeat counts are in the report, e.g. the speculative CPU run had 1 repeat).")
                ref_btn = gr.Button("Load stored benchmark CSVs")
                kv_df, kv_pl = gr.Dataframe(label="kv_bench.csv", interactive=False), gr.Plot()
                sp_df, sp_pl = gr.Dataframe(label="spec_bench.csv", interactive=False), gr.Plot()
                ref_btn.click(load_reference, None, [kv_df, kv_pl, sp_df, sp_pl])

            with gr.Tab("About"):
                gr.Markdown(f"**{DISCLAIMER}**\n\n- Evidence text is copied from your note by sentence ID; it is never generated.\n"
                            "- Labels: inclusion = included / not included / not enough information / not applicable; exclusion = excluded / "
                            "not excluded / not enough information / not applicable.\n- Missing information is shown as such; it is never treated as "
                            "eligibility or ineligibility.\n- The overall outcome is an explicit rule-based aggregation of criterion results.\n"
                            "- Model: Qwen2.5-1.5B-Instruct + LoRA adapter fine-tuned on the TrialGPT criterion annotations (train split only).")
    demo.queue(default_concurrency_limit=1)
    return demo


def comparison_figure(rows):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    names = [r[0] for r in rows]
    ax[0].bar(names, [r[1] for r in rows], label="prefill")
    ax[0].bar(names, [r[2] for r in rows], bottom=[r[1] for r in rows], label="decode")
    ax[0].set_ylabel("ms"); ax[0].set_title("Latency (prefill + decode)", fontsize=10); ax[0].legend()
    ax[1].bar(names, [r[3] for r in rows], color="tab:green"); ax[1].set_title("End-to-end tokens/s", fontsize=10)
    ax[2].bar(names, [r[4] for r in rows], color="tab:red"); ax[2].set_title("Peak memory above start (MB)", fontsize=10)
    for a in ax:
        plt.setp(a.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ entry point
def main() -> None:
    ap = argparse.ArgumentParser(description="TrialMatch browser app")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=str(ROOT / "weights" / "adapter"))
    ap.add_argument("--no-adapter", action="store_true", help="use the base model (not the fine-tuned one)")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no-draft", action="store_true", help="do not load the draft model (disables speculative mode)")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--dtype", default=None, choices=[None, "float16", "float32"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--share", action="store_true", help="BACKUP only: public gradio.live link (e.g. from Colab); the primary path is local")
    a = ap.parse_args()

    adapter = None if a.no_adapter else a.adapter
    if adapter and not (Path(adapter) / "adapter_model.safetensors").exists():
        sys.exit(f"ERROR: fine-tuned adapter not found at {adapter}.\n  Run:  python scripts/download_weights.py\n"
                 "  (or start with --no-adapter to use the base model; results will differ from the report).")
    from trialmatch.pipeline.engine import Engine  # heavy imports only after the quick checks above

    draft = None if a.no_draft else a.draft
    try:
        import psutil, torch
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if draft and dev == "cpu" and psutil.virtual_memory().available < 9.5 * 2**30:
            print("[warn] less than ~9.5 GB RAM available: skipping the draft model (speculative mode disabled). Use --draft to force.")
            draft = None
    except Exception:  # noqa: BLE001
        pass
    print("Loading models (first run downloads the base model from Hugging Face; this can take a few minutes) ...")
    try:
        engine = Engine.load(a.model, adapter, draft, device=a.device, dtype=a.dtype)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ERROR: could not load the model: {type(e).__name__}: {e}\n  Check internet access (first download), free RAM/VRAM, and that "
                 "`pip install -r requirements.txt` finished.")
    demo = build_ui(engine, a.max_new_tokens)
    print(f"Ready: open http://{a.host}:{a.port} in your browser")
    demo.launch(server_name=a.host, server_port=a.port, show_error=True, share=a.share)


if __name__ == "__main__":
    main()
