"""Readable eligibility screening report (HTML and Markdown) with provenance. Pure Python, no model needed."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from typing import Sequence

from trialmatch.pipeline.aggregate import DISCLAIMER, RULE_TEXT

CAVEAT = ("The overall outcome is a rule-based aggregation of criterion-level model outputs, not a ground-truth label. "
          "'Potentially eligible (preliminary)' can be wrong because the model misses some 'not included'/'excluded' cases "
          "(see error analysis). Absence of evidence is never treated as eligibility or ineligibility.")


def make_provenance(*, model_id: str, adapter: str | None, adapter_sha256: str | None, device: str, dtype: str, mode: str,
                    k: int | None, max_new_tokens: int, input_sha256: str, versions: dict, note_how: str,
                    prompt_budget: int, truncated: bool) -> dict:
    return {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "model_id": model_id,
            "adapter": adapter or "none (base model)", "adapter_sha256": adapter_sha256 or "n/a", "device": device,
            "dtype": dtype, "decoding_mode": mode, "speculative_k": k if mode == "speculative" else "n/a",
            "decoding": f"greedy, max_new_tokens={max_new_tokens}", "input_sha256": input_sha256,
            "note_numbering": note_how, "prompt_token_budget": prompt_budget,
            "note_truncated_for_some_criteria": truncated, "versions": versions}


def _ev_text(r: dict) -> str:
    return " | ".join(f"[{e['id']}] {e['text']}" for e in r["evidence"]) or "(none cited)"


def to_markdown(sentences: Sequence[dict], result: dict, prov: dict, warnings: Sequence[str] = ()) -> str:
    agg = result["aggregate"]
    L = ["# Eligibility screening report", "", f"> **{DISCLAIMER}**", "", f"## Overall preliminary outcome: {agg['outcome']}", "",
         f"{CAVEAT}", "", f"Rule: {RULE_TEXT}", ""]
    if agg["blocking_criteria"]:
        L.append("Blocking criteria: " + ", ".join(f"#{result['rows'][i]['index']}" for i in agg["blocking_criteria"]))
    if agg["undecided_criteria"]:
        L.append("Unresolved criteria (missing information / not evaluated): " +
                 ", ".join(f"#{result['rows'][i]['index']}" for i in agg["undecided_criteria"]))
    for w in warnings:
        L.append(f"\n**Warning:** {w}")
    L += ["", "## Criterion-level results", ""]
    for r in result["rows"]:
        L += [f"### {r['index']}. [{r['criterion_type']}] {r['criterion_text']}", "",
              f"- **Predicted label:** {r['pred_label'] or 'UNRESOLVED'}" + ("  (missing information)" if r["missing_info"] else ""),
              f"- **Cited sentence IDs:** {r['pred_evidence_ids'] if r['valid'] else '[]'}",
              f"- **Evidence (verbatim from note):** {_ev_text(r)}",
              f"- **Explanation (templated):** {r['explanation']}"]
        if r.get("error"):
            L.append(f"- **Note:** {r['error']}")
        if r.get("omitted_ids"):
            L.append(f"- **Truncation:** sentences {r['omitted_ids']} were omitted from the prompt (note too long).")
        L.append("")
    L += ["## Patient note (as numbered for the model)", ""] + [f"[{s['id']}] {s['text']}" for s in sentences]
    L += ["", "## Provenance", ""] + [f"- {k}: {v}" for k, v in prov.items() if k != "versions"]
    L += [f"- versions: {prov['versions']}", "", f"_{DISCLAIMER}_", ""]
    return "\n".join(L)


def to_html(sentences: Sequence[dict], result: dict, prov: dict, warnings: Sequence[str] = ()) -> str:
    e = escape
    agg = result["aggregate"]
    cited = {i for r in result["rows"] if r["valid"] for i in r["pred_evidence_ids"]}
    rows = []
    for r in result["rows"]:
        flag = "UNRESOLVED" if r["pred_label"] is None else e(r["pred_label"])
        warn = " <span class='warn'>missing information</span>" if r["missing_info"] else ""
        extra = (f"<div class='warn'>{e(r['error'])}</div>" if r.get("error") else "") + \
                (f"<div class='warn'>Sentences {r['omitted_ids']} omitted from the prompt (note too long).</div>" if r.get("omitted_ids") else "")
        rows.append(f"<tr><td>{r['index']}</td><td>{e(r['criterion_type'])}</td><td>{e(r['criterion_text'])}</td>"
                    f"<td><b>{flag}</b>{warn}</td><td>{e(str(r['pred_evidence_ids'] if r['valid'] else []))}</td>"
                    f"<td>{e(_ev_text(r))}</td><td>{e(r['explanation'])}{extra}</td></tr>")
    note = "".join(f"<div class='s{' c' if s['id'] in cited else ''}'><b>[{s['id']}]</b> {e(s['text'])}</div>" for s in sentences)
    prov_rows = "".join(f"<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>" for k, v in prov.items())
    warn_html = "".join(f"<p class='warn'>Warning: {e(w)}</p>" for w in warnings)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Eligibility screening report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 12px;color:#111}}
.banner{{background:#fef3c7;border:1px solid #d97706;padding:10px;border-radius:6px;font-weight:600}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:top;font-size:14px}}
th{{background:#f3f4f6}}.warn{{color:#b45309}}.s{{padding:2px 6px}}.c{{background:#fde68a}}.out{{font-size:20px;font-weight:700}}</style></head><body>
<div class="banner">{e(DISCLAIMER)}</div>
<h1>Eligibility screening report</h1>
<p class="out">Overall preliminary outcome: {e(agg['outcome'])}</p><p>{e(CAVEAT)}</p><p><small>{e(RULE_TEXT)}</small></p>{warn_html}
<h2>Criterion-level results</h2>
<table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Cited IDs</th><th>Evidence (verbatim from note)</th><th>Explanation (templated)</th></tr>{''.join(rows)}</table>
<h2>Patient note (as numbered for the model; cited sentences highlighted)</h2>{note}
<h2>Provenance</h2><table>{prov_rows}</table><p><i>{e(DISCLAIMER)}</i></p></body></html>"""
