"""Phase 1 step 1: inspect the raw dataset. Prints facts only. Run: python scripts/inspect_data.py"""
from __future__ import annotations

import collections

import pandas as pd

from trialmatch.data.load import RAW_CACHE, load_raw
from trialmatch.data.prompts import split_numbered_note
from trialmatch.data.schema import GPT4_COLUMNS, LABELS_BY_TYPE, parse_evidence_ids

pd.set_option("display.width", 200, "display.max_colwidth", 80)
df = load_raw()
print("== SHAPE / COLUMNS ==", df.shape, "cached at", RAW_CACHE)
print(df.dtypes.to_string())
print("\n== NULLS ==\n", df.isna().sum().to_string())
print("\nGPT-4 columns present (blacklisted):", sorted(set(df.columns) & GPT4_COLUMNS))
print("columns NOT in blacklist and not used:", sorted(set(df.columns) - GPT4_COLUMNS))

print("\n== FIRST 3 ROWS (GPT-4 columns hidden, text cut) ==")
show = df.drop(columns=[c for c in df.columns if c in GPT4_COLUMNS]).head(3)
for _, r in show.iterrows():
    print({k: (str(v)[:90] + "...") if len(str(v)) > 90 else v for k, v in r.items()})

print("\n== PATIENTS / TRIALS ==")
g = df.groupby("patient_id")
print("unique patients:", df.patient_id.nunique(), "| unique trials:", df.trial_id.nunique())
print("rows per patient: min/median/max =", g.size().min(), g.size().median(), g.size().max())
print("distinct notes per patient (must all be 1):", g.note.nunique().value_counts().to_dict())
ids = sorted(df.patient_id.unique())
print("sorted ids head/tail:", ids[:4], ids[-4:])
print("duplicate annotation_id:", int(df.annotation_id.duplicated().sum()))
key = ["patient_id", "trial_id", "criterion_type", "criterion_text"]
print("duplicate (patient, trial, type, text) rows:", int(df.duplicated(key).sum()))

print("\n== CRITERION TYPE / LABELS (expert) ==")
print(df.criterion_type.value_counts(dropna=False).to_string())
print(pd.crosstab(df.criterion_type, df.expert_eligibility, dropna=False).to_string())
bad = [(t, l) for t, l in zip(df.criterion_type, df.expert_eligibility) if l not in LABELS_BY_TYPE.get(t, ())]
print("rows whose label is not valid for their type:", len(bad), bad[:5])

print("\n== CRITERION TEXT ==")
ct = df.criterion_text
print("null:", int(ct.isna().sum()), "| empty after strip:", int((ct.fillna("").str.strip() == "").sum()))
print("length chars min/median/max:", ct.fillna("").str.len().min(), ct.fillna("").str.len().median(), ct.fillna("").str.len().max())

print("\n== EVIDENCE (expert_sentences) ==")
print("python type of field:", type(df.expert_sentences.iloc[0]).__name__, "| examples:", df.expert_sentences.drop_duplicates().head(6).tolist())
parsed, fails = [], 0
for v in df.expert_sentences:
    try:
        parsed.append(parse_evidence_ids(v))
    except Exception as e:  # noqa: BLE001
        fails += 1; parsed.append([]); print("PARSE FAIL", repr(v), e)
df["_gold"] = parsed
print("parse failures:", fails)
print("n evidence ids per row distribution:", collections.Counter(len(p) for p in parsed).most_common())
print("empty-evidence rate by label:\n", df.assign(empty=[len(p) == 0 for p in parsed]).groupby("expert_eligibility").empty.agg(["mean", "sum", "count"]).to_string())

print("\n== SENTENCE NUMBERING ==")
notes = df.drop_duplicates("patient_id")[["patient_id", "note"]]
counts, fail = {}, []
for pid, note in zip(notes.patient_id, notes.note):
    try:
        counts[pid] = len(split_numbered_note(note))
    except ValueError as e:
        fail.append((pid, str(e)))
print("notes parsed:", len(counts), "| failed:", fail[:3])
s = pd.Series(counts)
print("sentences per note min/median/max:", s.min(), s.median(), s.max())
oor = [(a, pid, max(gl)) for a, pid, gl in zip(df.annotation_id, df.patient_id, df._gold) if gl and pid in counts and max(gl) >= counts[pid]]
print("rows with gold id >= n_sentences:", len(oor), oor[:5])
print("note length (words) min/median/max:", notes.note.str.split().str.len().min(), notes.note.str.split().str.len().median(), notes.note.str.split().str.len().max())
p0 = notes.iloc[0]
print("\nexample parse of first note (id: first 70 chars):")
for sd in split_numbered_note(p0.note)[:4]:
    print(" ", sd["id"], sd["text"][:70])
print("\nlast 90 chars of first note:", repr(p0.note[-90:]))
print("\nDONE. Raw table cached to", RAW_CACHE)
