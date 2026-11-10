import json
import pytest
from trialmatch.data.load import REPO_ROOT, RAW_CACHE
from trialmatch.data.preprocess import build_records
from trialmatch.data.schema import GPT4_COLUMNS, RECORD_FIELDS, UNUSED_COLUMNS, USED_COLUMNS, assert_no_gpt4_keys
from conftest import FakeTok


def test_blacklist_and_used_columns_disjoint():
    assert not set(USED_COLUMNS) & (GPT4_COLUMNS | UNUSED_COLUMNS)
    assert not set(RECORD_FIELDS) & (GPT4_COLUMNS | UNUSED_COLUMNS)
    assert {"gpt4_explanation", "gpt4_sentences", "gpt4_eligibility", "explanation_correctness"} <= GPT4_COLUMNS


def test_guard_rejects_forbidden_and_unknown_keys():
    ok = {k: None for k in RECORD_FIELDS}
    assert_no_gpt4_keys(ok)
    for bad in ("gpt4_explanation", "gpt4_eligibility", "training"):
        with pytest.raises(AssertionError):
            assert_no_gpt4_keys({**ok, bad: 1})
    with pytest.raises(AssertionError):
        assert_no_gpt4_keys({**ok, "something_new": 1})


def test_records_have_no_gpt4_text_or_keys(synth_df):
    recs, dropped = build_records(synth_df, FakeTok(), max_len=4096)
    assert len(recs) == len(synth_df) and not dropped
    secrets = synth_df.gpt4_explanation.tolist()
    for r in recs:
        assert set(r) == set(RECORD_FIELDS)
        blob = r["prompt"] + r["target"]
        assert "GPT4" not in blob and not any(s in blob for s in secrets)


def test_gold_comes_from_expert_not_gpt4(synth_df):
    synth_df["gpt4_sentences"] = "[5]"   # poison the GPT-4 column: gold must not change
    a, _ = build_records(synth_df, FakeTok(), 4096)
    synth_df["gpt4_sentences"] = "[0, 1, 2, 3]"
    b, _ = build_records(synth_df, FakeTok(), 4096)
    assert a == b


def test_gpt4_columns_not_even_read(synth_df):
    recs, _ = build_records(synth_df.drop(columns=list(GPT4_COLUMNS | UNUSED_COLUMNS)), FakeTok(), 4096)
    assert len(recs) == len(synth_df)


def test_bad_label_and_bad_evidence_raise(synth_df):
    bad = synth_df.copy(); bad.loc[0, "expert_eligibility"] = "excluded"   # exclusion label on inclusion row
    with pytest.raises(ValueError):
        build_records(bad, FakeTok(), 4096)
    bad = synth_df.copy(); bad.loc[0, "expert_sentences"] = "[99]"
    with pytest.raises(ValueError):
        build_records(bad, FakeTok(), 4096)


def test_empty_criterion_text_dropped_and_reported(synth_df):
    d = synth_df.copy(); d.loc[0, "criterion_text"] = None; d.loc[1, "criterion_text"] = "  "
    recs, dropped = build_records(d, FakeTok(), 4096)
    assert len(dropped) == 2 and len(recs) == len(d) - 2


def test_truncation_protects_gold_and_is_recorded(synth_df):
    d = synth_df[synth_df.patient_id == "sigir-20143"].copy()
    d["expert_sentences"] = "[0, 8]"; d["expert_eligibility"] = d.criterion_type.map({"inclusion": "included", "exclusion": "excluded"})
    recs, _ = build_records(d, FakeTok(), max_len=170, reserve_tokens=10)
    assert any(r["truncated_ids"] for r in recs)
    for r in recs:
        assert not set(r["truncated_ids"]) & set(r["gold_evidence_ids"])
        if r["truncated_ids"]:
            assert "were omitted" in r["prompt"] and len(r["patient_sentences"]) == 9   # full note kept in record


REAL = REPO_ROOT / "data" / "processed"


@pytest.mark.skipif(not (REAL / "train.jsonl").exists() or not RAW_CACHE.exists(), reason="processed data not present")
def test_real_processed_files_are_clean_and_disjoint():
    """GPT-4 text must never reach a target; it may appear in a prompt ONLY if it is verbatim input text of that
    same record (the real data has one GPT-4 explanation equal to the boilerplate last sentence of the notes)."""
    import pandas as pd
    raw = pd.read_parquet(RAW_CACHE)
    expl = {e for e in raw.gpt4_explanation.dropna() if len(e) >= 30}
    seen = {}
    for sp in ("train", "val", "test"):
        for line in open(REAL / f"{sp}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            assert_no_gpt4_keys(r)
            allowed_input = " ".join(s["text"] for s in r["patient_sentences"]) + " " + r["criterion_text"]
            for e in expl:
                assert e not in r["target"]
                if e in r["prompt"]:
                    assert e in allowed_input, f"GPT-4 text in prompt that is not part of the record's own input: {e[:60]!r}"
            assert seen.setdefault(r["patient_id"], sp) == sp   # patient in exactly one file
