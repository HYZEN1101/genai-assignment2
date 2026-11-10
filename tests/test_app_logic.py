"""Phase 7: model-free app logic (note numbering, criteria parsing, validation, unresolved handling, report export)."""
import pytest

from trialmatch.pipeline.report import make_provenance, to_html, to_markdown
from trialmatch.pipeline.screening import (InputError, MAX_NOTE_CHARS, build_criteria, finalize, highlight_note_html, input_hash,
                                           note_to_sentences, parse_criteria_block, split_trial_text, unresolved_row,
                                           validate_inputs)

NUMBERED = "0. A 52-year-old woman has joint pain. 1. She takes methotrexate. 2. No cancer."


def test_numbered_note_matches_training_format():
    s, how = note_to_sentences(NUMBERED)
    assert [x["id"] for x in s] == [0, 1, 2] and s[1]["text"] == "She takes methotrexate." and "numbered" in how


def test_plain_note_autosplit_ids_from_zero():
    s, how = note_to_sentences("A man has cough. He smokes daily! Is he allergic? No known allergy.")
    assert [x["id"] for x in s] == [0, 1, 2, 3] and how.startswith("auto")


def test_bulleted_note_lines_renumbered_from_zero():
    s, how = note_to_sentences("1. First fact\n2. Second fact\n3. Third fact")   # 1-based list, not the dataset format
    assert [x["text"] for x in s] == ["First fact", "Second fact", "Third fact"] and [x["id"] for x in s] == [0, 1, 2]


def test_empty_note_raises():
    for bad in ("", "   \n "):
        with pytest.raises(InputError):
            note_to_sentences(bad)


def test_criteria_bullets_continuations_and_numbers():
    txt = "* Age over 18\n* Confirmed diagnosis of\n  type 2 diabetes\n- HbA1c > 7"
    assert parse_criteria_block(txt) == ["Age over 18", "Confirmed diagnosis of type 2 diabetes", "HbA1c > 7"]
    assert parse_criteria_block("1. a\n2) b") == ["a", "b"]
    assert parse_criteria_block("one per line\nsecond") == ["one per line", "second"]
    assert parse_criteria_block("\n \n") == []


def test_split_trial_text_headers():
    t = "Inclusion Criteria:\n* adult\n* diabetic\n\nExclusion Criteria:\n* pregnant\n"
    inc, exc = split_trial_text(t)
    assert parse_criteria_block(inc) == ["adult", "diabetic"] and parse_criteria_block(exc) == ["pregnant"]
    assert split_trial_text("only text")[0] == "only text"


def test_build_criteria_order_and_types():
    c = build_criteria("a\nb", "c")
    assert [(x["type"], x["text"]) for x in c] == [("inclusion", "a"), ("inclusion", "b"), ("exclusion", "c")]


def test_validation_blocking_and_warnings():
    crit = build_criteria("adult", "")
    assert validate_inputs("A long enough patient note about a person with cough today.", crit) == []
    with pytest.raises(InputError):
        validate_inputs("", crit)
    with pytest.raises(InputError):
        validate_inputs("note text here for the patient", [])
    with pytest.raises(InputError):
        validate_inputs("x" * (MAX_NOTE_CHARS + 1), crit)
    with pytest.raises(InputError):
        validate_inputs("note text here for the patient", [{"type": "inclusion", "text": "y" * 1001}])
    assert any("very short" in w for w in validate_inputs("short note", crit))
    assert any("non-English" in w for w in validate_inputs("\u0645\u0631\u064a\u0636 \u0639\u0646\u062f\u0647 \u0633\u0639\u0627\u0644 \u0634\u062f\u064a\u062f \u0645\u0646\u0630 \u0623\u0633\u0628\u0648\u0639\u064a\u0646 \u0648\u062d\u0645\u0649", crit))


def test_input_hash_stable_and_sensitive():
    c = build_criteria("a", "b")
    assert input_hash("n", c) == input_hash(" n ", c) != input_hash("n2", c)


def _row(ctype, label, ids, text="c"):
    return {"criterion_type": ctype, "criterion_text": text, "raw_output": "", "pred_label": label, "pred_evidence_ids": ids,
            "valid": label is not None, "error": None, "unknown_ids": [], "n_prompt_tokens": 1, "n_new_tokens": 1,
            "hit_max": False, "omitted_ids": [], "stats": None}


SENTS = [{"id": 0, "text": "A <b>bold</b> patient."}, {"id": 1, "text": "Second."}]


def test_unresolved_never_becomes_eligible():
    res = finalize([_row("inclusion", "included", [0]), unresolved_row("exclusion", "x", "timeout")], SENTS)
    assert res["aggregate"]["outcome"] == "Needs more information"
    assert res["rows"][1]["missing_info"] and res["rows"][1]["evidence"] == []
    blocked = finalize([_row("exclusion", "excluded", [1]), unresolved_row("inclusion", "x", "e")], SENTS)
    assert blocked["aggregate"]["outcome"] == "Likely ineligible"


def test_evidence_text_is_copied_by_id_and_hallucinated_ids_flagged():
    r = _row("inclusion", "included", [0, 7])
    r["unknown_ids"] = [7]
    res = finalize([r], SENTS)
    assert res["rows"][0]["evidence"] == [{"id": 0, "text": SENTS[0]["text"]}] and res["rows"][0]["unknown_ids"] == [7]


def test_highlight_escapes_html_and_marks_omitted():
    h = highlight_note_html(SENTS, [0], omitted=[1])
    assert "&lt;b&gt;" in h and "<b>bold" not in h and "omitted" in h and "fde68a" in h


def _prov():
    return make_provenance(model_id="m", adapter="a", adapter_sha256="abc", device="cpu", dtype="float32", mode="KV cache", k=4,
                           max_new_tokens=64, input_sha256="h", versions={"torch": "x"}, note_how="auto", prompt_budget=1024, truncated=False)


def test_report_contains_provenance_disclaimer_and_escapes():
    res = finalize([_row("inclusion", "included", [0], text="<script>alert(1)</script>")], SENTS)
    html, md = to_html(SENTS, res, _prov(), ["careful"]), to_markdown(SENTS, res, _prov(), ["careful"])
    for doc in (html, md):
        assert "not a clinical enrollment decision" in doc and "abc" in doc and "KV cache" in doc and "Potentially eligible" in doc
        assert "careful" in doc
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "greedy, max_new_tokens=64" in md
