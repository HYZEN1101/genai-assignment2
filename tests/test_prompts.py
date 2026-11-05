import pytest
from trialmatch.data.prompts import PromptTooLongError, build_prompt, count_tokens, fit_note_to_budget, split_numbered_note
from conftest import FakeTok, make_note


def test_split_basic_and_ids():
    s = split_numbered_note("0. A 58-year-old woman presents. 1. Pain started at age 58. 2. Denies diabetes.")
    assert [x["id"] for x in s] == [0, 1, 2]
    assert s[0]["text"] == "A 58-year-old woman presents."
    assert s[1]["text"] == "Pain started at age 58."   # "58." inside a sentence is not a marker


def test_split_many_sentences_and_numbers_in_text():
    note = make_note(12)
    s = split_numbered_note(note)
    assert [x["id"] for x in s] == list(range(12))
    assert all("aged 58. Hb low." in x["text"] for x in s)


def test_split_requires_zero_start():
    with pytest.raises(ValueError):
        split_numbered_note("1. no zero")


def test_user_message_contains_ids_and_labels():
    tok = FakeTok()
    s = split_numbered_note(make_note(3))
    p = build_prompt(tok, s, "exclusion", "Pregnant women")
    assert "[0] " in p and "[2] " in p and "excluded | not excluded | not enough information | not applicable" in p
    assert "included | not included" not in p


def test_fit_no_truncation_when_short():
    s = split_numbered_note(make_note(4)); tok = FakeTok()
    f = lambda ss: count_tokens(tok, build_prompt(tok, ss, "inclusion", "x"))
    kept, dropped = fit_note_to_budget(s, f, 10_000)
    assert kept == s and dropped == []


def test_fit_drops_from_end_and_protects_gold():
    s = split_numbered_note(make_note(30)); tok = FakeTok()
    f = lambda ss: count_tokens(tok, build_prompt(tok, ss, "inclusion", "x"))
    budget = f(s) - 60
    kept, dropped = fit_note_to_budget(s, f, budget)
    assert f(kept) <= budget and dropped == sorted(dropped) and max(dropped) == 29
    assert dropped == list(range(30 - len(dropped), 30))            # contiguous tail
    kept2, dropped2 = fit_note_to_budget(s, f, budget, protect_ids=[29])
    assert 29 not in dropped2 and f(kept2) <= budget and any(k["id"] == 29 for k in kept2)


def test_fit_raises_when_impossible():
    s = split_numbered_note(make_note(5)); tok = FakeTok()
    f = lambda ss: count_tokens(tok, build_prompt(tok, ss, "inclusion", "x"))
    with pytest.raises(PromptTooLongError):
        fit_note_to_budget(s, f, 5)
