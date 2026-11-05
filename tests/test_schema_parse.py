import pytest
from trialmatch.data.schema import ALL_LABELS, make_target, parse_evidence_ids, parse_output


def test_roundtrip():
    t = make_target("not enough information", [3, 1, 3])
    assert t == '{"label": "not enough information", "evidence_ids": [1, 3]}'
    r = parse_output(t, "inclusion", n_sentences=5)
    assert r["valid"] and r["label"] == "not enough information" and r["evidence_ids"] == [1, 3] and r["unknown_ids"] == []


def test_empty_evidence_is_valid():
    assert parse_output('{"label": "excluded", "evidence_ids": []}', "exclusion")["valid"]


@pytest.mark.parametrize("text,err", [
    ("", "empty_output"), ("   ", "empty_output"), ("I think it is included", "no_json_object"),
    ('{"label": "included", "evidence_ids": [1, 2', "no_json_object"),            # truncated generation
    ('{"label": 3, "evidence_ids": []}', "label_missing_or_not_string"),
    ('{"evidence_ids": []}', "label_missing_or_not_string"),
    ('{"label": "excluded", "evidence_ids": []}', "label_not_in_allowed_set"),   # exclusion label on inclusion criterion
    ('{"label": "maybe", "evidence_ids": []}', "label_not_in_allowed_set"),
    ('{"label": "included"}', "evidence_ids_missing_or_not_list"),
    ('{"label": "included", "evidence_ids": "1,2"}', "evidence_ids_missing_or_not_list"),
    ('{"label": "included", "evidence_ids": ["1"]}', "evidence_ids_not_all_int"),
    ('{"label": "included", "evidence_ids": [true]}', "evidence_ids_not_all_int"),
    ('{"label": "included", "evidence_ids": [1.5]}', "evidence_ids_not_all_int"),
    ('[1, 2, 3]', "no_json_object"),
])
def test_invalid(text, err):
    r = parse_output(text, "inclusion")
    assert not r["valid"] and r["error"] == err


def test_tolerates_fences_and_chatter():
    r = parse_output('Sure!\n```json\n{"label": "Included", "evidence_ids": [2, 0]}\n```\nHope this helps {x}', "inclusion")
    assert r["valid"] and r["label"] == "included" and r["evidence_ids"] == [0, 2]


def test_hallucinated_ids_flagged_not_invalid():
    r = parse_output('{"label": "included", "evidence_ids": [0, 9, -1]}', "inclusion", n_sentences=4)
    assert r["valid"] and r["unknown_ids"] == [-1, 9]


def test_untyped_accepts_all_six_labels():
    for l in ALL_LABELS:
        assert parse_output(make_target(l, []))["valid"]


@pytest.mark.parametrize("raw,exp", [("[0, 1, 2]", [0, 1, 2]), ("[]", []), ("", []), (None, []), ("[3, 3, 1]", [1, 3]), ([2, 1], [1, 2])])
def test_parse_evidence_ids(raw, exp):
    assert parse_evidence_ids(raw) == exp


@pytest.mark.parametrize("raw", ["[a]", "5", '["1"]', "[1.5]"])
def test_parse_evidence_ids_bad(raw):
    with pytest.raises(ValueError):
        parse_evidence_ids(raw)
