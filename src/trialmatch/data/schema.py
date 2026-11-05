"""Dataset schema: column roles, label sets, output format, and the output parser.

Source dataset: ncbi/TrialGPT-Criterion-Annotations (1,015 rows, 53 patients, 103 trials; one HF split "train").
Columns seen on the dataset card (3 Nov 2026):
  annotation_id, patient_id, note, trial_id, trial_title, criterion_type, criterion_text,
  gpt4_explanation, explanation_correctness, gpt4_sentences, expert_sentences,
  gpt4_eligibility, expert_eligibility, training
"""
from __future__ import annotations

import ast
import json
from typing import Any, Iterable

# Everything produced by the earlier GPT-4 system or derived from comparing it with the experts.
# NEVER an input feature, never in a prompt/target, never in evaluation. Enforced by tests/test_leakage.py.
GPT4_COLUMNS: frozenset[str] = frozenset(
    {"gpt4_explanation", "explanation_correctness", "gpt4_sentences", "gpt4_eligibility"}
)
# Original TrialGPT train/test flag. Not GPT-4 output, but not ours: we use our own patient-grouped split.
UNUSED_COLUMNS: frozenset[str] = frozenset({"training"})
USED_COLUMNS: tuple[str, ...] = (
    "annotation_id", "patient_id", "note", "trial_id",
    "criterion_type", "criterion_text", "expert_eligibility", "expert_sentences",
)
assert not (set(USED_COLUMNS) & (GPT4_COLUMNS | UNUSED_COLUMNS))

# Fields allowed in data/processed/*.jsonl (anything else fails the leakage test).
RECORD_FIELDS: tuple[str, ...] = (
    "annotation_id", "patient_id", "trial_id", "criterion_type", "criterion_text",
    "patient_sentences", "gold_label", "gold_evidence_ids", "prompt", "target",
    "truncated_ids", "n_prompt_tokens_full", "n_prompt_tokens", "n_target_tokens",
)

# Exact dataset strings of expert_eligibility; never collapsed to binary.
INCLUSION_LABELS: tuple[str, ...] = ("included", "not included", "not enough information", "not applicable")
EXCLUSION_LABELS: tuple[str, ...] = ("excluded", "not excluded", "not enough information", "not applicable")
LABELS_BY_TYPE: dict[str, tuple[str, ...]] = {"inclusion": INCLUSION_LABELS, "exclusion": EXCLUSION_LABELS}
ALL_LABELS: tuple[str, ...] = tuple(sorted(set(INCLUSION_LABELS) | set(EXCLUSION_LABELS)))
CRITERION_TYPES: tuple[str, ...] = ("inclusion", "exclusion")


def allowed_labels(criterion_type: str | None) -> tuple[str, ...]:
    """Label set for a criterion type; all six labels when the type is unknown."""
    if criterion_type is None:
        return ALL_LABELS
    return LABELS_BY_TYPE[criterion_type]


def parse_evidence_ids(raw: Any) -> list[int]:
    """expert_sentences is stored as a string like "[0, 1, 2]" (or "[]"). Returns sorted unique ints."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if s == "":
            return []
        try:
            val = json.loads(s)
        except json.JSONDecodeError:
            val = ast.literal_eval(s)  # raises ValueError on garbage: we want to know
    else:
        val = raw
    if hasattr(val, "tolist"):  # numpy array
        val = val.tolist()
    if not isinstance(val, (list, tuple)):
        raise ValueError(f"evidence field is not a list: {raw!r}")
    out = []
    for v in val:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"non-integer evidence id {v!r} in {raw!r}")
        out.append(v)
    return sorted(set(out))


def make_target(label: str, evidence_ids: Iterable[int]) -> str:
    """Compact JSON the model must produce. Key order fixed: label first, then evidence_ids."""
    return json.dumps({"label": label, "evidence_ids": sorted(set(int(i) for i in evidence_ids))})


def parse_output(text: str, criterion_type: str | None = None, n_sentences: int | None = None) -> dict:
    """Parse a model completion into {label, evidence_ids, valid, error, unknown_ids}.

    valid=False (counted as a wrong prediction by Phase 3 metrics) when: no JSON object, label missing / not a
    string / not in the allowed set for `criterion_type`, evidence_ids missing / not a list of ints.
    Never raises. unknown_ids = cited ids outside range(n_sentences) (hallucinated IDs) if n_sentences is given;
    they do NOT make the output invalid, Phase 3 counts them as false-positive evidence.
    """
    res: dict = {"label": None, "evidence_ids": [], "valid": False, "error": None, "unknown_ids": []}
    if not isinstance(text, str) or not text.strip():
        res["error"] = "empty_output"
        return res
    obj = _first_json_object(text)
    if obj is None:
        res["error"] = "no_json_object"
        return res
    label = obj.get("label")
    if not isinstance(label, str):
        res["error"] = "label_missing_or_not_string"
        return res
    res["label"] = label.strip().lower()
    if res["label"] not in allowed_labels(criterion_type):
        res["error"] = "label_not_in_allowed_set"
        return res
    ids = obj.get("evidence_ids")
    if not isinstance(ids, list):
        res["error"] = "evidence_ids_missing_or_not_list"
        return res
    if any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        res["error"] = "evidence_ids_not_all_int"
        return res
    res["evidence_ids"] = sorted(set(ids))
    if n_sentences is not None:
        res["unknown_ids"] = [i for i in res["evidence_ids"] if i < 0 or i >= n_sentences]
    res["valid"] = True
    return res


def _first_json_object(text: str) -> dict | None:
    """First '{' in `text` that starts a decodable JSON object (tolerates code fences / chatter around it)."""
    dec = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = dec.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def assert_no_gpt4_keys(record: dict) -> None:
    bad = set(record) & (GPT4_COLUMNS | UNUSED_COLUMNS)
    if bad:
        raise AssertionError(f"forbidden columns in record: {sorted(bad)}")
    extra = set(record) - set(RECORD_FIELDS)
    if extra:
        raise AssertionError(f"unexpected fields in record (not in RECORD_FIELDS): {sorted(extra)}")
