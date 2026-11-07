import pytest
import torch

from tiny_lm import make_model, make_records, make_tok
from trialmatch.pipeline.predictor import Predictor


def test_generate_and_parse_shapes():
    recs = make_records(6)
    tok = make_tok([r["prompt"] + r["target"] for r in recs])
    model = make_model(len(tok)).eval()
    p = Predictor(model, tok, max_new_tokens=8, batch_size=4)
    rows, timing = p.predict_records(recs)
    assert len(rows) == 6 and timing["n"] == 6
    assert all(r["n_new_tokens"] <= 8 for r in rows)
    assert [r["annotation_id"] for r in rows] == [r["annotation_id"] for r in recs]   # order preserved
    # a random model will not emit valid JSON: must be flagged invalid, not crash
    assert all((not r["valid"]) or r["pred_label"] is not None for r in rows)


def test_batching_does_not_change_greedy_output_fp32():
    recs = make_records(6)
    tok = make_tok([r["prompt"] + r["target"] for r in recs])
    model = make_model(len(tok)).eval()
    a = Predictor(model, tok, max_new_tokens=6, batch_size=1).generate_batch([r["prompt"] for r in recs])
    b = Predictor(model, tok, max_new_tokens=6, batch_size=6).generate_batch([r["prompt"] for r in recs])
    assert [x["text"] for x in a] == [x["text"] for x in b]
