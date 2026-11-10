"""Phase 7: Engine wiring on tiny random Qwen2 models (CPU, no downloads): all three decoders give identical tokens."""
import pytest

torch = pytest.importorskip("torch")

from tiny_lm import make_model, make_records, make_tok  # noqa: E402
from trialmatch.pipeline.engine import Engine, PeakMemory  # noqa: E402


@pytest.fixture(scope="module")
def eng():
    recs = make_records(4)
    tok = make_tok([r["prompt"] + r["target"] for r in recs])
    model = make_model(len(tok)).eval()
    draft = make_model(len(tok)).eval()
    e = Engine(model=model, tok=tok, model_id="tiny", draft=draft, draft_id="tiny-draft")
    e.end_id = tok.convert_tokens_to_ids("<|im_end|>")
    return e, recs


def test_three_modes_token_equal_and_stats(eng):
    e, recs = eng
    outs = {m: e.run_prompt(recs[0]["prompt"], m, k=3, max_new_tokens=12) for m in ("nocache", "kv", "speculative")}
    assert outs["nocache"]["tokens"] == outs["kv"]["tokens"] == outs["speculative"]["tokens"]
    s = outs["speculative"]["stats"]
    assert 0.0 <= s["acceptance_rate"] <= 1.0 and s["rounds"] >= 0 and s["peak_kind"] == "CPU peak RSS"
    assert outs["nocache"]["stats"]["kv_cache_mb_computed"] == 0.0 and outs["kv"]["stats"]["kv_cache_mb_computed"] > 0
    assert outs["kv"]["stats"]["cache_len_final"] == outs["kv"]["prompt_len"] + len(outs["kv"]["tokens"]) - 1


def test_speculative_without_draft_is_a_clear_error(eng):
    e, recs = eng
    nodraft = Engine(model=e.model, tok=e.tok, model_id="tiny")
    assert "speculative" not in nodraft.available_modes()
    with pytest.raises(RuntimeError, match="draft"):
        nodraft.run_prompt(recs[0]["prompt"], "speculative")
    with pytest.raises(ValueError):
        nodraft.run_prompt(recs[0]["prompt"], "bogus")


def test_peak_memory_context_runs_on_cpu():
    with PeakMemory(torch.device("cpu")) as m:
        _ = torch.zeros(1000, 1000)
    assert m.peak_mb >= m.base_mb > 0
