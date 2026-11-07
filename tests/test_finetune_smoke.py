"""CPU tests of the fine-tuning code on a tiny random Qwen2 (no network)."""
import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("peft")
from trialmatch.train import finetune as ft  # noqa: E402
from tiny_lm import make_model, make_records, make_tok  # noqa: E402


@pytest.fixture
def setup(tmp_path):
    recs = make_records(12)
    tok = make_tok([r["prompt"] + r["target"] for r in recs])
    for name in ("train", "val"):
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return recs, tok, tmp_path


def test_label_masking_and_eos(setup):
    recs, tok, _ = setup
    item = ft.tokenize_record(tok, recs[0], 256)
    n_prompt = len(tok(recs[0]["prompt"], add_special_tokens=False).input_ids)
    assert all(x == -100 for x in item["labels"][:n_prompt])
    assert all(x != -100 for x in item["labels"][n_prompt:])
    assert item["labels"][-1] == tok.convert_tokens_to_ids("<|im_end|>")
    assert item["input_ids"][:n_prompt] == tok(recs[0]["prompt"], add_special_tokens=False).input_ids


def test_too_long_raises(setup):
    recs, tok, _ = setup
    with pytest.raises(ValueError):
        ft.tokenize_record(tok, recs[0], 10)


def test_collate_right_padding(setup):
    recs, tok, _ = setup
    items = [ft.tokenize_record(tok, r, 256) for r in recs[:3]]
    b = ft.collate(items, tok.pad_token_id)
    for i, it in enumerate(items):
        n = len(it["input_ids"])
        assert b["attention_mask"][i, :n].all() and not b["attention_mask"][i, n:].any()
        assert (b["labels"][i, n:] == -100).all()


def test_selective_head_loss_equals_hf_loss(setup):
    """Our lm_head-on-supervised-positions loss must equal the standard full-logits HF loss."""
    recs, tok, _ = setup
    model = make_model(len(tok)).eval()
    items = [ft.tokenize_record(tok, r, 256) for r in recs[:4]]
    b = ft.collate(items, tok.pad_token_id)
    s, n = ft.token_loss_sum(model, b)
    ref = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss
    assert abs(float(s) / n - float(ref)) < 1e-5


def test_epoch_batches_deterministic():
    a = ft.epoch_batches(10, 4, 42, 0)
    assert a == ft.epoch_batches(10, 4, 42, 0) and a != ft.epoch_batches(10, 4, 42, 1)
    assert sorted(i for b in a for i in b) == list(range(10))


def test_lr_schedule():
    assert ft.lr_at(0, 100, 10, 1.0) == pytest.approx(0.1)
    assert ft.lr_at(9, 100, 10, 1.0) == pytest.approx(1.0)
    assert ft.lr_at(100, 100, 10, 1.0) == pytest.approx(0.0, abs=1e-9)


def _cfg(tmp):
    return {"run_name": "t", "model_id": "tiny", "train_file": str(tmp / "train.jsonl"), "val_file": str(tmp / "val.jsonl"),
            "out_dir": str(tmp / "run"), "repo_logs_dir": str(tmp / "logs"), "device": "cpu", "fp16": False, "seed": 1,
            "max_len": 256, "epochs": 2, "micro_batch": 2, "grad_accum": 2, "lr": 1e-2, "warmup_ratio": 0.0,
            "gradient_checkpointing": True, "eval_every_steps": 1, "lora": {"r": 4, "alpha": 8, "dropout": 0.0}}


def test_train_runs_logs_saves_and_loss_drops(setup):
    recs, tok, tmp = setup
    s = ft.train(_cfg(tmp), tok=tok, model=make_model(len(tok)))
    out = tmp / "run"
    assert (out / "best" / "adapter_config.json").exists() and (out / "last" / "state.pt").exists()
    assert (out / "epoch_2").exists() and (tmp / "logs" / "train_t.csv").exists()
    rows = (out / "train_log.csv").read_text().strip().splitlines()
    assert rows[0].startswith("step,epoch,train_loss,val_loss") and len(rows) == 1 + s["steps"]
    first = float(rows[1].split(",")[2])
    losses = [float(r.split(',')[2]) for r in rows[1:]]
    assert math.isfinite(first) and min(losses) < first
    assert s["trainable_params"] > 0 and s["trainable_params"] < s["total_params"] / 2


def test_resume_continues_step_counter(setup):
    recs, tok, tmp = setup
    cfg = _cfg(tmp)
    cfg["max_steps"] = 2
    ft.train(cfg, tok=tok, model=make_model(len(tok)))
    cfg["max_steps"] = 4
    s = ft.train(cfg, tok=tok, model=make_model(len(tok)))
    steps = [int(l.split(",")[0]) for l in (tmp / "run" / "train_log.csv").read_text().strip().splitlines()[1:]]
    assert steps == [1, 2, 3, 4] and s["steps"] == 4
