"""Tiny random Qwen2 + word-level tokenizer for CPU tests (no network, no downloads)."""
from __future__ import annotations

import json

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def make_tok(texts: list[str]) -> PreTrainedTokenizerFast:
    pre = pre_tokenizers.Whitespace()
    words = sorted({w for t in texts for w, _ in pre.pre_tokenize_str(t.replace("<|im_start|>", " ").replace("<|im_end|>", " "))})
    vocab = {s: i for i, s in enumerate(["<unk>"] + SPECIALS)}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<|endoftext|>")
    fast.add_special_tokens({"additional_special_tokens": SPECIALS[1:]})
    return fast


def make_model(vocab_size: int) -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg).float()


def make_records(n: int = 12) -> list[dict]:
    recs = []
    for i in range(n):
        lab = ["included", "not included", "not enough information", "not applicable"][i % 4]
        ids = [] if i % 3 == 0 else [0, 1][: 1 + i % 2]
        prompt = (f"<|im_start|>system\nscreen<|im_end|>\n<|im_start|>user\n[0] patient {i} has cough. [1] fever.\n"
                  f"Criterion: cough {i}\n<|im_end|>\n<|im_start|>assistant\n")
        target = json.dumps({"label": lab, "evidence_ids": ids})
        recs.append({"annotation_id": i, "patient_id": f"p{i}", "trial_id": "t", "criterion_type": "inclusion",
                     "criterion_text": f"cough {i}", "patient_sentences": [{"id": 0, "text": "a"}, {"id": 1, "text": "b"}],
                     "gold_label": lab, "gold_evidence_ids": ids, "prompt": prompt, "target": target,
                     "n_prompt_tokens": 30})
    return recs
