"""Predictor: load base model (+ optional LoRA adapter), greedy generation, parse_output.

LOCKED LOADING PROCEDURE (used by eval, benchmarks, speculative decoding and the app so every phase sees identical weights):
    1. AutoModelForCausalLM.from_pretrained(model_id, dtype=fp16 on cuda / fp32 on cpu)
    2. if adapter: PeftModel.from_pretrained(model, adapter) then model.merge_and_unload()   (adapter = local dir or HF repo id)
    3. model.eval()

Greedy decoding uses an explicit GenerationConfig with repetition_penalty=1.0: the Qwen2.5-Instruct default
generation_config has repetition_penalty=1.1, which would silently change "greedy" outputs (and break the later
token-equality tests against our own argmax loops).

NOTE: this file uses Hugging Face generate() ONLY as the baseline/eval predictor. The explicit KV-cache and speculative
decoders (Phases 5-6) are separate implementations; generate() is not a deliverable for those tasks.
"""
from __future__ import annotations

import time
from typing import Sequence

import torch

from trialmatch.data.schema import parse_output

END_TOKEN = "<|im_end|>"


def load_model(model_id: str, adapter: str | None = None, device: str | None = None, dtype: str | None = None,
               merge: bool = True):
    """Return (model, tokenizer). dtype: 'float16' | 'float32' | None (auto: fp16 on cuda, fp32 on cpu)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = "float16" if device == "cuda" else "float32"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"  # generation needs left padding so the last position is a real token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device)  # merge on the device
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return model, tok


class Predictor:
    def __init__(self, model, tok, max_new_tokens: int = 64, batch_size: int = 8):
        self.model, self.tok = model, tok
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.device = next(model.parameters()).device
        tok.padding_side = "left"
        self.end_id = tok.convert_tokens_to_ids(END_TOKEN)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.end_id

    def _gen_config(self):
        from transformers import GenerationConfig
        return GenerationConfig(do_sample=False, max_new_tokens=self.max_new_tokens, repetition_penalty=1.0,
                                eos_token_id=self.end_id, pad_token_id=self.pad_id)

    @torch.no_grad()
    def generate_batch(self, prompts: Sequence[str]) -> list[dict]:
        """Greedy completions, order preserved. Each item: {text, n_new_tokens, hit_max}.

        Prompts are length-sorted into batches (less padding). Left-padded batching can differ from batch size 1 by fp16
        rounding; base and fine-tuned runs use the identical procedure, so the comparison is fair.
        """
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        results: dict[int, dict] = {}
        gc = self._gen_config()
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            out = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], generation_config=gc)
            new = out[:, enc["input_ids"].shape[1]:]
            for row, i in zip(new, idx):
                ids = row.tolist()
                if self.end_id in ids:
                    ids = ids[: ids.index(self.end_id) + 1]
                    hit_max = False
                else:
                    ids = [t for t in ids if t != self.pad_id]
                    hit_max = len(ids) >= self.max_new_tokens
                results[i] = {"text": self.tok.decode(ids, skip_special_tokens=True),
                              "n_new_tokens": len(ids), "hit_max": hit_max}
        return [results[i] for i in range(len(prompts))]

    def predict_records(self, records: Sequence[dict], log_every: int = 0) -> tuple[list[dict], dict]:
        """Run the model on preprocessed records (they carry the rendered prompt). Returns (rows, timing)."""
        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gens = self.generate_batch([r["prompt"] for r in records])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        rows = []
        for r, g in zip(records, gens):
            p = parse_output(g["text"], criterion_type=r["criterion_type"], n_sentences=len(r["patient_sentences"]))
            rows.append({
                "annotation_id": r["annotation_id"], "patient_id": r["patient_id"], "trial_id": r["trial_id"],
                "criterion_type": r["criterion_type"], "gold_label": r["gold_label"],
                "gold_evidence_ids": r["gold_evidence_ids"], "raw_output": g["text"],
                "pred_label": p["label"] if p["valid"] else None, "pred_evidence_ids": p["evidence_ids"],
                "valid": p["valid"], "error": p["error"], "unknown_ids": p["unknown_ids"],
                "n_prompt_tokens": r["n_prompt_tokens"], "n_new_tokens": g["n_new_tokens"], "hit_max": g["hit_max"],
            })
        return rows, {"seconds": dt, "n": len(rows), "new_tokens": sum(g["n_new_tokens"] for g in gens)}
