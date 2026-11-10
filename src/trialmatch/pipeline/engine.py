"""Inference engine for the app: loads the target (+ optional draft) ONCE and runs one prompt through a chosen decoder.

Modes: 'nocache' (generate_greedy_nocache), 'kv' (generate_greedy_kv), 'speculative' (generate_speculative). All three are
the project's OWN decoders (Phases 5-6), batch size 1, greedy, and produce identical tokens (up to fp rounding, see P5/P6).
HF generate() is NOT used here.

Loading follows the LOCKED procedure through predictor.load_model (fp16 on CUDA, fp32 on CPU; adapter merged).
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from trialmatch.data.prompts import PromptTooLongError, build_prompt, count_tokens, fit_note_to_budget
from trialmatch.data.schema import parse_output
from trialmatch.decoding import (check_compatible, generate_greedy_kv, generate_greedy_nocache, generate_speculative,
                                 kv_bytes_per_token, tokenizers_identical)
from trialmatch.pipeline.predictor import END_TOKEN, load_model
from trialmatch.pipeline.screening import PROMPT_TOKEN_BUDGET, unresolved_row

MODES = ("nocache", "kv", "speculative")
MODE_LABELS = {"nocache": "No cache (full recompute)", "kv": "KV cache", "speculative": "Speculative (draft + KV cache)"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class PeakMemory:
    """Peak memory of one run. CUDA: torch max_memory_allocated (includes weights; 'above weights' = peak - allocated before).
    CPU: peak process RSS sampled every 20 ms in a thread (RSS includes weights and allocator slack; it is not a tensor count)."""

    def __init__(self, device: torch.device):
        self.dev, self.peak_mb, self.base_mb = device, float("nan"), float("nan")
        self._stop, self._thr = threading.Event(), None

    def __enter__(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.base_mb = torch.cuda.memory_allocated() / 2**20
        else:
            import psutil
            proc = psutil.Process()
            self.base_mb = proc.memory_info().rss / 2**20
            self._peak = self.base_mb

            def loop():
                while not self._stop.is_set():
                    self._peak = max(self._peak, proc.memory_info().rss / 2**20)
                    time.sleep(0.02)
            self._thr = threading.Thread(target=loop, daemon=True)
            self._thr.start()
        return self

    def __exit__(self, *exc):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / 2**20
        else:
            self._stop.set()
            self._thr.join()
            self.peak_mb = self._peak
        return False

    @property
    def above_base_mb(self) -> float:
        return self.peak_mb - self.base_mb

    @property
    def kind(self) -> str:
        return "CUDA max allocated" if self.dev.type == "cuda" else "CPU peak RSS"


@dataclass
class Engine:
    model: object
    tok: object
    model_id: str
    adapter: str | None = None
    adapter_sha256: str | None = None
    draft: object | None = None
    draft_id: str | None = None
    notes: list = field(default_factory=list)          # load-time messages shown in the UI

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_id: str, adapter: str | None, draft_id: str | None, device: str | None = None,
             dtype: str | None = None, log=print) -> "Engine":
        t0 = time.time()
        log(f"[load] target {model_id} + adapter {adapter or 'none'} ...")
        model, tok = load_model(model_id, adapter=adapter, device=device, dtype=dtype)
        log(f"[load] target ready in {time.time() - t0:.0f}s on {next(model.parameters()).device}")
        sha = None
        if adapter and (Path(adapter) / "adapter_model.safetensors").exists():
            sha = sha256_file(Path(adapter) / "adapter_model.safetensors")
        eng = cls(model=model, tok=tok, model_id=model_id, adapter=adapter, adapter_sha256=sha)
        if draft_id:
            t1 = time.time()
            log(f"[load] draft {draft_id} ...")
            draft, tok_d = load_model(draft_id, adapter=None, device=device, dtype=dtype)
            if not (tokenizers_identical(tok, tok_d) and check_compatible(model, draft)["equal"]):
                eng.notes.append("Draft model rejected: tokenizer/vocabulary differs from the target (speculative mode disabled).")
                del draft
            else:
                eng.draft, eng.draft_id = draft, draft_id
                log(f"[load] draft ready in {time.time() - t1:.0f}s")
        eng.end_id = tok.convert_tokens_to_ids(END_TOKEN)
        return eng

    # ------------------------------------------------------------------ info
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype_name(self) -> str:
        return str(next(self.model.parameters()).dtype).replace("torch.", "")

    @property
    def end_id(self) -> int:
        return self.__dict__.get("_end_id", self.tok.convert_tokens_to_ids(END_TOKEN))

    @end_id.setter
    def end_id(self, v: int) -> None:
        self.__dict__["_end_id"] = v

    def versions(self) -> dict:
        import transformers
        v = {"torch": torch.__version__, "transformers": transformers.__version__}
        try:
            import peft
            v["peft"] = peft.__version__
        except Exception:
            pass
        return v

    def available_modes(self) -> list[str]:
        return [m for m in MODES if m != "speculative" or self.draft is not None]

    # ------------------------------------------------------------------ one generation
    def run_prompt(self, prompt: str, mode: str, k: int = 4, max_new_tokens: int = 64, measure_memory: bool = True) -> dict:
        """Tokenise like the evaluation (add_special_tokens=False), decode with `mode`, return text/tokens/stats."""
        if mode not in MODES:
            raise ValueError(f"unknown decoding mode {mode!r}")
        if mode == "speculative" and self.draft is None:
            raise RuntimeError("speculative mode needs the draft model, which is not loaded (start the app without --no-draft).")
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        mem = PeakMemory(self.device) if measure_memory else None
        t0 = time.perf_counter()
        if mem:
            mem.__enter__()
        try:
            if mode == "nocache":
                toks, st = generate_greedy_nocache(self.model, ids, max_new_tokens, eos_id=self.end_id)
            elif mode == "kv":
                toks, st = generate_greedy_kv(self.model, ids, max_new_tokens, eos_id=self.end_id)
            else:
                toks, st = generate_speculative(self.model, self.draft, ids, max_new_tokens, k=k, eos_id=self.end_id)
        finally:
            if mem:
                mem.__exit__(None, None, None)
        wall = time.perf_counter() - t0
        stats = st.as_dict()
        # KV bytes are COMPUTED (formula x cache length), not measured: no-cache keeps no cache between steps.
        bpt = kv_bytes_per_token(self.model.config, next(self.model.parameters()).element_size())
        stats["kv_cache_mb_computed"] = 0.0 if mode == "nocache" else bpt * (stats.get("cache_len_final") or 0) / 2**20
        stats.update(mode=mode, wall_s=wall, peak_mb=mem.peak_mb if mem else float("nan"),
                     peak_above_base_mb=mem.above_base_mb if mem else float("nan"),
                     peak_kind=mem.kind if mem else "not measured")
        text = self.tok.decode(toks[: toks.index(self.end_id) + 1] if self.end_id in toks else toks, skip_special_tokens=True)
        return {"tokens": list(toks), "text": text, "stats": stats, "prompt_len": ids.shape[1]}

    # ------------------------------------------------------------------ one criterion (screening)
    def prepare(self, sentences: Sequence[dict], ctype: str, text: str, budget: int = PROMPT_TOKEN_BUDGET) -> tuple[str, list[int]]:
        """Training-format prompt. Over-long notes: trailing sentences are dropped (never silently: the ids are returned and
        the prompt itself says which were omitted). Raises PromptTooLongError if it cannot fit."""
        kept, dropped = fit_note_to_budget(
            sentences, lambda s: count_tokens(self.tok, build_prompt(self.tok, s, ctype, text)), budget)
        prompt = build_prompt(self.tok, kept, ctype, text, omitted_ids=dropped)
        if dropped and count_tokens(self.tok, prompt) > budget:       # the 'omitted' notice itself can push it over
            raise PromptTooLongError("prompt exceeds the token budget after truncation")
        return prompt, dropped

    def run_criterion(self, sentences: Sequence[dict], ctype: str, text: str, mode: str, k: int, max_new_tokens: int,
                      budget: int = PROMPT_TOKEN_BUDGET) -> dict:
        """prepare -> decode -> parse_output. Prompt problems return an unresolved row (never 'eligible')."""
        try:
            prompt, dropped = self.prepare(sentences, ctype, text, budget)
        except PromptTooLongError as e:
            return unresolved_row(ctype, text, f"Prompt too long: {e}")
        out = self.run_prompt(prompt, mode, k=k, max_new_tokens=max_new_tokens)
        p = parse_output(out["text"], criterion_type=ctype, n_sentences=len(sentences))
        return {"criterion_type": ctype, "criterion_text": text, "raw_output": out["text"],
                "pred_label": p["label"] if p["valid"] else None, "pred_evidence_ids": p["evidence_ids"], "valid": p["valid"],
                "error": p["error"], "unknown_ids": p["unknown_ids"], "n_prompt_tokens": out["prompt_len"],
                "n_new_tokens": len(out["tokens"]), "hit_max": len(out["tokens"]) >= max_new_tokens and self.end_id not in out["tokens"],
                "omitted_ids": dropped, "stats": out["stats"]}
