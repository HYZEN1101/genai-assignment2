"""LoRA supervised fine-tuning (custom loop, resumable). Phase 2.

Design (see docs/P2_DECISIONS_append.md):
- Input text = record["prompt"] (ends with "<|im_start|>assistant\\n") + record["target"] + "<|im_end|>".
  Prompt and completion are tokenized SEPARATELY and concatenated, so the boundary is exactly what generation sees.
- Loss only on target + <|im_end|> tokens (labels = -100 on the prompt).
- Memory trick: run the transformer body, then apply lm_head ONLY at supervised positions (~20-40 tokens per
  example instead of ~400), so we never build a (B, L, 151k) logits tensor.
- fp16 base weights, fp32 LoRA weights, torch.autocast + GradScaler (T4 has no native bf16).
- Resumable: `last/` (adapter + optimizer/scheduler/scaler state) is rewritten at every eval point.

Run:  python -m trialmatch.train.finetune --config configs/ft_A.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

END_TOKEN = "<|im_end|>"
DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ----------------------------------------------------------------------------- data
def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_record(tok, rec: dict, max_len: int) -> dict:
    """-> {input_ids, labels}. labels = -100 on prompt tokens. Raises if too long (P1: nothing is truncated)."""
    p_ids = tok(rec["prompt"], add_special_tokens=False).input_ids
    t_ids = tok(rec["target"] + END_TOKEN, add_special_tokens=False).input_ids
    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    if t_ids[-1] != end_id:
        raise ValueError(f"target does not end with {END_TOKEN} token (annotation {rec.get('annotation_id')})")
    ids = p_ids + t_ids
    if len(ids) > max_len:
        raise ValueError(f"record {rec.get('annotation_id')} has {len(ids)} tokens > max_len {max_len}")
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + t_ids}


def collate(items: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad. attention_mask is 1 on real tokens. Labels padded with -100."""
    L = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        lab[i, :n] = torch.tensor(x["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def epoch_batches(n_items: int, micro_batch: int, seed: int, epoch: int) -> list[list[int]]:
    """Deterministic shuffle per (seed, epoch) so a resumed run sees the same order."""
    order = np.random.default_rng(seed + 1000 * epoch).permutation(n_items).tolist()
    return [order[i:i + micro_batch] for i in range(0, n_items, micro_batch)]


# ----------------------------------------------------------------------------- loss
def _inner(model):
    """The underlying *ForCausalLM (works for a PeftModel and for a plain model)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def token_loss_sum(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Sum of cross-entropy over supervised tokens, and their count.

    Position t predicts token t+1, so labels are shifted left by one. Logits are computed only where the
    shifted label != -100 (lm_head applied to the selected hidden states only).
    """
    inner = _inner(model)
    out = inner.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    h = out.last_hidden_state[:, :-1]                      # (B, L-1, H)
    tgt = batch["labels"][:, 1:]                           # (B, L-1)
    sel = tgt != -100
    logits = inner.lm_head(h[sel]).float()                 # (n_supervised, V) only
    loss = F.cross_entropy(logits, tgt[sel], reduction="sum")
    return loss, int(sel.sum().item())


@torch.no_grad()
def evaluate_loss(model, items: list[dict], pad_id: int, micro_batch: int, device: str, use_amp: bool) -> float:
    """Token-weighted mean validation loss."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(items), micro_batch):
        b = {k: v.to(device) for k, v in collate(items[i:i + micro_batch], pad_id).items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            s, c = token_loss_sum(model, b)
        total += float(s.item())
        n += c
    model.train()
    return total / max(n, 1)


# ----------------------------------------------------------------------------- schedule
def lr_at(step: int, total: int, warmup: int, base_lr: float, min_ratio: float = 0.0) -> float:
    """Linear warm-up then cosine decay to min_ratio*base_lr. `step` = completed optimizer steps."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


# ----------------------------------------------------------------------------- main
def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def train(cfg: dict[str, Any], tok=None, model=None) -> dict:
    """Train. `tok`/`model` may be injected (CPU smoke test); otherwise loaded from cfg['model_id']."""
    from peft import LoraConfig, PeftModel, get_peft_model

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("fp16", True)) and device == "cuda"
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")

    max_len = int(cfg.get("max_len", 1024))
    train_recs = load_jsonl(cfg["train_file"])
    val_recs = load_jsonl(cfg["val_file"])
    if cfg.get("max_train_records"):
        train_recs = train_recs[: int(cfg["max_train_records"])]
    if cfg.get("max_val_records"):
        val_recs = val_recs[: int(cfg["max_val_records"])]
    train_items = [tokenize_record(tok, r, max_len) for r in train_recs]
    val_items = [tokenize_record(tok, r, max_len) for r in val_recs]
    print(f"[data] train {len(train_items)} val {len(val_items)} | longest train seq "
          f"{max(len(x['input_ids']) for x in train_items)} tokens")

    micro, accum = int(cfg["micro_batch"]), int(cfg["grad_accum"])
    steps_per_epoch = math.ceil(math.ceil(len(train_items) / micro) / accum)
    epochs = int(cfg["epochs"])
    total_steps = int(cfg.get("max_steps") or steps_per_epoch * epochs)
    warmup = int(round(float(cfg.get("warmup_ratio", 0.1)) * total_steps))
    eval_every = int(cfg.get("eval_every_steps", steps_per_epoch))
    lr = float(cfg["lr"])

    # ---- model (+ resume)
    last_dir, best_dir = out_dir / "last", out_dir / "best"
    resume = (last_dir / "state.pt").exists() and bool(cfg.get("resume", True))
    if model is None:
        from transformers import AutoModelForCausalLM
        dtype = torch.float16 if use_amp else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lc = cfg.get("lora", {})
    if resume:
        model = PeftModel.from_pretrained(model, str(last_dir), is_trainable=True)
        print(f"[resume] adapter loaded from {last_dir}")
    else:
        model = get_peft_model(model, LoraConfig(
            r=int(lc.get("r", 16)), lora_alpha=int(lc.get("alpha", 32)), lora_dropout=float(lc.get("dropout", 0.05)),
            target_modules=list(lc.get("target_modules", DEFAULT_TARGETS)), bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():  # LoRA master weights in fp32 (fp16 weights + GradScaler => NaN risk otherwise)
        if p.requires_grad:
            p.data = p.data.float()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {n_train / 1e6:.2f}M / {n_total / 1e6:.1f}M ({100 * n_train / n_total:.2f}%) "
          f"| device {device} amp {use_amp}")
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best_val, best_step, elapsed_prev = 0, float("inf"), -1, 0.0
    if resume:
        st = torch.load(last_dir / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step, best_val, best_step, elapsed_prev = st["step"], st["best_val"], st["best_step"], st["elapsed"]
        print(f"[resume] step {step}/{total_steps} best_val {best_val:.4f}@{best_step}")

    log_path = out_dir / "train_log.csv"
    cols = ["step", "epoch", "train_loss", "val_loss", "lr", "grad_norm", "elapsed_s", "peak_mem_gb"]
    new_log = not (resume and log_path.exists())
    logf = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(cols)
        logf.flush()

    def save_adapter(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(path))

    def do_eval_and_save(cur_step: int, elapsed: float) -> float:
        nonlocal best_val, best_step
        v = evaluate_loss(model, val_items, pad_id, micro, device, use_amp)
        if v < best_val:
            best_val, best_step = v, cur_step
            save_adapter(best_dir)
            (best_dir / "best_info.json").write_text(json.dumps({"step": cur_step, "val_loss": v}))
        save_adapter(last_dir)
        torch.save({"step": cur_step, "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best_val": best_val, "best_step": best_step, "elapsed": elapsed}, last_dir / "state.pt")
        if cfg.get("save_every_epoch", True) and cur_step % steps_per_epoch == 0:
            save_adapter(out_dir / f"epoch_{cur_step // steps_per_epoch}")
        return v

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, bad = time.time(), 0
    last_val = float("nan")
    while step < total_steps:
        ep = step // steps_per_epoch
        batches = epoch_batches(len(train_items), micro, seed, ep)
        k0 = (step % steps_per_epoch) * accum
        group = batches[k0:k0 + accum]          # the micro-batches of this optimizer step
        if not group:                           # epoch exhausted by rounding
            step = (ep + 1) * steps_per_epoch
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, warmup, lr)
        n_tok = sum(sum(1 for t in train_items[i]["labels"][1:] if t != -100) for mb in group for i in mb)
        loss_acc = 0.0
        for mb in group:
            b = {k: v.to(device) for k, v in collate([train_items[i] for i in mb], pad_id).items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                s, _ = token_loss_sum(model, b)
            loss = s / n_tok                    # token-weighted mean over the whole accumulation group
            scaler.scale(loss).backward()
            loss_acc += float(loss.item())
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0))))
        if not math.isfinite(loss_acc):
            bad += 1
            if bad >= 5:
                raise RuntimeError("5 consecutive non-finite losses: lower lr or train the small model in fp32")
        else:
            bad = 0
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        elapsed = elapsed_prev + time.time() - t0
        val = float("nan")
        if step % eval_every == 0 or step == total_steps:
            val = do_eval_and_save(step, elapsed)
            last_val = val
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        writer.writerow([step, f"{step / steps_per_epoch:.3f}", f"{loss_acc:.5f}",
                         "" if math.isnan(val) else f"{val:.5f}", f"{opt.param_groups[0]['lr']:.3e}",
                         f"{gn:.3f}", f"{elapsed:.1f}", f"{peak:.2f}"])
        logf.flush()
        if step % 5 == 0 or not math.isnan(val):
            vs = "-" if math.isnan(val) else f"{val:.4f}"
            print(f"step {step}/{total_steps} ep {step / steps_per_epoch:.2f} train {loss_acc:.4f} val {vs} "
                  f"lr {opt.param_groups[0]['lr']:.2e} gn {gn:.2f} {elapsed:.0f}s peak {peak:.2f}GB", flush=True)
    logf.close()

    summary = {
        "run": cfg.get("run_name", out_dir.name), "model_id": cfg["model_id"], "git": _git_hash(),
        "lora_r": int(lc.get("r", 16)), "lora_alpha": int(lc.get("alpha", 32)),
        "lora_dropout": float(lc.get("dropout", 0.05)),
        "lr": lr, "epochs": epochs, "micro_batch": micro, "grad_accum": accum, "effective_batch": micro * accum,
        "steps": total_steps, "steps_per_epoch": steps_per_epoch, "max_len": max_len, "seed": seed,
        "n_train": len(train_items), "n_val": len(val_items), "trainable_params": n_train, "total_params": n_total,
        "best_val_loss": best_val, "best_step": best_step, "last_val_loss": last_val,
        "train_seconds": elapsed_prev + time.time() - t0,
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None),
        "torch": torch.__version__,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # small files also go to the repo so they can be committed (adapters never go to git)
    repo_logs = Path(cfg.get("repo_logs_dir", "results/logs"))
    repo_logs.mkdir(parents=True, exist_ok=True)
    shutil.copy(log_path, repo_logs / f"train_{summary['run']}.csv")
    (repo_logs / f"summary_{summary['run']}.json").write_text(json.dumps(summary, indent=2))
    print("[done]", json.dumps({k: summary[k] for k in ("run", "best_val_loss", "best_step", "train_seconds", "peak_mem_gb")}))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. --set lr=1e-4 epochs=2")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    train(cfg)


if __name__ == "__main__":
    main()
