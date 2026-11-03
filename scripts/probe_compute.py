"""Compute probe (Phase 0). Run on COLAB (GPU) or LOCAL (CPU).

Prints and saves: hardware, library versions, RAM/disk, bf16 support, SDPA backends,
model download + one greedy generate per model (timing, peak memory).
Output: results/logs/probe_<tag>.json and probe_<tag>.txt

Usage:
  python scripts/probe_compute.py --tag colab
  python scripts/probe_compute.py --tag local --models Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers

LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LINES.append(msg)


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<failed: {e}>"


def hardware() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    for pkg in ("peft", "accelerate", "datasets", "gradio", "bitsandbytes"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:  # noqa: BLE001
            info[pkg] = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
        info["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception:  # noqa: BLE001
        info["ram_total_gb"] = None
    du = shutil.disk_usage(".")
    info["disk_free_gb"] = round(du.free / 1e9, 1)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            gpu=p.name,
            gpu_mem_gb=round(p.total_memory / 1e9, 2),
            gpu_capability=list(torch.cuda.get_device_capability(0)),
            gpu_count=torch.cuda.device_count(),
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
            cuda_version=torch.version.cuda,
        )
        info["nvidia_smi"] = sh("nvidia-smi")
    return info


def sdpa_backends() -> dict:
    """Actually try each SDPA backend on a tiny input (flags alone can be misleading)."""
    out: dict = {}
    if not torch.cuda.is_available():
        out["note"] = "no CUDA: only math/CPU path tested"
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as e:  # noqa: BLE001
        return {"error": f"sdpa_kernel unavailable: {e}"}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    q = torch.randn(1, 4, 64, 64, device=dev, dtype=dtype)
    for name in ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION"):
        be = getattr(SDPBackend, name, None)
        if be is None:
            out[name] = "not in this torch version"
            continue
        try:
            with sdpa_kernel(be):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            out[name] = "works"
        except Exception as e:  # noqa: BLE001
            out[name] = f"unsupported: {str(e).splitlines()[0][:100]}"
    return out


def try_model(name: str, new_tokens: int = 20) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    res: dict = {"model": name}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    res["dtype"] = str(dtype)
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
        res["load_s"] = round(time.time() - t0, 1)
        res["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        c = model.config
        res["config"] = {
            "layers": c.num_hidden_layers, "hidden": c.hidden_size, "heads": c.num_attention_heads,
            "kv_heads": getattr(c, "num_key_value_heads", None),
            "head_dim": getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
            "max_pos": c.max_position_embeddings, "vocab": c.vocab_size,
        }
        res["vocab_len_tokenizer"] = len(tok)
        msgs = [{"role": "user", "content": "Say hello in one short sentence."}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=new_tokens, do_sample=False)
        if dev == "cuda":
            torch.cuda.synchronize()
            res["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        dt = time.time() - t0
        n = out.shape[1] - ids.shape[1]
        res["gen_tokens"] = int(n)
        res["gen_s"] = round(dt, 2)
        res["tok_per_s"] = round(n / dt, 1)
        res["sample_output"] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        res["ok"] = True
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        res["ok"] = False
        res["error"] = repr(e)[:400]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="colab")
    ap.add_argument("--models", nargs="*",
                    default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--outdir", default="results/logs")
    a = ap.parse_args()

    report: dict = {"hardware": hardware()}
    log("== HARDWARE ==")
    for k, v in report["hardware"].items():
        if k != "nvidia_smi":
            log(f"{k}: {v}")
    if "nvidia_smi" in report["hardware"]:
        log(report["hardware"]["nvidia_smi"])
    log("\n== SDPA BACKENDS ==")
    report["sdpa"] = sdpa_backends()
    for k, v in report["sdpa"].items():
        log(f"{k}: {v}")
    log("\n== MODELS ==")
    report["models"] = []
    for m in a.models:
        r = try_model(m)
        report["models"].append(r)
        log(json.dumps(r, indent=2))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"probe_{a.tag}.json").write_text(json.dumps(report, indent=2))
    (out / f"probe_{a.tag}.txt").write_text("\n".join(LINES))
    log(f"\nsaved to {out}/probe_{a.tag}.json/.txt")


if __name__ == "__main__":
    main()
