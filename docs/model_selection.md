# Model selection (Phase 0)

Status: specs marked (M) were MEASURED by scripts/probe_compute.py on Colab (results/logs/probe_colab.json).
Specs marked (S) come from web pages seen on 3 Nov 2026, mostly mirrors/aggregators, NOT yet checked on the official repo
(verify on huggingface.co/<org>/<model> config.json + LICENSE). TBD = unknown, never filled from memory.

## Compute (M)
Colab free T4 15,360 MiB (sm75, fp16 only; torch reports bf16 "supported" via emulation, we do not use it), 13.6 GB RAM, 2 CPU cores,
torch 2.11.0+cu130, transformers 5.22.0, peft 0.21.2, Python 3.13. SDPA backends on T4: MATH works, EFFICIENT_ATTENTION works,
FLASH_ATTENTION unsupported (needs sm80+), CUDNN_ATTENTION unsupported. Laptop: Intel i7, 16 GB RAM, no GPU.

## Candidates
| | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct | Qwen2.5-3B-Instruct | Qwen3-1.7B | Llama-3.2-1B-Instruct | Gemma-2-2B-it |
|---|---|---|---|---|---|---|
| Params | 494.0M (M) | 1543.7M (M) | 3.09B (S) | 1.7B nominal, 1.4B non-emb (S) | 1.23B (S) | 2.6B (S) |
| Layers | 24 (M) | 28 (M) | 36 (S) | 28 (S; one site says 32: conflict, TBD) | TBD | TBD |
| Hidden | 896 (M) | 1536 (M) | 2048 (S) | TBD | TBD | TBD |
| Q heads / KV heads | 14 / 2 (M) | 12 / 2 (M) | 16 / 2 (S) | 16 / 8 (S) | TBD (GQA yes, S) | 8 / 4 (S, one source) |
| head_dim | 64 (M) | 128 (M) | 128 (derived 2048/16) | TBD | TBD | 256 (S, one source) |
| Context | 32,768 (M) | 32,768 (M) | 32,768 (S) | 32,768 (S) | 128k (S) | 8,192 (S) |
| Vocab (config) | 151,936 (M) | 151,936 (M) | 151,936 (S) | TBD | TBD | 256,128 (S, one source; verify) |
| License | TBD: verify | Apache-2.0 (S: secondary sources; check official LICENSE) | Qwen Research, non-commercial (S) | Apache-2.0 (S) | Llama 3.2 Community License, gated (S) | Gemma terms of use, gated (S) |

## Memory math (formulas; fp16 = 2 B/param, fp32 = 4 B/param)
KV bytes/token = 2 (K and V) x layers x kv_heads x head_dim x 2 B.
| Model | fp16 weights | KV B/token | KV @ 2048 tokens |
|---|---|---|---|
| Qwen2.5-0.5B | 0.99 GB | 12,288 | 25 MB |
| Qwen2.5-1.5B | 3.09 GB (measured peak 3.10 GB incl. runtime) | 28,672 | 59 MB |
| Qwen2.5-3B | 6.18 GB | 36,864 | 76 MB |
| Qwen3-1.7B | ~3.4 GB nominal (exact count TBD) | ~114,688 IF head_dim=128 (unverified) | ~235 MB |
Note: with 2 KV heads (GQA) the cache is 6x smaller than 12 KV heads would be for the 1.5B. Weights dominate memory at our context
lengths, so memory differences cached vs uncached show up mainly at long contexts (benchmark up to >=2048 in Phase 5).
Laptop CPU demo of the 1.5B in fp32: 1543.7M x 4 B = 6.17 GB RAM (fits 16 GB; speed TBD, measure in Phase 7).
LoRA fine-tune memory on T4: not measured yet (TBD in Phase 2). Expect weights 3.1 GB + activations; fits in 15 GB.

## Decision
Target = Qwen2.5-1.5B-Instruct, draft = Qwen2.5-0.5B-Instruct.
- Fits the T4 comfortably (3.1 GB fp16) and the laptop CPU in fp32 (6.2 GB).
- Only 2 KV heads: small cache, simple to implement and reason about (viva).
- Same family: identical config vocab (151,936) and identical tokenizer length (151,665, M); full tokenizer equality check is a Phase 6 test.
- Plain full-attention decoder: explicit KV cache, tiled attention and speculative rollback are straightforward.
- License still to be verified on the official cards before the report claims it.
Rejected: see docs/DECISIONS.md.
