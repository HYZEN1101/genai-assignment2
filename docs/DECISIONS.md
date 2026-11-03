# Decisions log (chosen, rejected, why, evidence)

## D1 Compute platform: Google Colab free T4 (not Kaggle)
Kaggle GPU needs phone verification the student declined. Colab needs only a Google account.
Consequences: sessions are not guaranteed and VM disk is wiped on disconnect, so the repo lives on Drive (MyDrive/genai_a2/repo)
and long jobs must checkpoint to Drive. Evidence: results/logs/probe_colab.json.

## D2 Everything runs in Colab; GitHub push from Colab with a token secret (GH_TOKEN)
No local clone (laptop has no GPU and the student works only in Colab). Token must have Contents: read/write on the repo.
First attempt failed with HTTP 403 (token without write permission); fixed by recreating the token. A token pasted in chat was revoked.

## D3 Target = Qwen2.5-1.5B-Instruct, draft = Qwen2.5-0.5B-Instruct (see docs/model_selection.md)
Measured: fp16 peak 3.10 GB on T4, 28 layers, 2 KV heads, head_dim 128; draft 24 layers, 2 KV heads, head_dim 64.
Rejected:
- Qwen2.5-3B: Qwen Research (non-commercial) license (S); 6.2 GB fp16 and roughly 2x training time with a short deadline; laptop CPU demo too slow.
- Qwen3-1.7B: viable close second. 8 KV heads => ~4x KV bytes/token (derived, head_dim unverified), plus thinking-mode chat template to control. Not rejected for quality (not measured).
- Llama-3.2-1B/3B: gated, custom license, different tokenizer family, layer/head specs TBD.
- Gemma-2-2B: gated, 8K context, alternating sliding-window attention and logit soft-capping make an explicit KV cache and tiled attention harder; one source reports cache problems.

## D4 fp16, not bf16
T4 is sm75 with no native bf16 (torch reports bf16 supported through emulation). fp16 training needs care (LoRA weights in fp32, lower LR if NaN).

## D5 Library versions are new: Python 3.13, torch 2.11, transformers 5.22.0
transformers 5 changed apply_chat_template (returns a dict-like unless tokenize=False) and deprecates torch_dtype in favour of dtype.
Probe bug found and fixed. Before Phase 5/6, inspect the installed Cache API (crop, update, get_seq_length) in the actual 5.22.0 source.

## D6 SDPA backends on T4
MATH and EFFICIENT_ATTENTION work; FLASH and CUDNN unsupported (sm75 < sm80). Phase 4 backend table will report this.
