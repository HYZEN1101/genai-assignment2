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

## D7 Causal-mask alignment (PyTorch SDPA docs)
For non-square (Lq != Lk) masks, is_causal=True in F.scaled_dot_product_attention is UPPER-LEFT aligned. Cached decoding needs
BOTTOM-RIGHT alignment (offset Lk - Lq). In Phase 4/5 tests, compare against SDPA with an explicit boolean mask, never is_causal=True when Lq != Lk.
Source: PyTorch docs (see docs/literature.md row 14).

## D3 addendum
Hugging Face KV-cache docs: for sliding-window models (Mistral, Gemma2) the cache never exceeds the window on those layers. This supports rejecting Gemma-2
for a hand-written cache. Qwen2.5 report table lists the 3B as Qwen Research (confirms the license concern).

## D8 (P2) Fine-tuning: custom loop, LoRA on all linear layers, fp16 + fp32 adapters
Custom ~250-line loop (src/trialmatch/train/finetune.py) instead of HF Trainer: transformers 5 changed Trainer/Cache APIs, and a loop I can
explain line by line is safer for the viva. LoRA r in {8,16}, alpha = 2r, dropout 0.05 on q,k,v,o,gate,up,down. Base fp16, adapter weights
fp32, torch.autocast(fp16) + GradScaler (T4 has no native bf16, D4). Cosine LR with 10% warm-up, AdamW, grad clip 1.0, effective batch 16.
Rejected for now: QLoRA (1.5B fp16 = 3.1 GB fits the T4; 4-bit adds bitsandbytes risk and noise), full fine-tuning (memory, 37 patients).

## D9 (P2) Loss only on target tokens; lm_head only at supervised positions
labels = -100 on the prompt; supervised span = JSON target + <|im_end|>. Prompt and target are tokenized separately and concatenated so the
boundary equals what generation sees. The loss runs the transformer body, then lm_head only on the ~20-40 supervised positions per example
instead of ~400 (avoids a B x L x 151,665 logits tensor). Unit test proves equality with the standard HF full-logits loss.

## D10 (P2) Greedy decoding must override repetition_penalty
Qwen2.5-Instruct's generation_config ships repetition_penalty=1.1 (verify in the model's generation_config.json). The predictor passes an explicit
GenerationConfig(do_sample=False, repetition_penalty=1.0), otherwise "greedy" would differ from our own argmax loops in P5/P6.

## D11 (P2) Merged adapter is the locked inference form
Eval, benchmarks, speculative decoding and the app load base fp16 -> PeftModel.from_pretrained -> merge_and_unload(). One set of weights everywhere,
no LoRA overhead in the hand-written decode loops.

## D12 (P2) Test split is guarded
scripts/run_eval.py refuses --split test unless --allow-test is given and the output file does not yet exist.
