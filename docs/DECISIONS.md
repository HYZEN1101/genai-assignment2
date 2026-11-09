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

## D16 (P3) Authoritative metric conventions
Own implementation (eval/metrics.py, eval/evidence.py), cross-checked against sklearn (tests) and against the P2 quick metrics on the saved
validation predictions (exact match, tests/test_p3_crosscheck.py). Headline macro-F1 = mean over classes in gold OR valid predictions
(sklearn default, same as the P2 selection rule). Secondary macro_f1_gold_support = classes with gold support > 0 (the test split has no gold
"excluded"). Invalid output = wrong, own confusion column, reported as invalid_rate. Evidence: frozen P1 empty-set policy; HEADLINE = micro P/R/F1
over examples with non-empty gold evidence; "all examples" scope and macro variants also reported.

## D17 (P3) Test-set freeze is enforced in code
scripts/freeze.py writes results/logs/FREEZE.json/.txt (git commit, sha256 of test.jsonl and of the Hub-downloaded adapter weights, decoding
config, versions) and refuses a dirty tree. run_eval.py --split test additionally refuses to run without a matching FREEZE.json
(--allow-test + non-existing output file were already required). Base and fine-tuned models run on identical records with identical decoding.

## D18 (P3) Aggregation rule (pipeline/aggregate.py)
First match wins: inclusion "not included" or exclusion "excluded" -> Likely ineligible; else any "not enough information"/invalid -> Needs more
information; else >=1 decided criterion -> Potentially eligible (preliminary); else No applicable criteria evaluated. Not a ground-truth label.
Known weakness: the model recalls "not included"/"excluded" poorly, so "Potentially eligible" can be wrong.

## D19 (P3) Patient-cluster bootstrap (OPTIONAL, cheap)
1000 resamples of the 8 patients (seed 42), paired for ft vs base, only to show uncertainty (8 patients, rare classes with 0-7 examples).

## D20 (P4) Attention implementation contract
reference_attention / tiled_attention: (B,H,L,D) tensors; scores, softmax, accumulators in fp32 regardless of input dtype, output cast back.
Causal mask bottom-right aligned (j <= i + Lk - Lq) so the same code serves prefill, Lq=1 cached decode and chunked prefill; tests compare to torch SDPA
with an EXPLICIT bool mask (D7). Fully masked rows return zeros (SDPA returns NaN), documented and tested. Tiled path: running max/denominator/accumulator,
causal key-block skipping, mask built only for blocks straddling the diagonal, no F.scaled_dot_product_attention inside. GQA head repetition is NOT part of
these kernels (done outside, OPTIONAL patch-into-model demo cut for time).
Alternatives rejected: Triton/CUDA kernel (out of scope, not required); is_causal=True in tests (wrong alignment for Lq != Lk).
Verification of the update rule: tests/test_attention.py (hand-computed case, 100 tiled-vs-reference shape/block/causal combos, SDPA cross-check, padding mask,
extreme logits, forced max-rescaling, fp16/bf16, decode-step and chunked-prefill equivalence).

## D21 (P5) Decoder design: batch 1, logits_to_keep=1, fixed-N benchmarks
Both explicit decoders (decoding/greedy_nocache.py, decoding/greedy_kv.py) are batch-size 1 greedy loops that I wrote; HF generate() appears only as a
reference in scripts/check_kv_equiv.py. Both call the model with logits_to_keep=1 (greedy decoding only needs the last row; avoids a [L, vocab] logits tensor,
and the same choice on both paths keeps the comparison fair). Stats definitions (identical for both paths): prefill = forward over the prompt + argmax of the
first new token; decode = the remaining N-1 steps; tokens/s reported for decode-only and end-to-end. Benchmarks disable EOS so both paths do exactly N new
tokens (fixed work); task-level equivalence uses EOS stopping on real validation records. Tokens stay on the device (no per-step host sync) unless EOS
checking is on. Rejected: batched/padded decoding (not needed by the rubric, adds mask bugs), static pre-allocated cache (OPTIONAL, cut).

## D22 (P5) transformers 5.22.1 cache API facts (read from the installed source, cache_utils.py / modeling_qwen2.py)
- Cache.crop(tokens_to_remove) takes a NEGATIVE count of tokens to remove; a positive value raises ValueError (the 4.x "keep max_length" meaning is gone).
  decoding.common.crop_cache(cache, n) computes the number to drop and asserts the resulting length. P6 must use it for rollback.
- Qwen2ForCausalLM.forward has no cache_position parameter any more (slot derived from the cache); we pass position_ids and a 2-D attention_mask of length
  past+new ourselves and assert cache_len == start before every call; cache_position is passed only if the model signature accepts it (4.x).
- DynamicCache(config=model.config) pre-creates one DynamicLayer per decoder layer; per-layer tensors are .keys/.values with shape [B, kv_heads, T, head_dim].
- HF skips the causal mask for SDPA only when q_len == 1 or kv_len == q_len; a multi-token chunk after a cached prefix (prefix reuse, speculative
  verification) gets an explicit mask. Tested (test_multi_token_chunk_equals_one_by_one).

## D23 (P5) Equivalence methodology (rubric D2)
(1) tiny random Qwen2, fp32, CPU: 5 length combos + 5 prompts + EOS + logits-per-step + negative controls (wrong position_ids, wrong start) in pytest;
(2) real model, scripts/check_kv_equiv.py: >=3 (prompt_len, new_tokens) combos on length-controlled real-token prompts, EOS off, plus N real validation records
with EOS on, cached vs uncached vs HF generate; on a mismatch it reports first divergent step, top-2 logit gap of both paths and the max logit difference before
the divergence; --recheck-fp32 reloads in fp32 and reruns. A near-tie flip (tiny gap, vanishes in fp32) is numerical; anything else is treated as a bug.
Why paths can differ in fp16: the uncached path multiplies [T, d] matrices, the cached path [1, d] x cached K/V; different GEMM shapes/tilings change fp16
accumulation order, so logits differ in the last bits and a near-tied argmax can flip. Step 0 (prefill) is the same computation in both paths.

## D24 (P5) Benchmark methodology (rubric D3)
scripts/bench_kv.py, decoding/bench.py: same model object, same input ids for both paths, 2 warm-up runs (executed, not recorded) + 5 timed repeats per cell,
torch.cuda.synchronize around every timing, torch.cuda.reset_peak_memory_stats before each run; reported peak = max_memory_allocated and "above weights" =
peak minus memory allocated before the run. Prompts: length-controlled REAL-token streams (concatenated validation prompts truncated to L, assistant header
appended): not valid task instances, they only fix L. Measured KV bytes (cache tensors after the run) are compared with the formula
2*layers*kv_heads*head_dim*dtype_bytes*(L+N-1) (1.5B fp16: 28,672 B/token). One CSV row is written per finished cell (resumable on Colab disconnects); OOM cells
are kept as rows with a status. Peak memory is not measured on CPU.

## P6 (speculative decoding)
- D25 Algorithm: greedy speculative decoding, own loop (decoding/speculative.py). Invariant: target cache = len(seq)-1 (last token "pending" never cached); draft cache is LAZY (dl <= len(seq)-1) and catches up with one multi-token forward at the start of the next round. Per round kk=min(k, remaining-1); ONE target verify forward over [pending, d1..dkk] with logits for all kk+1 positions; accept longest matching prefix, then append the target's token (correction on first mismatch, bonus if all matched); rollback = crop_cache(target, base+kept), crop_cache(draft, min(dl, base+kept)). EOS inside a block truncates it. Rejected: re-prefilling the draft each round; HF assisted generation as deliverable (reference only; cut for time and because of the open P5 HF-reference discrepancy).
- D26 Draft/target: Qwen2.5-0.5B-Instruct draft for fine-tuned Qwen2.5-1.5B target; identical tokenizer asserted (get_vocab + special tokens, equal config vocab_size). Both fp16 on one T4. Baseline for speed-ups = own generate_greedy_kv.
- D27 Benchmark protocol: workloads task (real val prompts, EOS, <=64 new), free (3 chat prompts, EOS, <=128), forced (512-token prompt, 128 new, no EOS); k in {1,2,4,8}; warm-up 1 + 3 repeats; cuda.synchronize; peak memory reset per run; greedy and speculative interleaved on the same inputs. Speed-up = pooled latency greedy/spec. fp16 near-tie mismatches are reported with the logit gap. Task-tuned draft (configs/ft_D.yaml) is OPTIONAL.
