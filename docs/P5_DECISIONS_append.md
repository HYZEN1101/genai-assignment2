
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
