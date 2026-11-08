# HANDOFF_P5.md: Phase 5 "KV-cached autoregressive decoding"   (save to docs/handoffs/)
Date: 8 Nov 2026 | Chat length/iterations: medium (one code drop, one patch, two Colab runs; sandbox had no working torch, all execution on Colab T4) | Status: DONE (D1-D3 satisfied; one non-blocking open item about the HF-generate reference, see section 8)

## 1. What was accomplished
- No-cache reference decoder and explicit KV-cache decoder (own loops; batch 1, greedy, logits_to_keep=1 in both), sharing one explicit cache-aware forward (`forward_with_cache`: position_ids, growing attention mask, cache-length assertion).
- Greedy token equality cached vs uncached on a fine-tuned 1.5B (fp16, 9 benchmark cells) and base 0.5B (fp16, 4 combos + 20 records), fp32 recheck on the fine-tuned 1.5B (4 combos + 20 records), plus agreement with HF generate where the reference works (0.5B). No mismatch observed.
- Part B mechanism demo: single GQA attention layer with a hand-written KV cache, RoPE position offset, bottom-right causal mask; equals full recompute, also with the P4 tiled kernel; negative controls (wrong RoPE offset / wrong position_ids) break the output.
- Rollback primitive for P6 (`crop_cache`) and multi-token verification chunk after a cached prefix tested.
- T4 benchmark, 9 cells (L 128/512/1024 x N 16/64/128), warm-up 2 + 5 repeats, synchronized, peak memory reset per run, resumable CSV; two figures.
- Measured KV bytes equal the formula exactly in all cells.
- Found and handled two transformers 5.22.1 API differences (crop takes a negative count; no cache_position argument).

## 2. Files created/changed (path : purpose : how to run)
- src/trialmatch/decoding/common.py : GenStats, cache helpers, crop_cache, forward_with_cache, kv_bytes_per_token
- src/trialmatch/decoding/greedy_nocache.py : reference decoder
- src/trialmatch/decoding/greedy_kv.py : explicit cached decoder
- src/trialmatch/decoding/cached_attention_demo.py : Part B demo layer
- src/trialmatch/decoding/equiv.py : compare_paths, first_divergence, length_prompt
- src/trialmatch/decoding/bench.py : bench_cell
- src/trialmatch/decoding/__init__.py : exports
- tests/test_kv_equiv.py : `python -m pytest tests/test_kv_equiv.py -q` (27 tests, CPU, ~35 s)
- scripts/check_kv_equiv.py : real-model equivalence (+HF reference, fp32 recheck)
- scripts/bench_kv.py : T4 benchmark + plots
- docs/P5_DECISIONS_append.md : D21-D24 (append to docs/DECISIONS.md)
- results/benchmarks/{kv_bench.csv, kv_equiv.csv, kv_equiv_detail.json, kv_env.json}, results/figures/{kv_time.png, kv_memory.png}, results/logs/{bench_kv.log, kv_equiv.log}
- Commits on GitHub: ... d09f420 (first push, partial results), 8050aa6 (final results + patch).

## 3. Commands to reproduce (WHERE: COLAB; T4 for 1.5B steps)
`!python -m pytest -q` (248 passed, 1 skipped) ; `!python scripts/check_kv_equiv.py --model Qwen/Qwen2.5-0.5B-Instruct` (seconds on T4) ; `!python scripts/check_kv_equiv.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter weights/adapter --tag ft1.5B --recheck-fp32` (~5-8 min) ; `!python scripts/bench_kv.py --adapter weights/adapter --tag ft1.5B` (~15-20 min, resumable) ; `!python scripts/bench_kv.py --mode plot --tag ft1.5B`.

## 4. Key numbers (value : source)
- Tests: 27 new, full suite 248 passed / 1 skipped : Colab output
- tokens_equal True in 9/9 fp16 benchmark cells (ft 1.5B) : kv_bench.csv
- 0.5B fp16: 4/4 combos equal, 20/20 records equal (nocache and HF), max|dlogit| 0.078-0.109, step-0 diff 0 : kv_equiv.csv
- ft 1.5B fp32: 4/4 combos equal, max|dlogit| 5.3e-5..6.3e-4, 20/20 records equal : kv_equiv.log
- Benchmark (see PROJECT_STATE "Key results: P5" for the full 9-cell list): speedup e2e 1.4-1.7x at L=128, 3.9-4.7x at 512, 5.6-8.2x at 1024; cached decode about 55-69 ms/token flat in L (~17 tok/s), uncached about 83-125 / ~250 / ~470-490 ms/token at L=128 / 512 / 1024; prefill equal (about 75-90 / 170-250 / 430-457 ms) : kv_bench.csv
- KV cache bytes measured == formula: 28,672 B/token; 33,001,472 B at L=1024, N=128 : kv_bench.csv
- Peak above weights nocache/kv at L=1024: 67-73 / 93 MB : kv_bench.csv

## 5. Decisions made + alternatives rejected + evidence (rubric A3)
- D21: batch 1; logits_to_keep=1 in both paths (fair, avoids [L, vocab] logits); fixed-N benchmarks with EOS off. Rejected: batched/padded decoding, static cache (optional).
- D22: transformers 5.22.1 cache facts read from source (crop negative; no cache_position; DynamicCache(config)); evidence: first draft of crop_cache would have raised ValueError.
- D23: equivalence methodology (tiny-model pytest incl. negative controls; real-model combos; fp32 recheck; HF generate as reference only).
- D24: benchmark methodology (length-controlled real-token prompts, warm-up/sync/peak reset, resumable CSV).
- Rejected: use_cache=True / HF generate as deliverable.

## 6. Problems hit / failed attempts / fixes (honest, feeds Limitations)
- Sandbox could not install torch; code was written without execution and validated on Colab. First Colab run: 1 failing test (negative-control threshold too strict for a tiny random model, fixed to 1e-4).
- HF generate reference first disagreed on every length combo: Qwen's shipped generation_config (repetition_penalty 1.05) overrides an explicit 1.0 because generate() treats a library-default value as unset; fixed by neutralising model.generation_config during the call. After the fix 0.5B agrees everywhere; ft1.5B fp32 still disagrees at L>=256 (open, section 8).
- Benchmark prompts are length-controlled concatenations of validation prompts, not valid task instances; EOS is off so the model keeps generating after <|im_end|> (junk continuation) to force N tokens.
- Colab note: first push attempt failed because GH_TOKEN/HF_TOKEN were not granted to the notebook; the first commit contained partial benchmark files (superseded by 8050aa6).
- Peak memory is the PyTorch allocator peak above weights, not nvidia-smi; includes prefill transients.

## 7. Interfaces later phases depend on
See PROJECT_STATE "Interfaces" (Decoding P5, P5 scripts). Rollback for P6: `crop_cache(cache, n)` (never call cache.crop with a positive number). Multi-token verify: `forward_with_cache(model, chunk[1,n], cache, start, logits_to_keep=n)` returns n rows of logits (tested equal to feeding the tokens one at a time, atol 1e-4 in fp32). Load the model only via `load_model(..., adapter="weights/adapter")`.

## 8. Open issues / TODO carried forward
- OPEN, non-blocking: HF-generate reference vs our decoders on ft1.5B fp32 for L>=256 (hf_equal=False while cached == uncached). Cause unknown; hypotheses untested: PEFT-merged model's generation_config object, fp32 path, forced-N junk continuations. Quick check before P6 relies on HF assisted decoding: print HF tokens vs ours, first divergent index, top-2 gap.
- fp16 ft1.5B rows of kv_equiv.csv were not seen in the pasted output: read the CSV before quoting them.
- Optional (cut): static cache, T4 profile showing where the ~57 ms/token goes, CPU-mode benchmark for the app.
- App note (P7): per-token cached decode is ~17 tok/s on T4; on laptop CPU expect much slower (not measured).

## 9. Report-ready material
- Figures: results/figures/kv_time.png (prefill / decode ms per token / speedup vs prompt length), kv_memory.png (peak above weights; formula vs measured KV bytes). Tables: equivalence table from kv_equiv.csv; benchmark table from kv_bench.csv.
- Method text (3-5 sentences): "Both decoders are our own greedy loops. The reference re-runs the full sequence at every step with use_cache=False. The cached decoder prefills once into a DynamicCache that the loop owns, then feeds only the newest token with explicit position_ids equal to the current length and an attention mask grown by one, asserting after every step that the cache length equals prompt plus tokens fed. Both decoders request only the last position's logits. Equality of the generated greedy tokens is verified on fine-tuned 1.5B in fp16 and fp32 and on the base 0.5B."
- Defense notes: (1) why position_ids must be the absolute position (RoPE rotates by position; negative-control test); (2) Lq=1 sees all cached keys, a chunk after a prefix needs an offset mask (HF only skips the mask when q_len==1 or kv_len==q_len); (3) cache bytes/token = 2*layers*kv_heads*head_dim*2 = 28,672 for the 1.5B; GQA (2 KV vs 12 Q heads) makes it 6x smaller than MHA; (4) fp16 mismatch mechanism = different GEMM shapes/accumulation order (logits differ ~0.1 yet argmax equal here), prefill identical; (5) cost per step O(T) cached vs full recompute; measured speedup grows with L (1.4x -> 8x); (6) flat ~57 ms/token cached = overhead-bound eager Python/kernels, not bandwidth (hypothesis); (7) live changes: break position_ids, drop mask growth, prefix reuse, crop after rejection.

## 10. Files to upload to the next chat (Phase 6)
PROJECT_STATE.md (new), docs/handoffs/HANDOFF_P5.md, src/trialmatch/decoding/{common,greedy_kv,greedy_nocache,equiv,bench,__init__}.py, scripts/check_kv_equiv.py, scripts/bench_kv.py, tests/test_kv_equiv.py, `head -3 results/benchmarks/kv_bench.csv`, adapter repo name hyzenOG/trialmatch-qwen2.5-1.5b-lora.
