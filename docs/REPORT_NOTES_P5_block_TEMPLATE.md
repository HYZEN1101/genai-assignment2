
## Phase 5: KV-cached decoding (append to docs/REPORT_NOTES.md AFTER the Colab runs; replace every TBD with numbers from the CSVs)
FACTS: own greedy loops. No-cache reference: each step re-runs the full sequence with use_cache=False. Cached: prefill once, then one token per step with explicit
position_ids = current length, 2-D attention_mask grown by one, cache object owned by the loop, cache length asserted every step. Both use logits_to_keep=1.
Part B: single GQA attention layer with a hand-written cache (RoPE offset, torch.cat growth, bottom-right causal mask) equals full recompute, also with the P4 tiled kernel.
NUMBERS (TBD, sources results/benchmarks/kv_equiv.csv, kv_bench.csv, kv_env.json): equality per combo (fp16 / fp32), mismatches + gaps, prefill ms, decode ms/token,
tokens/s, speedup, peak memory above weights, KV bytes measured vs formula.
FIGURES/TABLES: results/figures/kv_time.png, kv_memory.png; equivalence table; benchmark table.
EXPLANATIONS TO WRITE: positions (RoPE absolute position of the new token; wrong position -> different logits, see negative-control test), masking (Lq=1 sees all cached keys;
chunk after a prefix needs an explicit offset mask), cache growth (linear, bytes/token formula; GQA 2 KV heads vs 12 query heads = 6x smaller than MHA), numerical differences
(different GEMM shapes in fp16; near-tie flips; fp32 recheck), why decode is memory-bound and prefill compute-bound (state only what the numbers show).
DIFFICULTIES: transformers 5.22.1 API changes (crop takes a negative count, no cache_position argument); code was authored without a working torch in the sandbox, validated on Colab.
REJECTED: use_cache=True alone / HF generate as the deliverable; static cache (optional); batched decoding.
