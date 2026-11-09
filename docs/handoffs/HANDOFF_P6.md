# HANDOFF_P6.md: Phase 6 "Speculative decoding"   (save to docs/handoffs/)
Date: Mon 9 Nov 2026 | Chat length/iterations: one long chat, 2 code iterations (1 test fixed) | Status: PARTIAL (decoder + tests + real-model equality DONE; T4 benchmark, peak memory, free-text workload, k in {1,8}, HF reference, task-tuned draft NOT done: GPU quota exhausted)

## 1. What was accomplished
- Own greedy speculative decoder `generate_speculative` (drafting, one-pass target verification, accepted prefix, correction/bonus token, EOS truncation, cache rollback for BOTH models) built on P5's `forward_with_cache` and `crop_cache`.
- 43 unit tests (tiny random Qwen2, fp32): equality with greedy for oracle/noisy/random drafts, k in {1,2,4,8}, tiny N, k > remaining, EOS in many positions, cache-length invariants every round, stats identities. All pass.
- Compatibility check: identical tokenizers (vocab + special tokens), vocab_size 151936 both, target 1.544 B vs draft 0.494 B params.
- Real fine-tuned 1.5B target + base 0.5B draft (CPU fp32): token-identical to greedy on every real validation record tested (3 rows in spec_check.csv seen + 4 benchmark cells), acceptance 0.65-0.83.
- CPU latency comparison: speculative is SLOWER (0.72-0.79x per cell, 0.755x pooled), explained below.
- Resumable check/bench/plot script `scripts/bench_spec.py`.

## 2. Files created/changed (path : purpose : how to run)
- src/trialmatch/decoding/speculative.py : decoder + SpecStats + check_compatible + tokenizers_identical
- src/trialmatch/decoding/__init__.py : exports them
- tests/test_spec_equiv.py : `python -m pytest tests/test_spec_equiv.py -q` (43 tests, about 1 min CPU)
- scripts/bench_spec.py : `--mode check|bench|plot|all` (see PROJECT_STATE interfaces)
- configs/ft_D.yaml : OPTIONAL task-tuned 0.5B draft recipe (NOT run)
- docs/P6_DECISIONS_append.md : D25-D27
- results/benchmarks/spec_check.csv, spec_bench.csv, spec_env.json (+ spec_summary.csv and results/figures/spec_overview.png only if the plot mode was run)

## 3. Commands to reproduce
- LOCAL/COLAB CPU: `python -m pytest tests/test_spec_equiv.py -q`
- COLAB CPU (about 30-45 min, one job at a time, 8-9 GB RAM): `python scripts/bench_spec.py --mode check --adapter weights/adapter --dtype float32 --tag cpu_ft1.5B+base0.5B --n-task 3 --n-free 0 --no-forced --ks 2 4` then `... --mode bench ... --n-task 2 --n-free 0 --no-forced --ks 2 4 --warmup 0 --repeats 1`
- COLAB T4 (not done; about 20-30 min): notebook sections C and D of colab_session_p6.ipynb.
- Lesson: two parallel CPU jobs exhaust the 12.7 GB RAM; `pkill -f` inside `sh()` kills the shell running it (return code -15).

## 4. Key numbers (source results/benchmarks/spec_bench.csv, spec_env.json)
- Tests 43 passed. Pooled latency greedy 120,868 ms vs speculative 160,169 ms over 4 cells = 0.755x.
- val0 L=379 N=16: k=2 29,817 -> 41,672 ms (0.72x, acc 0.714, 7 rounds); k=4 30,767 -> 39,139 (0.79x, acc 0.688, 4 rounds). val1 L=358 N=15: k=2 29,980 -> 38,835 (0.77x, acc 0.833, 6 rounds); k=4 30,304 -> 40,522 (0.75x, acc 0.650, 5 rounds).
- Prefill greedy 20.6-21.7 s vs spec 26.8-29.5 s (draft prefill adds 5.4-8.3 s). Greedy decode 611-634 ms/token; a speculative round costs 1.94-2.53 s = 3.1-4.1 greedy steps for 2.1-3.8 tokens per round; decode-only speedup 0.68/0.99/0.71/0.68.
- Peak memory: NOT measured (CPU). Computed fp16 weights: target 3,087 MB + draft 988 MB; KV 28,672 + 12,288 B/token.

## 5. Decisions + alternatives + evidence (rubric A3 / F1)
D25 (lazy draft cache, one multi-token catch-up forward, kk = min(k, remaining-1), crop both caches), D26 (draft = base 0.5B, same tokenizer family, checked programmatically), D27 (benchmark protocol). Rejected: re-prefilling the draft every round (wasteful); HF assisted generation as deliverable (reference only, cut); task-tuned draft (cut, no GPU). The fixed test threshold is the only code change after the first Colab run (test bug, not decoder bug).

## 6. Problems / failed attempts (feeds Limitations)
- No T4 quota: all speculative timing is CPU fp32, 2 records, warm-up 0, repeats 1 (no error bars, cold-start effects possible in the first cell).
- Parallel CPU jobs ran out of RAM risk; first jobs were killed and restarted at reduced scope.
- Speculative decoding slower than greedy on CPU: explained, hypotheses NOT profiled.
- fp16 near-tie risk (verify of k+1 tokens vs 1-token matmul shapes) was NOT tested on real fp16 because the GPU was unavailable; fp32 equality only.

## 7. Interfaces later phases depend on
`generate_speculative(target, draft, input_ids[1,L], max_new_tokens, k=4, eos_id=None, check_cache=False) -> (tokens, SpecStats)`; `SpecStats.as_dict()` keys: path, prompt_len, new_tokens, k, rounds, proposed, accepted, target_forwards, draft_forwards, prefill_ms, decode_ms, stopped_on_eos, cache_len_final, draft_cache_len_final, total_ms, acceptance_rate, mean_accepted_per_round, tokens_per_target_forward, decode_tok_s, e2e_tok_s. spec_bench.csv columns: prompt_len,k,new_tokens,tokens_equal,first_div,warmup,repeats,{greedy,spec}_{prefill_ms,decode_ms,total_ms,e2e_tok_s}_median,{greedy,spec}_peak_mb_median,{greedy,spec}_peak_above_weights_mb_median,acceptance_rate,rounds,proposed,accepted,target_forwards,draft_forwards,mean_accepted_per_round,tokens_per_target_forward,speedup_total,tag,workload,item,status. App note: load both models once; draft = `load_model("Qwen/Qwen2.5-0.5B-Instruct")`.

## 8. Open issues / TODO carried forward
T4 benchmark with peak memory and free-text/forced workloads (notebook sections C, D) if quota returns; HF assisted-decoding sanity check (optional, see P5 open item); task-tuned draft (optional); real fp16 near-tie check. Commit/push of P6 files.

## 9. Report-ready material
Figures/tables: spec_summary table (acceptance, tokens/forward, latency, speedup per k), spec_overview.png (only after `--mode plot`), a speculative-round diagram (draft, verify, accept, rollback). Method text (3-5 sentences): see docs/REPORT_NOTES.md P6 block. Defense notes:
- Why output equals greedy: every emitted token is the target argmax given a prefix the target itself has verified; the draft only decides how many argmaxes we obtain per target forward.
- Round: draft kk tokens, one target forward over [pending, d1..dkk], accepted prefix = leading matches, plus the target's token (correction at the first mismatch, bonus if all matched): 1..kk+1 tokens per target forward.
- Cache after a round: target cache = base + kept (crop discards the rejected tokens' K/V); draft cache cropped to min(dl, base+kept) and caught up lazily next round.
- Why kk = min(k, remaining-1): at most kk+1 new tokens per round, so max_new_tokens is never exceeded.
- Why tokenizers must match: draft token ids are fed to the target and compared with target argmaxes; different vocabularies make the ids meaningless.
- Expected speed-up needs (tokens per round x greedy step cost) > (k x draft step + verify cost); here the round cost 3.1-4.1 greedy steps, so only the best cell reaches break-even.
- Live changes: change k, force a bad draft (acceptance drops, output unchanged), swap draft for the target (acceptance 1.0).
- Near-ties: fp16 can flip an argmax between the verify pass and a 1-token pass; report gap, re-run in fp32.

## 10. Files to upload to the next chat (Phase 7)
This PROJECT_STATE.md, HANDOFF_P3.md, HANDOFF_P5.md, HANDOFF_P6.md, src/trialmatch/decoding/*.py, src/trialmatch/pipeline/{predictor,aggregate}.py, src/trialmatch/eval/explain.py, src/trialmatch/data/{prompts,schema}.py, `head -n 3 results/predictions/test_ft_explained.jsonl`, `head -3 results/benchmarks/spec_bench.csv`, one example patient note + trial criteria.
