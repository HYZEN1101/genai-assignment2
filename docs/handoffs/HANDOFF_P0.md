# HANDOFF_P0.md: Phase 0 "Research, model choice, scaffold"   (save to docs/handoffs/)
Date: 3 Nov 2026 | Chat length/iterations: long (many iterations fixing Colab/GitHub setup) | Status: DONE (carried forward: user opens literature URLs + official 1.5B license card)

## 1. What was accomplished
- Repo scaffold created and pushed to GitHub (HYZEN1101/genai-assignment2) from Colab; repo kept on Drive (MyDrive/genai_a2/repo).
- Colab T4 compute probe ran for both Qwen2.5-0.5B and 1.5B; saved results/logs/probe_colab.{json,txt} (committed, both ok=true).
- Candidate comparison (6 candidates) with measured and sourced specs and memory math: docs/model_selection.md.
- Final choice: target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct.
- 14 literature rows located by web search with URLs: docs/literature.md.
- Decisions D1-D7 logged: docs/DECISIONS.md.
- 0.5B license confirmed Apache-2.0 from the official card (pasted by user).
- Reusable notebook colab_session.ipynb (setup, token write-check, install, probe, push).

## 2. Files created/changed (path : purpose : how to run)
- scaffold.py : creates folder layout, requirements.txt, .gitignore, pyproject.toml, README, placeholder tests/docs : `python scaffold.py` (idempotent)
- scripts/probe_compute.py : hardware/library/SDPA/model probe : `python scripts/probe_compute.py --tag colab`
- docs/model_selection.md, docs/literature.md, docs/DECISIONS.md, docs/REPORT_NOTES.md : research docs
- docs/PROJECT_STATE.md, docs/handoffs/HANDOFF_P0.md : state and handoff
- results/logs/probe_colab.json/.txt : measured probe results
- colab_session.ipynb (kept in Colab/Drive, not in repo): session notebook

## 3. Commands to reproduce (COLAB; runtime T4 GPU)
colab_session.ipynb cells 1-5 (GPU check, mount Drive + cd repo, git identity, token check, install), then
`!python scripts/probe_compute.py --tag colab 2>&1 | grep -v "UserWarning\|scaled_dot_product_attention"` (~3-8 min first time because of model downloads, ~1 min after).

## 4. Key numbers (value : source)
- T4 15,360 MiB (nvidia-smi), torch-visible 15.64 GB, sm75; RAM 13.61 GB; 2 CPU cores : results/logs/probe_colab.json
- Qwen2.5-0.5B: 494.0M params, 24 layers, hidden 896, 14Q/2KV, head_dim 64, fp16 peak 1.01 GB, load 5.2 s (cached) : probe json
- Qwen2.5-1.5B: 1543.7M params, 28 layers, hidden 1536, 12Q/2KV, head_dim 128, fp16 peak 3.10 GB, load 10.7 s (cached; 82.2 s on first run incl. download) : probe json
- Tokenizer length 151,665 both; config vocab 151,936 both : probe json (Qwen2.5 report: 151,643 regular + 22 control = 151,665)
- SDPA on T4: MATH ok, EFFICIENT ok, FLASH unsupported, CUDNN unsupported : probe json
- KV bytes/token fp16 (derived): 0.5B 12,288; 1.5B 28,672; 3B 36,864; Qwen3-1.7B ~114,688 only if head_dim=128 (unverified) : docs/model_selection.md
- One-shot generation of 10 tokens (0.5B 11.6 tok/s, 1.5B 23.9 tok/s) : probe json. NOT benchmarks (no warm-up; 0.5B ran first).

## 5. Decisions + alternatives rejected + evidence (feeds A3 and the report)
- D1 Colab, not Kaggle (phone verification declined). D2 everything in Colab, repo on Drive, push with a token secret.
- D3 target/draft pair. Rejected: Qwen2.5-3B (Qwen Research non-commercial license, per Qwen2.5 report table; 6.2 GB fp16, ~2x training time, CPU demo too slow);
  Qwen3-1.7B (viable second; 8 KV heads gives ~4x KV bytes/token if head_dim=128, thinking-mode template; not rejected on quality, which was not measured);
  Llama-3.2-1B/3B (gated, custom license, different tokenizer family, layer/head specs not obtained); Gemma-2-2B (gated, 8K ctx, alternating sliding-window + soft-capping; HF docs say cache is capped at the window on such layers).
- D4 fp16 not bf16 on T4. D5 new library versions (Python 3.13, torch 2.11, transformers 5.22.0). D6 SDPA backend support on T4. D7 causal mask alignment in PyTorch SDPA.

## 6. Problems hit / failed attempts / fixes (feeds Limitations)
- Cloned repo was empty because the scaffold had not been pushed from a local machine that does not exist; moved everything into Colab.
- git push 403 twice: fine-grained token without Contents write; my first "token check" used the repo `permissions` field, which reports the USER's role, not the token's. Fixed with a classic token (`repo` scope) and a `git push --dry-run` check.
- A token was pasted into chat and was revoked; a second token got stored in .git/config via `git push -u` and was cleaned.
- Probe AttributeError on transformers 5.22.0 (apply_chat_template returns a dict-like). Fixed with tokenize=False + tokenizer call, and dtype= instead of torch_dtype.
- The committed probe log was stale (ok=false) because a `%%writefile` cell re-wrote the unpatched script; re-patched, re-ran, re-committed (fd6ada2 contains the good log).
- Model-card specs came from mirrors/aggregators, not official repos. Qwen3-1.7B sources conflict on layer count (28 vs 32): official config.json not checked.

## 7. Interfaces later phases depend on
- Load convention: `AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")`; prompts via `tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)` then `tok(text, return_tensors="pt").input_ids`.
- Repo path in Colab: /content/drive/MyDrive/genai_a2/repo; import package after `pip install -e .` (package name `trialmatch`, under src/).
- Push pattern: `git push https://{USER}:{gh}@github.com/{USER}/{REPO}.git main` (no -u). Secrets: GH_TOKEN (set), HF_TOKEN (not set).

## 8. Open issues / TODO carried forward
- User: open every URL in docs/literature.md before it goes into refs.bib; open https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct and confirm the license; check official configs of Qwen3-1.7B / Llama-3.2 / Gemma-2 only if they will be cited.
- Create HF account + write token (HF_TOKEN secret) before Phase 2 ends. Measure Colab session limits. Find laptop Python version. Ask instructor about 135 vs 100.
- Schedule is about 1 day behind.

## 9. Report-ready material
- Tables: candidate comparison + memory math (docs/model_selection.md); SDPA backend support on T4; literature table.
- Method text: "We selected Qwen2.5-1.5B-Instruct (28 layers, 2 KV heads, head_dim 128, Apache-2.0) as the target and Qwen2.5-0.5B-Instruct as the draft because both share a tokenizer (length 151,665) and the grouped-query attention with two KV heads keeps the cache at 28,672 bytes per token in fp16. Experiments ran on a free Colab T4 (15 GB, no native bf16), so training and inference use fp16."
- Defense notes: why GQA shrinks the KV cache (cache size scales with KV heads, 2 vs 12 for the 1.5B); fp16 vs bf16 on sm75; why speculative decoding needs identical tokenizers/vocab; why FlashAttention backend is unavailable on T4 (needs sm80+); why Colab/Drive/GitHub flow was chosen.

## 10. Files to upload to the next chat (Phase 1)
HANDOFF_P0.md, requirements.txt, output of `!find src -name "*.py"` (src tree listing).
