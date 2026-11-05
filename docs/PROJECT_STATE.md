# PROJECT_STATE.md (master state; replace in Project knowledge after every phase)
Last updated: 3 Nov 2026 | Last closed phase: 0 (with 2 carried-forward verification items)

## Environment translation (READ FIRST; overrides wording in the PHASE_*.md files)
- No Kaggle (phone verification declined) and NO local repo/laptop GPU. Everywhere the phase files say KAGGLE or LOCAL, use a COLAB notebook cell instead.
- Repo lives on Drive: MyDrive/genai_a2/repo. Notebook: colab_session.ipynb (cells 1-3 + install at every session start; push cell at the end).
- GitHub push from Colab uses classic token in Colab secret GH_TOKEN, NO `-u` flag (it stored the token in .git/config once; cleaned).
- Never re-run `%%writefile` cells over files that were edited since. Long jobs write checkpoints/logs to Drive (MyDrive/genai_a2), not only /content.
- Python in Colab is 3.13.16 (laptop Python version unknown: needed for Phase 7).

## Phase status
| Phase | Name | Status | Handoff file |
|---|---|---|---|
| 0 | Research, model choice, scaffold | DONE (carried: user URL check, 1.5B card check) | docs/handoffs/HANDOFF_P0.md |
| 1 | Data, split, prompt/schema, truncation | TODO | |
| 2 | Baseline + fine-tuning | TODO | |
| 3 | Evaluation, evidence, error analysis (final test run) | TODO | |
| 4 | Tiled online-softmax attention | TODO | |
| 5 | KV-cached decoding + benchmarks | TODO | |
| 6 | Speculative decoding | TODO | |
| 7 | Browser app | TODO | |
| 8 | Report, repo, video, defense | TODO | |

## Locked decisions
- Target: Qwen/Qwen2.5-1.5B-Instruct | Draft: Qwen/Qwen2.5-0.5B-Instruct. Measured on Colab T4 (results/logs/probe_colab.json). Rationale in docs/model_selection.md.
- Licenses: 0.5B Apache-2.0 (official card header checked). 1.5B Apache-2.0 per secondary sources: official card check PENDING. 3B = Qwen Research (Qwen2.5 report table).
- Fine-tuning method (provisional until P2): LoRA on all linear layers, fp16 (T4 has no native bf16; torch's bf16_supported=True is emulation).
- Output schema (provisional): compact JSON {"label": "<dataset label>", "evidence_ids": [ints]}; explanation built from cited sentences by template.
- Split: seed 42, 37/8/8, patient-grouped, manifest at data/split_manifest.json (not created yet).
- Decoding for all evals: greedy, fixed max_new_tokens (TBD in P1/P2).
- App framework (provisional): Gradio Blocks, http://127.0.0.1:7860, must run on laptop CPU (i7, 16 GB RAM, no GPU). 1.5B fp32 = 6.17 GB RAM (speed TBD).
- Library quirks (transformers 5.22.0): apply_chat_template(..., tokenize=False) then tok(text, return_tensors="pt"); use from_pretrained(name, dtype=...) not torch_dtype; inspect the real Cache API (crop/update/get_seq_length) before P5/P6.
- D7: PyTorch SDPA is_causal is UPPER-LEFT aligned for Lq != Lk; cached decoding needs bottom-right (offset Lk-Lq). Tests use explicit masks.

## Compute (measured)
Colab T4: 15,360 MiB VRAM (nvidia-smi), sm75, fp16 only, 13.61 GB RAM, 2 CPU cores. torch 2.11.0+cu130, transformers 5.22.0, peft 0.21.2,
accelerate 1.15.0, datasets 4.8.5, gradio 6.31.0, bitsandbytes not installed. SDPA backends on T4: MATH ok, EFFICIENT_ATTENTION ok, FLASH unsupported, CUDNN unsupported.
Colab session length / quota: TBD (not measured). Drive free space shown by probe: 14.7 GB (measured at repo folder).

## Data facts (fill in P1)
columns / label sets per criterion type / GPT-4 columns blacklisted / sentence-ID format / counts per split / token-length stats / truncation policy: TBD

## Interfaces other phases rely on (fill as created)
- load_model(...) signature + adapter loading procedure: TBD (base load convention above)
- build_prompt(patient_sentences, criterion_type, criterion_text) -> str: TBD
- parse_output(text) -> {label, evidence_ids, valid}: TBD
- Predictor API used by eval + app: TBD

## Key results so far (source: results/logs/probe_colab.json)
0.5B: 494.0M params, 24 layers, hidden 896, 14Q/2KV heads, head_dim 64, fp16 peak 1.01 GB. 1.5B: 1543.7M params, 28 layers, hidden 1536, 12Q/2KV, head_dim 128, fp16 peak 3.10 GB.
Tokenizer length 151,665 for both; config vocab 151,936 for both. KV bytes/token fp16 (derived): 12,288 (0.5B), 28,672 (1.5B). One-shot 10-token tok/s numbers are NOT benchmarks.

## Artifacts / links
GitHub: https://github.com/HYZEN1101/genai-assignment2 (last commit fd6ada2 before P0 close) | HF adapter: TBD | Split manifest: TBD | Colab notebook: colab_session.ipynb | Video: TBD | Report: TBD

## Open issues / risks
- SCHEDULE: about 1 day behind (P0 finished 3 Nov, plan had P1 done and P2 launched on 3 Nov). Start P1 now; run P4 (CPU-capable) in parallel with P2 GPU runs.
- Rubric total printed as 135 vs "100": ask instructor (optional).
- User to verify literature URLs (docs/literature.md) and the official Qwen2.5-1.5B-Instruct license card.
- HF account/token needed in P2 to host the adapter (HF_TOKEN secret not set yet).
- Colab quota/session length unknown; training jobs must checkpoint to Drive and be resumable.
- App must run on laptop CPU; measure 1.5B fp32 latency in P7 (TBD); laptop Python version TBD.

## Next phase + files to upload to its chat
Phase 1: HANDOFF_P0.md, requirements.txt, output of `!find src -name "*.py"` (src tree listing). PHASE_1 file is already in Project knowledge.
