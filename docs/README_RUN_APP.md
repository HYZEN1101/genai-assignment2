## Run the app (browser, no code needed to use it)

Educational screening prototype, **not a clinical enrollment decision**.

```
git clone https://github.com/HYZEN1101/genai-assignment2.git && cd genai-assignment2
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_weights.py                        # LoRA adapter (HF: hyzenOG/trialmatch-qwen2.5-1.5b-lora) -> weights/adapter
python app/main.py                                        # then open http://127.0.0.1:7860
```

* First start downloads Qwen2.5-1.5B-Instruct (~3 GB) and the 0.5B draft (~1 GB) from Hugging Face, then loads both once (CPU: ~1-3 min).
* Memory: CPU fp32 needs about 8 GB RAM for target + draft (draft is skipped automatically below ~9.5 GB free; or pass `--no-draft`). GPU: fp16, about 4 GB VRAM.
* Speed on CPU: roughly 20 s prefill + 0.6 s/token per criterion (measured on Colab CPU); a 4-criterion screening takes a few minutes. A GPU is much faster.
* Flags: `--no-draft`, `--device cpu|cuda`, `--port N`, `--no-adapter` (base model, for comparison only), `--share` (public link, backup only).
* Tabs: (1) Screening: note + criteria in, per-criterion label / cited IDs / verbatim evidence / explanation, rule-based overall outcome, HTML + Markdown report with provenance. (2) Live decoding comparison: same prompt through no-cache, KV-cache and speculative decoding. (3) Stored benchmarks from `results/benchmarks/*.csv`.
* Problems: missing adapter -> run `download_weights.py`; out of memory -> `--device cpu` or `--no-draft`; port busy -> `--port 7861`.
