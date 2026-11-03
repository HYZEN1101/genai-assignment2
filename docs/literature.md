# Literature table
Status: VERIFIED-SEARCH = the bibliographic record was found by web search on 3 Nov 2026. UNVERIFIED = ID/URL from memory,
YOU must open it and confirm title/authors/venue before it goes into refs.bib. URL TBD = not located yet; do not invent.

| # | Citation | URL | Status | Contribution | Limitation | Used here |
|---|---|---|---|---|---|---|
| 1 | Jin et al., Matching patients to clinical trials with large language models, Nat. Commun. 15:9074 (2024) | https://arxiv.org/abs/2307.15051 ; doi 10.1038/s41467-024-53081-z | VERIFIED-SEARCH | TrialGPT: criterion-level eligibility + evidence sentences + explanation, trial-level aggregation | GPT-4 based, closed model; we cannot reuse its predictions | task definition, dataset origin, aggregation idea |
| 2 | Milakov & Gimelshein, Online normalizer calculation for softmax (2018) | https://arxiv.org/abs/1805.02867 | VERIFIED-SEARCH | single-pass running max + normalizer with rescaling | softmax only; attention form comes from FlashAttention | tiled attention (Phase 4) |
| 3 | Leviathan, Kalman, Matias, Fast Inference from Transformers via Speculative Decoding, ICML 2023 (PMLR 202:19274-19286) | https://arxiv.org/abs/2211.17192 | VERIFIED-SEARCH | draft+verify, outputs unchanged | speedup depends on acceptance and cost ratio | Phase 6 |
| 4 | Chen et al., Accelerating LLM Decoding with Speculative Sampling (2023) | https://arxiv.org/abs/2302.01318 | UNVERIFIED | concurrent speculative sampling work | same | Phase 6 related work |
| 5 | Dao et al., FlashAttention (2022) | https://arxiv.org/abs/2205.14135 | UNVERIFIED | IO-aware tiled exact attention | needs fused kernels; PyTorch tiling cannot match | Phase 4 |
| 6 | Hu et al., LoRA (2021) | https://arxiv.org/abs/2106.09685 | UNVERIFIED | low-rank adapters W + (alpha/r)BA | rank/target choice matters | Phase 2 |
| 7 | Dettmers et al., QLoRA (2023) | https://arxiv.org/abs/2305.14314 | UNVERIFIED | 4-bit base + LoRA | not needed at 1.5B | considered, rejected |
| 8 | Vaswani et al., Attention Is All You Need (2017) | https://arxiv.org/abs/1706.03762 | UNVERIFIED | scaled dot-product attention | quadratic cost | Phase 4 reference |
| 9 | Ainslie et al., GQA (2023) | https://arxiv.org/abs/2305.13245 | UNVERIFIED | grouped-query attention shrinks KV cache | | Phase 5 cache size |
| 10 | Su et al., RoFormer / RoPE (2021) | https://arxiv.org/abs/2104.09864 | UNVERIFIED | rotary position embedding | position offsets must be handled in cached decoding | Phase 5 |
| 11 | Qwen Team, Qwen2.5 Technical Report (2024) | arXiv ID TBD (check: model pages seen cite the Qwen2 report arXiv:2407.10671) | UNVERIFIED | model family description | | model choice |
| 12 | Hugging Face transformers docs: KV cache / cache classes | URL TBD | NOT LOCATED | cache API reference | API changed in v5 (installed 5.22.0) | Phase 5/6 |
| 13 | PyTorch docs: scaled_dot_product_attention, torch.nn.attention.sdpa_kernel | URL TBD | NOT LOCATED | backend selection | | Phase 4 |
