
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
