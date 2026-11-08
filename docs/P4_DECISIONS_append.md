
## D20 (P4) Attention implementation contract
reference_attention / tiled_attention: (B,H,L,D) tensors; scores, softmax, accumulators in fp32 regardless of input dtype, output cast back.
Causal mask bottom-right aligned (j <= i + Lk - Lq) so the same code serves prefill, Lq=1 cached decode and chunked prefill; tests compare to torch SDPA
with an EXPLICIT bool mask (D7). Fully masked rows return zeros (SDPA returns NaN), documented and tested. Tiled path: running max/denominator/accumulator,
causal key-block skipping, mask built only for blocks straddling the diagonal, no F.scaled_dot_product_attention inside. GQA head repetition is NOT part of
these kernels (done outside, OPTIONAL patch-into-model demo cut for time).
Alternatives rejected: Triton/CUDA kernel (out of scope, not required); is_causal=True in tests (wrong alignment for Lq != Lk).
Verification of the update rule: tests/test_attention.py (hand-computed case, 100 tiled-vs-reference shape/block/causal combos, SDPA cross-check, padding mask,
extreme logits, forced max-rescaling, fp16/bf16, decode-step and chunked-prefill equivalence).
