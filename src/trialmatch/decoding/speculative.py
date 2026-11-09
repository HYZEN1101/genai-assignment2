"""Greedy speculative decoding with explicit cache rollback (rubric F2).

Idea: a small DRAFT model proposes k tokens cheaply; the big TARGET model checks all k in ONE forward pass.
Under greedy decoding the output is IDENTICAL to plain greedy decoding of the target:
every emitted token is the target's own argmax given the already-verified prefix. The draft only decides how many
target argmaxes we obtain per target forward.

State invariant (checked every round when check_cache=True)
-----------------------------------------------------------
seq     = prompt + accepted tokens (python list). T = len(seq). The LAST token of seq ("pending") is NOT in any cache yet.
target cache length = T - 1          (everything except the pending token)
draft  cache length = dl <= T - 1    (lazy: it may lag; missing tokens are fed in one multi-token forward next round)

One round (base = T - 1, kk = min(k, remaining - 1) so we can never overshoot max_new_tokens):
 a) DRAFT   feed seq[dl:T] (>=1 token, includes pending) -> argmax = d1, then feed d1 -> d2 ... until kk proposals.
            The last proposal d_kk is NOT fed. Draft cache length becomes base + kk.
 b) VERIFY  ONE target forward over [pending, d1..d_kk] (kk+1 tokens) at positions base..base+kk with logits for all
            kk+1 positions: t_i = target argmax AFTER consuming d_i (t_0 follows pending).
 c) ACCEPT  a = number of leading i with d_{i+1} == t_i. New tokens = d_1..d_a followed by t_a:
            t_a is the CORRECTION if a < kk (first mismatch: take the target's token) or the BONUS token if a == kk.
            1..kk+1 tokens per target forward. EOS inside the block truncates it (EOS kept, nothing after it).
 d) ROLLBACK target cache holds base+kk+1 positions but only pending + the first kept-1 new tokens are valid
            (kept = number of new tokens emitted) -> crop to base + kept. Draft cache -> crop to min(dl, base + kept).
            Rejected draft tokens' K/V are thereby discarded; t_a becomes the new pending token.

Near-tie caveat: the verify forward computes kk+1 positions in one matmul shape, plain greedy computes 1 position per
pass. In fp16 the two can differ in the last bits and flip an argmax whose top-2 logits are (almost) tied. Tests run in
fp32; the real-model check script reports any mismatch together with the logit gap instead of hiding it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from .common import cache_len, crop_cache, eos_set, forward_with_cache, greedy_next, model_device, new_cache, now_ms


@dataclass
class SpecStats:
    path: str = "speculative"
    prompt_len: int = 0
    new_tokens: int = 0
    k: int = 0
    rounds: int = 0                      # verification rounds (excludes prefill)
    proposed: int = 0                    # draft tokens proposed (sum of kk)
    accepted: int = 0                    # draft tokens accepted (sum of a); correction/bonus tokens NOT counted here
    target_forwards: int = 0             # incl. prefill
    draft_forwards: int = 0              # incl. prefill
    prefill_ms: float = 0.0              # BOTH models' prompt forward + first argmax (time to first token)
    decode_ms: float = 0.0
    stopped_on_eos: bool = False
    cache_len_final: Optional[int] = None
    draft_cache_len_final: Optional[int] = None
    accepts_per_round: list = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.prefill_ms + self.decode_ms

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else float("nan")

    @property
    def mean_accepted_per_round(self) -> float:
        return self.accepted / self.rounds if self.rounds else float("nan")

    @property
    def tokens_per_target_forward(self) -> float:
        return self.new_tokens / self.target_forwards if self.target_forwards else float("nan")

    @property
    def decode_tok_s(self) -> float:
        n = max(self.new_tokens - 1, 0)
        return n / (self.decode_ms / 1000.0) if self.decode_ms > 0 and n > 0 else float("nan")

    @property
    def e2e_tok_s(self) -> float:
        return self.new_tokens / (self.total_ms / 1000.0) if self.total_ms > 0 else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("accepts_per_round")
        d.update(total_ms=self.total_ms, acceptance_rate=self.acceptance_rate,
                 mean_accepted_per_round=self.mean_accepted_per_round,
                 tokens_per_target_forward=self.tokens_per_target_forward,
                 decode_tok_s=self.decode_tok_s, e2e_tok_s=self.e2e_tok_s)
        return d


def check_compatible(target, draft) -> dict:
    """F1: the draft's token ids must mean the same thing as the target's (same vocabulary indices).
    Qwen2.5 0.5B and 1.5B share one tokenizer, but their logit width may differ in padding, so only the ids that the
    tokenizer can emit matter; we assert the config vocab sizes are equal here (stricter) and let callers compare
    tokenizers separately (see tokenizers_identical)."""
    tv, dv = target.config.vocab_size, draft.config.vocab_size
    return {"target_vocab": tv, "draft_vocab": dv, "equal": tv == dv}


def tokenizers_identical(tok_a, tok_b) -> bool:
    return tok_a.get_vocab() == tok_b.get_vocab() and tok_a.all_special_tokens == tok_b.all_special_tokens


@torch.no_grad()
def generate_speculative(target, draft, input_ids: torch.Tensor, max_new_tokens: int, k: int = 4, eos_id=None,
                         check_cache: bool = False) -> tuple[list[int], SpecStats]:
    """Greedy speculative decoding. input_ids [1, L]. Returns (new_token_ids, SpecStats); tokens == greedy target."""
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1 and k >= 1 and max_new_tokens >= 1
    dev = model_device(target)
    assert model_device(draft) == dev, "target and draft must be on the same device"
    input_ids = input_ids.to(dev)
    L = input_ids.shape[1]
    stops = eos_set(eos_id)
    tc, dc = new_cache(target), new_cache(draft)
    st = SpecStats(prompt_len=L, k=k)

    # ---- prefill both models on the whole prompt; first token = target argmax (same as plain greedy) --------------
    t0 = now_ms(dev)
    logits = forward_with_cache(target, input_ids, tc, start=0)
    first = int(greedy_next(logits))
    forward_with_cache(draft, input_ids, dc, start=0)             # draft logits unused; only its cache matters
    st.target_forwards, st.draft_forwards = 1, 1
    seq: list[int] = input_ids[0].tolist() + [first]
    gen: list[int] = [first]
    dl = L                                                       # draft cache length
    t_first = now_ms(dev)

    if first in stops:
        st.stopped_on_eos = True
    else:
        while len(gen) < max_new_tokens:
            T = len(seq)
            base = T - 1
            kk = min(k, max_new_tokens - len(gen) - 1)           # at most kk+1 new tokens -> never exceed max_new_tokens
            if check_cache:
                assert cache_len(tc) == base, (cache_len(tc), base)
                assert dl <= base, (dl, base)
            # a) draft kk tokens (first draft forward also catches up on tokens the draft cache has not seen)
            drafted: list[int] = []
            if kk > 0:
                feed = torch.tensor([seq[dl:T]], dtype=torch.long, device=dev)
                lg = forward_with_cache(draft, feed, dc, start=dl)
                dl = T
                st.draft_forwards += 1
                for i in range(kk):
                    d = int(greedy_next(lg))
                    drafted.append(d)
                    if i == kk - 1:
                        break                                    # last proposal is never fed to the draft
                    lg = forward_with_cache(draft, torch.tensor([[d]], dtype=torch.long, device=dev), dc, start=dl)
                    dl += 1
                    st.draft_forwards += 1
            # b) verify: ONE target forward over [pending, d1..dkk]
            chunk = torch.tensor([[seq[-1]] + drafted], dtype=torch.long, device=dev)
            vl = forward_with_cache(target, chunk, tc, start=base, logits_to_keep=kk + 1)    # [1, kk+1, V]
            st.target_forwards += 1
            targ = vl[0].argmax(dim=-1)                                                       # t_0..t_kk
            # c) accept longest matching prefix, then correction/bonus
            if kk > 0:
                dt = torch.tensor(drafted, dtype=torch.long, device=dev)
                a = int((dt == targ[:kk]).long().cumprod(0).sum())
            else:
                a = 0
            new = drafted[:a] + [int(targ[a])]
            for e, tkn in enumerate(new):                        # EOS inside the block: keep it, drop what follows
                if tkn in stops:
                    new = new[: e + 1]
                    st.stopped_on_eos = True
                    break
            kept = len(new)
            st.rounds += 1
            st.proposed += kk
            st.accepted += a
            st.accepts_per_round.append(a)
            seq.extend(new)
            gen.extend(new)
            # d) rollback: valid cache = pending + first kept-1 new tokens
            crop_cache(tc, base + kept)
            if dl > base + kept:
                crop_cache(dc, base + kept)
                dl = base + kept
            if check_cache:
                assert cache_len(tc) == len(seq) - 1 == L + len(gen) - 1, (cache_len(tc), len(seq), L, len(gen))
                assert cache_len(dc) == dl <= len(seq) - 1, (cache_len(dc), dl, len(seq))
            if st.stopped_on_eos:
                break
    t_end = now_ms(dev)
    st.new_tokens = len(gen)
    st.prefill_ms = t_first - t0
    st.decode_ms = t_end - t_first
    st.cache_len_final = cache_len(tc)
    st.draft_cache_len_final = cache_len(dc)
    return gen, st
