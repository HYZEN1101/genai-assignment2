"""Phase 6 tests (F2/F3): greedy speculative decoding == plain greedy target decoding. Tiny random Qwen2 on CPU, fp32.

Real-model evidence (fine-tuned 1.5B target + 0.5B draft) comes from `scripts/bench_spec.py --mode check`.
"""
import copy

import pytest
import torch

from tiny_lm import make_model
from trialmatch.decoding import (cache_len, generate_greedy_kv, generate_greedy_nocache, generate_speculative)

VOCAB = 60


@pytest.fixture(scope="module")
def target():
    m = make_model(VOCAB)
    m.eval()
    return m


def make_draft(target, noise: float, seed: int = 1):
    """Draft = copy of the target with Gaussian noise on every weight. noise=0 -> perfect (oracle) draft; large -> bad draft."""
    d = copy.deepcopy(target)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in d.parameters():
            p.add_(torch.randn(p.shape, generator=g) * noise)
    return d.eval()


def prompt(L: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, VOCAB, (1, L), generator=g)


# ---------------------------------------------------------------- oracle draft: 100% acceptance, minimal target forwards
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_oracle_draft_full_acceptance(target, k):
    N = 33
    ref, _ = generate_greedy_kv(target, prompt(24), N)
    out, st = generate_speculative(target, target, prompt(24), N, k=k, check_cache=True)
    assert out == ref
    assert st.acceptance_rate == 1.0 and st.accepted == st.proposed
    # 1 prefill forward (-> token 1) + each verify round yields <= k+1 tokens
    assert st.target_forwards <= 1 + -(-(N - 1) // (k + 1)) + 1


# ---------------------------------------------------------------- equality with greedy for any draft quality
@pytest.mark.parametrize("noise", [0.0, 0.02, 0.1, 1.0])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_equals_greedy_any_draft(target, noise, k):
    draft = make_draft(target, noise)
    for L, N in [(8, 5), (40, 24), (96, 40)]:
        p = prompt(L, seed=L)
        ref, _ = generate_greedy_nocache(target, p, N)
        out, st = generate_speculative(target, draft, p, N, k=k, check_cache=True)
        assert out == ref, (noise, k, L, N)
        assert st.new_tokens == N
        assert st.cache_len_final == L + N - 1               # same cache convention as generate_greedy_kv


def test_bad_draft_low_acceptance_still_equal(target):
    bad = make_draft(target, 1.0, seed=7)
    p = prompt(32)
    ref, _ = generate_greedy_kv(target, p, 30)
    out, st = generate_speculative(target, bad, p, 30, k=4, check_cache=True)
    assert out == ref
    assert st.acceptance_rate < 0.4                            # random-ish draft on a 60-token vocab
    assert st.rounds >= 30 / 5                                 # at least ~1 token per round; every round produced a correction


def test_partial_draft_exercises_rollback(target):
    """Some draft quality between 'oracle' and 'random' must give partial acceptance (0 < a < k in a round)."""
    seen = False
    for noise in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        d = make_draft(target, noise)
        for seed in range(6):
            p = prompt(30 + seed, seed=seed)
            ref, _ = generate_greedy_kv(target, p, 30)
            out, st = generate_speculative(target, d, p, 30, k=4, check_cache=True)
            assert out == ref, (noise, seed)
            seen |= any(0 < a < 4 for a in st.accepts_per_round)
    assert seen, "no draft produced a partial acceptance"


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_tiny_max_new_tokens_and_k_larger_than_remaining(target, N):
    d = make_draft(target, 0.05)
    ref, _ = generate_greedy_kv(target, prompt(16), N)
    out, st = generate_speculative(target, d, prompt(16), N, k=8, check_cache=True)
    assert out == ref and len(out) == N


# ---------------------------------------------------------------- EOS inside a drafted block / on a correction token
@pytest.mark.parametrize("pos", [0, 1, 2, 5, 9])
@pytest.mark.parametrize("noise", [0.0, 0.05, 1.0])
def test_eos_everywhere(target, pos, noise):
    p = prompt(20)
    free, _ = generate_greedy_kv(target, p, 14)
    eos = free[pos]
    ref, _ = generate_greedy_kv(target, p, 14, eos_id=eos)
    out, st = generate_speculative(target, make_draft(target, noise), p, 14, k=4, eos_id=eos, check_cache=True)
    assert out == ref and out[-1] == eos and st.stopped_on_eos
    assert st.cache_len_final == 20 + len(out) - 1


def test_eos_set_of_ids(target):
    p = prompt(20)
    free, _ = generate_greedy_kv(target, p, 14)
    eos = [free[3], 7]
    ref, _ = generate_greedy_kv(target, p, 14, eos_id=eos)
    out, _ = generate_speculative(target, make_draft(target, 0.05), p, 14, k=4, eos_id=eos, check_cache=True)
    assert out == ref


# ---------------------------------------------------------------- stats bookkeeping
def test_stats_consistency(target):
    d = make_draft(target, 0.05)
    out, st = generate_speculative(target, d, prompt(24), 40, k=4)
    assert st.accepted <= st.proposed
    assert sum(st.accepts_per_round) == st.accepted and len(st.accepts_per_round) == st.rounds
    # every round emits (accepted_in_round + 1) tokens (no EOS here), the prefill emits 1
    assert st.new_tokens == 1 + st.accepted + st.rounds
    assert st.target_forwards == 1 + st.rounds
    assert st.draft_cache_len_final <= st.cache_len_final
    assert {"acceptance_rate", "e2e_tok_s", "tokens_per_target_forward"} <= set(st.as_dict())
