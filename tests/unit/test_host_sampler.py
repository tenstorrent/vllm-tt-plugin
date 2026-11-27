# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks that unseeded top-k rows sample from their largest logits only.

``TTTopKTopPSampler`` samples a batch without seeded rows from each row's
``top_k + 64`` largest logits when every row has ``1 <= top_k <= 1024``.
These tests compare that path with vLLM's full-vocabulary path: the same kept
tokens, the same probabilities and the same sampled distribution, and the full
path for every batch outside those bounds.
"""

from typing import NamedTuple

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p

from vllm_tt_plugin.host_sampler import TTTopKTopPSampler


@pytest.fixture(autouse=True)
def restore_global_rng():
    """Restores the global torch generator that unseeded rows draw from."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def _logits(rows: int, vocab: int, scale: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, vocab, generator=generator) * scale


def _kept_and_probs(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns vLLM's full-vocabulary kept mask and probabilities, ``[B, V]``."""
    filtered = apply_top_k_top_p(logits.clone(), k, p)
    return torch.isfinite(filtered), filtered.softmax(dim=-1)


@pytest.mark.parametrize("vocab", [12000, 248320])
@pytest.mark.parametrize("scale", [1.0, 3.0, 6.0], ids=lambda s: f"scale{s}")
@pytest.mark.parametrize("k", [1, 20, 50, 1024], ids=lambda k: f"k{k}")
@pytest.mark.parametrize("p", [None, 0.5, 0.95, 1.0], ids=lambda p: f"p{p}")
def test_candidates_keep_the_full_vocabulary_tokens_and_probabilities(
    vocab, scale, k, p
):
    logits = _logits(2, vocab, scale, seed=17)
    k_rows = torch.full((2,), k, dtype=torch.int32)
    p_rows = None if p is None else torch.full((2,), p)
    expected_kept, expected_probs = _kept_and_probs(logits, k_rows, p_rows)

    candidates = TTTopKTopPSampler()._top_k_candidates(logits.clone(), k_rows, p_rows)

    assert candidates is not None
    values, ids = candidates
    kept = torch.zeros_like(expected_kept).scatter_(1, ids, torch.isfinite(values))
    probs = torch.zeros_like(expected_probs).scatter_(1, ids, values.softmax(dim=-1))
    assert torch.equal(kept, expected_kept)
    torch.testing.assert_close(probs, expected_probs, rtol=1e-5, atol=1e-7)


def test_sampled_frequencies_follow_the_full_vocabulary_probabilities():
    vocab, rows, rounds = 4096, 2000, 10
    row = _logits(1, vocab, scale=3.0, seed=5)
    k = torch.full((rows,), 20, dtype=torch.int32)
    p = torch.full((rows,), 0.95)
    kept, probs = _kept_and_probs(row, k[:1], p[:1])
    sampler = TTTopKTopPSampler()
    torch.manual_seed(7)

    counts = torch.zeros(vocab)
    for _ in range(rounds):
        tokens, scores = sampler(row.expand(rows, vocab).contiguous(), {}, k, p)
        assert scores is None
        counts += torch.bincount(tokens, minlength=vocab)

    assert bool((counts[~kept[0]] == 0).all())
    torch.testing.assert_close(counts / (rows * rounds), probs[0], rtol=0, atol=0.01)


def test_mixed_top_k_and_top_p_rows_sample_only_kept_tokens():
    logits = _logits(4, 12000, scale=3.0, seed=9)
    k = torch.tensor([1, 20, 50, 1024], dtype=torch.int32)
    p = torch.tensor([0.5, 0.95, 1.0, 0.9])
    kept, _ = _kept_and_probs(logits, k, p)
    sampler = TTTopKTopPSampler()
    torch.manual_seed(3)

    for _ in range(50):
        tokens, _ = sampler(logits.clone(), {}, k, p)
        assert kept[torch.arange(4), tokens].all()
    assert torch.equal(tokens[0], logits[0].argmax())


def test_candidate_rows_do_not_sample_the_full_vocabulary(monkeypatch):
    def full_vocabulary(*args):
        raise AssertionError("an unseeded top-k batch reached the full path")

    monkeypatch.setattr(TopKTopPSampler, "forward_native", full_vocabulary)
    k = torch.full((2,), 20, dtype=torch.int32)

    tokens, scores = TTTopKTopPSampler()(_logits(2, 8000, 3.0, seed=1), {}, k, None)

    assert tokens.shape == (2,)
    assert scores is None


class FullPathCase(NamedTuple):
    """A batch that must take vLLM's full-vocabulary path.

    Attributes
    ----------
    top_k : list[int] | None
        Per-row ``top_k``; ``None`` passes no top-k tensor.
    seeded_rows : tuple[int, ...]
        Rows that carry their own generator.
    tie_width : int
        Number of leading logits of row 0 set to one value, so that the k-th
        largest value continues past the candidates when it exceeds
        ``top_k + 64``.
    logprobs_mode : str
        The sampler's ``logprobs_mode``.
    """

    top_k: list[int] | None
    seeded_rows: tuple[int, ...] = ()
    tie_width: int = 0
    logprobs_mode: str = "raw_logprobs"


FULL_PATH_CASES = {
    "partly_seeded": FullPathCase([20, 20], seeded_rows=(1,)),
    "no_top_k": FullPathCase(None),
    "top_k_above_1024": FullPathCase([20, 1025]),
    "tie_past_candidates": FullPathCase([20, 20], tie_width=200),
    "processed_logits": FullPathCase([20, 20], logprobs_mode="processed_logits"),
    "processed_logprobs": FullPathCase([20, 20], logprobs_mode="processed_logprobs"),
}


@pytest.mark.parametrize(
    "case", FULL_PATH_CASES.values(), ids=list(FULL_PATH_CASES.keys())
)
def test_batches_outside_the_bounds_match_the_full_path_exactly(case):
    logits = _logits(2, 8000, scale=3.0, seed=1)
    logits[0, : case.tie_width] = 50.0
    k = None if case.top_k is None else torch.tensor(case.top_k, dtype=torch.int32)
    p = torch.full((2,), 0.9)

    def sample(sampler):
        torch.manual_seed(11)
        generators = {
            row: torch.Generator().manual_seed(100 + row) for row in case.seeded_rows
        }
        tokens, scores = sampler.forward_native(logits.clone(), generators, k, p)
        return tokens, scores, torch.get_rng_state()

    expected = sample(TopKTopPSampler(case.logprobs_mode))
    actual = sample(TTTopKTopPSampler(case.logprobs_mode))

    assert torch.equal(actual[0], expected[0])
    if expected[1] is None:
        assert actual[1] is None
    else:
        assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
