# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests: _get_output_tokens hands the host sampler no top-k/top-p
filter when no live row bounds it, as vLLM's GPU input batch does, so the
native sampler skips its full-vocabulary sort."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.host_sampler import TTTopKTopPSampler
from vllm_tt_plugin.model_runner import TTModelRunner

VOCAB = 4096
SLOT_ROWS = 4


def _runner(captured: dict) -> SimpleNamespace:
    def sampler(logits, sampling_metadata):
        captured["metadata"] = sampling_metadata
        return SimpleNamespace(
            sampled_token_ids=logits.argmax(dim=-1), logprobs_tensors=None
        )

    runner = SimpleNamespace(
        _num_speculative_tokens=0,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        tt_per_lane_max_num_seqs=SLOT_ROWS,
        vocab_size=VOCAB,
        model=SimpleNamespace(),
        host_sampler=sampler,
    )
    runner._check_host_logits = MethodType(TTModelRunner._check_host_logits, runner)
    return runner


def _metadata(top_k: list[int], top_p: list[float]):
    rows = len(top_k)
    captured: dict = {}
    params = SimpleNamespace(
        temperature=torch.ones(rows),
        top_k=torch.tensor(top_k, dtype=torch.int32),
        top_p=torch.tensor(top_p),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.zeros(rows, dtype=torch.int64),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )
    model_input = SimpleNamespace(
        intermediate_prefill_mask=None,
        grammar_bitmask=[None],
        output_tokens=None,
        prompt_tokens=None,
        max_num_logprobs=[None],
        allowed_token_ids_mask_list=[None],
        bad_words_token_ids_list=[None],
        logprob_token_ids_list=[{}],
        logitsprocs_list=[None],
        generators_list=[{}],
    )
    TTModelRunner._get_output_tokens(
        _runner(captured),
        torch.randn(rows, 1, VOCAB),
        None,
        params,
        model_input,
        [rows],
        perform_device_sampling=False,
        is_decode=True,
    )
    return captured["metadata"]


def test_unbounded_rows_hand_the_sampler_no_filters():
    md = _metadata([VOCAB, VOCAB], [1.0, 1.0])
    assert md.top_k is None
    assert md.top_p is None


def test_a_bounded_top_k_row_keeps_only_the_top_k_tensor():
    md = _metadata([50, VOCAB], [1.0, 1.0])
    assert md.top_k.tolist() == [50, VOCAB]
    assert md.top_p is None


def test_a_bounded_top_p_row_keeps_only_the_top_p_tensor():
    md = _metadata([VOCAB, VOCAB], [0.9, 1.0])
    assert md.top_k is None
    assert md.top_p.tolist() == pytest.approx([0.9, 1.0])


@pytest.mark.parametrize("seeded", [False, True])
def test_unbounded_filters_and_none_sample_the_same_tokens(seeded):
    """An all-vocab_size top_k and an all-1.0 top_p truncate nothing, so passing
    None instead leaves every probability, noise draw and token unchanged."""
    batch = 4
    torch.manual_seed(304)
    logits = torch.randn(batch, VOCAB)
    logits[:, 0] = -torch.inf  # a token masked by a grammar or bad words
    logits[1, 5] = logits[1, 6] = 8.0  # a tie at the top
    logits[2, 7] = 200.0  # so peaked that every other probability underflows
    k = torch.full((batch,), VOCAB, dtype=torch.int32)
    p = torch.ones(batch)

    def sample(top_k, top_p):
        sampler = TTTopKTopPSampler()
        generators = (
            {i: torch.Generator().manual_seed(10 + i) for i in range(batch)}
            if seeded
            else {}
        )
        torch.manual_seed(7)  # unseeded rows draw from the global generator
        try:
            tokens, _ = sampler(logits.clone(), generators, top_k, top_p)
        finally:
            if sampler._sampling_pool is not None:
                sampler._sampling_pool.shutdown()
        return tokens, [gen.get_state() for gen in generators.values()]

    tokens_tensors, states_tensors = sample(k, p)
    tokens_none, states_none = sample(None, None)

    torch.testing.assert_close(tokens_none, tokens_tensors, rtol=0, atol=0)
    assert all(
        torch.equal(a, b) for a, b in zip(states_none, states_tensors, strict=True)
    )
