# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for ``SamplingParams.logprob_token_ids`` on TT host sampling.

vLLM's ``/v1/systemone`` endpoint sends each question as a ``max_tokens=1``
request whose ``logprob_token_ids`` are the label tokens, and reads their
full-vocabulary logprobs from the first output token. These tests check that
both host-sampling paths (the merged lane batch and the per-rank
``_get_output_tokens``) return exactly ``log_softmax(logits)[ids]`` for each
row that names ids, and that such rows never take the device sampler.
"""

from itertools import repeat
from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.logprobs import append_logprobs_for_next_position
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_tt_plugin.host_sampler import create_host_sampler
from vllm_tt_plugin.input_batch import InputBatch, SamplingInputBatch, TTLaneInputBatch
from vllm_tt_plugin.model_runner import TTModelRunner

VOCAB = 64
BLOCK = 16
MAX_MODEL_LEN = 256


def _req(req_id, ids=None, **sp):
    sp.setdefault("temperature", 0.0)
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(logprob_token_ids=ids, max_tokens=1, **sp),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[],
    )


def _lane_batch():
    return TTLaneInputBatch(
        num_lanes=2,
        per_lane=4,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * 8,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
    )


def _host_sampler():
    sampler = create_host_sampler()
    # gather_specific_token_logprobs stages ids in pinned memory when asked;
    # a CPU-only host has none.
    sampler.pin_memory = False
    return sampler


def _assert_label_logprobs(logprobs_tensors, out_row, logits_row, ids):
    expected = torch.log_softmax(logits_row.float(), dim=-1)[ids]
    n = len(ids)
    assert logprobs_tensors.logprob_token_ids[out_row, 1 : n + 1].tolist() == ids
    torch.testing.assert_close(logprobs_tensors.logprobs[out_row, 1 : n + 1], expected)


def _public_logprobs(logprobs_tensors, out_row, num_logprobs):
    """The token -> logprob dict the output processor builds for one row."""
    position = []
    append_logprobs_for_next_position(
        position,
        logprobs_tensors.logprob_token_ids[out_row].tolist(),
        logprobs_tensors.logprobs[out_row].tolist(),
        repeat(None),
        int(logprobs_tensors.selected_token_ranks[out_row]),
        num_logprobs,
    )
    return {token: lp.logprob for token, lp in position[0].items()}


def _expected_public_logprobs(logits_row, ids=None, num_logprobs=None):
    """Greedy sampled token plus either the named ids or the top-N."""
    lp = torch.log_softmax(logits_row.float(), dim=-1)
    sampled = int(lp.argmax())
    if ids is None:
        n = lp.numel() if num_logprobs == -1 else num_logprobs
        ids = lp.topk(n).indices.tolist()
    return {token: lp[token].item() for token in [sampled, *ids]}


def _assert_public_logprobs(actual, expected):
    assert actual.keys() == expected.keys()
    for token, value in expected.items():
        assert actual[token] == pytest.approx(value, rel=1e-5), token


@pytest.mark.parametrize("compact", [False, True])
def test_merged_lane_metadata_returns_each_rows_label_logprobs(compact):
    """Rows 1 and 6 name ids on a gappy layout; row 3 names none."""
    batch = _lane_batch()
    batch.add_request_to_row(_req("a", [5, 9, 2]), 1)
    batch.add_request_to_row(_req("plain"), 3)
    batch.add_request_to_row(_req("b", [7]), 6)
    batch.refresh_logitsprocs()
    assert batch.max_num_logprobs is None

    torch.manual_seed(0)
    logits = torch.randn(batch.max_num_reqs, VOCAB)
    rows = [6, 1, 3]
    metadata = batch.build_merged_sampling_metadata(rows, compact=compact)
    out = _host_sampler()(
        logits=(logits[rows] if compact else logits).clone(),
        sampling_metadata=metadata,
    )

    position = {row: i for i, row in enumerate(rows)} if compact else None
    for row, ids in ((1, [5, 9, 2]), (6, [7])):
        out_row = position[row] if compact else row
        assert metadata.logprob_token_ids[out_row] == ids
        _assert_label_logprobs(out.logprobs_tensors, out_row, logits[row], ids)
    plain_row = position[3] if compact else 3
    assert plain_row not in metadata.logprob_token_ids


@pytest.mark.parametrize("plain_logprobs", [3, -1])
@pytest.mark.parametrize("compact", [False, True])
def test_mixed_lane_batch_keeps_top_n_rows_top_n(compact, plain_logprobs):
    """A row that names ids must not replace a neighbouring row's top-N. The
    plain row samples token 0, the id the wider layout pads with."""
    batch = _lane_batch()
    batch.add_request_to_row(_req("a", [5, 9, 2]), 1)
    batch.add_request_to_row(_req("plain", logprobs=plain_logprobs), 3)
    batch.add_request_to_row(_req("b", [7]), 6)
    batch.refresh_logitsprocs()

    torch.manual_seed(0)
    logits = torch.randn(batch.max_num_reqs, VOCAB)
    logits[3, 0] = 10.0
    rows = [6, 1, 3]
    metadata = batch.build_merged_sampling_metadata(rows, compact=compact)
    out = _host_sampler()(
        logits=(logits[rows] if compact else logits).clone(),
        sampling_metadata=metadata,
    )

    position = {row: i for i, row in enumerate(rows)} if compact else None
    cases = (
        (1, 3, {"ids": [5, 9, 2]}),
        (6, 1, {"ids": [7]}),
        (3, plain_logprobs, {"num_logprobs": plain_logprobs}),
    )
    for row, num_logprobs, want in cases:
        out_row = position[row] if compact else row
        _assert_public_logprobs(
            _public_logprobs(out.logprobs_tensors, out_row, num_logprobs),
            _expected_public_logprobs(logits[row], **want),
        )


def test_a_reused_lane_row_drops_the_previous_requests_ids():
    batch = _lane_batch()
    batch.add_request_to_row(_req("a", [5, 9]), 1)
    batch.remove_request("a")
    assert batch.sampling.logprob_token_ids == {}
    batch.add_request_to_row(_req("next"), 1)
    assert batch.build_merged_sampling_metadata().logprob_token_ids is None


def test_condense_moves_ids_with_their_request():
    batch = InputBatch(
        max_num_reqs=3,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
    )
    batch.add_request(_req("first"))
    batch.add_request(_req("second", [4, 8]))
    batch.condense([batch.remove_request("first")])
    assert batch.req_id_to_index["second"] == 0
    assert batch.sampling.logprob_token_ids == {0: [4, 8]}


def test_block_output_batch_ignores_ids():
    """Block-output models cannot return per-token logprobs (the request is
    also refused at admission); the worker-side guard keeps the row clean."""
    batch = InputBatch(
        max_num_reqs=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        disable_logprobs=True,
    )
    batch.add_request(_req("r", [4]))
    assert batch.sampling.logprob_token_ids == {}


def test_per_rank_host_sampling_returns_each_rows_label_logprobs():
    """``_get_output_tokens`` reads the ids from ``logprob_token_ids_list``."""
    rows = 3
    params = SimpleNamespace(
        temperature=torch.zeros(rows),
        top_k=torch.full((rows,), VOCAB, dtype=torch.int32),
        top_p=torch.ones(rows),
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
        bad_words_token_ids_list=[{}],
        logprob_token_ids_list=[{0: [11, 3], 2: [60]}],
        logitsprocs_list=[None],
        generators_list=[{}],
    )
    runner = SimpleNamespace(
        _num_speculative_tokens=0,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        tt_per_lane_max_num_seqs=rows,
        vocab_size=VOCAB,
        model=SimpleNamespace(),
        host_sampler=_host_sampler(),
    )
    runner._check_host_logits = MethodType(TTModelRunner._check_host_logits, runner)
    torch.manual_seed(1)
    logits = torch.randn(rows, 1, VOCAB)

    _, logprobs_per_dp = TTModelRunner._get_output_tokens(
        runner,
        logits.clone(),
        None,
        params,
        model_input,
        [rows],
        perform_device_sampling=False,
        is_decode=False,
    )

    (logprobs_tensors,) = logprobs_per_dp
    _assert_label_logprobs(logprobs_tensors, 0, logits[0, -1], [11, 3])
    _assert_label_logprobs(logprobs_tensors, 2, logits[2, -1], [60])


def test_per_rank_mixed_batch_keeps_top_n_rows_top_n():
    """``_get_output_tokens`` with one row naming ids and one asking top-3."""
    rows = 2
    params = SimpleNamespace(
        temperature=torch.zeros(rows),
        top_k=torch.full((rows,), VOCAB, dtype=torch.int32),
        top_p=torch.ones(rows),
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
        max_num_logprobs=[3],
        allowed_token_ids_mask_list=[None],
        bad_words_token_ids_list=[{}],
        logprob_token_ids_list=[{0: [11, 3]}],
        logitsprocs_list=[None],
        generators_list=[{}],
    )
    runner = SimpleNamespace(
        _num_speculative_tokens=0,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        tt_per_lane_max_num_seqs=rows,
        vocab_size=VOCAB,
        model=SimpleNamespace(),
        host_sampler=_host_sampler(),
    )
    runner._check_host_logits = MethodType(TTModelRunner._check_host_logits, runner)
    torch.manual_seed(1)
    logits = torch.randn(rows, 1, VOCAB)
    logits[1, -1, 0] = 10.0

    _, logprobs_per_dp = TTModelRunner._get_output_tokens(
        runner,
        logits.clone(),
        None,
        params,
        model_input,
        [rows],
        perform_device_sampling=False,
        is_decode=False,
    )

    (logprobs_tensors,) = logprobs_per_dp
    _assert_public_logprobs(
        _public_logprobs(logprobs_tensors, 0, 2),
        _expected_public_logprobs(logits[0, -1], ids=[11, 3]),
    )
    _assert_public_logprobs(
        _public_logprobs(logprobs_tensors, 1, 3),
        _expected_public_logprobs(logits[1, -1], num_logprobs=3),
    )


@pytest.mark.parametrize("with_ids", [False, True])
def test_ids_keep_the_step_on_host_sampling(with_ids):
    """Device samplers return the sampled token's logprob or a top-N, never a
    named token's; a step carrying ids must sample on the host."""
    sampling = SamplingInputBatch(2)
    if with_ids:
        sampling.logprob_token_ids[1] = [4, 5]
    runner = SimpleNamespace(
        sample_on_device_mode="all",
        seeded_sampling_policy="auto",
        num_devices=8,
        tt_data_parallel_size=1,
        model=SimpleNamespace(model_capabilities={}),
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            sampling=sampling,
            req_id_to_index={"a": 0, "b": 1},
            no_penalties=True,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
        ),
    )
    assert TTModelRunner.check_perform_device_sampling(runner, False, False) is (
        not with_ids
    )
