# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Logprobs from an ``argmax_ids`` verify (``supports_spec_device_logprobs``).

The capability is declared by the model; the runner decides per request
whether the verify's ``SpecLogprobs`` payload answers it
(``TTModelRunner._spec_serves_logprobs``), sends ``spec_num_logprobs``, and
packs the payload's committed columns in the id walk. The engine-level
behavior (certified rows chain, the payload reaches vLLM) is covered by the
``device_logprobs`` cases of ``tests/test_mamba_spec_async.py``; these cases
pin each decision and the payload layout on their own.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_tt_plugin.async_decode import _verify_output_tensor
from vllm_tt_plugin.model_input import spec_verify_num_logprobs
from vllm_tt_plugin.model_runner import SpecWalk, TTModelRunner
from vllm_tt_plugin.spec_admission import validate_spec_device_logprobs
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    MAX_SPEC_DEVICE_LOGPROBS,
    SpecLogprobs,
    VerifyOutput,
    check_spec_logprobs,
)

N = MAX_SPEC_DEVICE_LOGPROBS


def _runner(declared=True, modes=(ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS)):
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.model = SimpleNamespace(
        model_capabilities={"supports_spec_device_logprobs": declared}
    )
    runner._spec_accept_modes = modes
    runner.model_config = SimpleNamespace(logprobs_mode="raw_logprobs")
    return runner


def _payload(rows: int, width: int, n: int = N) -> SpecLogprobs:
    """Values that name their own row and column: logprob ``-(10 r + t)``,
    rank ``100 r + t``, top ids ``1000 r + 10 t + i``."""
    r = torch.arange(rows).reshape(rows, 1)
    t = torch.arange(width).reshape(1, width)
    i = torch.arange(n).reshape(1, 1, n)
    return SpecLogprobs(
        -(10 * r + t).to(torch.float32),
        (100 * r + t).to(torch.int32),
        -(10 * r + t).unsqueeze(-1).to(torch.float32) - 0.5 * i,
        (1000 * r + 10 * t).unsqueeze(-1).to(torch.int32) + i,
    )


@pytest.mark.parametrize(
    "params, served",
    [
        pytest.param({"logprobs": 0}, True, id="sampled-token-only"),
        pytest.param({"logprobs": N}, True, id="payload-width"),
        pytest.param({"logprobs": N + 1}, False, id="above-payload"),
        pytest.param({"logprobs": -1}, False, id="all-vocab"),
        pytest.param({}, False, id="no-logprobs"),
    ],
)
def test_the_payload_serves_logprobs_zero_to_its_width(params, served):
    assert _runner()._spec_serves_logprobs(SamplingParams(**params)) is served


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(_runner(declared=False), id="not-declared"),
        pytest.param(
            _runner(modes=(ACCEPT_MODE_FUSED_SAMPLE, ACCEPT_MODE_LOGITS)),
            id="fused-sample-launch",
        ),
    ],
)
def test_a_launch_without_an_argmax_ids_payload_serves_none(runner):
    assert not runner._spec_serves_logprobs(SamplingParams(logprobs=1))


def test_another_logprobs_mode_or_named_token_ids_keep_the_logits_verify():
    runner = _runner()
    params = SamplingParams(logprobs=1)
    params.logprob_token_ids = [5, 6]
    assert not runner._spec_serves_logprobs(params)
    runner.model_config = SimpleNamespace(logprobs_mode="processed_logprobs")
    assert not runner._spec_serves_logprobs(SamplingParams(logprobs=1))


def test_a_declaration_must_be_a_bool():
    validate_spec_device_logprobs({"supports_spec_device_logprobs": True}, "M")
    with pytest.raises(ValueError, match="is a bool"):
        validate_spec_device_logprobs({"supports_spec_device_logprobs": "yes"}, "M")


@pytest.mark.parametrize(
    "payload, match",
    [
        pytest.param((torch.zeros(2, 4),) * 4, "SpecLogprobs", id="plain-tuple"),
        pytest.param(_payload(3, 4), r"\[2, 4\]", id="other-rows"),
        pytest.param(_payload(2, 4, n=N - 1), "N >= 20", id="too-few-top"),
    ],
)
def test_a_payload_of_another_shape_is_refused(payload, match):
    with pytest.raises(ValueError, match=match):
        check_spec_logprobs(payload, 2, 4)


def _verify_input(spec_mode, num_logprobs, live_rows=None):
    rows = len(num_logprobs)
    return SimpleNamespace(
        spec_mode=spec_mode,
        input_tokens=torch.zeros(rows, 4, dtype=torch.int32),
        tt_sampling_params=SimpleNamespace(
            num_logprobs=torch.tensor(num_logprobs, dtype=torch.int32)
        ),
        row_req_ids=[f"r{i}" for i in range(len(live_rows or [0, 1]))],
        live_rows=live_rows,
    )


def test_the_verify_is_sent_its_live_rows_logprobs_requests():
    """Live rows of a lane block sit at their slots; an unscheduled slot's
    request (slot 1 here) asks for nothing in this step."""
    model_input = _verify_input(ACCEPT_MODE_ARGMAX_IDS, [5, 3, -2, 0], [0, 3])
    sent = spec_verify_num_logprobs(model_input)
    assert sent.dtype == torch.int32
    assert sent.tolist() == [5, -2, -2, 0]
    assert spec_verify_num_logprobs(_verify_input(ACCEPT_MODE_LOGITS, [5, 0])) is None
    assert (
        spec_verify_num_logprobs(_verify_input(ACCEPT_MODE_ARGMAX_IDS, [-2, -2]))
        is None
    )


def test_the_walk_publishes_each_committed_column_of_the_payload():
    """Rows 0 and 3 of a 4-row block are live; they commit 2 and 1 tokens.
    Each published row is that column's payload: the committed token, its
    logprob and rank, then the step's top ``N`` (the live rows' largest
    request, 2 here)."""
    model_input = _verify_input(ACCEPT_MODE_ARGMAX_IDS, [2, -2, -2, 0], [0, 3])
    committed = torch.tensor(
        [[7, 8, -1, -1], [9, -1, -1, -1], [9, -1, -1, -1], [6, -1, -1, -1]],
        dtype=torch.int32,
    )
    walk = SpecWalk(committed, torch.tensor([2, 1, 1, 1], dtype=torch.int32))

    result = TTModelRunner._spec_payload_logprobs(
        model_input, walk, _payload(4, 4), spec_verify_num_logprobs(model_input)
    )

    assert result.logprob_token_ids.tolist() == [
        [7, 0, 1],
        [8, 10, 11],
        [6, 3000, 3001],
    ]
    assert result.logprobs.tolist() == [
        [-0.0, -0.0, -0.5],
        [-1.0, -1.0, -1.5],
        [-30.0, -30.0, -30.5],
    ]
    assert result.selected_token_ranks.tolist() == [0, 1, 300]
    assert result.cu_num_generated_tokens == [0, 2, 3]


def test_a_host_answer_unwraps_to_the_pair_a_deferred_read_returns():
    ids = torch.zeros(2, 4, dtype=torch.int32)
    payload = _payload(2, 4)
    answer = VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids)
    assert _verify_output_tensor(answer, "M", ACCEPT_MODE_ARGMAX_IDS) is ids
    answer = VerifyOutput(
        spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids, logprobs=payload
    )
    pair = _verify_output_tensor(answer, "M", ACCEPT_MODE_ARGMAX_IDS)
    assert pair[0] is ids and pair[1] is payload
