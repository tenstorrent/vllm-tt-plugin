# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.worker import TTWorker


def _runner(width: int, *, num_tokens: int = 0, max_model_len: int = 32):
    output_tokens: list[int] = []
    return (
        SimpleNamespace(
            _output_tokens_per_step=width,
            input_batch=SimpleNamespace(
                num_reqs=1,
                req_ids=["req-0"],
                num_tokens=np.array([num_tokens], dtype=np.int32),
                token_ids_cpu=np.zeros((1, max_model_len), dtype=np.int32),
                req_output_token_ids=[output_tokens],
            ),
            model_config=SimpleNamespace(max_model_len=max_model_len),
        ),
        output_tokens,
    )


def test_full_block_updates_state_and_runner_output():
    runner, output_tokens = _runner(16)
    block = torch.arange(16, dtype=torch.int32).reshape(1, 16)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [16]
    assert runner.input_batch.token_ids_cpu[0, :16].tolist() == list(range(16))
    assert output_tokens == list(range(16))
    assert output.sampled_token_ids == [list(range(16))]


def test_output_width_is_strict():
    runner, _ = _runner(16)

    with pytest.raises(ValueError, match="violates output_tokens_per_step"):
        TTModelRunner._build_runner_output(
            runner, torch.zeros((1, 15), dtype=torch.int32)
        )


def test_capacity_overrun_is_rejected_before_writes():
    runner, output_tokens = _runner(16, num_tokens=17)
    block = torch.arange(16, dtype=torch.int32).reshape(1, 16)

    with pytest.raises(ValueError, match="exceed the max model length"):
        TTModelRunner._apply_sampled_tokens_to_state(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [17]
    assert not runner.input_batch.token_ids_cpu.any()
    assert output_tokens == []


def test_dp_packing_preserves_block_width_for_empty_rank():
    runner = SimpleNamespace(
        tt_data_parallel_size=2,
        tt_per_lane_max_num_seqs=1,
        _output_tokens_per_step=16,
    )
    block = torch.arange(16, dtype=torch.int32).reshape(1, 16)

    packed, logprobs = TTModelRunner.pack_dp_results(
        runner,
        [block, torch.empty((0, 16), dtype=torch.int32)],
        [None, None],
    )

    assert packed.shape == (2, 1, 16)
    assert packed[0, 0].tolist() == list(range(16))
    assert not packed[1].any()
    assert logprobs == [None, None]


def test_ar_k1_output_shape_is_unchanged():
    runner, output_tokens = _runner(1)
    token = torch.tensor([[7]], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, token)
    output = TTModelRunner._build_runner_output(runner, token)

    assert output_tokens == [7]
    assert output.sampled_token_ids == [[7]]


def test_worker_024_draft_token_rpc_returns_none():
    assert TTWorker.take_draft_token_ids(TTWorker.__new__(TTWorker)) is None


def test_update_states_accepts_absent_preemption_metadata():
    scheduler_output = SchedulerOutput.make_empty()
    assert scheduler_output.preempted_req_ids is None

    refreshed = []
    runner = SimpleNamespace(
        requests={},
        encoder_cache={},
        input_batch=SimpleNamespace(
            req_id_to_index={},
            refresh_logitsprocs=lambda: refreshed.append(True),
        ),
        _decode_layout_changed_since_last_decode=False,
    )

    TTModelRunner._update_states(runner, scheduler_output)

    assert refreshed == [True]


def test_worker_shutdown_releases_persistent_capture_once():
    releases = []
    runner = TTModelRunner.__new__(TTModelRunner)
    runner._persistent_capture_released = False
    runner.model = SimpleNamespace(
        release_persistent_capture=lambda: releases.append("released")
    )
    worker = SimpleNamespace(model_runner=runner)

    TTWorker.shutdown(worker)
    runner.shutdown()

    assert releases == ["released"]
