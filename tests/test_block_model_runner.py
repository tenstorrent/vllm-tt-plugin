# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

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


def test_ar_k1_output_shape_is_unchanged():
    runner, output_tokens = _runner(1)
    token = torch.tensor([[7]], dtype=torch.int32)

    TTModelRunner._apply_sampled_tokens_to_state(runner, token)
    output = TTModelRunner._build_runner_output(runner, token)

    assert output_tokens == [7]
    assert output.sampled_token_ids == [[7]]


def test_update_states_accepts_absent_preemption_metadata():
    scheduler_output = SchedulerOutput.make_empty()
    assert scheduler_output.preempted_req_ids is None

    refreshed = []
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.requests = {}
    runner.encoder_cache = {}
    runner.input_batch = SimpleNamespace(
        req_id_to_index={},
        refresh_logitsprocs=lambda: refreshed.append(True),
    )
    runner._decode_layout_changed_since_last_decode = False
    runner._req_state_slot = {}

    runner._update_states(scheduler_output)

    assert refreshed == [True]


@pytest.mark.parametrize(
    ("finished_req_ids", "preempted_req_ids", "request_retained"),
    [
        ({"req-0"}, None, False),
        (set(), {"req-0"}, True),
    ],
    ids=["finished", "preempted"],
)
def test_update_states_releases_model_request_before_removing_row(
    finished_req_ids, preempted_req_ids, request_retained
):
    events = []

    class InputBatchSpy:
        def __init__(self):
            self.req_id_to_index = {"req-0": 3}

        def remove_request(self, req_id):
            events.append(("remove", req_id))
            return self.req_id_to_index.pop(req_id, None)

        def condense(self, removed_req_indices):
            pass

        def refresh_logitsprocs(self):
            pass

    input_batch = InputBatchSpy()

    def release_request(slot):
        events.append(("release", slot))

    runner = TTModelRunner.__new__(TTModelRunner)
    runner.requests = {"req-0": object()}
    runner.encoder_cache = {}
    runner.input_batch = input_batch
    runner.model = SimpleNamespace(release_request=release_request)
    runner._decode_layout_changed_since_last_decode = False
    # State parked at a slot different from the batch row: release must
    # target the slot, not the row.
    runner._req_state_slot = {"req-0": 5}

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.finished_req_ids = finished_req_ids
    scheduler_output.preempted_req_ids = preempted_req_ids

    runner._update_states(scheduler_output)

    assert events == [("release", 5), ("remove", "req-0")]
    assert "req-0" not in input_batch.req_id_to_index
    assert ("req-0" in runner.requests) is request_retained
    assert runner._req_state_slot == {}


def test_release_model_request_never_falls_back_to_batch_row():
    released = []
    runner = TTModelRunner.__new__(TTModelRunner)
    runner._req_state_slot = {}
    runner.input_batch = SimpleNamespace(req_id_to_index={"req-0": 3})
    runner.model = SimpleNamespace(release_request=released.append)

    runner._release_model_request("req-0")

    assert released == []


def test_worker_wrapper_shutdown_releases_persistent_capture_once():
    releases = []
    runner = TTModelRunner.__new__(TTModelRunner)
    runner._persistent_capture_released = False
    runner.model = SimpleNamespace(
        release_persistent_capture=lambda: releases.append("released")
    )
    worker = TTWorker.__new__(TTWorker)
    worker.model_runner = runner
    wrapper = WorkerWrapperBase()
    wrapper.worker = worker

    wrapper.shutdown()
    assert releases == ["released"]

    runner.shutdown()
    assert releases == ["released"]

    wrapper.shutdown()
    assert releases == ["released"]
