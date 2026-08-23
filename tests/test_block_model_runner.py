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


@pytest.mark.parametrize("width", [1, 16], ids=["ar_k1", "block_k16"])
def test_apply_and_build_runner_output_widths(width):
    runner, output_tokens = _runner(width)
    block = torch.arange(1, width + 1, dtype=torch.int32).reshape(1, width)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)
    output = TTModelRunner._build_runner_output(runner, block)

    expected = list(range(1, width + 1))
    assert runner.input_batch.num_tokens.tolist() == [width]
    assert runner.input_batch.token_ids_cpu[0, :width].tolist() == expected
    assert output_tokens == expected
    assert output.sampled_token_ids == [expected]


@pytest.mark.parametrize(
    "actual_width",
    [
        pytest.param(15, id="narrow-output"),
        pytest.param(17, id="wide-output"),
    ],
)
def test_output_width_is_strict(actual_width):
    runner, _ = _runner(16)

    with pytest.raises(ValueError, match="violates output_tokens_per_step"):
        TTModelRunner._build_runner_output(
            runner, torch.zeros((1, actual_width), dtype=torch.int32)
        )


def _captured_runner(width: int, num_tokens: tuple[int, int]):
    outputs_a: list[int] = []
    outputs_b: list[int] = []
    state_a, state_b = (
        SimpleNamespace(output_token_ids=outputs_a),
        SimpleNamespace(output_token_ids=outputs_b),
    )
    runner = SimpleNamespace(
        _output_tokens_per_step=width,
        requests={"a": state_a, "b": state_b},
        input_batch=SimpleNamespace(
            req_id_to_index={"a": 0, "b": 1},
            num_tokens=np.array(num_tokens, dtype=np.int32),
            token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
        ),
        model_config=SimpleNamespace(max_model_len=32),
    )
    return runner, (state_a, state_b), (outputs_a, outputs_b)


def test_captured_ar_capacity_overrun_leaves_batch_unmutated():
    """The deferred-apply path validates every live row before mutating any:
    an oversized AR step on a later row must not leave earlier rows applied."""
    runner, states, (outputs_a, outputs_b) = _captured_runner(1, (0, 32))
    tokens = torch.arange(2, dtype=torch.int32).reshape(2, 1)

    with pytest.raises(ValueError, match="exceed the max model length"):
        TTModelRunner._apply_sampled_tokens_to_state(
            runner, tokens, req_ids=["a", "b"], request_states=states
        )

    # Row 0 fit, but must not have been applied before row 1's rejection.
    assert runner.input_batch.num_tokens.tolist() == [0, 32]
    assert not runner.input_batch.token_ids_cpu.any()
    assert outputs_a == [] and outputs_b == []


def test_captured_block_overrun_clips_only_the_overflowing_row():
    """A block canvas past max_model_len must not kill the engine: the
    deferred-apply path keeps only the slice that fits for that row and
    applies the others in full."""
    runner, states, (outputs_a, outputs_b) = _captured_runner(16, (0, 17))
    blocks = torch.arange(32, dtype=torch.int32).reshape(2, 16)

    TTModelRunner._apply_sampled_tokens_to_state(
        runner, blocks, req_ids=["a", "b"], request_states=states
    )

    assert runner.input_batch.num_tokens.tolist() == [16, 32]
    assert runner.input_batch.token_ids_cpu[0, :16].tolist() == list(range(16))
    assert runner.input_batch.token_ids_cpu[1, 17:32].tolist() == list(range(16, 31))
    assert outputs_a == list(range(16))
    assert outputs_b == list(range(16, 31))


def test_ar_capacity_overrun_is_rejected_before_writes():
    runner, output_tokens = _runner(1, num_tokens=32)
    token = torch.tensor([[7]], dtype=torch.int32)

    with pytest.raises(ValueError, match="exceed the max model length"):
        TTModelRunner._apply_sampled_tokens_to_state(runner, token)

    assert runner.input_batch.num_tokens.tolist() == [32]
    assert not runner.input_batch.token_ids_cpu.any()
    assert output_tokens == []


def test_block_capacity_overrun_clips_to_the_slice_that_fits():
    """A bypassed zero-canvas request still gets one full physical canvas
    back; the runner keeps the fitting slice so the scheduler's stop check
    can finish the request length-capped instead of the engine dying."""
    runner, output_tokens = _runner(16, num_tokens=17)
    block = torch.arange(1, 17, dtype=torch.int32).reshape(1, 16)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [32]
    assert runner.input_batch.token_ids_cpu[0, 17:32].tolist() == list(range(1, 16))
    assert output_tokens == list(range(1, 16))


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


def test_worker_del_closes_mesh_when_model_runner_was_never_assigned():
    """init_device raises between opening the mesh and assigning
    model_runner (block sizing validation); the destructor must still close
    the open mesh instead of the AttributeError short-circuiting it."""
    from unittest.mock import patch

    closed = []
    mesh = SimpleNamespace()
    worker = TTWorker.__new__(TTWorker)
    worker.mesh_device = mesh
    worker.vllm_config = SimpleNamespace(additional_config=None)

    with patch(
        "vllm_tt_plugin.worker.close_mesh_device",
        side_effect=lambda mesh_device, _cfg: closed.append(mesh_device),
    ):
        worker.__del__()

    assert closed == [mesh]
    assert not hasattr(worker, "mesh_device")


def test_worker_shutdown_closes_mesh_once_across_shutdown_and_del():
    """SIGTERM reaches worker.shutdown() (EngineCore's finally ->
    executor.shutdown()), the only hook guaranteed before process exit; an
    unclosed mesh wedges the board's ethernet cores for the next process.
    __del__ afterwards must not close the mesh a second time."""
    from unittest.mock import patch

    closed = []
    mesh = SimpleNamespace()
    worker = TTWorker.__new__(TTWorker)
    worker.mesh_device = mesh
    worker.vllm_config = SimpleNamespace(additional_config=None)

    with patch(
        "vllm_tt_plugin.worker.close_mesh_device",
        side_effect=lambda mesh_device, _cfg: closed.append(mesh_device),
    ):
        worker.shutdown()
        worker.shutdown()
        worker.__del__()

    assert closed == [mesh]


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


# --------------------------------------------------------------------------
# K>1 device-sampling extraction (_get_output_tokens)
# --------------------------------------------------------------------------


def _extract(
    width,
    tt_out,
    batch_size_per_dp,
    *,
    device_sampling=True,
    is_decode=False,
    enable_log_probs=False,
):
    runner = SimpleNamespace(
        _output_tokens_per_step=width,
        _is_block_output_model=width > 1,
        tt_per_lane_max_num_seqs=1,
    )
    sampling_params = SimpleNamespace(
        enable_log_probs=torch.tensor([enable_log_probs]),
    )
    model_input = SimpleNamespace(
        intermediate_prefill_mask=None,
        max_num_logprobs=[None],
    )
    return TTModelRunner._get_output_tokens(
        runner,
        tt_out,
        None,
        sampling_params,
        model_input,
        batch_size_per_dp,
        perform_device_sampling=device_sampling,
        is_decode=is_decode,
    )


def test_get_output_tokens_extracts_full_canvas_per_request():
    tt_out = torch.arange(4, dtype=torch.int32).reshape(1, 4)

    sampled, logprobs = _extract(4, tt_out, [1])

    assert len(sampled) == 1
    assert sampled[0].shape == (1, 4)
    assert sampled[0][0].tolist() == [0, 1, 2, 3]
    assert logprobs == [None]


@pytest.mark.parametrize(
    "actual_width",
    [
        pytest.param(3, id="narrow-output"),
        pytest.param(5, id="wide-output"),
    ],
)
def test_get_output_tokens_rejects_wrong_canvas_width(actual_width):
    with pytest.raises(ValueError, match="violates output_tokens_per_step"):
        _extract(4, torch.zeros((1, actual_width), dtype=torch.int32), [1])


def test_get_output_tokens_empty_rank_keeps_canvas_width():
    sampled, logprobs = _extract(
        4, torch.zeros((0,), dtype=torch.int32), [0], is_decode=True
    )

    assert sampled[0].shape == (0, 4)
    assert logprobs == [None]


def test_get_output_tokens_rejects_host_sampling_for_block_models():
    with pytest.raises(ValueError, match="host sampling cannot construct"):
        _extract(
            4,
            torch.zeros((1, 1, 8), dtype=torch.float32),
            [1],
            device_sampling=False,
            is_decode=True,
        )


def test_get_output_tokens_rejects_device_logprobs_for_block_models():
    with pytest.raises(ValueError, match="one output token per step"):
        _extract(4, torch.zeros((1, 4), dtype=torch.int32), [1], enable_log_probs=True)
