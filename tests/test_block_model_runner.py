# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import weakref
from collections import deque
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.worker import TTWorker


def _finish_step(runner, held, grammar=None):
    """Stand-in for the runner-bound finishers queued in ``_pending_samples``."""
    return held


class _Weakrefable:
    """Attribute bag that supports weak references, unlike SimpleNamespace."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


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
    runner, _, (outputs_a, outputs_b) = _captured_runner(1, (0, 32))
    tokens = torch.arange(2, dtype=torch.int32).reshape(2, 1)

    with pytest.raises(ValueError, match="exceed the max model length"):
        TTModelRunner._apply_sampled_tokens_to_state(runner, tokens, req_ids=["a", "b"])

    # Row 0 fit, but must not have been applied before row 1's rejection.
    assert runner.input_batch.num_tokens.tolist() == [0, 32]
    assert not runner.input_batch.token_ids_cpu.any()
    assert outputs_a == [] and outputs_b == []


def test_captured_block_overrun_clips_only_the_overflowing_row():
    """A block canvas past max_model_len must not kill the engine: the
    deferred-apply path keeps only the slice that fits for that row and
    applies the others in full."""
    runner, _, (outputs_a, outputs_b) = _captured_runner(16, (0, 17))
    blocks = torch.arange(32, dtype=torch.int32).reshape(2, 16)

    TTModelRunner._apply_sampled_tokens_to_state(runner, blocks, req_ids=["a", "b"])

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
    """shutdown() must close the mesh (an unclosed mesh wedges the board for
    the next process) exactly once, with __del__ as an idempotent backstop."""
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
        # Pin *when*: the first shutdown() closes the mesh itself. Deferring the
        # close to __del__ would satisfy the counts below, but __del__ is not
        # guaranteed to run at interpreter exit -- the mesh could stay open and
        # wedge the board for the next process.
        assert closed == [mesh]
        assert worker.mesh_device is None
        worker.shutdown()
        worker.__del__()

    assert closed == [mesh]


def test_worker_shutdown_releases_the_model_before_closing_the_mesh():
    """The model owns tensors, traces and captures allocated on the mesh, so
    they must be released before close_mesh_device -- and dropping references
    cannot achieve that, since the runner and its async-decode controller
    reference each other. So observe actual release with weakrefs.

    Every liveness fact is sampled inside the patched close, not asserted after
    it: "was it released" stays true once true, but "is it still alive" does
    not, because the runner and its queued step form an isolated cycle any
    gen-0 pass would collect. Cyclic collection is disabled across the call for
    the same reason.
    """
    import gc
    from unittest.mock import patch

    at_close: dict = {}
    order: list[str] = []
    mesh = SimpleNamespace()
    worker = TTWorker.__new__(TTWorker)
    worker.mesh_device = mesh
    worker.device = mesh
    worker.device_config = SimpleNamespace(device=mesh)
    worker.vllm_config = SimpleNamespace(additional_config=None)

    def _attach_runner():
        """Build the runner in a scope that leaves no strong reference."""
        runner = TTModelRunner.__new__(TTModelRunner)
        model = _Weakrefable(
            release_persistent_capture=lambda: order.append("capture-released")
        )
        runner.model = model
        runner.kv_caches = [SimpleNamespace()]
        runner._shutdown_complete = False
        # An async step that ran but was never sampled, with the ``_controller``
        # back-edge an engine-held output really has.
        controller = _Weakrefable(runner=runner)
        in_flight = _Weakrefable(_controller=controller)
        runner._pending_samples = deque([partial(_finish_step, runner, in_flight)])
        runner.async_decode = controller
        worker.model_runner = runner
        return (
            weakref.ref(model),
            weakref.ref(runner),
            weakref.ref(in_flight),
            controller,
        )

    model_ref, runner_ref, in_flight_ref, controller = _attach_runner()

    gc.disable()
    try:
        with patch(
            "vllm_tt_plugin.worker.close_mesh_device",
            side_effect=lambda _mesh, _cfg: (
                order.append("mesh-closed"),
                at_close.update(
                    model_alive=model_ref() is not None,
                    in_flight_alive=in_flight_ref() is not None,
                    runner_alive=runner_ref() is not None,
                    controller_cut=in_flight_ref()._controller.runner is None,
                ),
            ),
        ):
            worker.shutdown()
    finally:
        gc.enable()

    assert order == ["capture-released", "mesh-closed"]
    # The safety property: the model is gone before the close, not after.
    assert at_close["model_alive"] is False
    assert model_ref() is None
    # The deliberate leak: an in-flight async read is not freed here.
    assert at_close["in_flight_alive"] is True
    assert at_close["runner_alive"] is True
    # The cut: an engine-held output must not reach a runner whose model is gone.
    assert controller.runner is None
    assert at_close["controller_cut"] is True
    assert not hasattr(worker, "model_runner")
    assert worker.mesh_device is None
    assert worker.device is None
    assert worker.device_config.device is None


def test_worker_shutdown_closes_the_mesh_even_if_the_runner_raises():
    """A failing runner release must not stop the mesh from being closed."""
    from unittest.mock import patch

    closed = []
    worker = TTWorker.__new__(TTWorker)
    worker.mesh_device = SimpleNamespace()
    worker.device = worker.mesh_device
    worker.device_config = SimpleNamespace(device=worker.mesh_device)
    worker.vllm_config = SimpleNamespace(additional_config=None)
    worker.model_runner = SimpleNamespace(
        shutdown=lambda: (_ for _ in ()).throw(RuntimeError("device fault"))
    )

    with patch(
        "vllm_tt_plugin.worker.close_mesh_device",
        side_effect=lambda mesh_device, _cfg: closed.append(mesh_device),
    ):
        worker.shutdown()

    assert len(closed) == 1
    assert worker.mesh_device is None


def test_runner_shutdown_is_idempotent():
    """Second call must not re-release the capture."""
    releases = []
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.model = SimpleNamespace(
        release_persistent_capture=lambda: releases.append("released")
    )

    runner.shutdown()
    runner.shutdown()

    assert releases == ["released"]
    assert runner.model is None
    assert runner.kv_caches == []


def test_worker_shutdown_releases_admission_handle(monkeypatch):
    """Shutdown must free the process-level admission handle immediately so an
    in-process successor engine is admitted without waiting for GC."""
    from vllm_tt_plugin.platform import TTPlatform

    config = SimpleNamespace(additional_config=None)
    worker = TTWorker.__new__(TTWorker)
    worker.vllm_config = config
    monkeypatch.setattr(TTPlatform, "_tt_vllm_config", config)

    worker.shutdown()

    assert TTPlatform._tt_vllm_config is None


def test_worker_wrapper_shutdown_releases_persistent_capture_once():
    releases = []
    runner = TTModelRunner.__new__(TTModelRunner)
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
        grammar_bitmask=[None],
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
