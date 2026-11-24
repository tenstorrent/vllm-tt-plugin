# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The KV connector lifecycle on the single-process TT path.

No bytes move here. A fake connector records what the plugin asks of it and
reports the transfers each test scripts.
"""

import threading
from collections import deque
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.test_block_scheduler import BLOCK_SIZE, _request
from tests.test_block_scheduler import _runner_output as _block_runner_output
from tests.test_block_scheduler import _scheduler as _real_scheduler
from tests.test_sample_time_grammar import _model_input, _sync_forward
from vllm_tt_plugin import kv_layout
from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput
from vllm_tt_plugin.model_runner import TTModelRunner


class FakeKVMetadata(KVConnectorMetadata):
    def __init__(self, loads: dict[str, list[int]]):
        # Request ID to the block IDs its external KV should be written into.
        self.loads = loads


class FakeKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Both halves of a connector, with the transfers scripted by the test."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.calls: list[str] = []
        # Scheduler half: tokens of external KV to offer per request ID.
        self.external_tokens: dict[str, int] = {}
        self.pending_loads: dict[str, list[int]] = {}
        # Either half: finished requests it was told of. The scheduler half hears
        # when building metadata, the worker half when asked what has finished.
        self.finished_requests: set[str] = set()
        # Worker half: loads it was told to start, and what to report next.
        self.loads_started: dict[str, list[int]] = {}
        self.finished_recving: set[str] = set()
        self.finished_sending: set[str] = set()
        # Scheduler half: keep a finished request's blocks until its KV is sent.
        self.delay_free = False

    # Worker half.

    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches

    def bind_connector_metadata(self, connector_metadata):
        self.calls.append("bind")
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context, **kwargs):
        self.calls.append("start_load_kv")
        self.loads_started.update(self._get_connector_metadata().loads)

    def wait_for_layer_load(self, layer_name):
        raise AssertionError("TT never calls the per-layer connector hooks")

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        raise AssertionError("TT never calls the per-layer connector hooks")

    def wait_for_save(self):
        self.calls.append("wait_for_save")

    def get_finished(self, finished_req_ids):
        self.calls.append("get_finished")
        self.finished_requests |= finished_req_ids
        finished_sending, self.finished_sending = self.finished_sending, set()
        finished_recving, self.finished_recving = self.finished_recving, set()
        return finished_sending, finished_recving

    def clear_connector_metadata(self):
        self.calls.append("clear")
        super().clear_connector_metadata()

    # Scheduler half.

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        num_tokens = self.external_tokens.get(request.request_id, 0)
        return num_tokens, num_tokens > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            (block_ids,) = blocks.get_block_ids()
            self.pending_loads[request.request_id] = block_ids

    def build_connector_meta(self, scheduler_output):
        self.finished_requests |= scheduler_output.finished_req_ids
        meta = FakeKVMetadata(self.pending_loads)
        self.pending_loads = {}
        return meta

    def request_finished_all_groups(self, request, block_ids):
        return self.delay_free, None


@pytest.fixture
def vllm_config():
    return _real_scheduler(output_width=1).vllm_config


@pytest.fixture
def worker_connector(monkeypatch, vllm_config):
    """The worker-side connector, installed where ``TTWorker`` would put it."""
    config = SimpleNamespace(kv_transfer_config=object())
    connector = FakeKVConnector(config, KVConnectorRole.WORKER, None)
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector)
    return connector


def _scheduler_output() -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.kv_connector_metadata = FakeKVMetadata({"parked": [3]})
    output.num_scheduled_tokens = {"req-0": 1}
    output.total_num_scheduled_tokens = 1
    return output


def _runner(vllm_config, calls: list[str], *, async_decode: bool) -> SimpleNamespace:
    """A runner whose forward and sampling are doubles, as in the grammar tests."""
    model_input = _model_input()
    scheduled: list[str] = []

    def build_model_input(scheduler_output, _grammar):
        scheduled[:] = scheduler_output.num_scheduled_tokens
        return model_input if scheduled else None

    def _runner_output(*_args, **_kwargs):
        return ModelRunnerOutput(
            req_ids=list(scheduled),
            req_id_to_index={req_id: i for i, req_id in enumerate(scheduled)},
            sampled_token_ids=[[7] for _ in scheduled],
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def forward(received_input):
        calls.append("forward")
        return _sync_forward(received_input)

    def submit_async_decode(received_input, **_kwargs):
        calls.append("forward")
        controller = SimpleNamespace(
            complete_decode_step=lambda **_kwargs: SimpleNamespace(),
            build_runner_output_from_completed_step=_runner_output,
            enqueue_completed_decode_step=lambda _done: None,
        )
        return AsyncTTModelRunnerOutput(
            controller, object(), received_input, threading.Event(), object()
        )

    runner = SimpleNamespace(
        vllm_config=vllm_config,
        async_decode=SimpleNamespace(
            can_attempt_steady_decode_from_scheduler=lambda _output: False,
            must_drain_pending_async_steps=lambda *_args: False,
            can_use_steady_decode_fast_path=lambda _input: False,
            submit_async_decode=submit_async_decode,
        ),
        async_decode_scheduling=async_decode,
        scheduler_config=SimpleNamespace(async_scheduling=async_decode),
        _pending_samples=deque(),
        build_model_input=build_model_input,
        _forward_with_model_input=forward,
        _sample_sync_forward=lambda _fwd: ([torch.tensor([[42]])], [None]),
        apply_and_build_runner_output=_runner_output,
        _num_speculative_tokens=0,
    )
    for name in (
        "_reorder_grammar_bitmask",
        "_apply_grammar_to_input",
        "_finish_front_packed_sync",
        "_finish_async_decode",
    ):
        setattr(runner, name, partial(getattr(TTModelRunner, name), runner))
    return runner


@pytest.mark.parametrize("async_decode", [False, True], ids=["sync", "async_decode"])
def test_a_forward_step_runs_inside_the_connector_step(
    vllm_config, worker_connector, async_decode
):
    worker_connector.finished_recving = {"parked"}
    calls = worker_connector.calls
    runner = _runner(vllm_config, calls, async_decode=async_decode)

    assert TTModelRunner.execute_model(runner, _scheduler_output()) is None
    output = TTModelRunner.sample_tokens(runner, None)
    if async_decode:
        output = output.get_output()

    assert calls == [
        "bind",
        "forward",
        "start_load_kv",
        "wait_for_save",
        "get_finished",
        "clear",
    ]
    assert worker_connector.loads_started == {"parked": [3]}
    assert output.req_ids == ["req-0"]
    assert output.kv_connector_output.finished_recving == {"parked"}


def test_an_async_step_reports_its_connector_output_without_a_readback(
    vllm_config, worker_connector
):
    # vLLM reads the connector output this way when ``get_output`` raised, so
    # finished transfers are not lost with a failed readback.
    worker_connector.finished_recving = {"parked"}
    runner = _runner(vllm_config, worker_connector.calls, async_decode=True)

    assert TTModelRunner.execute_model(runner, _scheduler_output()) is None
    output = TTModelRunner.sample_tokens(runner, None)

    assert output.get_kv_connector_output().finished_recving == {"parked"}


def test_a_step_without_a_connector_reports_no_connector_output(vllm_config):
    runner = _runner(vllm_config, [], async_decode=False)

    assert TTModelRunner.execute_model(runner, _scheduler_output()) is None
    output = TTModelRunner.sample_tokens(runner, None)

    assert output.kv_connector_output is None


CHIPS = 2
NUM_BLOCKS = 12
LAYER_NAMES = ["model.layers.1.self_attn", "model.layers.0.self_attn"]


class FakeKVTensor:
    """One layer's bfloat8_b K or V on a 2-chip mesh, one head per chip."""

    dtype = kv_layout.ttnn.bfloat8_b
    layout = kv_layout.ttnn.TILE_LAYOUT
    shape = (NUM_BLOCKS, 1, BLOCK_SIZE, 32)
    block_bytes = (BLOCK_SIZE // 32) * 1088

    def device(self):
        return SimpleNamespace(get_num_devices=lambda: CHIPS)


def _initialize_kv_cache(kv_caches: list, model=None) -> None:
    """``TTModelRunner.initialize_kv_cache`` over a model holding ``kv_caches``."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=CHIPS, head_size=32, dtype=torch.float32
    )
    runner = SimpleNamespace(
        tt_max_batch_size=2,
        model_config=SimpleNamespace(max_model_len=BLOCK_SIZE),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=BLOCK_SIZE),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        vocab_size=8,
        _is_lane_mode=False,
        _host_logitsprocs=None,
        _is_block_output_model=False,
        _output_tokens_per_step=1,
        _validate_kv_cache_groups=TTModelRunner._validate_kv_cache_groups,
        _allocate_kv_caches=lambda _kv_cache_config: kv_caches,
        model=model if model is not None else SimpleNamespace(),
    )
    TTModelRunner.initialize_kv_cache(
        runner,
        KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(LAYER_NAMES, spec)],
        ),
    )


def _kv_caches(num_layers: int = 2) -> list:
    return [[FakeKVTensor(), FakeKVTensor()] for _ in range(num_layers)]


def test_the_connector_is_handed_a_description_of_each_layers_kv(worker_connector):
    _initialize_kv_cache(_kv_caches())

    assert list(worker_connector.kv_caches) == LAYER_NAMES
    for description in worker_connector.kv_caches.values():
        assert description.shape[:3] == (NUM_BLOCKS, CHIPS, BLOCK_SIZE)
        assert description.dtype == torch.uint8
        # A block is the K and V of that block on every chip.
        assert description[0].nbytes == 2 * CHIPS * FakeKVTensor.block_bytes
        # One element stands in for the whole shape.
        assert description.untyped_storage().nbytes() == 1


def test_a_connector_refuses_a_kv_cache_it_cannot_describe(
    worker_connector, monkeypatch
):
    monkeypatch.setattr(FakeKVTensor, "dtype", "uint32")
    with pytest.raises(ValueError, match="cannot describe a uint32"):
        _initialize_kv_cache(_kv_caches())


def test_a_connector_refuses_kv_it_cannot_map_to_layer_names(worker_connector):
    with pytest.raises(ValueError, match="allocated 3 \\[k, v\\] entries for 2"):
        _initialize_kv_cache(_kv_caches(num_layers=3))


def test_a_model_layout_replaces_the_default_one(worker_connector):
    class Model:
        def get_kv_transfer_layout(self, kv_caches, kv_cache_config):
            # One latent per layer instead of a K and V pair.
            return {name: [(kv_caches[0][0], 4096, [[0, 1]])] for name in LAYER_NAMES}

    _initialize_kv_cache(_kv_caches(), Model())

    for description in worker_connector.kv_caches.values():
        assert description[0].nbytes == 4096


def test_no_connector_is_handed_kv_without_a_transfer_group(vllm_config):
    # Without --kv-transfer-config the model's KV caches are not inspected.
    _initialize_kv_cache([["not", "a", "kv", "cache"]])


def _scheduler_with_running_decode():
    """A two-slot scheduler with a connector, whose request ``A`` is decoding."""
    scheduler = _real_scheduler(
        output_width=1,
        max_num_seqs=2,
        kv_transfer_config=KVTransferConfig(
            kv_connector="FakeKVConnector",
            kv_connector_module_path=__name__,
            kv_role="kv_both",
        ),
    )
    scheduler.add_request(_request(8, request_id="A"))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _block_runner_output(prefill, [7]))
    return scheduler


def _run_step(scheduler, runner) -> SchedulerOutput:
    """One engine step: schedule, run the forward, sample, report back."""
    step = scheduler.schedule()
    output = TTModelRunner.execute_model(runner, step)
    if output is None:
        output = TTModelRunner.sample_tokens(runner, None)
    scheduler.update_from_output(step, output)
    return step


@pytest.mark.parametrize("another_request_is_decoding", [False, True])
def test_a_request_loads_remote_kv_then_holds_its_blocks_until_sent(
    worker_connector, another_request_is_decoding
):
    """One request through the whole connector lifecycle, scheduler to runner."""
    scheduler = _scheduler_with_running_decode()
    runner = _runner(scheduler.vllm_config, [], async_decode=False)
    # C is aborted before the step, so the step must tell both halves about it.
    scheduler.add_request(_request(8, request_id="C"))
    aborted = {"C"} if another_request_is_decoding else {"A", "C"}
    scheduler.finish_requests(aborted, RequestStatus.FINISHED_ABORTED)
    scheduler.connector.external_tokens["B"] = BLOCK_SIZE
    scheduler.connector.delay_free = True
    request = _request(1, request_id="B", prompt_length=BLOCK_SIZE + 40)
    scheduler.add_request(request)

    # B is parked on its load. The instruction reaches the worker half, either
    # with A's decode or on a step that schedules no tokens at all.
    step = _run_step(scheduler, runner)
    scheduled = {"A"} if another_request_is_decoding else set()
    assert set(step.num_scheduled_tokens) == scheduled
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    (b_block_ids,) = scheduler.kv_cache_manager.get_block_ids("B")
    assert worker_connector.loads_started == {"B": b_block_ids}
    assert scheduler.connector.finished_requests == aborted
    assert worker_connector.finished_requests == aborted

    # The worker half reports the load; the scheduler hears it on that step's
    # output and prefills the rest of B's prompt on the next one.
    worker_connector.finished_recving = {"B"}
    _run_step(scheduler, runner)
    step = _run_step(scheduler, runner)
    assert step.num_scheduled_tokens["B"] == 40

    # B's one token finished it, but the connector asked to send its KV first.
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    block_pool = scheduler.kv_cache_manager.block_pool
    free_blocks_while_held = block_pool.get_num_free_blocks()
    assert "B" in scheduler.requests
    worker_connector.finished_sending = {"B"}
    _run_step(scheduler, runner)
    assert "B" not in scheduler.requests
    assert block_pool.get_num_free_blocks() == free_blocks_while_held + 2


class NotifyingFakeKVConnector(FakeKVConnector):
    """Like NixlConnector: a finished request leaves a notice for the next build."""

    def request_finished_all_groups(self, request, block_ids):
        self.pending_loads[request.request_id] = []
        return False, None


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_a_notice_recorded_between_steps_reaches_the_returned_step(async_scheduling):
    """Work the connector recorded before ``schedule`` reaches the step it returns,
    also when an async release pass runs first and is then dropped."""
    scheduler = _real_scheduler(
        output_width=1,
        max_num_seqs=1,
        async_scheduling=async_scheduling,
        kv_transfer_config=KVTransferConfig(
            kv_connector="NotifyingFakeKVConnector",
            kv_connector_module_path=__name__,
            kv_role="kv_both",
        ),
    )
    scheduler.add_request(_request(8, request_id="A"))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _block_runner_output(prefill, [7]))
    decode = scheduler.schedule()  # Under async scheduling this step stays held.
    if not async_scheduling:
        scheduler.update_from_output(decode, _block_runner_output(decode, [7]))
    # B cannot be admitted (max_num_seqs=1); C is aborted before admission.
    scheduler.add_request(_request(8, request_id="B"))
    scheduler.add_request(_request(8, request_id="C"))
    scheduler.finish_requests({"C"}, RequestStatus.FINISHED_ABORTED)

    step = scheduler.schedule()

    assert set(step.num_scheduled_tokens) == {"A"}
    assert step.kv_connector_metadata.loads == {"C": []}
