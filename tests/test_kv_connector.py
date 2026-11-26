# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The KV connector lifecycle on the single-process TT path.

A fake connector records what the plugin asks of it and reports the transfers
each test scripts. KV bytes move only between fake chips and CPU buffers.
"""

import threading
from collections import deque
from functools import partial
from types import SimpleNamespace

import numpy as np
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
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.support.kv_cache import FakeKVCache, fake_region_copies
from tests.test_block_scheduler import BLOCK_SIZE, _request
from tests.test_block_scheduler import _runner_output as _block_runner_output
from tests.test_block_scheduler import _scheduler as _real_scheduler
from tests.test_sample_time_grammar import _model_input, _sync_forward
from vllm_tt_plugin import kv_transfer
from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput
from vllm_tt_plugin.kv_layout import default_kv_layout
from vllm_tt_plugin.kv_transfer import kv_cache_descriptions
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import _install_tt_nixl_single_rank_patch


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

    def set_host_xfer_buffer_ops(self, copy_operation):
        self.copy_blocks = copy_operation

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
    runner._kv_caches_by_lane = partial(TTModelRunner._kv_caches_by_lane, runner)
    TTModelRunner.initialize_kv_cache(
        runner,
        KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(LAYER_NAMES, spec)],
        ),
    )


def _kv_caches(num_layers: int = 2) -> list:
    """A bfloat8_b K and V per layer, one head on each chip."""
    return [
        [
            FakeKVCache(blocks=NUM_BLOCKS, heads=1, chips=CHIPS, block_size=BLOCK_SIZE)
            for _ in "kv"
        ]
        for _ in range(num_layers)
    ]


# Bytes of one block of one K or V on one chip.
BLOCK_BYTES = _kv_caches(num_layers=1)[0][0].block_bytes


def test_the_connector_is_handed_a_description_of_each_layers_kv(worker_connector):
    _initialize_kv_cache(_kv_caches())

    assert list(worker_connector.kv_caches) == LAYER_NAMES
    for description in worker_connector.kv_caches.values():
        assert description.shape[:3] == (NUM_BLOCKS, CHIPS, BLOCK_SIZE)
        assert description.dtype == torch.uint8
        # A block is the K and V of that block on every chip.
        assert description[0].nbytes == 2 * CHIPS * BLOCK_BYTES
        # One element stands in for the whole shape.
        assert description.untyped_storage().nbytes() == 1


def test_a_connector_refuses_a_kv_cache_it_cannot_describe(
    worker_connector, monkeypatch
):
    monkeypatch.setattr(FakeKVCache, "dtype", "uint32")
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


@pytest.mark.parametrize("one_list_per_submesh", [False, True])
def test_kv_moves_between_two_servers_through_their_cpu_buffers(
    worker_connector, monkeypatch, one_list_per_submesh
):
    """KV crosses a sender's and a receiver's CPU buffer as the chips' bytes."""
    fake_region_copies(monkeypatch)

    def server():
        kv_caches = _kv_caches()
        _initialize_kv_cache([kv_caches] if one_list_per_submesh else kv_caches)
        # NixlConnector's buffer: tokens before heads in memory, viewed as
        # (blocks, heads, tokens, n) like the description.
        cpu_buffer = {
            name: torch.zeros(
                (d.shape[0], d.shape[2], d.shape[1], d.shape[3]), dtype=d.dtype
            ).permute(0, 2, 1, 3)
            for name, d in worker_connector.kv_caches.items()
        }
        return kv_caches, cpu_buffer, worker_connector.copy_blocks

    sent, sender_buffer, sender_copy = server()
    received, receiver_buffer, receiver_copy = server()
    rng = np.random.default_rng(0)
    for cache in (cache for layer in sent for cache in layer):
        for chip in cache.chips:
            chip[:] = rng.integers(0, 256, chip.size, dtype=np.uint8)

    sent_ids = [0, 1, 2, 9, 8, 7, 5]
    received_ids = [4, 5, 6, 11, 10, 9, 0]
    sender_copy(None, sender_buffer, sent_ids, sent_ids, "d2h")
    # The connector moves buffer blocks between servers; block IDs differ there.
    for name in LAYER_NAMES:
        receiver_buffer[name][received_ids] = sender_buffer[name][sent_ids]
    receiver_copy(receiver_buffer, None, received_ids, received_ids, "h2d")

    for sent_layer, received_layer in zip(sent, received):
        for sent_cache, received_cache in zip(sent_layer, received_layer):
            for chip in range(CHIPS):
                sent_blocks = sent_cache.blocks_on(chip)
                received_blocks = received_cache.blocks_on(chip)
                assert (received_blocks[received_ids] == sent_blocks[sent_ids]).all()
                # Blocks nothing was sent to stay empty.
                assert not received_blocks[[1, 2, 3, 7, 8]].any()


class TestNixlConnectorOnTT:
    """What vLLM's NixlConnector worker asks of TT when it starts."""

    def test_nixl_accepts_tt_with_a_cpu_buffer_only(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
            _NIXL_SUPPORTED_DEVICE,
        )
        from vllm.platforms import current_platform

        # NIXL cannot register TT device memory.
        assert _NIXL_SUPPORTED_DEVICE[current_platform.device_type] == ("cpu",)

    def test_nixl_finds_tt_attention_and_keeps_the_block_size(self, vllm_config):
        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backends,
        )
        from vllm.v1.worker.utils import select_common_block_size

        # TT runs no vLLM attention layer, so NIXL asks the platform. Two
        # servers compare the backend name; a kernel block size other than the
        # block size would split each block.
        (backend,) = get_current_attn_backends(vllm_config)
        assert backend.get_name() == "TT"
        assert select_common_block_size(BLOCK_SIZE, [backend]) == BLOCK_SIZE

    def test_the_nixl_worker_is_rank_0_of_1(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

        # TT creates no tensor-parallel group to ask.
        _install_tt_nixl_single_rank_patch()

        assert base_worker.get_tensor_model_parallel_rank() == 0
        assert base_worker.get_tensor_model_parallel_world_size() == 1

    @pytest.mark.parametrize(
        "connector_name",
        ["NixlConnector", "TTNixlPushConnector"],
        ids=["mirror_of_the_pool", "staging_slots"],
    )
    def test_the_nixl_connector_stages_tt_kv_through_a_cpu_buffer(
        self, vllm_config, connector_name
    ):
        pytest.importorskip("nixl")
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
            NixlConnector,
        )

        from vllm_tt_plugin.nixl_staging import TTNixlPushConnector

        connector_cls = {
            "NixlConnector": NixlConnector,
            "TTNixlPushConnector": TTNixlPushConnector,
        }[connector_name]
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector=connector_name,
            kv_role="kv_producer",
            kv_buffer_device="cpu",
        )
        spec = FullAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=CHIPS, head_size=32, dtype=torch.float32
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(LAYER_NAMES, spec)],
        )
        layout = default_kv_layout(_kv_caches(), kv_cache_config)
        # The platform config hook installs this in a server.
        _install_tt_nixl_single_rank_patch()

        connector = connector_cls(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
        try:
            connector.register_kv_caches(kv_cache_descriptions(kv_cache_config, layout))

            # NIXL transfers its own CPU buffer, one transferred block per block:
            # the K and V of that block on every chip. With staging slots it
            # has one row per slot, fewer than the pool's blocks.
            worker = connector.connector_worker
            rows = NUM_BLOCKS
            if connector_name == "TTNixlPushConnector":
                rows = worker.slots.num_slots
                assert rows < NUM_BLOCKS
            for buffer in worker.host_xfer_buffers.values():
                assert buffer.device.type == "cpu"
                assert buffer.shape[0] == rows
                assert buffer[0].nbytes == 2 * CHIPS * BLOCK_BYTES
                # The copy reads it as one block after another.
                assert kv_transfer._buffer_bytes(buffer).nbytes == buffer.nbytes
        finally:
            connector.shutdown()


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


class CountingFakeKVConnector(FakeKVConnector):
    """Counts how often the scheduler half packs its metadata."""

    builds = 0

    def build_connector_meta(self, scheduler_output):
        CountingFakeKVConnector.builds += 1
        return super().build_connector_meta(scheduler_output)


def _lane_coordinator():
    """Two real lanes under a coordinator, sharing one fake connector."""
    from vllm_tt_plugin.config import store_tt_lane_count
    from vllm_tt_plugin.lane_scheduler import TTLaneCoordinator

    base = _real_scheduler(
        output_width=1,
        max_num_seqs=4,
        kv_transfer_config=KVTransferConfig(
            kv_connector="CountingFakeKVConnector",
            kv_connector_module_path=__name__,
            kv_role="kv_both",
        ),
    )
    store_tt_lane_count(base.vllm_config, 2)
    coordinator = TTLaneCoordinator(
        base.vllm_config,
        base.kv_cache_config,
        base.structured_output_manager,
        BLOCK_SIZE,
    )
    # Both lanes load "A" (lane 0) and "B" (lane 1) from a remote engine.
    for lane, req_id in enumerate(("A", "B")):
        coordinator._req_to_lane[req_id] = lane
        coordinator.connector.external_tokens[req_id] = BLOCK_SIZE
        coordinator.add_request(
            _request(1, request_id=req_id, prompt_length=BLOCK_SIZE + 8)
        )
    return coordinator


def test_lanes_share_one_connector_and_build_its_metadata_once_per_step():
    coordinator = _lane_coordinator()
    assert coordinator.get_kv_connector() is coordinator.connector
    assert all(lane.connector is coordinator.connector for lane in coordinator.lanes)
    CountingFakeKVConnector.builds = 0

    step = coordinator.schedule()

    assert CountingFakeKVConnector.builds == 1
    loads = step.kv_connector_metadata.loads
    assert set(loads) == {"A", "B"}
    # Lanes number their blocks independently, so the lane rides along.
    assert step.kv_connector_metadata.tt_lane_of == {"A": 0, "B": 1}


def test_a_finished_load_reaches_only_the_lane_that_holds_the_request():
    coordinator = _lane_coordinator()
    step = coordinator.schedule()
    pool = coordinator.lanes[0].kv_cache_config.num_blocks
    runner_output = ModelRunnerOutput.with_kv_conn_output_only(
        KVConnectorOutput(finished_recving={"B"}, invalid_block_ids={pool + 3})
    )

    lane0, lane1 = coordinator._runner_output_per_lane(runner_output)
    coordinator.update_from_output(step, runner_output)

    assert lane0.kv_connector_output.finished_recving == set()
    assert lane1.kv_connector_output.finished_recving == {"B"}
    assert lane1.kv_connector_output.invalid_block_ids == {3}
    assert "B" in coordinator.lanes[1].finished_recving_kv_req_ids


def _lane_runner(vllm_config, calls: list[str], *, async_decode: bool):
    """A runner for ``_execute_lane_step`` whose forward and sampling are doubles."""
    model_input = _model_input(
        prompt_lens=None if async_decode else [1],
        tt_sampling_params=SimpleNamespace(enable_log_probs=torch.zeros(1, dtype=bool)),
    )

    def forward(*_args, **_kwargs):
        calls.append("forward")
        return torch.zeros(1)

    def submit_async_lane_decode(*_args):
        calls.append("forward")
        controller = SimpleNamespace(
            complete_decode_step=lambda **_kwargs: SimpleNamespace(),
            build_runner_output_from_completed_step=lambda *_a, **_k: _lane_output(),
            enqueue_completed_decode_step=lambda _done: None,
        )
        return AsyncTTModelRunnerOutput(
            controller, object(), model_input, threading.Event(), object()
        )

    runner = SimpleNamespace(
        vllm_config=vllm_config,
        request_specific_rope=False,
        model=SimpleNamespace(),
        requests={},
        encoder_cache={},
        lane_batch=SimpleNamespace(
            apply_step_plan=lambda *_args, **_kwargs: False,
            build_model_input=lambda *_args: model_input,
            max_num_reqs=2,
        ),
        async_decode=SimpleNamespace(
            can_attempt_steady_lane_decode_from_scheduler=lambda *_a, **_k: False,
            must_drain_pending_async_steps=lambda *_args: False,
            apply_completed_decode_steps_before_build=lambda _output: None,
            capture_submitted_step_context=lambda _req_ids: object(),
            submit_async_lane_decode=submit_async_lane_decode,
        ),
        async_decode_scheduling=async_decode,
        _pending_samples=deque(),
        submit_prefill=forward,
    )
    for name in (
        "_reorder_grammar_bitmask",
        "_finish_lane_sync",
        "_finish_async_decode",
    ):
        setattr(runner, name, partial(getattr(TTModelRunner, name), runner))
    return runner


def _lane_output() -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        sampled_token_ids=[[7]],
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_a_lane_step_without_tokens_still_runs_the_connector(
    vllm_config, worker_connector
):
    # Both requests wait for remote KV, so the lanes schedule no tokens. The
    # worker must still bind this step's metadata and start its loads: a push
    # connector's decode worker sends its block registrations from there.
    step = _lane_coordinator().schedule()
    assert step.total_num_scheduled_tokens == 0
    worker_connector.finished_recving = {"A"}
    runner = _lane_runner(vllm_config, worker_connector.calls, async_decode=False)

    output = TTModelRunner._execute_lane_step(runner, step)

    assert set(worker_connector.loads_started) == {"A", "B"}
    assert output.kv_connector_output.finished_recving == {"A"}
    assert not runner._pending_samples


@pytest.mark.parametrize("async_decode", [False, True], ids=["prefill", "async_decode"])
def test_a_lane_forward_runs_inside_the_connector_step(
    vllm_config, worker_connector, monkeypatch, async_decode
):
    from vllm_tt_plugin.lane_scheduler import TTStepPlan

    monkeypatch.setattr(
        TTModelRunner, "_finish_lane_sync_impl", lambda *_a, **_k: _lane_output()
    )
    worker_connector.finished_recving = {"parked"}
    plan = TTStepPlan(
        is_decode=async_decode,
        capacity=2,
        scheduled_req_ids=("req-0",),
        scheduled_rows=(0,),
        input_rows=(0, 1),
        req_id_to_row={"req-0": 0},
        batch_size_per_dp=(1, 1),
        prefill_empty_slots=None,
    )
    step = _scheduler_output()
    step._tt_step_state = SimpleNamespace(plan=plan)
    runner = _lane_runner(
        vllm_config, worker_connector.calls, async_decode=async_decode
    )

    assert TTModelRunner._execute_lane_step(runner, step) is None
    output = TTModelRunner.sample_tokens(runner, None)
    if async_decode:
        output = output.get_output()

    assert worker_connector.calls == [
        "bind",
        "forward",
        "start_load_kv",
        "wait_for_save",
        "get_finished",
        "clear",
    ]
    assert worker_connector.loads_started == {"parked": [3]}
    assert output.kv_connector_output.finished_recving == {"parked"}
