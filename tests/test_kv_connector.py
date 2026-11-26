# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The KV connector lifecycle on the single-process TT path.

No bytes move here. A fake connector records what the plugin asks of it and
reports the transfers each test scripts. vLLM's NixlConnector worker runs with
a fake NIXL agent that records the card memory it registers and the
descriptors it prepares.
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
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.support.kv_cache import (
    BANKS,
    TILE_BYTES,
    Cluster,
    FakeKVCache,
    fake_cluster,
)
from tests.test_block_scheduler import BLOCK_SIZE, _request
from tests.test_block_scheduler import _runner_output as _block_runner_output
from tests.test_block_scheduler import _scheduler as _real_scheduler
from tests.test_sample_time_grammar import _model_input, _sync_forward
from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput
from vllm_tt_plugin.kv_dram import DramRegion, KVDramLayout, kv_dram_layout
from vllm_tt_plugin.kv_layout import KVLayout, KVPart
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import (
    TTPlatform,
    _install_tt_nixl_dram_registration_patch,
    _install_tt_nixl_single_rank_patch,
)


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
NUM_BLOCKS = 4
HEAD_SIZE = 128  # With BLOCK_SIZE, 16 pages a head: 2 in each DRAM bank.
LAYER_NAMES = ["model.layers.1.self_attn", "model.layers.0.self_attn"]


@pytest.fixture
def cluster(monkeypatch, tmp_path) -> Cluster:
    """Devices 0 to 3 on cards 10 to 13, each with 8 banks at distinct bases."""
    return fake_cluster(monkeypatch, tmp_path)


def _initialize_kv_cache(kv_caches: list, model=None, tt_kv_parts=None) -> None:
    """``TTModelRunner.initialize_kv_cache`` over a model holding ``kv_caches``.

    ``tt_kv_parts`` is the connector's ``"tt_kv_parts"`` extra config.
    """
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=CHIPS,
        head_size=HEAD_SIZE,
        dtype=torch.float32,
    )
    extra = {} if tt_kv_parts is None else {"tt_kv_parts": tt_kv_parts}
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector="NixlConnector",
                kv_role="kv_producer",
                kv_connector_extra_config=extra,
            )
        ),
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


def _kv_caches(num_layers: int = 2, heads: int = 1, chips: int = CHIPS) -> list:
    """A bfloat8_b K and V per layer, each at its own buffer address."""
    return [
        [
            FakeKVCache(
                blocks=NUM_BLOCKS,
                heads=heads,
                chips=chips,
                block_size=BLOCK_SIZE,
                head_dim=HEAD_SIZE,
                address=0x100000 * (2 * layer + i + 1),
            )
            for i in range(2)
        ]
        for layer in range(num_layers)
    ]


def _offset(region: DramRegion) -> int:
    """Where block 0 of ``region`` is in its bank."""
    return region.addr & 0xFFFFFFFF


class TestKVDramRegistration:
    """``initialize_kv_cache`` hands the connector the KV's DRAM regions."""

    def test_each_layer_name_gets_its_caches_regions_on_each_chip(
        self, worker_connector, cluster
    ):
        kv_caches = _kv_caches()
        _initialize_kv_cache(kv_caches)

        layout = worker_connector.kv_caches
        assert isinstance(layout, KVDramLayout)
        assert list(layout.regions) == LAYER_NAMES
        bank_0 = cluster.banks[0][0]["base_addr"]
        for name in LAYER_NAMES:
            k, v = kv_caches[int(name.split(".")[2])]
            regions = layout.regions[name]
            # K then V; in each, chip 0's 8 banks, then chip 1's.
            assert len(regions) == 2 * CHIPS * BANKS
            assert [_offset(r) - bank_0 for r in regions[::BANKS]] == [
                k.address,
                k.address,
                v.address,
                v.address,
            ]
            assert [r.card for r in regions[::BANKS]] == [10, 11, 10, 11]
            # A head of a block is 2 pages in each bank.
            assert {(r.block_bytes, r.block_stride) for r in regions} == {
                (2 * TILE_BYTES, 2 * TILE_BYTES)
            }

    def test_tt_kv_parts_lists_a_chip_as_that_many_chips(
        self, worker_connector, cluster
    ):
        # One chip with both heads, listed as the two chips of a 2-chip server
        # hold them.
        _initialize_kv_cache(_kv_caches(heads=CHIPS, chips=1), tt_kv_parts=CHIPS)

        for regions in worker_connector.kv_caches.regions.values():
            assert len(regions) == 2 * CHIPS * BANKS
            assert {(r.block_bytes, r.block_stride) for r in regions} == {
                (2 * TILE_BYTES, 4 * TILE_BYTES)
            }

    def test_a_model_layout_replaces_the_default_one(self, worker_connector, cluster):
        class Model:
            def get_kv_transfer_layout(self, kv_caches, kv_cache_config):
                # One latent per layer instead of a K and V pair.
                k = kv_caches[0][0]
                return {name: [(k, k.block_bytes, [[0], [1]])] for name in LAYER_NAMES}

        _initialize_kv_cache(_kv_caches(), Model())

        for regions in worker_connector.kv_caches.regions.values():
            assert len(regions) == CHIPS * BANKS

    def test_kv_that_cannot_be_mapped_to_layer_names_is_refused(self, worker_connector):
        with pytest.raises(ValueError, match="allocated 3 \\[k, v\\] entries for 2"):
            _initialize_kv_cache(_kv_caches(num_layers=3))

    def test_no_connector_is_handed_kv_without_a_transfer_group(self):
        # Without --kv-transfer-config the model's KV caches are not inspected.
        _initialize_kv_cache([["not", "a", "kv", "cache"]])


class FakeNixlAgent:
    """Stands in for NIXL's agent, ``nixl._api.nixl_agent``.

    It records the memory a connector worker registers and the descriptor lists
    it prepares, and moves no bytes.

    Attributes
    ----------
    registered
        Per ``register_memory`` call still registered, the ``(ranges,
        mem_type)`` given to ``get_reg_descs``.
    prepared
        Per agent name, the last ``prep_xfer_dlist`` rows ``(address, bytes,
        device ID, stride, count)`` and memory type. The local agent is
        ``"NIXL_INIT_AGENT"``; a remote one is named by its metadata.
    """

    def __init__(self, name: str, config=None):
        self.name = name
        self.registered: list[tuple[list, str]] = []
        self.prepared: dict[str, tuple[list[list[int]], str]] = {}

    def get_agent_metadata(self) -> bytes:
        return self.name.encode()

    def add_remote_agent(self, metadata: bytes) -> str:
        return metadata.decode()

    def remove_remote_agent(self, name: str) -> None:
        pass

    def get_reg_descs(self, ranges: list, mem_type: str) -> tuple[list, str]:
        return ranges, mem_type

    def register_memory(self, descs: tuple[list, str], backends=None) -> None:
        self.registered.append(descs)

    def deregister_memory(self, descs: tuple[list, str]) -> None:
        self.registered.remove(descs)

    def prep_xfer_dlist(self, agent_name: str, descs: np.ndarray, mem_type: str):
        self.prepared[agent_name] = (descs.tolist(), mem_type)
        return len(self.prepared)

    def release_dlist_handle(self, handle) -> None:
        pass

    def release_xfer_handle(self, handle) -> None:
        pass


@pytest.fixture
def fake_nixl(monkeypatch) -> None:
    """NIXL connector workers get a ``FakeNixlAgent``, with or without nixl."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    monkeypatch.setattr(base_worker, "NixlWrapper", FakeNixlAgent)
    monkeypatch.setattr(base_worker, "nixl_agent_config", None)


def _nixl_connector(kv_role: str = "kv_producer"):
    """A worker-side NixlConnector for one attention layer, as a server builds it.

    Each call has its own ``VllmConfig``, so its own engine ID.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
    )

    vllm_config = _real_scheduler(output_width=1).vllm_config
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector", kv_role=kv_role, kv_buffer_device="tt"
    )
    model_config = vllm_config.model_config
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([LAYER_NAMES[1]], spec)],
    )
    # The platform config hook installs these in a server.
    _install_tt_nixl_single_rank_patch()
    _install_tt_nixl_dram_registration_patch()
    return NixlConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)


def _dram_layout(device_ids: list[int], heads: int = 1, parts=None) -> KVDramLayout:
    """The DRAM layout of one layer's K and V, split over ``device_ids``."""
    (k, v) = _kv_caches(num_layers=1, heads=heads, chips=len(device_ids))[0]
    shards = tuple((chip,) for chip in range(len(device_ids)))
    for cache in (k, v):
        cache.device_ids = device_ids
    layout = KVLayout(
        {LAYER_NAMES[1]: tuple(KVPart(c, c.block_bytes, shards) for c in (k, v))}
    )
    return kv_dram_layout(layout, NUM_BLOCKS, parts)


def _handshake_metadata(connector):
    """What the worker sends another server in a NIXL handshake."""
    import msgspec
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )

    payload = connector.connector_worker.xfer_handshake_metadata
    return msgspec.msgpack.decode(payload.agent_metadata_bytes, type=NixlAgentMetadata)


def _rows(layout: KVDramLayout, reader: KVDramLayout | None = None) -> list[list[int]]:
    """The NIXL descriptor of each region of ``layout``'s one layer.

    A row is ``(bank address, bytes, card, block stride, blocks)``. With
    ``reader``, the bytes are those of the reader's region in the same place:
    a pull reads as many bytes as the local region holds.
    """
    (regions,) = layout.regions.values()
    (sizes,) = (reader or layout).regions.values()
    return [
        [r.addr, size.block_bytes, r.card, r.block_stride, NUM_BLOCKS]
        for r, size in zip(regions, sizes, strict=True)
    ]


class TestNixlConnectorOnTT:
    """What vLLM's NixlConnector worker asks of TT, and records, when it starts."""

    def test_nixl_accepts_tt_with_its_kv_in_card_dram_only(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
            _NIXL_SUPPORTED_DEVICE,
        )
        from vllm.platforms import current_platform

        # No CPU buffer: NIXL transfers card memory.
        assert _NIXL_SUPPORTED_DEVICE[current_platform.device_type] == ("tt",)

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

    def test_the_nixl_worker_transfers_card_memory(self, fake_nixl):
        connector = _nixl_connector()
        try:
            worker = connector.connector_worker
            # What NIXL's Tenstorrent build calls card memory.
            assert not worker.use_host_buffer
            assert worker.nixl_memory_type == "VRAM"
            # The worker hands the platform its NIXL device ID before a
            # handshake; TT needs no device context for it.
            TTPlatform.set_device(worker.device_id)
        finally:
            connector.shutdown()

    def test_the_nixl_worker_registers_banks_and_describes_regions(
        self, fake_nixl, cluster
    ):
        connector = _nixl_connector()
        # Mesh index 0 is device 2 (card 12), index 1 is device 1 (card 11).
        layout = _dram_layout([2, 1])
        try:
            connector.register_kv_caches(layout)

            agent = connector.connector_worker.nixl_wrapper
            # Each bank once, whole, as card memory.
            assert agent.registered == [
                ([(*bank, "") for bank in layout.banks], "VRAM")
            ]
            # A strided descriptor per region, on that region's card: K's
            # banks on each chip, then V's.
            assert agent.prepared["NIXL_INIT_AGENT"] == (_rows(layout), "VRAM")
            # What the other server learns in the handshake.
            metadata = _handshake_metadata(connector)
            (regions,) = layout.regions.values()
            assert metadata.block_lens == [r.block_bytes for r in regions]
            assert metadata.block_strides == [r.block_stride for r in regions]
            assert metadata.region_mem_types == ["VRAM"] * len(regions)
            assert metadata.region_names == [LAYER_NAMES[1]] * len(regions)
        finally:
            connector.shutdown()

    def test_a_decode_worker_reads_each_region_from_the_prefill_cards(
        self, fake_nixl, cluster
    ):
        """A 1-chip prefill server listed in 2 parts hands its KV to a 2-chip
        decode server: the handshake passes vLLM's checks, and the decode
        worker reads each of its regions from the prefill region in the same
        place, on the prefill server's card."""
        prefill = _nixl_connector("kv_producer")
        decode = _nixl_connector("kv_consumer")
        sent = _dram_layout([0], heads=CHIPS, parts=CHIPS)
        received = _dram_layout([1, 2])
        try:
            prefill.register_kv_caches(sent)
            decode.register_kv_caches(received)

            worker = decode.connector_worker
            agent_name = worker.add_remote_agent(_handshake_metadata(prefill))

            # Each read takes the decode region's bytes, from the prefill
            # region's address and card, block after block at its stride.
            assert worker.nixl_wrapper.prepared[agent_name] == (
                _rows(sent, reader=received),
                "VRAM",
            )
            assert worker.nixl_wrapper.prepared["NIXL_INIT_AGENT"] == (
                _rows(received),
                "VRAM",
            )
        finally:
            prefill.shutdown()
            decode.shutdown()

    def test_registering_card_dram_records_what_upstream_records_for_tensors(
        self, fake_nixl, cluster
    ):
        """The DRAM registration replaces an upstream method whose records
        other upstream code reads. It fails when a vLLM update makes the
        upstream method record more."""
        upstream = _nixl_connector()
        ours = _nixl_connector()
        (group,) = upstream.connector_worker.kv_cache_config.kv_cache_groups
        page_bytes = group.kv_cache_spec.page_size_bytes
        # A torch cache NIXL registers in place, one page per block.
        kv_caches = {
            LAYER_NAMES[1]: torch.zeros(NUM_BLOCKS, page_bytes, dtype=torch.uint8)
        }
        try:
            by_upstream = _changed_attributes(
                upstream.connector_worker,
                lambda: upstream.register_kv_caches(kv_caches),
            )
            by_ours = _changed_attributes(
                ours.connector_worker,
                lambda: ours.register_kv_caches(_dram_layout([0])),
            )
            assert by_upstream - by_ours == set()
        finally:
            upstream.shutdown()
            ours.shutdown()


def _changed_attributes(worker, register) -> set[str]:
    """Names of ``worker``'s attributes that ``register()`` sets or resizes."""

    def snapshot():
        return {
            name: (id(value), len(value) if hasattr(value, "__len__") else None)
            for name, value in vars(worker).items()
        }

    before = snapshot()
    register()
    after = snapshot()
    return {name for name in after if before.get(name) != after[name]}


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
