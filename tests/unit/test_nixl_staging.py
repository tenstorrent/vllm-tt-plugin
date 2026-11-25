# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request host staging for the NIXL push connector.

The worker tests run the TT overrides on a worker built without NIXL: vLLM's
push worker methods the overrides call are replaced by recorders, so each
test sees which IDs reach vLLM and which copies the overrides make. The
scheduler tests do the same with vLLM's push scheduler. The cancellation tests
link a D worker and a P worker through ``FakeNixl`` agents and run their
writer threads' steps by hand, so each test picks the order in which
notifications arrive.
"""

import queue
import re
import threading
from collections import defaultdict, deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest
import torch
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.v1.core.sched.output import KVConnectorBlockState
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from tests.support.nixl_staging import (
    Recorder,
    push_metadata,
    record_push_worker,
    staging_worker,
)
from tests.support.nixl_writer import (
    D_ENGINE,
    P_ENGINE,
    FakeNixl,
    Write,
    push_reg_notif,
    push_writer,
)
from tests.support.nixl_writer import registration as push_registration
from vllm_tt_plugin import nixl_staging
from vllm_tt_plugin.nixl_staging import (
    CANCEL,
    CANCEL_ACK,
    REG_ACK,
    TT_NOTIF_PREFIX,
    NixlPushChunkJob,
    StagingSlots,
    TTNixlPushConnectorScheduler,
    TTNixlPushConnectorWorker,
)

# vLLM's WRITE submission, which ``recorder`` replaces.
_SUBMIT_PUSH = NixlPushConnectorWorker._do_start_push_kv


class TestStagingSlots:
    def test_takes_all_or_nothing(self):
        slots = StagingSlots(4)
        assert slots.take(3) == [0, 1, 2]
        assert slots.take(2) is None
        assert slots.free == 1

    def test_returned_slots_are_taken_again(self):
        slots = StagingSlots(2)
        taken = slots.take(2)
        slots.give_back(taken)
        assert sorted(slots.take(2)) == [0, 1]

    def test_refuses_an_empty_pool(self):
        with pytest.raises(ValueError, match="got 0"):
            StagingSlots(0)


@pytest.mark.parametrize(
    ("num_blocks", "expected"),
    [(10_000, 4 * 2 * 4), (10, 10)],
    ids=["four_longest_requests", "capped_at_the_pool"],
)
def test_slot_count(num_blocks, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=250),
        cache_config=SimpleNamespace(block_size=64),
    )
    # 250 tokens are 4 blocks per group; two groups, four requests.
    assert nixl_staging.staging_slot_count(config, num_blocks, 2) == expected


def test_p_keeps_its_last_blocks_per_group():
    kept = nixl_staging._last_blocks(([9, 8, 7, 6], [3]), ([40, 41], []))
    assert kept == ([7, 6], [])


def test_p_refuses_a_registration_longer_than_its_request():
    with pytest.raises(ValueError, match="registered 3 blocks"):
        nixl_staging._last_blocks(([1, 2],), ([5, 6, 7],))


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    return record_push_worker(monkeypatch)


class Clock:
    """``nixl_staging``'s ``time.perf_counter``, moved by the test."""

    def __init__(self):
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(nixl_staging, "time", fake)
    return fake


def _held_entries(worker) -> dict[str, int]:
    """Entries per container and queue of ``worker`` that holds any, apart from
    the test's record of copies and the peer agents ``_linked`` gives it."""
    held = {}
    for name, value in vars(worker).items():
        if name in ("copies", "_remote_agents"):
            continue
        if isinstance(value, queue.Queue):
            size = value.qsize()
        elif isinstance(value, (dict, set, list, deque)):
            size = len(value)
        else:
            continue
        if size:
            held[name] = size
    return held


def _registration(block_ids, request_id="d-rid"):
    """D's registration of ``block_ids``, as vLLM's push scheduler on D builds it."""
    return {
        "request_id": request_id,
        "decode_engine_id": D_ENGINE,
        "decode_host": "d-host",
        "decode_port": 1,
        "decode_tp_size": 1,
        "local_block_ids": (block_ids,),
        "remote_engine_id": P_ENGINE,
        "remote_host": "p-host",
        "remote_port": 2,
        "remote_tp_size": 1,
    }


def _linked(worker, engine_id: str, peer_engine: str) -> TTNixlPushConnectorWorker:
    """Give ``worker`` a NIXL agent with a done handshake to ``peer_engine``.

    NIXL's ``add_remote_agent`` returns the peer's name as bytes, while the
    peer's own agent names itself with a str, so the peer agent is recorded as
    bytes, as vLLM records it.
    """
    worker.engine_id = engine_id
    worker.nixl_wrapper = FakeNixl(f"{engine_id}-agent")
    worker._remote_agents = {peer_engine: {(0, 0): f"{peer_engine}-agent".encode()}}
    worker._handshake_lock = threading.Lock()
    worker._ensure_handshake = lambda *args, **kwargs: None
    worker.xfer_stats = MagicMock()
    worker._log_failure = lambda **kwargs: None
    return worker


def _writer_pass(worker) -> None:
    """vLLM's writer loop steps before it reads notifications: the queued
    registrations and TT tasks, then the queued staged pushes."""
    while not worker._reg_send_inbox.empty():
        worker._send_registration_to_p(*worker._reg_send_inbox.get_nowait())
    while not worker._deferred_push_inbox.empty():
        worker._do_start_push_kv(*worker._deferred_push_inbox.get_nowait())


def _deliver(sender, receiver) -> None:
    """Hand ``receiver``'s writer thread what ``sender`` has sent, as vLLM's
    writer loop routes a notification with the ``PUSH_REG`` prefix."""
    notifs, sender.nixl_wrapper.sent = sender.nixl_wrapper.sent, []
    for notif in notifs:
        receiver._handle_push_reg_notif(notif)


def _kinds(worker) -> list[str]:
    """What ``worker`` has sent and not yet delivered: ``PUSH_REG`` or the TT kind."""
    return [
        msgspec.msgpack.decode(notif[len(TT_NOTIF_PREFIX) :])["kind"]
        if notif.startswith(TT_NOTIF_PREFIX)
        else "PUSH_REG"
        for notif in worker.nixl_wrapper.sent
    ]


def _tt_notif(kind: str, request_id: str) -> bytes:
    """A TT notification from D about ``request_id``."""
    message = {
        key: value
        for key, value in _registration([], request_id).items()
        if key.startswith("decode_")
    }
    message.update(kind=kind, request_id=request_id)
    return TT_NOTIF_PREFIX + msgspec.msgpack.encode(message)


class TestDecodeSide:
    def test_registration_carries_slots_and_the_blocks_wait_for_them(self, recorder):
        worker = staging_worker(num_slots=4)
        original = push_metadata({"r": _registration([17, 4, 92])})

        worker.start_load_kv(original)

        (sent,) = recorder.metadata
        assert sent.push_registrations["r"]["local_block_ids"] == ([0, 1, 2],)
        # The scheduler's metadata keeps the block IDs.
        assert original.push_registrations["r"]["local_block_ids"] == ([17, 4, 92],)

    def test_a_received_request_lands_in_its_blocks_and_frees_its_slots(self, recorder):
        worker = staging_worker(num_slots=4)
        worker.start_load_kv(push_metadata({"r": _registration([17, 4, 92])}))
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == [("h2d", [0, 1, 2], [17, 4, 92], 0)]
        assert worker.slots.free == 4

    def test_a_registration_waits_while_the_pool_is_full(self, recorder):
        worker = staging_worker(num_slots=3)
        worker.start_load_kv(push_metadata({"a": _registration([1, 2])}))
        worker.start_load_kv(push_metadata({"b": _registration([3, 4])}))
        assert recorder.metadata[-1].push_registrations == {}

        recorder.results.finished_recving = {"a"}
        worker.get_transfer_results()
        worker.start_load_kv(push_metadata())

        assert list(recorder.metadata[-1].push_registrations) == ["b"]


class TestLanes:
    """Lane DP: block IDs repeat across lanes, so each copy names its lane."""

    def test_d_copies_a_received_request_into_its_lane(self, recorder):
        worker = staging_worker(num_slots=4)
        worker.start_load_kv(
            push_metadata({"r": _registration([5, 6])}, lanes={"r": 1})
        )
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == [("h2d", [0, 1], [5, 6], 1)]

    def test_a_failed_receive_reports_its_blocks_numbered_by_lane(self, recorder):
        worker = staging_worker(num_slots=4, pool_blocks=100)
        worker.start_load_kv(
            push_metadata({"r": _registration([5, 6])}, lanes={"r": 2})
        )
        recorder.results.failed_recving = {"r"}
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.get_block_ids_with_load_errors() == {205, 206}
        assert worker.get_block_ids_with_load_errors() == set()

    def test_p_stages_a_matched_request_from_its_lane(self, recorder):
        worker = staging_worker(num_slots=4)
        worker.start_load_kv(push_metadata(finished={"p": ([30, 31],)}, lanes={"p": 3}))
        worker._do_start_push_kv("p", ([30, 31],), _registration([8, 9]))

        worker.get_transfer_results()

        assert worker.copies == [("d2h", [30, 31], [0, 1], 3)]


class TestPrefillSide:
    def test_a_match_is_staged_on_the_runner_thread_then_written(self, recorder):
        worker = staging_worker(num_slots=4)
        worker.start_load_kv(push_metadata(finished={"p": ([30, 31, 32],)}))
        # D already holds the first block of this prompt, so it registers two.
        worker._do_start_push_kv("p", ([30, 31, 32],), _registration([8, 9]))
        assert recorder.pushes == []

        worker.get_transfer_results()

        assert worker.copies == [("d2h", [31, 32], [0, 1], 0)]
        request_id, slots, registration = worker._deferred_push_inbox.get_nowait()
        worker._do_start_push_kv(request_id, slots, registration)
        assert recorder.pushes == [("p", ([0, 1],))]

    def test_slots_are_freed_after_the_send_and_its_last_write(self, recorder):
        worker = staging_worker(num_slots=2)
        worker.start_load_kv(push_metadata(finished={"p": ([30, 31],)}))
        worker._do_start_push_kv("p", ([30, 31],), _registration([8, 9]))
        worker.get_transfer_results()
        worker._sending_transfers["p"] = ["handle"]
        recorder.results.finished_sending = {"p"}

        worker.get_transfer_results()
        assert worker.slots.free == 0

        worker._sending_transfers["p"] = []
        worker.get_transfer_results()
        assert worker.slots.free == 2

    def test_a_send_that_already_ended_is_never_staged(self, recorder):
        worker = staging_worker(num_slots=2)
        worker.start_load_kv(push_metadata(finished={"p": ([30, 31],)}))
        recorder.results.finished_sending = {"p"}
        worker.get_transfer_results()

        worker._do_start_push_kv("p", ([30, 31],), _registration([8, 9]))
        worker.get_transfer_results()

        assert worker.copies == []
        assert worker.slots.free == 2


BLOCK = 64
# P's block table of a 300-token prompt: five blocks, the last one partial.
TABLE = [10, 11, 12, 13, 14]
_FULL = FullAttentionSpec(
    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
)


@pytest.fixture
def push_scheduler(monkeypatch):
    """A builder of ``TTNixlPushConnectorScheduler`` over vLLM's scheduler stubs.

    vLLM's push scheduler methods the overrides call are replaced: they keep
    no state, return a marker from ``request_finished``, and build empty
    metadata. The builder takes the KV cache group specs (one full-attention
    group by default), ``chunked`` for ``enable_chunked_push`` and ``role``.
    """

    def init(self, vllm_config, engine_id, kv_cache_config):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config

    monkeypatch.setattr(NixlPushConnectorScheduler, "__init__", init)
    monkeypatch.setattr(NixlPushConnectorScheduler, "_prefill_backoff", lambda _: 0)
    monkeypatch.setattr(
        NixlPushConnectorScheduler,
        "update_state_after_alloc",
        lambda self, request, blocks, num_external_tokens: None,
    )
    monkeypatch.setattr(
        NixlPushConnectorScheduler,
        "request_finished",
        lambda self, request, block_ids: (True, {"from": "vllm"}),
    )
    monkeypatch.setattr(
        NixlPushConnectorScheduler,
        "build_connector_meta",
        lambda self, scheduler_output: NixlConnectorMetadata(),
    )

    def build(*specs, chunked=True, role="kv_producer"):
        config = SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector=nixl_staging.CONNECTOR_NAME,
                kv_role=role,
                kv_connector_extra_config={"enable_chunked_push": chunked},
            ),
            additional_config={},
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([f"layer.{i}"], spec)
                for i, spec in enumerate(specs or (_FULL,))
            ],
        )
        return TTNixlPushConnectorScheduler(config, "engine", kv_cache_config)

    return build


def _p_request(prompt: int = 300, computed: int = 0) -> SimpleNamespace:
    """A prefill request ``p`` whose KV goes to a decode server."""
    return SimpleNamespace(
        request_id="p",
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        kv_transfer_params={"do_remote_decode": True},
    )


def _step(scheduled, tables=None, preempted=None) -> SimpleNamespace:
    """What ``build_connector_meta`` reads of a ``SchedulerOutput``.

    ``tables`` maps requests to their block tables. Without it the step has
    no block state, as a TT release step has none.
    """
    block_state = None
    if tables is not None:
        block_state = KVConnectorBlockState(
            req_ids=set(tables),
            resolve_block_ids=lambda req_id: (tables[req_id],),
            boundary_state_offloads={},
        )
    return SimpleNamespace(
        num_scheduled_tokens=scheduled,
        preempted_req_ids=preempted,
        kv_connector_block_state=block_state,
    )


def _prefill(scheduler, request, chunks, table=TABLE) -> list[list[tuple[int, int]]]:
    """Each step's jobs as ``(start_block, end_block)``, one chunk per step.

    Like vLLM's scheduler, a step's metadata is built before the step's tokens
    are added to ``num_computed_tokens``.
    """
    ranges = []
    for chunk in chunks:
        meta = scheduler.build_connector_meta(
            _step({request.request_id: chunk}, {request.request_id: table})
        )
        ranges.append(
            [(job.start_block, job.end_block) for job in meta.push_chunk_jobs]
        )
        request.num_computed_tokens += chunk
    return ranges


class TestChunkJobs:
    """P's scheduler names the prompt blocks each prefill step completes."""

    def test_each_prefill_step_gets_a_job_for_the_full_blocks_it_completed(
        self, push_scheduler
    ):
        scheduler = push_scheduler()
        request = _p_request(prompt=300)
        scheduler.update_state_after_alloc(request, None, 0)

        jobs = []
        for chunk in (100, 100, 100):
            meta = scheduler.build_connector_meta(_step({"p": chunk}, {"p": TABLE}))
            jobs += meta.push_chunk_jobs
            request.num_computed_tokens += chunk

        assert jobs == [
            NixlPushChunkJob("p", [10], start_block=0, end_block=1, total_blocks=5),
            NixlPushChunkJob("p", [11, 12], start_block=1, end_block=3, total_blocks=5),
            NixlPushChunkJob("p", [13, 14], start_block=3, end_block=5, total_blocks=5),
        ]

    def test_the_partial_last_block_goes_with_the_job_that_completes_the_prompt(
        self, push_scheduler
    ):
        scheduler = push_scheduler()
        request = _p_request(prompt=130)
        scheduler.update_state_after_alloc(request, None, 0)

        # 100 tokens fill one block; the last 30 complete the prompt, whose
        # third block holds only 2 tokens.
        assert _prefill(scheduler, request, [100, 30]) == [[(0, 1)], [(1, 3)]]

    def test_a_block_a_chunk_leaves_partial_goes_with_the_next_job(
        self, push_scheduler
    ):
        scheduler = push_scheduler()
        request = _p_request(prompt=300)
        scheduler.update_state_after_alloc(request, None, 0)

        # 32 tokens complete no block; 96 complete block 0 but not block 1.
        assert _prefill(scheduler, request, [32, 64]) == [[], [(0, 1)]]

    def test_the_first_job_includes_p_prefix_cache_hits(self, push_scheduler):
        scheduler = push_scheduler()
        # P's prefix cache holds the first 128 tokens; the step computes 128 more.
        request = _p_request(prompt=300, computed=128)
        scheduler.update_state_after_alloc(request, None, 0)

        meta = scheduler.build_connector_meta(_step({"p": 128}, {"p": TABLE}))

        (job,) = meta.push_chunk_jobs
        assert (job.start_block, job.end_block, job.block_ids) == (0, 4, TABLE[:4])

    def test_a_preempted_request_is_listed_and_restarts_at_block_zero(
        self, push_scheduler
    ):
        scheduler = push_scheduler()
        request = _p_request(prompt=300)
        scheduler.update_state_after_alloc(request, None, 0)
        _prefill(scheduler, request, [100])

        # vLLM's preemption frees the blocks and forgets the computed tokens.
        request.num_computed_tokens = 0
        meta = scheduler.build_connector_meta(_step({}, preempted={"p", "other"}))
        assert meta.push_chunk_preemptions == {"p"}

        # Scheduled again, the request is allocated again, in new blocks.
        scheduler.update_state_after_alloc(request, None, 0)
        assert _prefill(scheduler, request, [200], table=[20, 21, 22, 23, 24]) == [
            [(0, 3)]
        ]

    def test_a_finished_request_gets_no_more_jobs(self, push_scheduler):
        scheduler = push_scheduler()
        request = _p_request(prompt=300)
        scheduler.update_state_after_alloc(request, None, 0)
        _prefill(scheduler, request, [100])

        assert scheduler.request_finished(request, (TABLE,)) == (True, {"from": "vllm"})
        assert _prefill(scheduler, request, [200]) == [[]]

    def test_a_step_without_prefill_needs_no_block_state(self, push_scheduler):
        scheduler = push_scheduler()
        request = _p_request(prompt=300)
        scheduler.update_state_after_alloc(request, None, 0)
        _prefill(scheduler, request, [100])

        # A TT release step schedules nothing; a decode step schedules another
        # request. Neither carries block state.
        assert scheduler.build_connector_meta(_step({})).push_chunk_jobs == []
        assert scheduler.build_connector_meta(_step({"d": 1})).push_chunk_jobs == []

        # Under async scheduling a request whose prompt is complete may be
        # scheduled again before its output is known.
        _prefill(scheduler, request, [200])
        assert scheduler.build_connector_meta(_step({"p": 1})).push_chunk_jobs == []


_SLIDING = SlidingWindowSpec(
    block_size=BLOCK,
    num_kv_heads=1,
    head_size=1,
    dtype=torch.float32,
    sliding_window=128,
)
_GDN = MambaSpec(block_size=BLOCK, shapes=((1,),), dtypes=(torch.float32,))


@pytest.mark.parametrize(
    ("specs", "backoff"),
    [((_FULL, _FULL), 0), ((_SLIDING,), 0), ((_GDN,), 0), ((_FULL,), 1)],
    ids=["two_groups", "sliding_window", "gdn", "mtp_lookahead"],
)
def test_chunked_push_is_refused_for_a_sliding_window_gdn_or_mtp_model(
    push_scheduler, monkeypatch, specs, backoff
):
    # A prefill backoff is the prompt tail P leaves for D: MTP's lookahead
    # window, or the token from which D derives a Mamba state.
    monkeypatch.setattr(
        NixlPushConnectorScheduler, "_prefill_backoff", lambda _: backoff
    )
    with pytest.raises(ValueError, match="enable_chunked_push"):
        push_scheduler(*specs)


@pytest.mark.parametrize(
    ("value", "role"),
    [("false", "kv_producer"), (1, "kv_producer"), ("true", "kv_consumer")],
    ids=["string_on_p", "integer_on_p", "string_on_d"],
)
def test_a_non_boolean_enable_chunked_push_is_refused(push_scheduler, value, role):
    # Read as a truth value, the JSON string "false" would turn the setting on.
    with pytest.raises(ValueError, match=re.escape(f"got {value!r}")):
        push_scheduler(chunked=value, role=role)


@pytest.mark.parametrize(
    ("chunked", "role"),
    [(False, "kv_producer"), (True, "kv_consumer")],
    ids=["off", "decode_server"],
)
def test_with_chunked_push_off_no_job_is_built_and_nothing_is_copied_per_step(
    push_scheduler, chunked, role
):
    # Off, or on a decode server, it refuses no model either.
    scheduler = push_scheduler(_SLIDING, chunked=chunked, role=role)
    request = _p_request(prompt=300)
    scheduler.update_state_after_alloc(request, None, 0)
    meta = scheduler.build_connector_meta(_step({"p": 300}, {"p": TABLE}))
    worker = staging_worker(num_slots=8)

    worker.save_kv_to_host(meta)

    assert getattr(meta, "push_chunk_jobs", []) == []
    assert worker.copies == []
    assert worker.slots.free == 8


def _job(start, end, block_ids, total=5, req_id="p") -> NixlPushChunkJob:
    return NixlPushChunkJob(
        request_id=req_id,
        block_ids=block_ids,
        start_block=start,
        end_block=end,
        total_blocks=total,
    )


class TestChunkStaging:
    """P copies each prefill step's completed blocks into its request's slots."""

    def test_a_job_is_copied_into_the_slots_of_its_prompt_positions(self):
        worker = staging_worker(num_slots=8)

        worker.save_kv_to_host(push_metadata(jobs=[_job(0, 1, [30])]))
        worker.save_kv_to_host(push_metadata(jobs=[_job(1, 3, [31, 32])]))

        assert worker.copies == [("d2h", [30], [0], 0), ("d2h", [31, 32], [1, 2], 0)]
        # The first job took a slot for each of the prompt's five blocks.
        assert worker.slots.free == 3

    def test_the_first_job_takes_slots_for_the_whole_prompt_or_none(self, recorder):
        worker = staging_worker(num_slots=6)
        worker.save_kv_to_host(
            push_metadata(jobs=[_job(0, 1, [30], req_id="a"), _job(0, 1, [40], 3, "b")])
        )
        worker.save_kv_to_host(push_metadata(jobs=[_job(1, 3, [41, 42], 3, "b")]))

        # One slot was left for b's three blocks: b stages nothing, then or later.
        assert worker.copies == [("d2h", [30], [0], 0)]
        assert worker.slots.free == 1

        # Once matched, b is copied from its blocks, as without chunked push.
        worker.start_load_kv(push_metadata(finished={"b": ([40, 41, 42],)}))
        worker._do_start_push_kv("b", ([40, 41, 42],), _registration([8]))
        worker.get_transfer_results()
        assert worker.copies[-1] == ("d2h", [42], [5], 0)

    def test_a_matched_request_is_written_from_its_staged_slots_without_a_second_copy(  # noqa: E501
        self, recorder
    ):
        worker = staging_worker(num_slots=8)
        for job in (_job(0, 1, [30]), _job(1, 3, [31, 32]), _job(3, 5, [33, 34])):
            worker.save_kv_to_host(push_metadata(jobs=[job]))
        staged = list(worker.copies)

        # D holds the first two blocks of this prompt, so it registers three.
        worker.start_load_kv(push_metadata(finished={"p": ([30, 31, 32, 33, 34],)}))
        worker._do_start_push_kv(
            "p", ([30, 31, 32, 33, 34],), _registration([8, 9, 10])
        )
        worker.get_transfer_results()

        assert worker.copies == staged
        request_id, slots, registration = worker._deferred_push_inbox.get_nowait()
        worker._do_start_push_kv(request_id, slots, registration)
        assert recorder.pushes == [("p", ([2, 3, 4],))]
        # All of the prompt's slots go back once the send and its writes end.
        recorder.results.finished_sending = {"p"}
        worker.get_transfer_results()
        assert worker.slots.free == 8

    @pytest.mark.parametrize("listed", [True, False], ids=["listed", "not_listed"])
    def test_a_preempted_request_gets_fresh_slots_and_its_old_slots_go_back(
        self, recorder, listed
    ):
        worker = push_writer(staging_worker(num_slots=16, chunked=True))
        worker.save_kv_to_host(push_metadata(jobs=[_job(0, 1, [30])]))

        # The recompute's first job starts at block 0 again; a job there always
        # begins the request anew, whether or not the preemption was listed.
        recompute = push_metadata(
            preempted={"p"} if listed else (), jobs=[_job(0, 2, [40, 41])]
        )
        worker.start_load_kv(recompute)
        worker.save_kv_to_host(recompute)
        assert worker.copies[-1] == ("d2h", [40, 41], [5, 6], 0)
        assert worker.slots.free == 6

        # Slots 0 to 4 go back once the writer thread has seen the preemption.
        worker._push_writer_pass()
        worker.get_transfer_results()
        assert worker.slots.free == 11

    @pytest.mark.parametrize("ending", ["aborted", "lease_expired"])
    def test_a_request_that_ends_unmatched_gives_its_slots_back(self, recorder, ending):
        worker = push_writer(staging_worker(num_slots=8, chunked=True))
        worker.save_kv_to_host(push_metadata(jobs=[_job(0, 1, [30])]))
        assert worker.slots.free == 3

        if ending == "aborted":
            worker.start_load_kv(push_metadata(aborted={"p"}))
        else:
            # vLLM reports a finished request whose lease expired as sent.
            recorder.results.finished_sending = {"p"}
        worker.get_transfer_results()
        worker._push_writer_pass()
        worker.get_transfer_results()

        assert worker.slots.free == 8

    def test_a_job_that_does_not_continue_the_staged_blocks_is_refused(self):
        worker = staging_worker(num_slots=8)
        worker.save_kv_to_host(push_metadata(jobs=[_job(0, 1, [30])]))

        with pytest.raises(RuntimeError, match="block 3"):
            worker.save_kv_to_host(push_metadata(jobs=[_job(3, 5, [33, 34])]))


# P's device blocks of a 300-token prompt in blocks of 64.
PROMPT_BLOCKS = [30, 31, 32, 33, 34]
# Its prefill in chunks of 100 tokens: the jobs of three steps.
CHUNK_JOBS = (_job(0, 1, [30]), _job(1, 3, [31, 32]), _job(3, 5, [33, 34]))


def _p_worker(num_slots: int = 8):
    """P's worker with chunked push and a writer whose passes the test runs."""
    return push_writer(staging_worker(num_slots, chunked=True))


def _run_step(worker, jobs=(), **fields) -> None:
    """One step on P's model runner thread, in the order vLLM calls the worker."""
    meta = push_metadata(jobs=jobs, **fields)
    worker.start_load_kv(meta)
    worker.save_kv_to_host(meta)
    worker.finish_chunk_step(meta)
    worker.get_transfer_results()


def _finish(worker, req_id: str, blocks: list[int]) -> None:
    """P's request finishes: vLLM's ``start_load_kv`` also hands its blocks to
    the writer thread."""
    worker.start_load_kv(push_metadata(finished={req_id: (blocks,)}))
    worker._finished_blocks_inbox.put((req_id, (blocks,)))


def _on_writer_thread(worker) -> None:
    """Run one writer pass on another thread, as the writer thread does."""
    thread = threading.Thread(target=worker._push_writer_pass, name="writer")
    thread.start()
    thread.join(timeout=5)


class TestChunkWrites:
    """P writes each prefill step's chunk from its slots while it prefills."""

    def test_a_chunk_is_written_from_the_slots_it_was_staged_into(self, recorder):
        worker = _p_worker()
        # D holds the prompt's first two blocks, so it registers three.
        worker._pending_d_registrations["p"] = push_registration([100, 101, 102])

        for job in CHUNK_JOBS:
            _run_step(worker, jobs=[job])
            worker._push_writer_pass()
            worker.nixl_wrapper.finish()
        worker._push_writer_pass()

        assert worker.copies == [
            ("d2h", [30], [0], 0),
            ("d2h", [31, 32], [1, 2], 0),
            ("d2h", [33, 34], [3, 4], 0),
        ]
        assert worker.nixl_wrapper.writes == [
            Write(1, [2], [100], b""),
            Write(2, [3, 4], [101, 102], b"p:1"),
        ]

    @pytest.mark.parametrize("cause", ["late_registration", "failed_write"])
    def test_a_request_written_whole_is_written_from_its_slots_without_a_copy(
        self, recorder, cause
    ):
        worker = _p_worker()
        reg = push_registration([100, 101, 102])
        if cause == "failed_write":
            worker._pending_d_registrations["p"] = reg
        for job in CHUNK_JOBS:
            _run_step(worker, jobs=[job])
            worker._push_writer_pass()
            worker.nixl_wrapper.finish("ERR")
        worker._push_writer_pass()
        worker.nixl_wrapper.finish()
        staged = list(worker.copies)

        _finish(worker, "p", PROMPT_BLOCKS)
        worker._push_writer_pass()
        if cause == "late_registration":
            worker.nixl_wrapper.notifs = {"d": [push_reg_notif(reg)]}
        worker._push_writer_pass()
        worker.get_transfer_results()
        worker._push_writer_pass()

        assert all(write.notif == b"" for write in worker.nixl_wrapper.writes)
        assert recorder.pushes == [("p", ([2, 3, 4],))]
        assert worker.copies == staged

    def test_a_handed_off_request_gives_its_slots_back_once_reported_sent(
        self, recorder
    ):
        worker = _p_worker()
        worker._pending_d_registrations["p"] = push_registration(PROMPT_BLOCKS)
        _run_step(worker, jobs=[_job(0, 5, PROMPT_BLOCKS)])
        worker._push_writer_pass()
        worker.nixl_wrapper.finish()
        _finish(worker, "p", PROMPT_BLOCKS)
        worker._push_writer_pass()
        assert worker._sending_transfers["p"] == []

        # vLLM reports the request sent once the empty handle list is polled.
        recorder.results.finished_sending = {"p"}
        worker.get_transfer_results()
        worker._push_writer_pass()
        worker.get_transfer_results()

        assert worker.slots.free == 8
        assert recorder.pushes == []

    @pytest.mark.parametrize("ending", ["preempted", "aborted", "lease_expired"])
    def test_slots_go_back_only_after_every_write_of_the_request_ended(
        self, recorder, ending
    ):
        worker = _p_worker()
        worker._pending_d_registrations["p"] = push_registration(PROMPT_BLOCKS)
        _run_step(worker, jobs=[_job(0, 2, [30, 31])])
        worker._push_writer_pass()
        assert worker.slots.free == 3

        if ending == "preempted":
            _run_step(worker, preempted={"p"})
        elif ending == "aborted":
            _run_step(worker, aborted={"p"})
        else:
            # vLLM reports a finished request whose lease expired as sent.
            recorder.results.finished_sending = {"p"}
            worker.get_transfer_results()
        worker._push_writer_pass()
        worker.get_transfer_results()
        # The WRITE of slots 0 and 1 may still be reading them.
        assert worker.slots.free == 3

        worker.nixl_wrapper.finish()
        worker._push_writer_pass()
        worker.get_transfer_results()
        assert worker.slots.free == 8

    def test_handle_preemptions_never_blocks_the_model_runner_thread(self, recorder):
        worker = _p_worker(num_slots=16)
        worker._pending_d_registrations["p"] = push_registration(PROMPT_BLOCKS)
        _run_step(worker, jobs=[_job(0, 2, [30, 31])])
        # This WRITE never completes.
        worker._push_writer_pass()

        step = threading.Thread(
            target=_run_step, args=(worker,), kwargs={"preempted": {"p"}}
        )
        step.start()
        step.join(timeout=1)
        assert not step.is_alive()

        # The recompute stages into fresh slots; the writer holds its chunk
        # until the old WRITE has ended.
        _run_step(worker, jobs=[_job(0, 2, [40, 41])])
        worker._push_writer_pass()
        assert worker.copies[-1] == ("d2h", [40, 41], [5, 6], 0)
        assert len(worker.nixl_wrapper.writes) == 1

    def test_the_writer_thread_never_touches_the_device(self, recorder):
        worker = _p_worker()
        copy, threads = worker.copy_blocks, []

        def copy_blocks(*args, **kwargs):
            threads.append(threading.current_thread().name)
            copy(*args, **kwargs)

        worker.copy_blocks = copy_blocks
        # Request a is staged by chunks; b found no slots and is copied whole.
        _run_step(worker, jobs=[_job(0, 5, PROMPT_BLOCKS, req_id="a")])
        _on_writer_thread(worker)
        for req_id, blocks in (("a", PROMPT_BLOCKS), ("b", [40, 41])):
            _finish(worker, req_id, blocks)
        worker.nixl_wrapper.notifs = {
            "d": [
                push_reg_notif(push_registration(PROMPT_BLOCKS, request_id="a")),
                push_reg_notif(push_registration([8, 9], request_id="b")),
            ]
        }
        _on_writer_thread(worker)
        worker.get_transfer_results()
        _on_writer_thread(worker)

        assert len(worker.nixl_wrapper.writes) == 1
        assert recorder.pushes == [("b", ([5, 6],))]
        assert worker.copies[-1] == ("d2h", [40, 41], [5, 6], 0)
        assert set(threads) == {threading.current_thread().name}


class TestSendSlotLifetime:
    """P's slots stay with a request while its write is being submitted or runs."""

    def test_a_lease_that_expires_during_submission_keeps_the_slots(
        self, recorder, monkeypatch
    ):
        p = staging_worker(num_slots=2)
        registration = {
            **_registration([8, 9]),
            "decode_engine_id": "d-engine",
            "decode_host": "d-host",
            "decode_port": 1,
            "decode_tp_size": 1,
        }
        p.start_load_kv(push_metadata(finished={"p": ([30, 31],)}))
        p._do_start_push_kv("p", ([30, 31],), registration)
        p.get_transfer_results()
        pushed = p._deferred_push_inbox.get_nowait()
        # vLLM's own submission, paused where it waits for P's handshake with
        # D, after the TT override has let it use the slots.
        monkeypatch.setattr(NixlPushConnectorWorker, "_do_start_push_kv", _SUBMIT_PUSH)
        p.world_size = 1
        p._physical_blocks_per_logical_kv_block = 1
        p._engine_last_active = {}
        p._sending_transfers = defaultdict(list)
        paused, resume = threading.Event(), threading.Event()
        submitted = []

        def handshake(*args):
            paused.set()
            resume.wait(5)

        def write(req_id, meta):
            submitted.append((req_id, meta.local_block_ids, p.slots.free))
            p._sending_transfers[req_id].append("handle")

        p._ensure_handshake = handshake
        p._xfer_blocks_for_req = write
        writer = threading.Thread(target=p._do_start_push_kv, args=pushed)
        writer.start()
        assert paused.wait(5)

        recorder.results.finished_sending = {"p"}
        p.get_transfer_results()
        free_during_submission = p.slots.free
        resume.set()
        writer.join(5)
        p.get_transfer_results()
        free_while_writing = p.slots.free
        del p._sending_transfers["p"]
        p.get_transfer_results()

        assert free_during_submission == 0
        assert submitted == [("p", ([0, 1],), 0)]
        assert free_while_writing == 0
        assert p.slots.free == 2

    def test_completed_sends_leave_no_state(self, recorder):
        p = staging_worker(num_slots=2)
        for i in range(10_000):
            req_id = f"p{i}"
            p.start_load_kv(push_metadata(finished={req_id: ([30, 31],)}))
            p._do_start_push_kv(req_id, ([30, 31],), _registration([8, 9]))
            p.get_transfer_results()
            p._do_start_push_kv(*p._deferred_push_inbox.get_nowait())
            recorder.results.finished_sending = {req_id}
            p.get_transfer_results()

        assert len(recorder.pushes) == 10_000
        assert _held_entries(p) == {}
        assert p.slots.free == 2


class TestCancellation:
    """D gives a failed receive's slots to another request only once P has
    confirmed that it writes no more into them."""

    @pytest.fixture
    def d(self):
        return _linked(
            staging_worker(num_slots=2, recv_timeout=60.0), D_ENGINE, P_ENGINE
        )

    @pytest.fixture
    def p(self):
        return _linked(staging_worker(num_slots=2), P_ENGINE, D_ENGINE)

    def test_a_registration_p_reads_late_never_lands_in_reused_slots(
        self, recorder, clock, d, p
    ):
        d.start_load_kv(push_metadata({"old": _registration([5, 6], "old")}))
        _writer_pass(d)
        # P's writer thread waits for a wake that its idle engine never gives,
        # so the registration stays unread on P.
        clock.now += 61.0
        d.get_transfer_results()
        recorder.results.failed_recving = recorder.results.finished_recving = {"old"}
        d.get_transfer_results()
        _writer_pass(d)
        clock.now += 10_000.0
        d.get_transfer_results()

        d.start_load_kv(push_metadata({"new": _registration([7, 8], "new")}))
        assert recorder.metadata[-1].push_registrations == {}

        # P finishes "old", then reads the registration and later the cancel.
        p.start_load_kv(push_metadata(finished={"old": ([30, 31],)}))
        p._push_finished_blocks["old"] = ([30, 31],)
        _deliver(d, p)
        _deliver(p, d)
        _deliver(d, p)
        p.get_transfer_results()
        _writer_pass(p)
        _deliver(p, d)
        d.get_transfer_results()
        d.start_load_kv(push_metadata())

        assert recorder.pushes == []
        assert p.copies == []
        new = recorder.metadata[-1].push_registrations["new"]
        assert new["local_block_ids"] == ([0, 1],)

    def test_d_cancels_only_a_registration_p_has_acknowledged(self, recorder, d, p):
        d.start_load_kv(push_metadata({"r": _registration([5, 6], "r")}))
        _writer_pass(d)
        recorder.results.failed_recving = recorder.results.finished_recving = {"r"}
        d.get_transfer_results()
        _writer_pass(d)
        # NIXL does not order notifications: a cancel sent now could reach P
        # before the registration.
        assert _kinds(d) == ["PUSH_REG"]

        _deliver(d, p)
        assert _kinds(p) == [REG_ACK]
        _deliver(p, d)
        assert _kinds(d) == [CANCEL]

    def test_p_acknowledges_a_cancel_once_its_write_has_ended(self, recorder, p):
        p.start_load_kv(push_metadata(finished={"r": ([30, 31],)}))
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(
            PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(_registration([8, 9], "r"))
        )
        p.get_transfer_results()
        _writer_pass(p)
        assert recorder.pushes == [("r", ([0, 1],))]
        p._sending_transfers["r"] = ["handle"]

        p._handle_push_reg_notif(_tt_notif(CANCEL, "r"))
        p.get_transfer_results()
        _writer_pass(p)
        assert _kinds(p) == [REG_ACK]

        del p._sending_transfers["r"]
        p.get_transfer_results()
        _writer_pass(p)
        assert _kinds(p) == [REG_ACK, CANCEL_ACK]

    @pytest.mark.parametrize("staged", [False, True], ids=["matched", "staged"])
    def test_p_writes_nothing_after_a_cancel(self, recorder, p, staged):
        p.start_load_kv(push_metadata(finished={"r": ([30, 31],)}))
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(
            PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(_registration([8, 9], "r"))
        )
        if staged:
            p.get_transfer_results()

        p._handle_push_reg_notif(_tt_notif(CANCEL, "r"))
        p.get_transfer_results()
        _writer_pass(p)

        assert recorder.pushes == []
        assert len(p.copies) == int(staged)
        assert _kinds(p) == [REG_ACK, CANCEL_ACK]
        recorder.results.finished_sending = {"r"}
        p.get_transfer_results()
        assert p.slots.free == 2

    def test_a_cancel_drops_a_pending_registration(self, recorder, p):
        p._handle_push_reg_notif(
            PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(_registration([8, 9], "r"))
        )

        p._handle_push_reg_notif(_tt_notif(CANCEL, "r"))

        # P's request finishes later and finds no registration to match.
        assert p._pop_matching_registration("r") is None
        assert _kinds(p) == [REG_ACK, CANCEL_ACK]

    def test_a_received_request_leaves_no_state(self, recorder, d):
        d.start_load_kv(push_metadata({"r": _registration([5, 6], "r")}))
        _writer_pass(d)
        # P's acknowledgement of the registration never arrives.
        recorder.results.finished_recving = {"r"}

        d.get_transfer_results()
        _writer_pass(d)

        assert _held_entries(d) == {}
        assert d.slots.free == 2

    def test_a_registration_that_never_reached_p_frees_its_slots(
        self, recorder, clock, d
    ):
        # P stopped while D still holds its agent: NIXL refuses the send.
        d.nixl_wrapper.disconnected = True
        d.start_load_kv(push_metadata({"r": _registration([5, 6], "r")}))
        _writer_pass(d)

        clock.now += 61.0
        d.get_transfer_results()
        recorder.results.failed_recving = recorder.results.finished_recving = {"r"}
        d.get_transfer_results()
        _writer_pass(d)
        d.get_transfer_results()

        assert d.slots.free == 2

    @pytest.mark.parametrize(
        "dies", ["before_the_cancel", "after_the_cancel"], ids=lambda dies: dies
    )
    def test_slots_of_a_cancel_p_can_no_longer_answer_are_freed(
        self, recorder, clock, d, p, dies
    ):
        d.start_load_kv(push_metadata({"r": _registration([5, 6], "r")}))
        _writer_pass(d)
        _deliver(d, p)
        _deliver(p, d)
        if dies == "before_the_cancel":
            d.nixl_wrapper.disconnected = True

        recorder.results.failed_recving = recorder.results.finished_recving = {"r"}
        d.get_transfer_results()
        _writer_pass(d)
        d.nixl_wrapper.disconnected = True
        for _ in range(3):
            clock.now += 1.0
            d.get_transfer_results()
            _writer_pass(d)
        # NIXL reported the disconnect; a WRITE P posted before may still land.
        assert d.slots.free == 0

        for _ in range(12):
            clock.now += 1.0
            d.get_transfer_results()
            _writer_pass(d)

        d.get_block_ids_with_load_errors()  # vLLM reads them each step
        assert d.slots.free == 2
        assert _held_entries(d) == {}

    def test_a_silent_p_that_is_alive_keeps_the_slots(self, recorder, clock, d, p):
        d.start_load_kv(push_metadata({"r": _registration([5, 6], "r")}))
        _writer_pass(d)
        _deliver(d, p)
        _deliver(p, d)
        recorder.results.failed_recving = recorder.results.finished_recving = {"r"}
        d.get_transfer_results()
        _writer_pass(d)

        # P never answers the cancel, but D's notifications still reach it.
        for _ in range(100):
            clock.now += 10.0
            d.get_transfer_results()
            _writer_pass(d)

        assert "probe" in _kinds(d)
        assert d.slots.free == 0

    def test_a_registration_not_yet_sent_is_withdrawn(self, recorder, d):
        handshake = Future()
        d._ensure_handshake = lambda *args, **kwargs: handshake
        d.start_load_kv(push_metadata({"r": _registration([5, 6], "r")}))
        _writer_pass(d)

        recorder.results.failed_recving = recorder.results.finished_recving = {"r"}
        d.get_transfer_results()
        _writer_pass(d)
        d.get_transfer_results()
        assert d.slots.free == 2
        assert d.copies == []

        # vLLM queues the registration again once the handshake is done.
        handshake.set_result(({(0, 0): f"{P_ENGINE}-agent"}, 0.0))
        _writer_pass(d)
        assert d.nixl_wrapper.sent == []


class TestChunkCancellation:
    """Once D cancels a registration, P writes no chunk for it and acknowledges
    the cancel only after the chunk WRITEs in flight have ended."""

    @pytest.mark.parametrize(
        ("cancel_at_step", "written"),
        [
            (None, [Write(1, [2], [100], b""), Write(2, [3, 4], [101, 102], b"p:1")]),
            (2, [Write(1, [2], [100], b"")]),
            (1, []),
        ],
        ids=["not_cancelled", "cancelled_before_the_last_chunk", "cancelled_early"],
    )
    def test_p_writes_no_chunk_after_a_cancel(self, recorder, cancel_at_step, written):
        # D holds the prompt's first two blocks and registers the other three;
        # P prefills the prompt in three steps.
        worker = _p_worker()
        worker.nixl_wrapper.notifs = {
            "d": [push_reg_notif(push_registration([100, 101, 102]))]
        }

        for step, job in enumerate(CHUNK_JOBS):
            if step == cancel_at_step:
                worker.nixl_wrapper.notifs = {"d": [_tt_notif(CANCEL, "p")]}
            _run_step(worker, jobs=[job])
            worker._push_writer_pass()
            worker.nixl_wrapper.finish()
        _finish(worker, "p", PROMPT_BLOCKS)
        worker._push_writer_pass()
        worker.get_transfer_results()
        worker._push_writer_pass()

        # A cancelled chunk leaves the notification out, so D never decodes on
        # partial KV, and the whole-request push that follows is dropped.
        assert worker.nixl_wrapper.writes == written
        assert recorder.pushes == []
        acks = [REG_ACK] if cancel_at_step is None else [REG_ACK, CANCEL_ACK]
        assert _kinds(worker) == acks

    def test_p_acknowledges_a_cancel_once_its_chunk_writes_have_ended(self, recorder):
        worker = _p_worker()
        worker.nixl_wrapper.notifs = {"d": [push_reg_notif(push_registration([100]))]}
        _run_step(worker, jobs=[_job(0, 5, PROMPT_BLOCKS)])
        worker._push_writer_pass()
        assert len(worker.nixl_wrapper.writes) == 1

        worker.nixl_wrapper.notifs = {"d": [_tt_notif(CANCEL, "p")]}
        worker._push_writer_pass()
        assert _kinds(worker) == [REG_ACK]

        worker.nixl_wrapper.finish()
        worker._push_writer_pass()
        assert _kinds(worker) == [REG_ACK, CANCEL_ACK]
