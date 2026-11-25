# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request host staging for the NIXL push connector.

The worker tests run the TT overrides on a worker built without NIXL: vLLM's
push worker methods the overrides call are replaced by recorders, so each
test sees which IDs reach vLLM and which copies the overrides make. The
scheduler tests do the same with vLLM's push scheduler.
"""

import re
import threading
from types import SimpleNamespace

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
from tests.support.nixl_writer import Write, push_reg_notif, push_writer
from tests.support.nixl_writer import registration as push_registration
from vllm_tt_plugin import nixl_staging
from vllm_tt_plugin.nixl_staging import (
    NixlPushChunkJob,
    StagingSlots,
    TTNixlPushConnectorScheduler,
)


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


def _registration(block_ids, request_id="d-rid"):
    return {"request_id": request_id, "local_block_ids": (block_ids,)}


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

    def test_a_failed_receive_frees_its_slots_without_a_copy(self, recorder):
        worker = staging_worker(num_slots=4)
        worker.start_load_kv(push_metadata({"r": _registration([5, 6])}))
        recorder.results.failed_recving = {"r"}
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == []
        assert worker.slots.free == 4

    def test_a_receive_that_never_comes_fails_and_its_slots_rest(self, recorder):
        worker = staging_worker(num_slots=2, recv_timeout=0.0, quarantine=3600.0)
        worker.start_load_kv(push_metadata({"r": _registration([5, 6])}))

        worker.get_transfer_results()
        assert worker._failed_recv_reqs.get_nowait() == "r"

        # vLLM reports it failed; P's late write could still land in its slots.
        recorder.results.failed_recving = {"r"}
        recorder.results.finished_recving = {"r"}
        worker.get_transfer_results()
        assert worker.copies == []
        assert worker.slots.free == 0


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

        # vLLM's start_load_kv hands the finished request to the writer.
        worker._finished_blocks_inbox.put(("p", (PROMPT_BLOCKS,)))
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
        worker._finished_blocks_inbox.put(("p", (PROMPT_BLOCKS,)))
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
            worker._finished_blocks_inbox.put((req_id, (blocks,)))
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


class TestReceiveDeadline:
    """P writes nothing once D may have given the receive's slots away.

    D fails a receive ``recv_timeout`` seconds after registering it, and its
    slots go to another request once they have rested. D's registration tells
    P how many seconds are left; P counts them from receipt.
    """

    @staticmethod
    def _register(d, recorder) -> bytes:
        """D registers request ``r`` and sends the registration to P."""
        d.start_load_kv(push_metadata({"r": _registration([5, 6], request_id="r")}))
        d._do_send_reg_notif("r", recorder.metadata[-1].push_registrations["r"])
        return recorder.notifs[-1]

    @staticmethod
    def _write_staged(p) -> None:
        """P's runner thread stages the match; P's writer thread writes it."""
        p.get_transfer_results()
        while not p._deferred_push_inbox.empty():
            p._do_start_push_kv(*p._deferred_push_inbox.get_nowait())

    def test_p_finishing_after_ds_deadline_writes_nothing(self, recorder, clock):
        d = staging_worker(num_slots=4, recv_timeout=60.0)
        p = staging_worker(num_slots=4)
        # D registers while P still computes the prompt.
        p._handle_push_reg_notif(self._register(d, recorder))

        clock.now += 61.0
        d.get_transfer_results()
        assert d._failed_recv_reqs.get_nowait() == "r"
        # P's writer matches the finished request with the waiting registration.
        p._do_start_push_kv("r", ([30, 31],), p._pop_matching_registration("r"))
        self._write_staged(p)

        assert recorder.pushes == []
        assert p.copies == []

    @pytest.mark.parametrize(
        ("write_after", "written"),
        [(30.0, True), (61.0, False)],
        ids=["in_time", "after_the_deadline"],
    )
    def test_a_staged_write_is_submitted_only_before_ds_deadline(
        self, recorder, clock, write_after, written
    ):
        d = staging_worker(num_slots=4, recv_timeout=60.0)
        p = staging_worker(num_slots=4)
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(self._register(d, recorder))
        p.get_transfer_results()

        # The write waits, for example for P's handshake with D.
        clock.now += write_after
        p._do_start_push_kv(*p._deferred_push_inbox.get_nowait())

        assert recorder.pushes == ([("r", ([0, 1],))] if written else [])

    def test_p_stops_a_margin_before_ds_deadline(self, recorder, clock):
        d = staging_worker(num_slots=4, recv_timeout=60.0)
        p = staging_worker(num_slots=4)
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(self._register(d, recorder))

        clock.now += 60.0 - nixl_staging._DEADLINE_MARGIN_S + 0.5
        self._write_staged(p)

        assert recorder.pushes == []

    @pytest.mark.parametrize(
        ("write_after", "written"),
        [(5.0, True), (7.0, False)],
        ids=["before_the_margin", "inside_the_margin"],
    )
    def test_a_short_receive_timeout_shrinks_the_margin(
        self, recorder, clock, write_after, written
    ):
        # 8 s left: a quarter of it, 2 s, is the margin.
        d = staging_worker(num_slots=4, recv_timeout=8.0)
        p = staging_worker(num_slots=4)
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(self._register(d, recorder))

        clock.now += write_after
        self._write_staged(p)

        assert recorder.pushes == ([("r", ([0, 1],))] if written else [])

    def test_d_sends_the_seconds_left_when_it_sends(self, recorder, clock):
        d = staging_worker(num_slots=4, recv_timeout=60.0)
        d.start_load_kv(push_metadata({"r": _registration([5, 6], request_id="r")}))

        # D's first registration to a P waits for their handshake.
        clock.now += 5.0
        d._do_send_reg_notif("r", recorder.metadata[-1].push_registrations["r"])

        (notif,) = recorder.notifs
        sent = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        assert sent[nixl_staging.RECV_SECONDS_LEFT] == 55.0
        # D's own clock reading stays on D.
        assert set(sent) == {
            "request_id",
            "local_block_ids",
            nixl_staging.RECV_SECONDS_LEFT,
        }

    def test_a_registration_without_seconds_left_sets_no_deadline(
        self, recorder, clock
    ):
        # vLLM's own push worker on D sends no deadline.
        notif = push_reg_notif(_registration([8, 9], request_id="r"))
        p = staging_worker(num_slots=4)
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(notif)

        clock.now += 10_000.0
        self._write_staged(p)

        assert recorder.pushes == [("r", ([0, 1],))]

    @pytest.mark.parametrize(
        ("late_from_step", "written"),
        [
            (None, [Write(1, [2], [100], b""), Write(2, [3, 4], [101, 102], b"p:1")]),
            (2, [Write(1, [2], [100], b"")]),
            (1, []),
        ],
        ids=["in_time", "last_chunk_late", "every_written_chunk_late"],
    )
    def test_p_writes_no_chunk_after_ds_deadline(
        self, recorder, clock, late_from_step, written
    ):
        # D holds the prompt's first two blocks and registers the other three,
        # with 60 s left; P prefills the prompt in three steps.
        reg = {**push_registration([100, 101, 102]), nixl_staging.RECV_SECONDS_LEFT: 60}
        worker = _p_worker()
        worker.nixl_wrapper.notifs = {"d": [push_reg_notif(reg)]}

        for step, job in enumerate(CHUNK_JOBS):
            if step == late_from_step:
                clock.now += 60.0 - nixl_staging._DEADLINE_MARGIN_S + 0.5
            _run_step(worker, jobs=[job])
            worker._push_writer_pass()
            worker.nixl_wrapper.finish()
        worker._finished_blocks_inbox.put(("p", (PROMPT_BLOCKS,)))
        worker._push_writer_pass()
        worker.get_transfer_results()
        worker._push_writer_pass()

        # A late chunk leaves the notification out, so D never decodes on
        # partial KV, and the whole-request push that follows is dropped.
        assert worker.nixl_wrapper.writes == written
        assert recorder.pushes == []
