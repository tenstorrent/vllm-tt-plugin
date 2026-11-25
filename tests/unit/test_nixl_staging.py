# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request host staging for the NIXL push connector.

The worker tests run the TT overrides on a worker built without NIXL: vLLM's
push worker methods the overrides call are replaced by recorders, so each
test sees which IDs reach vLLM and which copies the overrides make. The
scheduler tests do the same with vLLM's push scheduler.
"""

import re
from types import SimpleNamespace

import pytest
import torch
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
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

from tests.support.nixl_staging import Recorder, record_push_worker, staging_worker
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


def _metadata(
    registrations=None, finished=None, lanes=None, jobs=(), preempted=(), aborted=()
):
    """The connector metadata fields the TT worker reads."""
    return SimpleNamespace(
        push_registrations=registrations or {},
        push_finished_blocks=finished or {},
        tt_lane_of=lanes or {},
        push_chunk_jobs=list(jobs),
        push_chunk_preemptions=set(preempted),
        reqs_not_processed=set(aborted),
    )


def _registration(block_ids):
    return {"request_id": "d-rid", "local_block_ids": (block_ids,)}


class TestDecodeSide:
    def test_registration_carries_slots_and_the_blocks_wait_for_them(self, recorder):
        worker = staging_worker(num_slots=4)
        original = _metadata({"r": _registration([17, 4, 92])})

        worker.start_load_kv(original)

        (sent,) = recorder.metadata
        assert sent.push_registrations["r"]["local_block_ids"] == ([0, 1, 2],)
        # The scheduler's metadata keeps the block IDs.
        assert original.push_registrations["r"]["local_block_ids"] == ([17, 4, 92],)

    def test_a_received_request_lands_in_its_blocks_and_frees_its_slots(self, recorder):
        worker = staging_worker(num_slots=4)
        worker.start_load_kv(_metadata({"r": _registration([17, 4, 92])}))
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == [("h2d", [0, 1, 2], [17, 4, 92], 0)]
        assert worker.slots.free == 4

    def test_a_registration_waits_while_the_pool_is_full(self, recorder):
        worker = staging_worker(num_slots=3)
        worker.start_load_kv(_metadata({"a": _registration([1, 2])}))
        worker.start_load_kv(_metadata({"b": _registration([3, 4])}))
        assert recorder.metadata[-1].push_registrations == {}

        recorder.results.finished_recving = {"a"}
        worker.get_transfer_results()
        worker.start_load_kv(_metadata())

        assert list(recorder.metadata[-1].push_registrations) == ["b"]

    def test_a_failed_receive_frees_its_slots_without_a_copy(self, recorder):
        worker = staging_worker(num_slots=4)
        worker.start_load_kv(_metadata({"r": _registration([5, 6])}))
        recorder.results.failed_recving = {"r"}
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == []
        assert worker.slots.free == 4

    def test_a_receive_that_never_comes_fails_and_its_slots_rest(self, recorder):
        worker = staging_worker(num_slots=2, recv_timeout=0.0, quarantine=3600.0)
        worker.start_load_kv(_metadata({"r": _registration([5, 6])}))

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
        worker.start_load_kv(_metadata({"r": _registration([5, 6])}, lanes={"r": 1}))
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == [("h2d", [0, 1], [5, 6], 1)]

    def test_a_failed_receive_reports_its_blocks_numbered_by_lane(self, recorder):
        worker = staging_worker(num_slots=4, pool_blocks=100)
        worker.start_load_kv(_metadata({"r": _registration([5, 6])}, lanes={"r": 2}))
        recorder.results.failed_recving = {"r"}
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.get_block_ids_with_load_errors() == {205, 206}
        assert worker.get_block_ids_with_load_errors() == set()

    def test_p_stages_a_matched_request_from_its_lane(self, recorder):
        worker = staging_worker(num_slots=4)
        worker.start_load_kv(_metadata(finished={"p": ([30, 31],)}, lanes={"p": 3}))
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

        worker.save_kv_to_host(_metadata(jobs=[_job(0, 1, [30])]))
        worker.save_kv_to_host(_metadata(jobs=[_job(1, 3, [31, 32])]))

        assert worker.copies == [("d2h", [30], [0], 0), ("d2h", [31, 32], [1, 2], 0)]
        # The first job took a slot for each of the prompt's five blocks.
        assert worker.slots.free == 3

    def test_the_first_job_takes_slots_for_the_whole_prompt_or_none(self, recorder):
        worker = staging_worker(num_slots=6)
        worker.save_kv_to_host(
            _metadata(jobs=[_job(0, 1, [30], req_id="a"), _job(0, 1, [40], 3, "b")])
        )
        worker.save_kv_to_host(_metadata(jobs=[_job(1, 3, [41, 42], 3, "b")]))

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
            worker.save_kv_to_host(_metadata(jobs=[job]))
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
        worker = staging_worker(num_slots=8)
        worker.save_kv_to_host(_metadata(jobs=[_job(0, 1, [30])]))

        # The recompute's first job starts at block 0 again; a job there always
        # begins the request anew, whether or not the preemption was listed.
        recompute = _metadata(
            preempted={"p"} if listed else (), jobs=[_job(0, 2, [40, 41])]
        )
        worker.start_load_kv(recompute)
        worker.save_kv_to_host(recompute)

        # Slots 0 to 4 went back behind the free ones; the recompute took 5 on.
        assert worker.copies[-1] == ("d2h", [40, 41], [5, 6], 0)
        assert worker.slots.free == 3

    @pytest.mark.parametrize("ending", ["aborted", "lease_expired"])
    def test_a_request_that_ends_unmatched_gives_its_slots_back(self, recorder, ending):
        worker = staging_worker(num_slots=8)
        worker.save_kv_to_host(_metadata(jobs=[_job(0, 1, [30])]))
        assert worker.slots.free == 3

        if ending == "aborted":
            worker.start_load_kv(_metadata(aborted={"p"}))
        else:
            # vLLM reports a finished request whose lease expired as sent.
            recorder.results.finished_sending = {"p"}
            worker.get_transfer_results()

        assert worker.slots.free == 8

    def test_a_job_that_does_not_continue_the_staged_blocks_is_refused(self):
        worker = staging_worker(num_slots=8)
        worker.save_kv_to_host(_metadata(jobs=[_job(0, 1, [30])]))

        with pytest.raises(RuntimeError, match="block 3"):
            worker.save_kv_to_host(_metadata(jobs=[_job(3, 5, [33, 34])]))
