# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request host staging for the NIXL push connector.

The worker tests run the TT overrides on a worker built without NIXL: vLLM's
push worker methods the overrides call are replaced by recorders, so each
test sees which IDs reach vLLM and which copies the overrides make.
"""

import queue
import threading
from collections import defaultdict, deque
from types import SimpleNamespace

import msgspec
import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

from vllm_tt_plugin import nixl_staging
from vllm_tt_plugin.nixl_staging import StagingSlots, TTNixlPushConnectorWorker

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
    [(10_000, 2 * 2 * 4), (10, 10)],
    ids=["two_longest_requests", "capped_at_the_pool"],
)
def test_slot_count(num_blocks, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=250),
        cache_config=SimpleNamespace(block_size=64),
    )
    # 250 tokens are 4 blocks per group; two groups, two requests.
    assert nixl_staging.staging_slot_count(config, num_blocks, 2) == expected


def test_p_keeps_its_last_blocks_per_group():
    kept = nixl_staging._last_blocks(([9, 8, 7, 6], [3]), ([40, 41], []))
    assert kept == ([7, 6], [])


def test_p_refuses_a_registration_longer_than_its_request():
    with pytest.raises(ValueError, match="registered 3 blocks"):
        nixl_staging._last_blocks(([1, 2],), ([5, 6, 7],))


class Recorder:
    """What the overrides handed to vLLM's push worker."""

    def __init__(self):
        self.metadata = []
        self.pushes = []
        self.notifs = []
        self.results = KVConnectorTransferResults()


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()

    def start_load_kv(self, metadata):
        rec.metadata.append(metadata)

    def get_transfer_results(self):
        # vLLM's own results also run D's copy into the chips.
        for req_id in rec.results.finished_recving - rec.results.failed_recving:
            self.sync_recved_kv_to_device(req_id, None)
        results, rec.results = rec.results, KVConnectorTransferResults()
        return results

    def do_start_push_kv(self, request_id, local_block_ids, registration_data):
        rec.pushes.append((request_id, local_block_ids))

    def do_send_reg_notif(self, req_id, reg_data):
        rec.notifs.append(PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg_data))

    monkeypatch.setattr(NixlPushConnectorWorker, "start_load_kv", start_load_kv)
    monkeypatch.setattr(
        NixlPushConnectorWorker, "get_transfer_results", get_transfer_results
    )
    monkeypatch.setattr(NixlPushConnectorWorker, "_do_start_push_kv", do_start_push_kv)
    monkeypatch.setattr(
        NixlPushConnectorWorker, "_do_send_reg_notif", do_send_reg_notif
    )
    return rec


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


def _worker(num_slots: int, **staging) -> TTNixlPushConnectorWorker:
    """A TT push worker with staging state and vLLM's fields the overrides use."""
    worker = TTNixlPushConnectorWorker.__new__(TTNixlPushConnectorWorker)
    worker._init_staging(num_slots, **staging)
    worker._failed_recv_reqs = queue.Queue()
    worker._invalid_block_ids = queue.Queue()
    worker.copies = []
    worker.copy_blocks = lambda src, dst, src_ids, dst_ids, direction, lane=0: (
        worker.copies.append((direction, list(src_ids), list(dst_ids), lane))
    )
    worker.host_xfer_buffers, worker.device_kv_caches = "host", "device"
    worker._deferred_push_inbox = queue.Queue()
    worker._push_writer_wake = threading.Event()
    worker._sending_transfers = {}
    worker._sending_transfers_lock = threading.Lock()
    worker._push_finished_blocks = {}
    worker._pending_d_registrations = {}
    return worker


def _metadata(registrations=None, finished=None, lanes=None):
    return SimpleNamespace(
        push_registrations=registrations or {},
        push_finished_blocks=finished or {},
        tt_lane_of=lanes or {},
    )


def _held_entries(worker) -> dict[str, int]:
    """Entries per container and queue of ``worker`` that holds any."""
    held = {}
    for name, value in vars(worker).items():
        if isinstance(value, queue.Queue):
            size = value.qsize()
        elif isinstance(value, (dict, set, list, deque)) and name != "copies":
            size = len(value)
        else:
            continue
        if size:
            held[name] = size
    return held


def _registration(block_ids, request_id="d-rid"):
    return {"request_id": request_id, "local_block_ids": (block_ids,)}


class TestDecodeSide:
    def test_registration_carries_slots_and_the_blocks_wait_for_them(self, recorder):
        worker = _worker(num_slots=4)
        original = _metadata({"r": _registration([17, 4, 92])})

        worker.start_load_kv(original)

        (sent,) = recorder.metadata
        assert sent.push_registrations["r"]["local_block_ids"] == ([0, 1, 2],)
        # The scheduler's metadata keeps the block IDs.
        assert original.push_registrations["r"]["local_block_ids"] == ([17, 4, 92],)

    def test_a_received_request_lands_in_its_blocks_and_frees_its_slots(self, recorder):
        worker = _worker(num_slots=4)
        worker.start_load_kv(_metadata({"r": _registration([17, 4, 92])}))
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == [("h2d", [0, 1, 2], [17, 4, 92], 0)]
        assert worker.slots.free == 4

    def test_a_registration_waits_while_the_pool_is_full(self, recorder):
        worker = _worker(num_slots=3)
        worker.start_load_kv(_metadata({"a": _registration([1, 2])}))
        worker.start_load_kv(_metadata({"b": _registration([3, 4])}))
        assert recorder.metadata[-1].push_registrations == {}

        recorder.results.finished_recving = {"a"}
        worker.get_transfer_results()
        worker.start_load_kv(_metadata())

        assert list(recorder.metadata[-1].push_registrations) == ["b"]

    def test_a_failed_receive_frees_its_slots_without_a_copy(self, recorder):
        worker = _worker(num_slots=4)
        worker.start_load_kv(_metadata({"r": _registration([5, 6])}))
        recorder.results.failed_recving = {"r"}
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == []
        assert worker.slots.free == 4

    def test_a_receive_that_never_comes_fails_and_its_slots_rest(self, recorder):
        worker = _worker(num_slots=2, recv_timeout=0.0, quarantine=3600.0)
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
        worker = _worker(num_slots=4)
        worker.start_load_kv(_metadata({"r": _registration([5, 6])}, lanes={"r": 1}))
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.copies == [("h2d", [0, 1], [5, 6], 1)]

    def test_a_failed_receive_reports_its_blocks_numbered_by_lane(self, recorder):
        worker = _worker(num_slots=4, pool_blocks=100)
        worker.start_load_kv(_metadata({"r": _registration([5, 6])}, lanes={"r": 2}))
        recorder.results.failed_recving = {"r"}
        recorder.results.finished_recving = {"r"}

        worker.get_transfer_results()

        assert worker.get_block_ids_with_load_errors() == {205, 206}
        assert worker.get_block_ids_with_load_errors() == set()

    def test_p_stages_a_matched_request_from_its_lane(self, recorder):
        worker = _worker(num_slots=4)
        worker.start_load_kv(_metadata(finished={"p": ([30, 31],)}, lanes={"p": 3}))
        worker._do_start_push_kv("p", ([30, 31],), _registration([8, 9]))

        worker.get_transfer_results()

        assert worker.copies == [("d2h", [30, 31], [0, 1], 3)]


class TestPrefillSide:
    def test_a_match_is_staged_on_the_runner_thread_then_written(self, recorder):
        worker = _worker(num_slots=4)
        worker.start_load_kv(_metadata(finished={"p": ([30, 31, 32],)}))
        # D already holds the first block of this prompt, so it registers two.
        worker._do_start_push_kv("p", ([30, 31, 32],), _registration([8, 9]))
        assert recorder.pushes == []

        worker.get_transfer_results()

        assert worker.copies == [("d2h", [31, 32], [0, 1], 0)]
        request_id, slots, registration = worker._deferred_push_inbox.get_nowait()
        worker._do_start_push_kv(request_id, slots, registration)
        assert recorder.pushes == [("p", ([0, 1],))]

    def test_slots_are_freed_after_the_send_and_its_last_write(self, recorder):
        worker = _worker(num_slots=2)
        worker.start_load_kv(_metadata(finished={"p": ([30, 31],)}))
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
        worker = _worker(num_slots=2)
        worker.start_load_kv(_metadata(finished={"p": ([30, 31],)}))
        recorder.results.finished_sending = {"p"}
        worker.get_transfer_results()

        worker._do_start_push_kv("p", ([30, 31],), _registration([8, 9]))
        worker.get_transfer_results()

        assert worker.copies == []
        assert worker.slots.free == 2


class TestSendSlotLifetime:
    """P's slots stay with a request while its write is being submitted or runs."""

    def test_a_lease_that_expires_during_submission_keeps_the_slots(
        self, recorder, monkeypatch
    ):
        p = _worker(num_slots=2)
        registration = {
            **_registration([8, 9]),
            "decode_engine_id": "d-engine",
            "decode_host": "d-host",
            "decode_port": 1,
            "decode_tp_size": 1,
        }
        p.start_load_kv(_metadata(finished={"p": ([30, 31],)}))
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
        p = _worker(num_slots=2)
        for i in range(10_000):
            req_id = f"p{i}"
            p.start_load_kv(_metadata(finished={req_id: ([30, 31],)}))
            p._do_start_push_kv(req_id, ([30, 31],), _registration([8, 9]))
            p.get_transfer_results()
            p._do_start_push_kv(*p._deferred_push_inbox.get_nowait())
            recorder.results.finished_sending = {req_id}
            p.get_transfer_results()

        assert len(recorder.pushes) == 10_000
        assert _held_entries(p) == {}
        assert p.slots.free == 2


class TestReceiveDeadline:
    """P writes nothing once D may have given the receive's slots away.

    D fails a receive ``recv_timeout`` seconds after registering it, and its
    slots go to another request once they have rested. D's registration tells
    P how many seconds are left; P counts them from receipt.
    """

    @staticmethod
    def _register(d, recorder) -> bytes:
        """D registers request ``r`` and sends the registration to P."""
        d.start_load_kv(_metadata({"r": _registration([5, 6], request_id="r")}))
        d._do_send_reg_notif("r", recorder.metadata[-1].push_registrations["r"])
        return recorder.notifs[-1]

    @staticmethod
    def _write_staged(p) -> None:
        """P's runner thread stages the match; P's writer thread writes it."""
        p.get_transfer_results()
        while not p._deferred_push_inbox.empty():
            p._do_start_push_kv(*p._deferred_push_inbox.get_nowait())

    def test_p_finishing_after_ds_deadline_writes_nothing(self, recorder, clock):
        d, p = _worker(num_slots=4, recv_timeout=60.0), _worker(num_slots=4)
        # D registers while P still computes the prompt.
        p._handle_push_reg_notif(self._register(d, recorder))

        clock.now += 61.0
        d.get_transfer_results()
        assert d._failed_recv_reqs.get_nowait() == "r"
        # P's writer matches the finished request with the waiting registration.
        p.start_load_kv(_metadata(finished={"r": ([30, 31],)}))
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
        d, p = _worker(num_slots=4, recv_timeout=60.0), _worker(num_slots=4)
        p.start_load_kv(_metadata(finished={"r": ([30, 31],)}))
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(self._register(d, recorder))
        p.get_transfer_results()

        # The write waits, for example for P's handshake with D.
        clock.now += write_after
        p._do_start_push_kv(*p._deferred_push_inbox.get_nowait())

        assert recorder.pushes == ([("r", ([0, 1],))] if written else [])

    def test_p_stops_a_margin_before_ds_deadline(self, recorder, clock):
        d, p = _worker(num_slots=4, recv_timeout=60.0), _worker(num_slots=4)
        p.start_load_kv(_metadata(finished={"r": ([30, 31],)}))
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
        d, p = _worker(num_slots=4, recv_timeout=8.0), _worker(num_slots=4)
        p.start_load_kv(_metadata(finished={"r": ([30, 31],)}))
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(self._register(d, recorder))

        clock.now += write_after
        self._write_staged(p)

        assert recorder.pushes == ([("r", ([0, 1],))] if written else [])

    def test_d_sends_the_seconds_left_when_it_sends(self, recorder, clock):
        d = _worker(num_slots=4, recv_timeout=60.0)
        d.start_load_kv(_metadata({"r": _registration([5, 6], request_id="r")}))

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
        notif = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(
            _registration([8, 9], request_id="r")
        )
        p = _worker(num_slots=4)
        p.start_load_kv(_metadata(finished={"r": ([30, 31],)}))
        p._push_finished_blocks["r"] = ([30, 31],)
        p._handle_push_reg_notif(notif)

        clock.now += 10_000.0
        self._write_staged(p)

        assert recorder.pushes == [("r", ([0, 1],))]
