# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request host staging for the NIXL push connector.

The worker tests run the TT overrides on a worker built without NIXL: vLLM's
push worker methods the overrides call are replaced by recorders, so each
test sees which IDs reach vLLM and which copies the overrides make. The
cancellation tests link a D worker and a P worker through ``Wire`` agents and
run their writer threads' steps by hand, so each test picks the order in which
notifications arrive.
"""

import queue
import threading
from collections import defaultdict, deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

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
from vllm_tt_plugin.nixl_staging import (
    CANCEL,
    CANCEL_ACK,
    REG_ACK,
    TT_NOTIF_PREFIX,
    StagingSlots,
    TTNixlPushConnectorWorker,
)

D_ENGINE, P_ENGINE = "d-engine", "p-engine"
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
        # vLLM queues D's registrations for the writer thread.
        for req_id, reg in metadata.push_registrations.items():
            self._reg_send_inbox.put((req_id, reg))

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
    worker._reg_send_inbox = queue.Queue()
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


class nixlRemoteDisconnectError(Exception):  # noqa: N801
    """Stands in for the exception NIXL's bindings raise when the UCX endpoint to
    the remote agent has failed; the worker recognizes it by its name."""


class Wire:
    """A NIXL agent's ``send_notif``: notifications wait in ``sent`` until delivered.

    Attributes
    ----------
    name
        The agent's name, as its peer knows it.
    disconnected
        Every send raises ``nixlRemoteDisconnectError`` and delivers nothing, as
        NIXL does once the endpoint to a dead peer has failed.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.sent: list[bytes] = []
        self.disconnected = False

    def send_notif(self, agent_name, notif_msg):
        if self.disconnected:
            raise nixlRemoteDisconnectError(agent_name)
        self.sent.append(notif_msg)


def _linked(worker, engine_id: str, peer_engine: str) -> TTNixlPushConnectorWorker:
    """Give ``worker`` a NIXL agent with a done handshake to ``peer_engine``."""
    worker.engine_id = engine_id
    worker.nixl_wrapper = Wire(f"{engine_id}-agent")
    worker._remote_agents = {peer_engine: {(0, 0): f"{peer_engine}-agent"}}
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


class TestCancellation:
    """D gives a failed receive's slots to another request only once P has
    confirmed that it writes no more into them."""

    @pytest.fixture
    def d(self):
        return _linked(_worker(num_slots=2, recv_timeout=60.0), D_ENGINE, P_ENGINE)

    @pytest.fixture
    def p(self):
        return _linked(_worker(num_slots=2), P_ENGINE, D_ENGINE)

    def test_a_registration_p_reads_late_never_lands_in_reused_slots(
        self, recorder, clock, d, p
    ):
        d.start_load_kv(_metadata({"old": _registration([5, 6], "old")}))
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

        d.start_load_kv(_metadata({"new": _registration([7, 8], "new")}))
        assert recorder.metadata[-1].push_registrations == {}

        # P finishes "old", then reads the registration and later the cancel.
        p.start_load_kv(_metadata(finished={"old": ([30, 31],)}))
        p._push_finished_blocks["old"] = ([30, 31],)
        _deliver(d, p)
        _deliver(p, d)
        _deliver(d, p)
        p.get_transfer_results()
        _writer_pass(p)
        _deliver(p, d)
        d.get_transfer_results()
        d.start_load_kv(_metadata())

        assert recorder.pushes == []
        assert p.copies == []
        new = recorder.metadata[-1].push_registrations["new"]
        assert new["local_block_ids"] == ([0, 1],)

    def test_d_cancels_only_a_registration_p_has_acknowledged(self, recorder, d, p):
        d.start_load_kv(_metadata({"r": _registration([5, 6], "r")}))
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
        p.start_load_kv(_metadata(finished={"r": ([30, 31],)}))
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
        p.start_load_kv(_metadata(finished={"r": ([30, 31],)}))
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
        d.start_load_kv(_metadata({"r": _registration([5, 6], "r")}))
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
        d.start_load_kv(_metadata({"r": _registration([5, 6], "r")}))
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
        d.start_load_kv(_metadata({"r": _registration([5, 6], "r")}))
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
        d.start_load_kv(_metadata({"r": _registration([5, 6], "r")}))
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
        d.start_load_kv(_metadata({"r": _registration([5, 6], "r")}))
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
