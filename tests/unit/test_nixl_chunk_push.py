# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""P's NIXL writer thread writing prompt chunks to D before the request finishes.

Each test runs the writer's passes on its own thread, on a worker built
without NIXL: ``FakeNixl`` records each WRITE and keeps it in flight until the
test finishes it. A job's block IDs are P's IDs of the host rows NIXL reads.
"""

import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

from tests.support.nixl_writer import (
    D_ENGINE,
    Write,
    push_reg_notif,
    push_writer,
    registration,
)
from vllm_tt_plugin.nixl_chunk_push import ChunkedPushWorker, NixlPushChunkJob

# P's block IDs of a five-block prompt.
TABLE = [10, 11, 12, 13, 14]
# D's blocks when it holds none of the prompt.
D_BLOCKS = [100, 101, 102, 103, 104]


def _writer(chunked: bool = True) -> ChunkedPushWorker:
    """P's writer; ``worker.pushes`` records each whole-request push it starts."""
    worker = ChunkedPushWorker.__new__(ChunkedPushWorker)
    worker._init_chunk_push(chunked)
    worker.pushes = []
    worker._do_start_push_kv = lambda req_id, blocks, reg: worker.pushes.append(
        (req_id, blocks, reg)
    )
    return push_writer(worker)


def _job(start: int, end: int, base: int = 10, total: int = 5) -> NixlPushChunkJob:
    """Prompt blocks ``[start, end)`` of request ``p``, held in rows ``base + i``."""
    return NixlPushChunkJob(
        request_id="p",
        block_ids=[base + i for i in range(start, end)],
        start_block=start,
        end_block=end,
        total_blocks=total,
    )


def _passes(worker: ChunkedPushWorker, count: int = 1) -> list[Write]:
    """Run ``count`` writer passes; return every WRITE so far."""
    for _ in range(count):
        worker._push_writer_pass()
    return worker.nixl_wrapper.writes


class TestChunkWrites:
    def test_a_chunk_is_written_to_d_without_a_notification(self):
        worker = _writer()
        worker._pending_d_registrations["p"] = registration(D_BLOCKS)
        worker._queue_chunk_jobs([_job(0, 2)])

        assert _passes(worker) == [Write(1, [10, 11], [100, 101], b"")]
        # A chunk WRITE never reports the request as sent.
        assert dict(worker._sending_transfers) == {}
        assert list(worker._chunk_transfers.values()) == [[1]]

    def test_chunks_map_onto_ds_uncached_suffix(self):
        # D holds the prompt's first two blocks, so it registers three.
        worker = _writer()
        worker._pending_d_registrations["p"] = registration([100, 101, 102])

        for job in (_job(0, 1), _job(1, 3), _job(3, 5)):
            worker._queue_chunk_jobs([job])
            _passes(worker)
            worker.nixl_wrapper.finish()

        # Block 0 is before D's blocks: nothing is written for the first job.
        assert _passes(worker) == [
            Write(1, [12], [100], b""),
            Write(2, [13, 14], [101, 102], b"p:1"),
        ]

    def test_only_the_last_write_notifies_and_only_after_the_earlier_writes_complete(
        self,
    ):
        worker = _writer()
        worker._pending_d_registrations["p"] = registration(D_BLOCKS)
        worker._queue_chunk_jobs([_job(0, 2), _job(2, 5)])

        assert _passes(worker, 2) == [Write(1, [10, 11], [100, 101], b"")]

        worker.nixl_wrapper.finish()
        assert _passes(worker, 2)[1:] == [
            Write(2, [12, 13, 14], [102, 103, 104], b"p:1")
        ]

    def test_staged_chunks_are_written_when_a_late_push_reg_arrives(self):
        worker = _writer()
        worker._queue_chunk_jobs([_job(0, 2), _job(2, 4)])
        assert _passes(worker) == []

        # D registers while P still computes the prompt's last block.
        worker.nixl_wrapper.notifs = {"d": [push_reg_notif(registration(D_BLOCKS))]}

        assert _passes(worker) == [
            Write(1, [10, 11], [100, 101], b""),
            Write(2, [12, 13], [102, 103], b""),
        ]

    def test_a_handed_off_request_is_reported_once(self):
        worker = _writer()
        worker._pending_d_registrations["p"] = registration(D_BLOCKS)
        worker._queue_chunk_jobs([_job(0, 5)])
        _passes(worker)
        worker.nixl_wrapper.finish()

        worker._finished_blocks_inbox.put(("p", (TABLE,)))
        _passes(worker)
        # Reported like a whole-request push whose WRITEs completed.
        assert worker._sending_transfers["p"] == []

        # vLLM's get_transfer_results reports it and drops the entry.
        del worker._sending_transfers["p"]
        _passes(worker)
        assert "p" not in worker._sending_transfers
        assert len(worker.nixl_wrapper.writes) == 1
        assert worker.pushes == []

    @pytest.mark.parametrize(
        "cause", ["failed_write", "failed_handshake", "incompatible_peer"]
    )
    def test_a_failed_chunk_withholds_the_notification_and_pushes_the_request_whole(
        self, cause
    ):
        worker = _writer()
        reg = registration(D_BLOCKS)
        worker._pending_d_registrations["p"] = reg
        if cause == "failed_handshake":
            handshake: Future = Future()
            handshake.set_exception(RuntimeError("D is unreachable"))
            worker._ensure_handshake = lambda *args, **kwargs: handshake
        elif cause == "incompatible_peer":
            worker.transfer_topo.get_engine_info = lambda *args: SimpleNamespace(
                remote_tp_size=2, remote_dcp_size=1
            )

        worker._queue_chunk_jobs([_job(0, 2)])
        _passes(worker)
        worker.nixl_wrapper.finish("ERR")
        worker._queue_chunk_jobs([_job(2, 5)])
        _passes(worker, 2)
        worker.nixl_wrapper.finish()
        worker._finished_blocks_inbox.put(("p", (TABLE,)))
        _passes(worker)

        assert all(write.notif == b"" for write in worker.nixl_wrapper.writes)
        assert worker.pushes == [("p", (TABLE,), reg)]

    def test_a_request_that_finishes_before_d_registers_waits_as_a_whole_push(self):
        worker = _writer()
        worker._queue_chunk_jobs([_job(0, 5)])
        worker._finished_blocks_inbox.put(("p", (TABLE,)))

        _passes(worker, 2)

        assert worker._push_finished_blocks == {"p": (TABLE,)}
        assert worker._chunk_deferred == []
        assert worker.nixl_wrapper.writes == []

    def test_a_request_that_finishes_before_the_writer_takes_its_last_job(self):
        # One prefill step computed the whole prompt; its finish reaches the
        # writer in the same pass as its only job.
        worker = _writer()
        worker._pending_d_registrations["p"] = registration(D_BLOCKS)
        worker._queue_chunk_jobs([_job(0, 5)])
        worker._finished_blocks_inbox.put(("p", (TABLE,)))

        assert _passes(worker) == [Write(1, TABLE, D_BLOCKS, b"p:1")]
        assert worker.pushes == []

    def test_an_expired_lease_drops_the_chunk_state(self):
        worker = _writer()
        worker._queue_chunk_jobs([_job(0, 2)])
        _passes(worker)
        assert worker._chunk_deferred

        # vLLM reports the request sent when its lease expires.
        worker._evict_finished_inbox.put("p")
        _passes(worker)

        assert worker._chunk_deferred == []
        assert "p" not in worker._chunk_states


class TestRetirement:
    """Preempted and aborted requests, retired without blocking the runner."""

    @staticmethod
    def _writing(base: int = 10) -> ChunkedPushWorker:
        """A writer with the WRITE of ``p``'s blocks 0 and 1 in flight."""
        worker = _writer()
        worker._pending_d_registrations["p"] = registration(D_BLOCKS)
        worker._queue_chunk_jobs([_job(0, 2, base)])
        _passes(worker)
        return worker

    @pytest.mark.parametrize("preempted", [True, False], ids=["preempted", "aborted"])
    def test_a_retirement_settles_once_no_write_of_its_requests_is_in_flight(
        self, preempted
    ):
        worker = self._writing()
        worker._queue_chunk_jobs([_job(2, 5)])
        retired = {"p"}
        retirement = worker._queue_chunk_retirement(
            retired if preempted else set(), set() if preempted else retired
        )

        _passes(worker, 2)
        assert not retirement.done.is_set()

        worker.nixl_wrapper.finish()
        _passes(worker)
        assert retirement.done.is_set()
        # The job queued before the retirement is dropped.
        assert len(_passes(worker, 2)) == 1

    def test_a_preempted_request_writes_its_new_version_after_its_old_writes_end(
        self,
    ):
        worker = self._writing()
        worker._queue_chunk_retirement({"p"}, set())
        # The recompute stages its first chunk while the old WRITE is in flight.
        worker._queue_chunk_jobs([_job(0, 3, base=20)])

        assert len(_passes(worker, 2)) == 1

        worker.nixl_wrapper.finish()
        assert _passes(worker, 2)[1:] == [Write(2, [20, 21, 22], [100, 101, 102], b"")]

    @pytest.mark.parametrize(
        "handed_off",
        [True, False],
        ids=["notifying_write_landed", "notifying_write_failed"],
    )
    def test_a_request_handed_off_before_preemption_is_not_pushed_again(
        self, handed_off
    ):
        worker = _writer()
        worker._pending_d_registrations["p"] = registration(D_BLOCKS)
        worker._queue_chunk_jobs([_job(0, 5)])
        _passes(worker)
        worker._queue_chunk_retirement({"p"}, set())
        worker._queue_chunk_jobs([_job(0, 5, base=20)])
        _passes(worker)

        worker.nixl_wrapper.finish("DONE" if handed_off else "ERR")
        writes = _passes(worker, 2)

        if handed_off:
            assert len(writes) == 1
        else:
            assert writes[1:] == [Write(2, [20, 21, 22, 23, 24], D_BLOCKS, b"p:1")]

    def test_a_finished_request_is_pushed_whole_only_after_its_old_writes_end(self):
        # The recompute found no free staging slots, so it has no chunk jobs.
        worker = self._writing()
        reg = worker._chunk_states["p"].reg_data
        worker._queue_chunk_retirement({"p"}, set())
        worker._finished_blocks_inbox.put(("p", (TABLE,)))

        _passes(worker, 2)
        assert worker.pushes == []

        worker.nixl_wrapper.finish()
        _passes(worker)
        assert worker.pushes == [("p", (TABLE,), reg)]


def test_a_whole_request_write_keeps_vllms_notification_and_tracking():
    worker = _writer()
    meta = ReqMeta(
        local_block_ids=([10, 11],),
        local_physical_block_ids=([10, 11],),
        tp_size=1,
        remote=RemoteMeta(
            block_ids=([100, 101],),
            host="",
            port=0,
            engine_id=D_ENGINE,
            request_id="d-p",
        ),
    )

    worker._xfer_blocks_for_req(req_id="p", meta=meta)

    assert worker.nixl_wrapper.writes == [Write(1, [10, 11], [100, 101], b"d-p:1")]
    assert worker._sending_transfers == {"p": [1]}


def test_with_chunked_push_off_a_finished_request_is_pushed_whole():
    worker = _writer(chunked=False)
    reg = registration(D_BLOCKS)
    worker._pending_d_registrations["p"] = reg
    worker._finished_blocks_inbox.put(("p", (TABLE,)))

    _passes(worker)

    assert worker.pushes == [("p", (TABLE,), reg)]


def test_the_writer_thread_polls_while_a_chunk_waits_for_an_earlier_write():
    worker = _writer()
    worker._pending_d_registrations["p"] = registration(D_BLOCKS)
    thread = threading.Thread(target=worker._push_writer_loop, daemon=True)
    thread.start()
    try:
        worker._queue_chunk_jobs([_job(0, 2), _job(2, 5)])
        _wait_for(lambda: len(worker.nixl_wrapper.writes) == 1)

        # Nothing wakes the writer when the WRITE completes; it polls.
        worker.nixl_wrapper.finish()
        _wait_for(lambda: len(worker.nixl_wrapper.writes) == 2)
    finally:
        worker._push_writer_stop.set()
        worker._push_writer_wake.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_shutdown_releases_chunk_writes_still_in_flight(monkeypatch):
    released = []
    monkeypatch.setattr(NixlPushConnectorWorker, "shutdown", lambda self: None)
    worker = _writer()
    worker._push_writer_thread = None
    worker._pending_d_registrations["p"] = registration(D_BLOCKS)
    worker._queue_chunk_jobs([_job(0, 2)])
    _passes(worker)
    worker.nixl_wrapper.release_xfer_handle = released.append

    worker.shutdown()

    assert released == [1]
    assert worker._chunk_transfers == {}


def _wait_for(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.001)
