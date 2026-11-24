# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request host staging for the NIXL push connector.

The worker tests run the TT overrides on a worker built without NIXL: vLLM's
push worker methods the overrides call are replaced by recorders, so each
test sees which IDs reach vLLM and which copies the overrides make.
"""

import queue
import threading
from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

from vllm_tt_plugin import nixl_staging
from vllm_tt_plugin.nixl_staging import StagingSlots, TTNixlPushConnectorWorker


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

    monkeypatch.setattr(NixlPushConnectorWorker, "start_load_kv", start_load_kv)
    monkeypatch.setattr(
        NixlPushConnectorWorker, "get_transfer_results", get_transfer_results
    )
    monkeypatch.setattr(NixlPushConnectorWorker, "_do_start_push_kv", do_start_push_kv)
    return rec


def _worker(num_slots: int, **staging) -> TTNixlPushConnectorWorker:
    """A TT push worker with staging state and vLLM's fields the overrides use."""
    worker = TTNixlPushConnectorWorker.__new__(TTNixlPushConnectorWorker)
    worker._init_staging(num_slots, **staging)
    worker._failed_recv_reqs = queue.Queue()
    worker.copies = []
    worker.copy_blocks = lambda src, dst, src_ids, dst_ids, direction: (
        worker.copies.append((direction, list(src_ids), list(dst_ids)))
    )
    worker.host_xfer_buffers, worker.device_kv_caches = "host", "device"
    worker._deferred_push_inbox = queue.Queue()
    worker._push_writer_wake = threading.Event()
    worker._sending_transfers = {}
    worker._sending_transfers_lock = threading.Lock()
    return worker


def _metadata(registrations=None):
    return SimpleNamespace(push_registrations=registrations or {})


def _registration(block_ids):
    return {"request_id": "d-rid", "local_block_ids": (block_ids,)}


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

        assert worker.copies == [("h2d", [0, 1, 2], [17, 4, 92])]
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


class TestPrefillSide:
    def test_a_match_is_staged_on_the_runner_thread_then_written(self, recorder):
        worker = _worker(num_slots=4)
        # D already holds the first block of this prompt, so it registers two.
        worker._do_start_push_kv("p", ([30, 31, 32],), _registration([8, 9]))
        assert recorder.pushes == []

        worker.get_transfer_results()

        assert worker.copies == [("d2h", [31, 32], [0, 1])]
        request_id, slots, registration = worker._deferred_push_inbox.get_nowait()
        worker._do_start_push_kv(request_id, slots, registration)
        assert recorder.pushes == [("p", ([0, 1],))]

    def test_slots_are_freed_after_the_send_and_its_last_write(self, recorder):
        worker = _worker(num_slots=2)
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
        recorder.results.finished_sending = {"p"}
        worker.get_transfer_results()

        worker._do_start_push_kv("p", ([30, 31],), _registration([8, 9]))
        worker.get_transfer_results()

        assert worker.copies == []
        assert worker.slots.free == 2
