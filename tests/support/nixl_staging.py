# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A ``TTNixlPushConnectorWorker`` without NIXL, for unit and runner tests.

``record_push_worker`` replaces the vLLM push worker methods that the TT
overrides call with recorders, so a test sees which IDs reach vLLM.
``staging_worker`` builds a TT worker with only the state the overrides use;
it records each copy between chips and slots instead of making it.
``push_metadata`` builds the connector metadata fields the TT worker reads.
"""

import queue
import threading
from types import SimpleNamespace

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

from vllm_tt_plugin.nixl_staging import TTNixlPushConnectorWorker


class Recorder:
    """What the overrides handed to vLLM's push worker.

    Attributes
    ----------
    metadata
        Each metadata passed to vLLM's ``start_load_kv``, in call order.
    pushes
        ``(request ID, block IDs)`` of each write vLLM was asked to start.
    notifs
        Each registration vLLM was asked to send to P, as P receives it.
    results
        What vLLM's ``get_transfer_results`` reports next; reset after each call.
    """

    def __init__(self):
        self.metadata = []
        self.pushes = []
        self.notifs = []
        self.results = KVConnectorTransferResults()


def record_push_worker(monkeypatch) -> Recorder:
    """Replace the vLLM push worker methods the TT overrides call; return the record."""
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


def staging_worker(
    num_slots: int, chunked: bool = False, **staging
) -> TTNixlPushConnectorWorker:
    """A TT push worker with staging state and vLLM's fields the overrides use.

    ``chunked`` is ``enable_chunked_push`` on P. ``worker.copies`` lists each
    copy as ``(direction, source IDs, destination IDs, lane)``.
    """
    worker = TTNixlPushConnectorWorker.__new__(TTNixlPushConnectorWorker)
    worker._init_chunk_push(chunked)
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


def push_metadata(
    registrations=None, finished=None, lanes=None, jobs=(), preempted=(), aborted=()
) -> SimpleNamespace:
    """The connector metadata fields the TT worker reads."""
    return SimpleNamespace(
        push_registrations=registrations or {},
        push_finished_blocks=finished or {},
        tt_lane_of=lanes or {},
        push_chunk_jobs=list(jobs),
        push_chunk_preemptions=set(preempted),
        reqs_not_processed=set(aborted),
    )
