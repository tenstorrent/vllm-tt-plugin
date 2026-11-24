# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request host staging for vLLM's NIXL push connector on TT.

NIXL cannot register TT device memory, so KV crosses the network through a
host buffer. vLLM's own host buffer (``"kv_buffer_device": "cpu"``) mirrors the
whole KV pool, one host row per vLLM block. ``TTNixlPushConnector`` replaces
the mirror with a small pool of host rows, called staging slots here, sized for
the requests in flight:

* D (decode) takes slots for the blocks it registers with P, hands P the slot
  numbers instead of block IDs, and copies the slots into its chips once P's
  write has landed.
* P (prefill) waits until its finished request is matched with D's
  registration, then takes slots for only the blocks D asked for, copies them
  from its chips, and lets vLLM write the slots to D.

vLLM's scheduler keeps using real block IDs; only the IDs that reach NIXL are
slot numbers. Copies between chips and slots run on the model runner thread,
the only thread that may touch the device.

Principal entry points: ``TTNixlPushConnector`` (selected with
``"kv_connector": "TTNixlPushConnector"``) and ``register_kv_connectors``.
"""

import copy
import dataclasses
import queue
import threading
import time
from collections import deque
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.utils.math_utils import cdiv

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

CONNECTOR_NAME = "TTNixlPushConnector"

# A request in flight on each side while the next one is being staged.
_REQUESTS_PER_POOL = 2


def register_kv_connectors() -> None:
    """Make ``TTNixlPushConnector`` selectable by name in ``--kv-transfer-config``."""
    if CONNECTOR_NAME not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            CONNECTOR_NAME, __name__, TTNixlPushConnector.__name__
        )


def staging_slot_count(vllm_config, num_blocks: int, num_groups: int) -> int:
    """Number of staging slots: room for two longest requests, at most the pool.

    One request crosses in one NIXL write, which cannot be split, so the pool
    must hold the longest request whole.

    Parameters
    ----------
    vllm_config
        The engine's ``VllmConfig``; ``max_model_len`` and ``block_size`` set
        the longest request in blocks.
    num_blocks
        Blocks in the KV pool; more slots than blocks are never useful.
    num_groups
        KV cache groups; a request takes blocks in each.
    """
    per_request = num_groups * cdiv(
        vllm_config.model_config.max_model_len, vllm_config.cache_config.block_size
    )
    return min(num_blocks, _REQUESTS_PER_POOL * per_request)


class StagingSlots:
    """A pool of host staging rows, taken and returned whole per request.

    Owned by the model runner thread: only that thread takes and returns slots.

    Examples
    --------
    >>> slots = StagingSlots(4)
    >>> slots.take(3)
    [0, 1, 2]
    >>> slots.take(2) is None
    True
    >>> slots.give_back([1])
    >>> slots.free
    2
    """

    def __init__(self, num_slots: int):
        if num_slots <= 0:
            raise ValueError(f"A staging pool needs slots; got {num_slots}.")
        self.num_slots = num_slots
        self._free = deque(range(num_slots))

    @property
    def free(self) -> int:
        """Slots not taken."""
        return len(self._free)

    def take(self, count: int) -> list[int] | None:
        """Take ``count`` slots, or none at all when fewer are free."""
        if count > len(self._free):
            return None
        return [self._free.popleft() for _ in range(count)]

    def give_back(self, slots: list[int]) -> None:
        """Return slots taken earlier."""
        self._free.extend(slots)


class TTNixlPushConnectorWorker(NixlPushConnectorWorker):
    """vLLM's push worker with its host buffer cut down to staging slots.

    Attributes
    ----------
    slots
        The slot pool; NIXL's host buffer has one row per slot.

    Notes
    -----
    State and threads:

    * ``_recv_slots``, ``_recv_deadlines``, ``_expired``, ``_quarantined`` and
      ``_waiting_registrations`` (D), ``_waiting_stage`` (P) and ``slots``
      belong to the model runner thread.
    * ``_to_stage`` carries matched P requests from the writer thread to the
      model runner thread.
    * ``_send_slots`` and ``_released`` (P) are written by the model runner
      thread and read by the writer thread, under ``_lock``.
    """

    def __init__(self, vllm_config, engine_id: str, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        num_slots = staging_slot_count(
            vllm_config,
            kv_cache_config.num_blocks,
            len(kv_cache_config.kv_cache_groups),
        )
        # NIXL registers and addresses the host buffer by these counts; the
        # engine's own KVCacheConfig is left untouched.
        self.num_blocks = num_slots * self._physical_blocks_per_logical_kv_block
        self._logical_num_blocks = num_slots
        self.kv_cache_config = dataclasses.replace(
            kv_cache_config, num_blocks=num_slots
        )
        extra = vllm_config.kv_transfer_config
        ttl = extra.get_from_extra_config("decoder_kv_blocks_ttl", 480)
        self._init_staging(
            num_slots,
            recv_timeout=float(
                extra.get_from_extra_config("push_registration_timeout", ttl)
            ),
            quarantine=float(extra.get_from_extra_config("kv_lease_duration", 30)),
        )
        logger.info(
            "NIXL host staging: %d slots instead of a %d-block mirror.",
            num_slots,
            kv_cache_config.num_blocks,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register a host buffer of ``slots.num_slots`` rows per layer.

        ``kv_caches`` describes each layer's KV with the KV pool's block count
        in dimension 0; only that count changes.
        """
        staged = {
            name: torch.empty(1, dtype=layer.dtype).expand(
                (self.slots.num_slots, *layer.shape[1:])
            )
            for name, layer in kv_caches.items()
        }
        super().register_kv_caches(staged)

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        """Send D's registrations with slot numbers, and stage P's matched sends."""
        waiting = list(self._waiting_registrations)
        self._waiting_registrations.clear()
        registrations = {}
        for req_id, reg in [*waiting, *(metadata.push_registrations or {}).items()]:
            device = self._as_grouped_block_ids(reg["local_block_ids"])
            slots = self._take_slots(device)
            if slots is None:
                self._waiting_registrations.append((req_id, reg))
                continue
            self._recv_slots[req_id] = (slots, device)
            self._recv_deadlines[req_id] = time.perf_counter() + self._recv_timeout
            registrations[req_id] = {**reg, "local_block_ids": slots}
        # The scheduler shares ``metadata``; hand vLLM a copy.
        metadata = copy.copy(metadata)
        metadata.push_registrations = registrations
        super().start_load_kv(metadata)
        self._stage_matched_sends()

    def save_kv_to_host(self, metadata: NixlConnectorMetadata) -> None:
        """Copy nothing per step: P stages a request once D's blocks are known."""

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta) -> None:
        """Copy a received request from its slots into the chips; free the slots."""
        slots, device = self._recv_slots.pop(req_id)
        self._recv_deadlines.pop(req_id, None)
        flat_slots = _flatten(slots)
        self.copy_blocks(
            self.host_xfer_buffers,
            self.device_kv_caches,
            flat_slots,
            _flatten(device),
            "h2d",
        )
        self.slots.give_back(flat_slots)

    def get_transfer_results(self) -> KVConnectorTransferResults:
        """Report vLLM's results; free slots of failed receives and done sends.

        A D request whose KV has not arrived by its deadline is reported as a
        failed receive (vLLM's push path waits for it indefinitely). Its slots
        rest for ``kv_lease_duration``, since P's write may still land.
        """
        now = time.perf_counter()
        for req_id, deadline in list(self._recv_deadlines.items()):
            if now > deadline:
                del self._recv_deadlines[req_id]
                self._expired.add(req_id)
                self._failed_recv_reqs.put(req_id)
                logger.warning(
                    "No KV arrived for %s within %.0f s; failing its receive.",
                    req_id,
                    self._recv_timeout,
                )
        while self._quarantined and self._quarantined[0][0] <= now:
            self.slots.give_back(self._quarantined.popleft()[1])
        results = super().get_transfer_results()
        for req_id in results.failed_recving:
            self._recv_deadlines.pop(req_id, None)
            entry = self._recv_slots.pop(req_id, None)
            if entry is None:
                continue
            if req_id in self._expired:
                self._expired.discard(req_id)
                self._quarantined.append((now + self._quarantine, _flatten(entry[0])))
            else:
                self.slots.give_back(_flatten(entry[0]))
        with self._lock:
            self._released |= results.finished_sending
        self._release_pending |= results.finished_sending & self._send_slots.keys()
        with self._sending_transfers_lock:
            writing = {r for r, handles in self._sending_transfers.items() if handles}
        for req_id in list(self._release_pending - writing):
            self._release_pending.discard(req_id)
            with self._lock:
                slots = self._send_slots.pop(req_id)
            self.slots.give_back(slots)
        self._stage_matched_sends()
        return results

    def _init_staging(
        self, num_slots: int, recv_timeout: float = 480.0, quarantine: float = 30.0
    ) -> None:
        """Create the slot pool and the per-request staging state.

        Parameters
        ----------
        num_slots
            Host rows of the pool.
        recv_timeout
            Seconds D waits for a registered request's KV before failing it.
        quarantine
            Seconds the slots of a failed receive rest before reuse.
        """
        self.slots = StagingSlots(num_slots)
        self._recv_timeout = recv_timeout
        self._quarantine = quarantine
        # D: deadline per request whose registration went out.
        self._recv_deadlines: dict[str, float] = {}
        # D: requests this worker failed at their deadline.
        self._expired: set[str] = set()
        # D: (release time, slots) of failed receives, oldest first.
        self._quarantined: deque[tuple[float, list[int]]] = deque()
        self._lock = threading.Lock()
        # D: slots and device blocks per request whose registration went out.
        self._recv_slots: dict[str, tuple[BlockIds, BlockIds]] = {}
        # D: registrations waiting for free slots, oldest first.
        self._waiting_registrations: deque[tuple[str, dict[str, Any]]] = deque()
        # P: matched requests (device blocks, D's registration) to stage.
        self._to_stage: queue.Queue[tuple[str, BlockIds, dict[str, Any]]] = (
            queue.Queue()
        )
        self._waiting_stage: deque[tuple[str, BlockIds, dict[str, Any]]] = deque()
        # P: slots of requests staged for, or being written to, D.
        self._send_slots: dict[str, list[int]] = {}
        # P: requests whose send finished or expired; never staged again.
        self._released: set[str] = set()
        # P: finished requests whose slots wait for their last write handle.
        self._release_pending: set[str] = set()

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Write staged slots to D; hand an unstaged match to the runner thread.

        Runs on the writer thread, once a finished P request meets D's
        registration (``local_block_ids`` are device blocks) and again once
        ``_stage_matched_sends`` has staged it (``local_block_ids`` are slots).
        """
        with self._lock:
            released = request_id in self._released
            staged = request_id in self._send_slots
        if released and not staged:
            logger.debug("Dropping push of %s: its send already ended.", request_id)
            return
        if not staged:
            self._to_stage.put((request_id, local_block_ids, registration_data))
            return
        super()._do_start_push_kv(request_id, local_block_ids, registration_data)

    def _stage_matched_sends(self) -> None:
        """Copy matched P requests from the chips into slots and queue their writes.

        D registers only blocks it has no prefix-cache hit for, so P stages
        the same number of its last blocks per group, the blocks vLLM's push
        path would pair with D's.
        """
        pending = list(self._waiting_stage)
        self._waiting_stage.clear()
        while True:
            try:
                pending.append(self._to_stage.get_nowait())
            except queue.Empty:
                break
        queued = False
        for req_id, device_ids, reg in pending:
            with self._lock:
                if req_id in self._released:
                    continue
            device = _last_blocks(
                self._as_grouped_block_ids(device_ids),
                self._as_grouped_block_ids(reg["local_block_ids"]),
            )
            slots = self._take_slots(device)
            if slots is None:
                self._waiting_stage.append((req_id, device_ids, reg))
                continue
            flat_slots = _flatten(slots)
            self.copy_blocks(
                self.device_kv_caches,
                self.host_xfer_buffers,
                _flatten(device),
                flat_slots,
                "d2h",
            )
            with self._lock:
                self._send_slots[req_id] = flat_slots
            self._deferred_push_inbox.put((req_id, slots, reg))
            queued = True
        if queued:
            self._push_writer_wake.set()

    def _take_slots(self, device: BlockIds) -> BlockIds | None:
        """Slots for ``device``'s blocks, grouped like it, or None when full."""
        flat = self.slots.take(sum(len(group) for group in device))
        if flat is None:
            return None
        grouped, start = [], 0
        for group in device:
            grouped.append(flat[start : start + len(group)])
            start += len(group)
        return tuple(grouped)


class TTNixlPushConnector(NixlPushConnector):
    """``NixlPushConnector`` whose worker stages KV through slots.

    Raises
    ------
    ValueError
        With decode context parallelism, as ``NixlPushConnector`` does.
    """

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        # NixlPushConnector.__init__ would build vLLM's worker and its NIXL
        # agent; build the TT worker in its place.
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise ValueError(
                f"{CONNECTOR_NAME} does not support decode_context_parallel_size > 1."
            )
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPushConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = TTNixlPushConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")


def _flatten(grouped: BlockIds) -> list[int]:
    """All block IDs of all groups, group after group."""
    return [block for group in grouped for block in group]


def _last_blocks(device: BlockIds, remote: BlockIds) -> BlockIds:
    """Per group, the last ``len(remote group)`` blocks of ``device``.

    Raises
    ------
    ValueError
        When D asks for more blocks of a group than P has.
    """
    if len(device) != len(remote):
        raise ValueError(
            f"P has {len(device)} KV cache groups, D registered {len(remote)}."
        )
    kept = []
    for mine, theirs in zip(device, remote):
        if len(theirs) > len(mine):
            raise ValueError(
                f"D registered {len(theirs)} blocks of a group, P has {len(mine)}."
            )
        kept.append(list(mine[len(mine) - len(theirs) :]))
    return tuple(kept)
