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
* With ``"enable_chunked_push": true`` in ``kv_connector_extra_config``
  (experimental), P instead copies the prompt blocks each prefill step
  completes into slots for the whole prompt, right after the step, so a
  matched request is written from its slots without a copy. The write itself
  still starts after the match. Names follow vLLM's chunked push proposal
  (``NixlPushChunkJob``, ``_build_chunk_jobs``).

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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

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
from vllm.v1.kv_cache_interface import KVCacheConfig, is_full_attention_spec

from vllm_tt_plugin.config import get_tt_data_parallel_size
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_tt_logger(__name__)

CONNECTOR_NAME = "TTNixlPushConnector"

# Longest requests the pool holds at once. P, with chunked push, holds a
# request's slots from its first prefill step until its write ends; D, behind
# a proxy that sends both legs at once, holds them from the start of P's
# prefill. One request being written, one waiting for its write and one
# starting its prefill need three; the fourth is headroom. A request that finds
# too few free slots is not refused: P copies it whole once matched, and D's
# registration waits for slots.
_REQUESTS_PER_POOL = 4


def register_kv_connectors() -> None:
    """Make ``TTNixlPushConnector`` selectable by name in ``--kv-transfer-config``."""
    if CONNECTOR_NAME not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            CONNECTOR_NAME, __name__, TTNixlPushConnector.__name__
        )


def staging_slot_count(vllm_config, num_blocks: int, num_groups: int) -> int:
    """Number of staging slots: room for four longest requests, at most the pool.

    One request crosses in one NIXL write, which cannot be split, so the pool
    must hold the longest request whole.

    Parameters
    ----------
    vllm_config
        The engine's ``VllmConfig``; ``max_model_len`` and ``block_size`` set
        the longest request in blocks.
    num_blocks
        Blocks in the KV pool; more slots than blocks are not useful, since a
        request holds its slots while it holds, or is prefilling into, as many
        blocks.
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


@dataclasses.dataclass(frozen=True)
class NixlPushChunkJob:
    """Prompt blocks ``[start_block, end_block)`` of a P request that one
    prefill step completes.

    P's scheduler lists a step's jobs in its connector metadata
    (``push_chunk_jobs``); P's worker copies them into staging slots after
    the step's forward.

    Attributes
    ----------
    request_id
        P's request ID.
    block_ids
        P's device block IDs of the job's blocks, in prompt order.
    start_block, end_block
        Positions of the blocks in the prompt.
    total_blocks
        Blocks of the whole prompt. The job that ends there is the last one.
    """

    request_id: str
    block_ids: list[int]
    start_block: int
    end_block: int
    total_blocks: int


@dataclasses.dataclass
class _ChunkPushTracker:
    """A P request with chunked push and the prompt blocks it has jobs for.

    Attributes
    ----------
    request
        The scheduler's request; its ``num_computed_tokens`` is read per step.
    next_block
        Prompt blocks already put into jobs.
    """

    request: "Request"
    next_block: int = 0


class TTNixlPushConnectorScheduler(NixlPushConnectorScheduler):
    """vLLM's push scheduler that also names the blocks each P step completes.

    With ``enable_chunked_push`` on a producer, each step's metadata gets two
    fields: ``push_chunk_jobs``, a ``NixlPushChunkJob`` per P request whose
    prefill in the step completes prompt blocks, and
    ``push_chunk_preemptions``, the P requests preempted since the previous
    step. Off, or on a consumer, it is vLLM's push scheduler.

    Raises
    ------
    ValueError
        When ``enable_chunked_push`` is not a boolean, on either side, and
        when it is on for a producer whose KV is not one full-attention group,
        whose prompt tail is left to D (a prefill backoff, for a Mamba state
        or a multi-module MTP lookahead), or that runs TT lane DP.

    Notes
    -----
    Runs on the engine core thread. A request is in ``_chunk_trackers`` from
    its block allocation (``update_state_after_alloc``, called again when a
    preempted request is scheduled again) to ``request_finished``.
    """

    def __init__(self, vllm_config, engine_id: str, kv_cache_config: KVCacheConfig):
        transfer_config = vllm_config.kv_transfer_config
        enabled = transfer_config.get_from_extra_config("enable_chunked_push", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                "enable_chunked_push in kv_connector_extra_config must be true or "
                f"false; got {enabled!r}."
            )
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._chunked_push = enabled and transfer_config.kv_role == "kv_producer"
        self._chunk_trackers: dict[str, _ChunkPushTracker] = {}
        if self._chunked_push:
            self._refuse_unsupported_chunked_push(vllm_config, kv_cache_config)

    def update_state_after_alloc(
        self, request: "Request", blocks, num_external_tokens: int
    ) -> None:
        """Start a P request's jobs at block 0, also after a preemption."""
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        params = request.kv_transfer_params
        if self._chunked_push and params and params.get("do_remote_decode"):
            self._chunk_trackers[request.request_id] = _ChunkPushTracker(request)

    def request_finished(
        self, request: "Request", block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        """Plan no more jobs for the request; vLLM's push scheduler does the rest."""
        self._chunk_trackers.pop(request.request_id, None)
        return super().request_finished(request, block_ids)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> NixlConnectorMetadata:
        """vLLM's push metadata, with the step's chunk jobs when chunked push is on."""
        meta = super().build_connector_meta(scheduler_output)
        if self._chunked_push:
            self._build_chunk_jobs(meta, scheduler_output)
        return meta

    def _build_chunk_jobs(
        self, meta: NixlConnectorMetadata, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Set ``meta.push_chunk_jobs`` and ``meta.push_chunk_preemptions``.

        vLLM builds a step's metadata before it adds the step's tokens to
        ``num_computed_tokens``, so the prompt tokens computed once the step
        has run are the two summed. A job takes the blocks those tokens fill;
        the prompt's partial last block goes with the job that completes the
        prompt. Block IDs are read only for a step with a job: ``TTScheduler``
        can build a release step's metadata after its block state is dropped.

        Raises
        ------
        RuntimeError
            When the step carries no block IDs for a request it has a job for.
        """
        meta.push_chunk_preemptions = {
            req_id
            for req_id in scheduler_output.preempted_req_ids or ()
            if req_id in self._chunk_trackers
        }
        meta.push_chunk_jobs = []
        block_size = self.kv_cache_config.transfer_groups[0].kv_cache_spec.block_size
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            tracker = self._chunk_trackers.get(req_id)
            if tracker is None:
                continue
            prompt = tracker.request.num_prompt_tokens
            computed = min(tracker.request.num_computed_tokens + num_tokens, prompt)
            end = (
                cdiv(computed, block_size)
                if computed >= prompt
                else computed // block_size
            )
            start = tracker.next_block
            if end <= start:
                continue
            block_state = scheduler_output.kv_connector_block_state
            block_ids = (
                None if block_state is None else block_state.get_block_ids(req_id)
            )
            if block_ids is None:
                raise RuntimeError(
                    f"The step completes blocks {start} to {end - 1} of {req_id} "
                    "but carries no block IDs for it."
                )
            table = self.kv_cache_config.select_transfer_block_ids(block_ids)[0]
            meta.push_chunk_jobs.append(
                NixlPushChunkJob(
                    request_id=req_id,
                    block_ids=list(table[start:end]),
                    start_block=start,
                    end_block=end,
                    total_blocks=cdiv(prompt, block_size),
                )
            )
            tracker.next_block = end

    def _refuse_unsupported_chunked_push(
        self, vllm_config, kv_cache_config: KVCacheConfig
    ) -> None:
        """Refuse KV that a step's completed blocks do not hold in full.

        Raises
        ------
        ValueError
            With more than one transfer group or one that is not full
            attention, with a prefill backoff, or with TT lane DP, whose
            coordinator builds metadata without block state.
        """
        groups = kv_cache_config.transfer_groups
        if len(groups) != 1 or not is_full_attention_spec(groups[0].kv_cache_spec):
            raise ValueError(
                "enable_chunked_push needs one full-attention KV cache group; got "
                f"{[type(group.kv_cache_spec).__name__ for group in groups]}."
            )
        backoff = self._prefill_backoff()
        if backoff > 0:
            raise ValueError(
                "enable_chunked_push does not support a prefill backoff (Mamba "
                f"state, multi-module MTP lookahead); got {backoff} tokens."
            )
        lanes = get_tt_data_parallel_size(vllm_config)
        if lanes > 1:
            raise ValueError(
                f"enable_chunked_push does not support TT lane DP; got {lanes} lanes."
            )


class TTNixlPushConnectorWorker(NixlPushConnectorWorker):
    """vLLM's push worker with its host buffer cut down to staging slots.

    Attributes
    ----------
    slots
        The slot pool; NIXL's host buffer has one row per slot.

    Notes
    -----
    State and threads:

    * ``_recv_slots``, ``_recv_deadlines``, ``_expired``, ``_quarantined``,
      ``_load_errors`` and ``_waiting_registrations`` (D), ``_waiting_stage``,
      ``_send_lane``, ``_chunk_slots``, ``_chunk_staged`` and
      ``_chunk_unstaged`` (P) and ``slots`` belong to the model runner thread.
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
            pool_blocks=kv_cache_config.num_blocks,
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
        """Send D's registrations with slot numbers, and stage P's matched sends.

        Under lane DP, ``metadata.tt_lane_of`` names each request's lane: block
        IDs are lane-local. Without it every request is on lane 0. P first
        gives back the chunk slots of requests preempted or aborted.
        """
        self.handle_preemptions(metadata)
        lane_of = getattr(metadata, "tt_lane_of", None) or {}
        now = time.perf_counter()
        queued = list(self._waiting_registrations)
        self._waiting_registrations.clear()
        queued += [
            (req_id, reg, lane_of.get(req_id, 0))
            for req_id, reg in (metadata.push_registrations or {}).items()
        ]
        registrations = {}
        for req_id, reg, lane in queued:
            self._recv_deadlines.setdefault(req_id, now + self._recv_timeout)
            if req_id in self._expired:
                continue
            device = self._as_grouped_block_ids(reg["local_block_ids"])
            slots = self._take_slots(device)
            if slots is None:
                self._waiting_registrations.append((req_id, reg, lane))
                continue
            self._recv_slots[req_id] = (slots, device, lane)
            registrations[req_id] = {**reg, "local_block_ids": slots}
        for req_id in metadata.push_finished_blocks or {}:
            self._send_lane[req_id] = lane_of.get(req_id, 0)
        # The scheduler shares ``metadata``; hand vLLM a copy.
        metadata = copy.copy(metadata)
        metadata.push_registrations = registrations
        super().start_load_kv(metadata)
        self._stage_matched_sends()

    def save_kv_to_host(self, metadata: NixlConnectorMetadata) -> None:
        """Copy the step's chunk jobs from the chips into their requests' slots.

        vLLM calls this after the step's forward is enqueued; the copy runs
        after the forward on the same command queue and returns once the
        blocks are in the slots. Without chunked push there are no jobs: P
        stages a request once D's blocks are known.
        """
        self._stage_chunk_jobs(getattr(metadata, "push_chunk_jobs", ()))

    def handle_preemptions(self, metadata: NixlConnectorMetadata) -> None:
        """Give back the chunk slots of preempted and aborted P requests.

        Nothing waits: no write reads chunk slots before the request is
        matched, and a preempted request's recompute takes new slots.
        """
        retired = set(getattr(metadata, "push_chunk_preemptions", ()))
        for req_id in retired | set(metadata.reqs_not_processed or ()):
            self._drop_chunk_slots(req_id)

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta) -> None:
        """Copy a received request from its slots into the chips; free the slots."""
        slots, device, lane = self._recv_slots.pop(req_id)
        self._recv_deadlines.pop(req_id, None)
        flat_slots = _flatten(slots)
        self.copy_blocks(
            self.host_xfer_buffers,
            self.device_kv_caches,
            flat_slots,
            _flatten(device),
            "h2d",
            lane=lane,
        )
        self.slots.give_back(flat_slots)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Blocks of failed receives, as ``lane * pool blocks + block ID``.

        vLLM's own report lists every block of a failed request without its
        lane; this one lists the blocks that were to be received, and the lane
        coordinator splits it by lane. Without lanes the IDs are the blocks.
        """
        super().get_block_ids_with_load_errors()
        errors, self._load_errors = self._load_errors, set()
        return errors

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
                # One still waiting for slots never registered with P.
                self._waiting_registrations = deque(
                    entry for entry in self._waiting_registrations if entry[0] != req_id
                )
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
            expired = req_id in self._expired
            self._expired.discard(req_id)
            entry = self._recv_slots.pop(req_id, None)
            if entry is None:
                continue
            slots, device, lane = entry
            self._load_errors |= {
                lane * self._pool_blocks + block for block in _flatten(device)
            }
            if expired:
                self._quarantined.append((now + self._quarantine, _flatten(slots)))
            else:
                self.slots.give_back(_flatten(slots))
        with self._lock:
            self._released |= results.finished_sending
        for req_id in results.finished_sending:
            # Still here only when its lease expired before D registered.
            self._drop_chunk_slots(req_id)
        self._release_pending |= results.finished_sending & self._send_slots.keys()
        with self._sending_transfers_lock:
            writing = {r for r, handles in self._sending_transfers.items() if handles}
        for req_id in list(self._release_pending - writing):
            self._release_pending.discard(req_id)
            with self._lock:
                slots = self._send_slots.pop(req_id)
            self.slots.give_back(slots)
        for req_id in results.finished_sending - self._send_slots.keys():
            self._send_lane.pop(req_id, None)
        self._stage_matched_sends()
        return results

    def _init_staging(
        self,
        num_slots: int,
        recv_timeout: float = 480.0,
        quarantine: float = 30.0,
        pool_blocks: int = 0,
    ) -> None:
        """Create the slot pool and the per-request staging state.

        Parameters
        ----------
        num_slots
            Host rows of the pool.
        recv_timeout
            Seconds D waits for a request's KV, from its registration, before
            failing it.
        quarantine
            Seconds the slots of a failed receive rest before reuse.
        pool_blocks
            Blocks in each lane's KV pool; numbers the blocks of failed
            receives per lane.
        """
        self.slots = StagingSlots(num_slots)
        self._pool_blocks = pool_blocks
        # D: lane-numbered blocks of failed receives, for vLLM to recompute.
        self._load_errors: set[int] = set()
        # P: lane of each finished request until its send ends.
        self._send_lane: dict[str, int] = {}
        self._recv_timeout = recv_timeout
        self._quarantine = quarantine
        # D: deadline per request whose registration went out.
        self._recv_deadlines: dict[str, float] = {}
        # D: requests this worker failed at their deadline.
        self._expired: set[str] = set()
        # D: (release time, slots) of failed receives, oldest first.
        self._quarantined: deque[tuple[float, list[int]]] = deque()
        self._lock = threading.Lock()
        # D: slots, device blocks and lane per request whose registration went out.
        self._recv_slots: dict[str, tuple[BlockIds, BlockIds, int]] = {}
        # D: (request, registration, lane) waiting for free slots, oldest first.
        self._waiting_registrations: deque[tuple[str, dict[str, Any], int]] = deque()
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
        # P, chunked push: a slot per prompt block of each staged request, slot
        # i holding block i, until the request is matched or ends.
        self._chunk_slots: dict[str, list[int]] = {}
        # P, chunked push: prompt blocks of each request copied into its slots.
        self._chunk_staged: dict[str, int] = {}
        # P, chunked push: requests whose first job found too few free slots;
        # they are staged whole once matched.
        self._chunk_unstaged: set[str] = set()

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
        path would pair with D's. A request staged chunk by chunk is written
        from the last of its slots instead, without a copy.
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
            remote = self._as_grouped_block_ids(reg["local_block_ids"])
            flat_slots = self._take_staged_chunks(req_id)
            if flat_slots is not None:
                slots = _last_blocks((flat_slots,), remote)
            else:
                device = _last_blocks(self._as_grouped_block_ids(device_ids), remote)
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
                    lane=self._send_lane.get(req_id, 0),
                )
            with self._lock:
                self._send_slots[req_id] = flat_slots
            self._deferred_push_inbox.put((req_id, slots, reg))
            queued = True
        if queued:
            self._push_writer_wake.set()

    def _stage_chunk_jobs(self, jobs: Iterable[NixlPushChunkJob]) -> None:
        """Copy each job's blocks into its request's slots at their prompt positions.

        A job at block 0 begins its request: it takes ``total_blocks`` slots,
        or none when fewer are free, and then the request is staged whole once
        matched. Slots the request held from before a preemption go back
        first, whether or not the preemption was listed.

        Raises
        ------
        RuntimeError
            When a later job does not start where the request's staged blocks
            end, or counts another number of prompt blocks.
        """
        for job in jobs:
            req_id = job.request_id
            if job.start_block == 0:
                self._drop_chunk_slots(req_id)
                taken = self.slots.take(job.total_blocks)
                if taken is None:
                    self._chunk_unstaged.add(req_id)
                    logger.debug(
                        "No %d free staging slots for %s; staging it once matched.",
                        job.total_blocks,
                        req_id,
                    )
                    continue
                self._chunk_slots[req_id] = taken
                self._chunk_staged[req_id] = 0
            elif req_id in self._chunk_unstaged:
                continue
            slots = self._chunk_slots.get(req_id, [])
            staged = self._chunk_staged.get(req_id, 0)
            if job.start_block != staged or job.total_blocks != len(slots):
                raise RuntimeError(
                    f"A chunk job for {req_id} starts at block {job.start_block} "
                    f"of {job.total_blocks}; its slots hold {staged} staged "
                    f"blocks of {len(slots)}."
                )
            self.copy_blocks(
                self.device_kv_caches,
                self.host_xfer_buffers,
                job.block_ids,
                slots[job.start_block : job.end_block],
                "d2h",
            )
            self._chunk_staged[req_id] = job.end_block

    def _take_staged_chunks(self, req_id: str) -> list[int] | None:
        """A matched request's slots for its whole prompt, or None if not staged.

        Raises
        ------
        RuntimeError
            When the request was matched before all its blocks were staged.
        """
        self._chunk_unstaged.discard(req_id)
        slots = self._chunk_slots.pop(req_id, None)
        staged = self._chunk_staged.pop(req_id, None)
        if slots is not None and staged != len(slots):
            raise RuntimeError(
                f"{req_id} was matched with {staged} of its {len(slots)} prompt "
                "blocks staged."
            )
        return slots

    def _drop_chunk_slots(self, req_id: str) -> None:
        """Forget ``req_id``'s chunk staging and give its slots back."""
        self._chunk_unstaged.discard(req_id)
        self._chunk_staged.pop(req_id, None)
        slots = self._chunk_slots.pop(req_id, None)
        if slots is not None:
            self.slots.give_back(slots)

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
        With decode context parallelism, as ``NixlPushConnector`` does, and
        for a configuration ``TTNixlPushConnectorScheduler`` refuses.
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
            self.connector_scheduler = TTNixlPushConnectorScheduler(
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
