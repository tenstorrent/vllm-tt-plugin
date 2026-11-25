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
* D fails a receive whose KV has not arrived within
  ``push_registration_timeout``. It gives the receive's slots to another
  request only once P has confirmed that it writes nothing more into them
  (``TTNixlPushConnectorWorker``, Notes).

vLLM's scheduler keeps using real block IDs; only the IDs that reach NIXL are
slot numbers. Copies between chips and slots run on the model runner thread,
the only thread that may touch the device. P and D must both run
``TTNixlPushConnector``: they exchange notifications that vLLM's push worker
does not send or understand.

Principal entry points: ``TTNixlPushConnector`` (selected with
``"kv_connector": "TTNixlPushConnector"``) and ``register_kv_connectors``.
"""

import copy
import dataclasses
import functools
import queue
import threading
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future
from typing import Any

import msgspec
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
    PUSH_REG_NOTIF_PREFIX,
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

TT_NOTIF_PREFIX = PUSH_REG_NOTIF_PREFIX + b"tt:"
"""Prefix of the notifications that only ``TTNixlPushConnectorWorker`` sends.

They start like a ``PUSH_REG``, so that vLLM's writer loop hands them to
``_handle_push_reg_notif`` on the writer thread, which owns the registrations
they concern. The rest is a msgpack map: ``kind`` (``REG_ACK``, ``CANCEL``,
``CANCEL_ACK`` or ``PROBE``), ``request_id`` (D's request ID), in a ``CANCEL``
D's ``decode_*`` address fields from the registration, and in an acknowledgement
the name of P's NIXL agent (``agent``)."""

REG_ACK = "reg-ack"
"""P to D: P has recorded D's registration; a later ``CANCEL`` finds it."""
CANCEL = "cancel"
"""D to P: D failed the receive; P must write nothing more into its slots."""
CANCEL_ACK = "cancel-ack"
"""P to D: no write into the cancelled receive's slots is in flight or will be
submitted; D may give them to another request."""
PROBE = "probe"
"""D to P, while a cancellation waits: tells D whether P's agent is still
reachable. P ignores it."""

# NIXL's bindings raise these, by status, when a send was refused before
# anything was posted: the UCX endpoint to the agent has failed, which is
# sticky (NIXL_ERR_REMOTE_DISCONNECT), or NIXL knows no such agent
# (NIXL_ERR_NOT_FOUND). The bindings module differs per build, so match names.
_DISCONNECTED = "nixlRemoteDisconnectError"
_NOT_DELIVERED = frozenset({_DISCONNECTED, "nixlNotFoundError"})

# Seconds between D's probes of the P agents a cancellation waits for.
_PROBE_INTERVAL_S = 1.0
# Seconds D keeps the slots after NIXL first reports the endpoint to a P agent
# failed, before it treats that agent as gone. D's writer thread drives NIXL's
# progress on every pass meanwhile, so that data of a WRITE that reached D's
# host before the disconnect is applied before the slots change hands.
_GONE_GRACE_S = 10.0

# The address fields of D, from a registration, that P needs to answer.
_DECODE_ADDRESS = ("decode_engine_id", "decode_host", "decode_port", "decode_tp_size")

# Keys TT adds to registration dicts on their own side; none is ever sent.
# D: the names of the P agents the registration may have reached.
_SENT_TO = "tt_sent_to"
# D: the receive failed before the registration was sent; never send it.
_WITHDRAWN = "tt_withdrawn"
# P: D cancelled the registration; write nothing for it.
_CANCELLED = "tt_cancelled"
# Kinds of queued ``_WriterTask`` besides the notifications. D: the receive
# succeeded, so it is never cancelled; probe P and free what is due. P: send
# the cancel acknowledgements due.
_RECEIVED = "received"
_CHECK_CANCELS = "check-cancels"
_SETTLE = "settle"

# Why D's model runner thread gives back a cancelled receive's slots.
_FREED_WHY = {
    _WITHDRAWN: "its registration never reached P",
    CANCEL_ACK: "P acknowledged the cancel",
    _DISCONNECTED: "NIXL reports P's agent disconnected",
}


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


@dataclasses.dataclass
class _Receive:
    """D's record of a request whose registration holds staging slots.

    Attributes
    ----------
    slots
        The staging slots, grouped like ``device``.
    device
        D's blocks that the slots are copied into.
    lane
        The lane whose chips hold ``device``.
    reg
        The registration with slot numbers, as handed to vLLM. vLLM's writer
        thread sends this dict, and TT's writer thread marks it.
    """

    slots: BlockIds
    device: BlockIds
    lane: int
    reg: dict[str, Any]


@dataclasses.dataclass
class _Unacked:
    """D's record of a sent registration that awaits P's acknowledgements.

    It lives on the writer thread from the send until every P agent in
    ``reg[_SENT_TO]`` has sent ``REG_ACK`` or the receive has succeeded, and,
    once cancelled, until every one has sent ``CANCEL_ACK`` or is gone.

    Attributes
    ----------
    reg
        The registration; ``reg[_SENT_TO]`` names the P agents it may have
        reached.
    reg_acks, cancel_acks
        The P agents that sent ``REG_ACK`` and ``CANCEL_ACK``.
    cancelled
        D failed the receive. ``CANCEL`` goes out once every agent has sent
        ``REG_ACK`` or is disconnected.
    cancel_sent
        ``CANCEL`` went out.
    disconnected
        P agents a send to which NIXL refused as disconnected, with the
        ``time.perf_counter`` of the first refusal. An agent is gone
        ``_GONE_GRACE_S`` after it.
    probed
        ``time.perf_counter`` of the last ``PROBE``.
    """

    reg: dict[str, Any]
    reg_acks: set[str] = dataclasses.field(default_factory=set)
    cancelled: bool = False
    cancel_sent: bool = False
    cancel_acks: set[str] = dataclasses.field(default_factory=set)
    disconnected: dict[str, float] = dataclasses.field(default_factory=dict)
    probed: float = float("-inf")


@dataclasses.dataclass
class _Send:
    """P's record of a finished request, from its finish until its slots are free.

    Attributes
    ----------
    lane
        The lane whose chips hold the request's KV.
    reg
        D's registration, once the writer thread has matched the request with
        it.
    slots
        The staging slots the request was copied into, once staged.
    submitting
        The writer thread is submitting the request's write: the slots stay,
        even if the send ends meanwhile.
    ended
        vLLM reported the send finished, written or past its lease. The slots
        go back once no write reads them.
    """

    lane: int
    reg: dict[str, Any] | None = None
    slots: list[int] | None = None
    submitting: bool = False
    ended: bool = False


@dataclasses.dataclass(frozen=True)
class _WriterTask:
    """Work for the writer thread, queued on vLLM's ``_reg_send_inbox``.

    vLLM's writer loop hands each item of ``_reg_send_inbox`` to
    ``_send_registration_to_p``, which runs a task instead of sending a
    registration.

    Attributes
    ----------
    kind
        ``CANCEL``: D withdraws or cancels ``reg``. ``_RECEIVED``: D's receive
        succeeded; D awaits no acknowledgement of it. ``_CHECK_CANCELS``: D
        probes the P agents its cancellations wait for and frees what is due.
        ``REG_ACK`` or ``CANCEL_ACK``: P sends it for ``reg`` now that its
        handshake with D is done. ``_SETTLE``: P sends the cancel
        acknowledgements now due.
    reg
        The registration, or for ``CANCEL_ACK`` D's ``CANCEL``; empty for
        ``_RECEIVED``, ``_CHECK_CANCELS`` and ``_SETTLE``.
    """

    kind: str
    reg: dict[str, Any] = dataclasses.field(default_factory=dict)


class TTNixlPushConnectorWorker(NixlPushConnectorWorker):
    """vLLM's push worker with its host buffer cut down to staging slots.

    Attributes
    ----------
    slots
        The slot pool; NIXL's host buffer has one row per slot.

    Notes
    -----
    D's receive of a request fails when no KV arrives within
    ``push_registration_timeout``, or when vLLM reports it failed. P may still
    write into its slots then: P can read the registration late, or have a
    write in flight. D therefore gives the slots to another request only on
    P's word, given in these notifications (``TT_NOTIF_PREFIX``):

    1. D's writer thread sends the registration, ``PUSH_REG``, as vLLM does,
       and records the P agents it may have reached: NIXL refuses a send to an
       agent whose endpoint has failed, or that it does not know, before it
       posts anything.
    2. P's writer thread records it, matched or pending, then sends
       ``REG_ACK``.
    3. D's model runner thread fails the receive and queues its cancellation
       to D's writer thread. A registration that reached no P agent (not yet
       sent, or refused) is withdrawn, and the slots are free at once. For a
       sent one, D's writer thread sends ``CANCEL`` once every agent has sent
       ``REG_ACK`` or is disconnected.
    4. P's writer thread marks the registration cancelled, or drops it if it
       is pending, and sends ``CANCEL_ACK`` once no write of the P requests
       matched with it has a handle in flight.
    5. Once every agent has sent ``CANCEL_ACK`` or is gone, D's writer thread
       hands the receive back to D's model runner thread, which frees its
       slots.

    While a cancellation waits, D's writer thread sends the agents it waits
    for a ``PROBE`` every ``_PROBE_INTERVAL_S``. NIXL refuses a send to an
    agent whose UCX endpoint has failed (the peer process died, the connection
    was reset, or keepalive timed out), and that state is sticky. D counts the
    agent gone ``_GONE_GRACE_S`` after the first refusal; meanwhile its writer
    thread keeps driving NIXL's progress, so that WRITE data that reached D's
    host before the disconnect is applied before the slots change hands.

    NIXL does not order notifications, so ``CANCEL`` waits for ``REG_ACK``:
    P reads a ``CANCEL`` only after it has recorded the registration, and no
    copy of the registration can reach P afterwards. P submits writes only on
    its writer thread, and each submission point (``_do_start_push_kv``)
    checks the mark there. Any write into the slots was therefore submitted
    before P read ``CANCEL`` and ended before P sent ``CANCEL_ACK``, which is
    before D frees the slots. A write ends when vLLM releases its handle, after
    NIXL reports it done or failed; as vLLM's own release of P's blocks, this
    takes NIXL's word that a done write has landed. A P that is alive but
    never answers leaves the slots reserved: D loses capacity, never data. A
    gone P is the exception: D frees without its word, on NIXL's report that
    the endpoint failed. That P's own endpoint to D has failed then too, or
    that P's process is gone, NIXL does not tell D. TT engines run one NIXL
    agent each; the acknowledgements name the agent, so several are allowed.

    State and threads:

    * ``_receives``, ``_recv_deadlines``, ``_cancelling``,
      ``_cancels_checked``, ``_load_errors`` and ``_waiting_registrations``
      (D), ``_waiting_stage`` (P) and ``slots`` belong to the model runner
      thread.
    * ``_unacked`` (D) belongs to the writer thread. ``_cancel_done`` carries
      settled cancellations from the writer thread to the model runner
      thread; ``_reg_send_inbox`` carries ``_WriterTask`` items to the writer
      thread. D's model runner thread does not touch a registration after
      handing it to vLLM; the writer thread marks it.
    * ``_to_stage`` carries matched P requests from the writer thread to the
      model runner thread.
    * ``_sends`` (P) holds a ``_Send`` per finished request, under ``_lock``.
      The model runner thread adds it when the request finishes, stages the
      request, marks it ended and removes it; the writer thread sets its
      registration when it matches the request and marks it submitting while
      vLLM submits its write. The model runner thread removes it, and frees
      its slots, once it has ended, is not being submitted and has no write
      handle in flight. A request finishes once, so work that finds no record
      belongs to a send that is over and is dropped.
    * ``_cancel_waits`` (P), and the ``_CANCELLED`` mark on P's registrations,
      are written by the writer thread under ``_lock``. The model runner
      thread reads them, under the same lock, only to skip work.
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

        A registration waits until enough slots are free. Under lane DP,
        ``metadata.tt_lane_of`` names each request's lane: block IDs are
        lane-local. Without it every request is on lane 0.
        """
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
            device = self._as_grouped_block_ids(reg["local_block_ids"])
            slots = self._take_slots(device)
            if slots is None:
                self._waiting_registrations.append((req_id, reg, lane))
                continue
            registrations[req_id] = {**reg, "local_block_ids": slots}
            self._receives[req_id] = _Receive(
                slots, device, lane, registrations[req_id]
            )
        with self._lock:
            for req_id in metadata.push_finished_blocks or {}:
                self._sends[req_id] = _Send(lane_of.get(req_id, 0))
        # The scheduler shares ``metadata``; hand vLLM a copy.
        metadata = copy.copy(metadata)
        metadata.push_registrations = registrations
        super().start_load_kv(metadata)
        self._stage_matched_sends()

    def save_kv_to_host(self, metadata: NixlConnectorMetadata) -> None:
        """Copy nothing per step: P stages a request once D's blocks are known."""

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta) -> None:
        """Copy a received request from its slots into the chips; free the slots.

        P has written the request, so D no longer awaits P's acknowledgement
        of its registration.
        """
        receive = self._receives.pop(req_id)
        self._recv_deadlines.pop(req_id, None)
        flat_slots = _flatten(receive.slots)
        self.copy_blocks(
            self.host_xfer_buffers,
            self.device_kv_caches,
            flat_slots,
            _flatten(receive.device),
            "h2d",
            lane=receive.lane,
        )
        self.slots.give_back(flat_slots)
        self._reg_send_inbox.put((req_id, _WriterTask(_RECEIVED)))

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
        """Report vLLM's results; free slots of done sends and cancelled receives.

        A D request whose KV has not arrived by its deadline is reported as a
        failed receive (vLLM's push path waits for it indefinitely). The slots
        of a failed receive go back once P has acknowledged its cancellation,
        or once P is gone (Notes). A P request reported sent keeps its slots
        while the writer thread is submitting its write or a write handle is in
        flight.
        """
        now = time.perf_counter()
        for req_id, deadline in list(self._recv_deadlines.items()):
            if now > deadline:
                del self._recv_deadlines[req_id]
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
        self._give_back_cancelled()
        results = super().get_transfer_results()
        for req_id in results.failed_recving:
            self._recv_deadlines.pop(req_id, None)
            receive = self._receives.pop(req_id, None)
            if receive is None:
                continue
            self._load_errors |= {
                receive.lane * self._pool_blocks + block
                for block in _flatten(receive.device)
            }
            self._cancelling[req_id] = _flatten(receive.slots)
            self._reg_send_inbox.put((req_id, _WriterTask(CANCEL, receive.reg)))
            self._push_writer_wake.set()
        self._end_sends(results.finished_sending)
        with self._lock:
            settle = bool(self._cancel_waits)
        if settle:
            # vLLM has just released the handles of writes that ended.
            self._reg_send_inbox.put(("", _WriterTask(_SETTLE)))
            self._push_writer_wake.set()
        self._stage_matched_sends()
        return results

    def _init_staging(
        self,
        num_slots: int,
        recv_timeout: float = 480.0,
        pool_blocks: int = 0,
    ) -> None:
        """Create the slot pool and the per-request staging state.

        Parameters
        ----------
        num_slots
            Host rows of the pool.
        recv_timeout
            Seconds D waits for a request's KV, from its first registration,
            before failing it.
        pool_blocks
            Blocks in each lane's KV pool; numbers the blocks of failed
            receives per lane.
        """
        self.slots = StagingSlots(num_slots)
        self._pool_blocks = pool_blocks
        self._recv_timeout = recv_timeout
        self._lock = threading.Lock()
        # D: lane-numbered blocks of failed receives, for vLLM to recompute.
        self._load_errors: set[int] = set()
        # D: deadline per request, set when D first tries to register it.
        self._recv_deadlines: dict[str, float] = {}
        # D: each request whose registration holds slots.
        self._receives: dict[str, _Receive] = {}
        # D: (request, registration, lane) waiting for free slots, oldest first.
        self._waiting_registrations: deque[tuple[str, dict[str, Any], int]] = deque()
        # D: slots of failed receives, until their cancellation settles.
        self._cancelling: dict[str, list[int]] = {}
        # D: (request, why) of cancellations the writer thread has settled;
        # why is a key of _FREED_WHY.
        self._cancel_done: queue.Queue[tuple[str, str]] = queue.Queue()
        # D: when this thread last queued _CHECK_CANCELS.
        self._cancels_checked = float("-inf")
        # D, writer thread: sent registrations that await P's acknowledgements.
        self._unacked: dict[str, _Unacked] = {}
        # P: matched requests (device blocks, D's registration) to stage.
        self._to_stage: queue.Queue[tuple[str, BlockIds, dict[str, Any]]] = (
            queue.Queue()
        )
        self._waiting_stage: deque[tuple[str, BlockIds, dict[str, Any]]] = deque()
        # P: each finished request until its send has ended and its slots are free.
        self._sends: dict[str, _Send] = {}
        # P: per request ID of D that D cancelled, D's CANCEL and the P requests
        # matched with it, whose writes must end before P acknowledges it.
        self._cancel_waits: dict[str, tuple[dict[str, Any], list[str]]] = {}

    def _send_registration_to_p(
        self, req_id: str, reg_data: dict[str, Any] | _WriterTask
    ) -> None:
        """Send D's registration, or run a ``_WriterTask``.

        Runs on the writer thread for each item of ``_reg_send_inbox``. A
        withdrawn registration, queued again by vLLM once its handshake with P
        is done, is not sent.
        """
        if not isinstance(reg_data, _WriterTask):
            if not reg_data.get(_WITHDRAWN):
                super()._send_registration_to_p(req_id, reg_data)
        elif reg_data.kind == CANCEL:
            self._cancel_registration(req_id, reg_data.reg)
        elif reg_data.kind == _RECEIVED:
            self._unacked.pop(req_id, None)
        elif reg_data.kind == _CHECK_CANCELS:
            self._check_cancels()
        elif reg_data.kind == _SETTLE:
            self._settle_cancels()
        else:
            self._send_tt_notif(reg_data.kind, reg_data.reg)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        """Send D's registration to P's agents and await a ``REG_ACK`` from each.

        Runs on the writer thread once the handshake with P is done. It
        replaces vLLM's method, which sends to the same agents, to learn which
        agents it may have reached. A registration that reached none fails its
        receive at once, as vLLM's method does for an engine without agents.
        """
        engine_id = reg_data["remote_engine_id"]
        agents = self._engine_agents(engine_id)
        notif = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg_data)
        refused = self._notify_agents(agents, notif, req_id, "push_reg_notif_failed")
        sent_to = [
            agent for agent in agents if refused.get(agent) not in _NOT_DELIVERED
        ]
        if not sent_to:
            logger.error(
                "The registration of %s reached no NIXL agent of engine %s; "
                "failing its receive.",
                req_id,
                engine_id,
            )
            if not agents:
                self.xfer_stats.record_failed_notification()
            self._failed_recv_reqs.put(req_id)
            return
        reg_data[_SENT_TO] = sent_to
        self._unacked[req_id] = _Unacked(reg_data)

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        """Record D's registration as vLLM does and acknowledge it; or read a TT
        notification.

        Runs on the writer thread for each notification with the ``PUSH_REG``
        prefix. Apart from the TT notifications and the ``REG_ACK``, this is
        vLLM's method, which decodes the registration inside and so leaves no
        hook after it.
        """
        if notif.startswith(TT_NOTIF_PREFIX):
            self._handle_tt_notif(notif)
            return
        try:
            reg = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        except Exception:
            logger.exception("Failed to decode PUSH_REG notification payload")
            return
        rid = reg.get("request_id") if isinstance(reg, dict) else None
        if not isinstance(rid, str):
            logger.warning("PUSH_REG notif missing request_id; dropping")
            return
        match = self._pop_matching_finished_blocks(rid)
        if match is None:
            self._pending_d_registrations[rid] = reg
        else:
            fin_id, blocks = match
            self._do_start_push_kv(fin_id, blocks, reg)
        self._send_tt_notif(REG_ACK, reg)

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
        A request without a ``_sends`` record, whose send has ended, or whose
        registration D cancelled is dropped; a dropped staged request keeps its
        slots until vLLM reports the send finished, at the end of P's lease.
        While vLLM's method submits the write, the request's record is marked
        submitting, so that ``get_transfer_results`` keeps its slots.
        """
        with self._lock:
            send = self._sends.get(request_id)
            if send is None or send.ended or registration_data.get(_CANCELLED):
                logger.debug(
                    "Dropping push of %s: its send ended or D cancelled it.",
                    request_id,
                )
                return
            staged = send.slots is not None
            send.submitting = staged
            if not staged:
                send.reg = registration_data
        if not staged:
            self._to_stage.put((request_id, local_block_ids, registration_data))
            return
        try:
            super()._do_start_push_kv(request_id, local_block_ids, registration_data)
        finally:
            with self._lock:
                send.submitting = False

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
                send = self._sends.get(req_id)
                cancelled = reg.get(_CANCELLED)
            # Only this thread ends a send or stages it.
            if send is None or send.ended or send.slots is not None or cancelled:
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
                lane=send.lane,
            )
            with self._lock:
                send.slots = flat_slots
            self._deferred_push_inbox.put((req_id, slots, reg))
            queued = True
        if queued:
            self._push_writer_wake.set()

    def _end_sends(self, finished: set[str]) -> None:
        """Mark P's sends vLLM reports finished; free those no write reads.

        Parameters
        ----------
        finished
            vLLM's ``finished_sending``: requests whose write completed or
            whose lease expired. A request may still have a write being
            submitted or in flight; its record stays until neither holds.
        """
        freed = []
        with self._lock:
            for req_id in finished:
                if (send := self._sends.get(req_id)) is not None:
                    send.ended = True
            # Under ``_lock``: the writer thread either finds the record gone
            # or has marked it submitting before this looks.
            with self._sending_transfers_lock:
                for req_id, send in list(self._sends.items()):
                    if (
                        send.ended
                        and not send.submitting
                        and not self._sending_transfers.get(req_id)
                    ):
                        del self._sends[req_id]
                        freed += send.slots or []
        self.slots.give_back(freed)

    def _give_back_cancelled(self) -> None:
        """Free the slots of D's receives whose cancellation has settled.

        While cancellations wait, also has the writer thread check them every
        ``_PROBE_INTERVAL_S``.
        """
        while True:
            try:
                req_id, why = self._cancel_done.get_nowait()
            except queue.Empty:
                break
            slots = self._cancelling.pop(req_id)
            self.slots.give_back(slots)
            logger.info(
                "Freed the %d staging slots of %s: %s.",
                len(slots),
                req_id,
                _FREED_WHY[why],
            )
        now = time.perf_counter()
        if self._cancelling and now - self._cancels_checked >= _PROBE_INTERVAL_S:
            self._cancels_checked = now
            self._reg_send_inbox.put(("", _WriterTask(_CHECK_CANCELS)))
            self._push_writer_wake.set()

    def _cancel_registration(self, req_id: str, reg: dict[str, Any]) -> None:
        """Withdraw D's registration of a failed receive, or cancel it at P.

        Runs on the writer thread. A registration that reached no P agent is
        never sent, and the receive's slots are free at once. A sent one is
        cancelled as ``_advance_cancel`` describes. Running again, as after a
        handshake, sends what is due.
        """
        sent_to = reg.get(_SENT_TO)
        if sent_to is None:
            reg[_WITHDRAWN] = True
            self._cancel_done.put((req_id, _WITHDRAWN))
            return
        unacked = self._unacked.setdefault(req_id, _Unacked(reg, reg_acks=set(sent_to)))
        unacked.cancelled = True
        self._advance_cancel(req_id, unacked)

    def _advance_cancel(self, req_id: str, unacked: _Unacked) -> None:
        """D: send ``CANCEL`` and settle the cancellation when they are due.

        Runs on the writer thread. ``CANCEL`` goes out once every P agent has
        sent ``REG_ACK`` or is disconnected. The receive goes back to the
        model runner thread once every agent has sent ``CANCEL_ACK`` or has
        been disconnected for ``_GONE_GRACE_S``.
        """
        sent_to = set(unacked.reg[_SENT_TO])
        now = time.perf_counter()
        if not unacked.cancel_sent and sent_to <= (
            unacked.reg_acks | unacked.disconnected.keys()
        ):
            refused = self._send_tt_notif(CANCEL, unacked.reg)
            if refused is None:
                # Queued again once the handshake with P is done.
                return
            unacked.cancel_sent = True
            self._note_disconnects(unacked, refused, now)
            num_slots = len(_flatten(unacked.reg["local_block_ids"]))
            if refused:
                logger.info(
                    "The cancel of %s did not reach every NIXL agent of P (%s); "
                    "its %d staging slots stay reserved until P acknowledges it "
                    "or is gone.",
                    req_id,
                    ", ".join(sorted(set(refused.values()))),
                    num_slots,
                )
            else:
                logger.info(
                    "Sent the cancel of %s to P; its %d staging slots stay "
                    "reserved until P acknowledges it.",
                    req_id,
                    num_slots,
                )
        gone = {
            agent
            for agent, since in unacked.disconnected.items()
            if now - since >= _GONE_GRACE_S
        }
        if sent_to <= unacked.cancel_acks | gone:
            del self._unacked[req_id]
            why = CANCEL_ACK if sent_to <= unacked.cancel_acks else _DISCONNECTED
            self._cancel_done.put((req_id, why))

    def _check_cancels(self) -> None:
        """D: probe the P agents that cancellations wait for; settle what is due.

        Runs on the writer thread, queued by the model runner thread every
        ``_PROBE_INTERVAL_S`` while a cancellation waits.
        """
        now = time.perf_counter()
        for req_id, unacked in list(self._unacked.items()):
            if not unacked.cancelled:
                continue
            waiting = (
                set(unacked.reg[_SENT_TO])
                - unacked.cancel_acks
                - unacked.disconnected.keys()
            )
            if waiting and now - unacked.probed >= _PROBE_INTERVAL_S:
                unacked.probed = now
                probe = TT_NOTIF_PREFIX + msgspec.msgpack.encode(
                    {"kind": PROBE, "request_id": req_id}
                )
                refused = self._notify_agents(waiting, probe, req_id, "tt_probe_failed")
                self._note_disconnects(unacked, refused, now)
            self._advance_cancel(req_id, unacked)

    def _handle_tt_notif(self, notif: bytes) -> None:
        """Act on a TT notification from the peer engine, on the writer thread."""
        try:
            message = msgspec.msgpack.decode(notif[len(TT_NOTIF_PREFIX) :])
            kind, req_id = message["kind"], message["request_id"]
        except Exception:
            logger.exception("Failed to decode a TT NIXL notification; dropping it")
            return
        if kind == REG_ACK:
            self._on_reg_ack(req_id, message.get("agent"))
        elif kind == CANCEL:
            self._on_cancel(message)
        elif kind == CANCEL_ACK:
            self._on_cancel_ack(req_id, message.get("agent"))
        elif kind != PROBE:
            logger.warning("Dropping TT NIXL notification %r for %s.", kind, req_id)

    def _on_reg_ack(self, req_id: str, agent: str | None) -> None:
        """D: record P's acknowledgement; cancel a failed receive once all are in."""
        unacked = self._unacked.get(req_id)
        if unacked is None or agent not in unacked.reg[_SENT_TO]:
            return
        unacked.reg_acks.add(agent)
        if unacked.cancelled:
            self._advance_cancel(req_id, unacked)
        elif unacked.reg_acks >= set(unacked.reg[_SENT_TO]):
            del self._unacked[req_id]

    def _on_cancel(self, message: dict[str, Any]) -> None:
        """P: write nothing more for D's cancelled registration.

        Drops a pending registration and marks a matched one, which every
        submission point checks; ``_settle_cancels`` then acknowledges the
        cancellation once the matched requests' writes have ended.
        """
        req_id = message["request_id"]
        self._pending_d_registrations.pop(req_id, None)
        writing = []
        with self._lock:
            for p_req_id, send in self._sends.items():
                if send.reg is not None and send.reg["request_id"] == req_id:
                    send.reg[_CANCELLED] = True
                    writing.append(p_req_id)
            self._cancel_waits[req_id] = (message, writing)
        self._settle_cancels()

    def _settle_cancels(self) -> None:
        """P: send ``CANCEL_ACK`` for each cancellation with no write in flight.

        Runs on the writer thread, the only thread that submits writes, so no
        write of a cancelled registration starts while this looks.
        """
        due = []
        with self._lock, self._sending_transfers_lock:
            for req_id, (message, writing) in list(self._cancel_waits.items()):
                if not any(self._sending_transfers.get(p) for p in writing):
                    del self._cancel_waits[req_id]
                    due.append(message)
        for message in due:
            self._send_tt_notif(CANCEL_ACK, message)

    def _on_cancel_ack(self, req_id: str, agent: str | None) -> None:
        """D: record P's acknowledgement; free the slots once all are in."""
        unacked = self._unacked.get(req_id)
        if (
            unacked is None
            or not unacked.cancelled
            or agent not in unacked.reg[_SENT_TO]
        ):
            return
        unacked.cancel_acks.add(agent)
        self._advance_cancel(req_id, unacked)

    def _send_tt_notif(self, kind: str, reg: dict[str, Any]) -> dict[str, str] | None:
        """Send the peer engine a TT notification about ``reg``, after any handshake.

        Runs on the writer thread. ``CANCEL`` goes from D to P, over the
        notification-only handshake vLLM uses for ``PUSH_REG``; ``REG_ACK``
        and ``CANCEL_ACK`` go from P to D, over the handshake P's writes use,
        and name P's agent. While a handshake runs, the notification is queued
        again for when it is done.

        Parameters
        ----------
        kind
            ``REG_ACK``, ``CANCEL`` or ``CANCEL_ACK``.
        reg
            The registration; for ``CANCEL_ACK``, D's ``CANCEL``. Either names
            D's request and the address of the peer engine.

        Returns
        -------
        dict[str, str] or None
            None while the handshake runs; else ``_notify_agents``'s refusals.
        """
        req_id = reg["request_id"]
        if kind == CANCEL:
            engine_id = reg["remote_engine_id"]
            handshake = self._ensure_handshake(
                engine_id,
                reg["remote_host"],
                reg["remote_port"],
                reg["remote_tp_size"],
                pp_size=reg.get("remote_pp_size", 1),
                notif_agents_only=True,
            )
            message = {key: reg[key] for key in _DECODE_ADDRESS}
        else:
            engine_id = reg["decode_engine_id"]
            handshake = self._ensure_handshake(
                engine_id, reg["decode_host"], reg["decode_port"], reg["decode_tp_size"]
            )
            message = {"agent": self.nixl_wrapper.name}
        if handshake is not None:
            handshake.add_done_callback(
                functools.partial(
                    self._queue_after_handshake, req_id, _WriterTask(kind, reg)
                )
            )
            return None
        message.update(kind=kind, request_id=req_id)
        notif = TT_NOTIF_PREFIX + msgspec.msgpack.encode(message)
        agents = self._engine_agents(engine_id)
        if not agents:
            logger.warning(
                "No NIXL agent of engine %s for the %s of %s; the request's "
                "staging slots stay reserved on D.",
                engine_id,
                kind,
                req_id,
            )
        return self._notify_agents(agents, notif, req_id, "tt_notif_failed")

    def _queue_after_handshake(
        self, req_id: str, task: _WriterTask, handshake: Future
    ) -> None:
        """Queue ``task`` for the writer thread once its handshake has succeeded.

        Runs on the thread that completes the handshake.
        """
        if (error := handshake.exception()) is not None:
            logger.warning(
                "Cannot send the %s of %s: the NIXL handshake failed (%s). The "
                "request's staging slots stay reserved on D.",
                task.kind,
                req_id,
                error,
            )
            return
        self._reg_send_inbox.put((req_id, task))
        self._push_writer_wake.set()

    def _engine_agents(self, engine_id: str) -> list[str]:
        """The names of ``engine_id``'s NIXL agents, as D or P knows them now."""
        # _remote_agents is mutated on other threads; snapshot under the lock.
        with self._handshake_lock:
            return list((self._remote_agents.get(engine_id) or {}).values())

    def _notify_agents(
        self, agents: Iterable[str], notif: bytes, req_id: str, failure_type: str
    ) -> dict[str, str]:
        """Send ``notif`` to each of ``agents``.

        Returns
        -------
        dict[str, str]
            The agents whose send raised, with the name of the exception's
            class; each is also logged under ``failure_type``. A name in
            ``_NOT_DELIVERED`` means NIXL posted nothing; any other failed
            send may still have been delivered.
        """
        refused = {}
        for agent in agents:
            try:
                self.nixl_wrapper.send_notif(agent, notif_msg=notif)
            except Exception as e:
                refused[agent] = type(e).__name__
                self._log_failure(
                    failure_type=failure_type, req_id=req_id, error=e, agent=agent
                )
                self.xfer_stats.record_failed_notification()
        return refused

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

    @staticmethod
    def _note_disconnects(
        unacked: _Unacked, refused: dict[str, str], now: float
    ) -> None:
        """Record the first time NIXL refused a send to each agent as disconnected."""
        for agent, error in refused.items():
            if error == _DISCONNECTED:
                unacked.disconnected.setdefault(agent, now)


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
