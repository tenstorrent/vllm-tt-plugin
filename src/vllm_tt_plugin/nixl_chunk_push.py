# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM 0.33's NIXL push worker with a writer that sends prompt chunks early.

vLLM's push worker on P writes a request's KV to D in one NIXL WRITE, once the
request has finished and D's registration (``PUSH_REG``) has arrived.
``ChunkedPushWorker`` adds the writer side of vLLM's chunked push proposal,
vllm-project/vllm#67203 at 53921cad, which vLLM 0.33 does not have:

1. Each prefill step's completed prompt blocks reach the writer thread as a
   ``NixlPushChunkJob`` (``_queue_chunk_jobs``).
2. The writer WRITEs a job's blocks into D's blocks as soon as D's
   registration is known, without a completion notification. A job that
   arrives before the registration waits; once the registration arrives, the
   writer catches up on every waiting job.
3. The WRITE of the prompt's last blocks carries the completion notification,
   and the writer submits it only after every earlier WRITE of the request has
   completed. D therefore never decodes on partial KV.
4. Once the request has finished and the notifying WRITE has completed, the
   writer reports the request as sent. If a chunk was not written (a failed
   WRITE or handshake, or a D that the chunk path does not support), the
   writer pushes the whole request with vLLM's own push instead.

D is unchanged: chunk WRITEs without a notification are invisible to it.

Adapted from #67203: ``NixlPushChunkJob``, ``_ChunkRequestState``,
``_ChunkTask``, ``_ChunkRetirement`` and the writer's chunk methods keep its
names and logic. ``_push_writer_loop``, ``_xfer_blocks_for_req`` and
``_xfer_blocks`` are vLLM 0.33.0's with #67203's changes (the
``notif_msg`` and ``track_request`` keywords). This module differs from #67203
in these points:

* A job carries no ``torch.Event``: the caller queues a job only once NIXL can
  read its blocks.
* Retiring a preempted or aborted request does not block the caller
  (``_queue_chunk_retirement``). Jobs that arrive for a request with an
  unsettled retirement wait in ``_chunk_held``, and the request's hand-off
  waits too, so the new version of the request reaches D after its old WRITEs.
* Before recording a finished request, the writer takes the queued jobs, so a
  request whose last job and finish reach the writer in the same pass is still
  handed off through its chunks.
* The body of the writer loop is ``_push_writer_pass``.
* Upstream's ``assert`` statements raise ``RuntimeError``.

Principal entry points: ``ChunkedPushWorker`` and ``NixlPushChunkJob``.
"""

import dataclasses
import queue
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    RemoteMeta,
    ReqMeta,
    TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    _PUSH_WRITER_POLL_INTERVAL_MS,
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
)

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


@dataclasses.dataclass(frozen=True)
class NixlPushChunkJob:
    """Prompt blocks ``[start_block, end_block)`` of a P request that one
    prefill step completes.

    P's scheduler lists a step's jobs in its connector metadata
    (``push_chunk_jobs``); P's worker makes the blocks readable by NIXL and
    queues the jobs to its writer thread.

    Attributes
    ----------
    request_id
        P's request ID.
    block_ids
        The job's blocks, in prompt order: P's device block IDs in the
        scheduler's metadata, the IDs of the host rows that hold them once
        queued to the writer.
    start_block, end_block
        Positions of the blocks in the prompt.
    total_blocks
        Blocks of the whole prompt. The job that ends there is the last one;
        its WRITE carries the completion notification.
    """

    request_id: str
    block_ids: list[int]
    start_block: int
    end_block: int
    total_blocks: int


@dataclasses.dataclass
class _ChunkRequestState:
    """The writer's record of one version of a P request with chunk jobs.

    A preempted request gets a new record when it is computed again.

    Attributes
    ----------
    pending_jobs
        Jobs admitted and not finished: waiting, or with a WRITE in flight.
    failed
        A job of the request was not written to D.
    notified
        The WRITE of the prompt's last blocks, which carries the completion
        notification, was submitted.
    retired
        The record was dropped; its remaining jobs are skipped.
    reg_data
        D's registration, taken from the pending ones once a job finds it.
    finished_blocks
        P's block IDs, set once the request finishes; its hand-off then waits
        for its pending jobs.
    """

    pending_jobs: int = 0
    failed: bool = False
    notified: bool = False
    retired: bool = False
    reg_data: dict[str, Any] | None = None
    finished_blocks: BlockIds | None = None


@dataclasses.dataclass
class _ChunkTask:
    """An admitted job, waiting for submission or with a WRITE in flight.

    Attributes
    ----------
    job
        The job, with the host rows NIXL reads.
    state
        The record of the request version the job belongs to.
    handshake
        The P-to-D handshake the job waits for. Each job tries it once.
    """

    job: NixlPushChunkJob
    state: _ChunkRequestState
    handshake: Future | None = None


@dataclasses.dataclass
class _ChunkRetirement:
    """Preempted and aborted requests whose queued jobs are dropped.

    Attributes
    ----------
    preempted
        Requests computed again from the start. Each keeps D's registration
        and is written again from its first block, unless its notifying WRITE
        already went out.
    aborted
        Requests that end; the writer drops their records.
    done
        Set by the writer thread once none of their WRITEs is in flight.
    """

    preempted: set[str]
    aborted: set[str]
    done: threading.Event = dataclasses.field(default_factory=threading.Event)


class ChunkedPushWorker(NixlPushConnectorWorker):
    """vLLM's push worker whose writer also WRITEs prompt chunks before the finish.

    With chunked push off it behaves as vLLM's push worker.

    Notes
    -----
    Threads and state:

    * The caller of ``_queue_chunk_jobs`` and ``_queue_chunk_retirement``
      (vLLM's model runner thread) hands jobs and retirements to the writer
      thread through ``_chunk_job_inbox``, in step order.
    * ``_chunk_deferred``, ``_chunk_held``, ``_chunk_retirements``,
      ``_chunk_states``, ``_chunk_transfers``, ``_chunk_transfer_tasks`` and
      ``_next_chunk_transfer_id`` belong to the writer thread
      (``nixl-push-writer``).
    * A retirement's ``done`` event is set by the writer thread and read by the
      caller.
    """

    def __init__(self, vllm_config, engine_id: str, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        transfer = vllm_config.kv_transfer_config
        self._init_chunk_push(
            transfer.get_from_extra_config("enable_chunked_push", False) is True
            and transfer.kv_role == "kv_producer"
        )

    def _init_chunk_push(self, chunked: bool) -> None:
        """Create the chunk writer state.

        Parameters
        ----------
        chunked
            Whether P writes prompt chunks; ``enable_chunked_push`` on a
            producer.
        """
        self._chunked_push = chunked
        self._chunk_job_inbox: queue.Queue[NixlPushChunkJob | _ChunkRetirement] = (
            queue.Queue()
        )
        # Admitted jobs not yet submitted, in arrival order.
        self._chunk_deferred: list[_ChunkTask] = []
        # Jobs of requests whose retirement has not settled, in arrival order.
        self._chunk_held: list[NixlPushChunkJob] = []
        self._chunk_retirements: list[_ChunkRetirement] = []
        self._chunk_states: dict[str, _ChunkRequestState] = {}
        # In-flight WRITE handles and the task of each chunk WRITE.
        self._chunk_transfers: dict[str, list[TransferHandle]] = {}
        self._chunk_transfer_tasks: dict[str, _ChunkTask] = {}
        self._next_chunk_transfer_id = 0

    def shutdown(self) -> None:
        """Stop the writer thread and release chunk WRITE handles, then vLLM's."""
        self._push_writer_stop.set()
        self._push_writer_wake.set()
        if self._push_writer_thread is not None:
            self._push_writer_thread.join(timeout=2)
            self._push_writer_thread = None
        for handles in self._chunk_transfers.values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._chunk_transfers.clear()
        self._chunk_transfer_tasks.clear()
        super().shutdown()

    def _queue_chunk_jobs(self, jobs: Iterable[NixlPushChunkJob]) -> None:
        """Hand jobs to the writer thread.

        Parameters
        ----------
        jobs
            Jobs whose ``block_ids`` NIXL can read now, in step order.
        """
        queued = False
        for job in jobs:
            self._chunk_job_inbox.put(job)
            queued = True
        if queued:
            self._push_writer_wake.set()

    def _queue_chunk_retirement(
        self, preempted: set[str], aborted: set[str]
    ) -> _ChunkRetirement:
        """Have the writer drop the queued jobs of requests; do not wait.

        Returns
        -------
        _ChunkRetirement
            Its ``done`` event is set once none of the requests' WRITEs is in
            flight. Until then, NIXL may still read the requests' blocks.
        """
        retirement = _ChunkRetirement(set(preempted), set(aborted))
        self._chunk_job_inbox.put(retirement)
        self._push_writer_wake.set()
        return retirement

    def _push_writer_loop(self) -> None:
        """Run writer passes until shutdown.

        Between passes the writer polls every
        ``_PUSH_WRITER_POLL_INTERVAL_MS`` while a finished request waits for
        D, or a chunk job waits, a chunk WRITE is in flight or a retirement is
        unsettled; otherwise it waits for a wake.
        """
        sleep_s = _PUSH_WRITER_POLL_INTERVAL_MS / 1000.0
        while not self._push_writer_stop.is_set():
            try:
                self._push_writer_pass()
            except Exception:
                logger.exception("nixl-push-writer error; continuing")
            if self._push_finished_blocks or (
                self._chunked_push
                and (
                    self._chunk_deferred
                    or self._chunk_transfers
                    or self._chunk_retirements
                )
            ):
                self._push_writer_stop.wait(timeout=sleep_s)
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _push_writer_pass(self) -> None:
        """One pass of the writer thread: vLLM's steps, then the chunk steps."""
        # 1. D registrations to send.
        while True:
            try:
                rid, rd = self._reg_send_inbox.get_nowait()
            except queue.Empty:
                break
            self._send_registration_to_p(rid, rd)

        # 2. Deferred P->D pushes whose handshake just completed.
        while True:
            try:
                rid, blocks, rd = self._deferred_push_inbox.get_nowait()
            except queue.Empty:
                break
            self._do_start_push_kv(rid, blocks, rd)

        # 3. P-side finished blocks; match against pending registrations.
        while True:
            try:
                rid, blocks = self._finished_blocks_inbox.get_nowait()
            except queue.Empty:
                break
            if self._chunked_push:
                # The request's last job was queued before its finish; without
                # it, a request with one job has no record yet.
                self._take_chunk_inbox()
                if self._set_chunk_finished(rid, blocks):
                    continue
            matched = self._pop_matching_registration(rid)
            if matched is not None:
                self._do_start_push_kv(rid, blocks, matched)
            else:
                self._push_finished_blocks[rid] = blocks

        # 3b. Requests sent or whose lease expired.
        while True:
            try:
                rid = self._evict_finished_inbox.get_nowait()
            except queue.Empty:
                break
            self._push_finished_blocks.pop(rid, None)
            self._pending_d_registrations.pop(rid, None)
            if self._chunked_push:
                self._drop_chunk_request(rid)

        # 4. NIXL notifications: route PUSH_REG; forward the rest.
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                    self._handle_push_reg_notif(notif)
                else:
                    self._pending_completion_notifs.put(notif)

        if self._chunked_push:
            self._process_chunk_jobs()
            self._poll_chunk_transfers()
            self._process_chunk_handoffs()

    def _take_chunk_inbox(self) -> None:
        """Admit queued jobs and record queued retirements, in queue order.

        A job of a request with an unsettled retirement is held until the
        retirement settles; a retirement drops the held jobs of its requests,
        which belong to the version it retires.
        """
        while True:
            try:
                item = self._chunk_job_inbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _ChunkRetirement):
                retired = item.preempted | item.aborted
                self._chunk_held = [
                    job for job in self._chunk_held if job.request_id not in retired
                ]
                for req_id in item.aborted:
                    self._drop_chunk_request(req_id)
                for req_id in item.preempted:
                    state = self._chunk_states.get(req_id)
                    if state is not None and not state.notified:
                        self._restart_chunk_request(req_id)
                self._chunk_retirements.append(item)
            elif self._retiring(item.request_id):
                self._chunk_held.append(item)
            else:
                self._admit_chunk_job(item)

    def _admit_chunk_job(self, job: NixlPushChunkJob) -> None:
        """Count the job on its request's record and defer it for submission."""
        state = self._chunk_states.setdefault(job.request_id, _ChunkRequestState())
        if state.notified:
            # Handed off before a preemption; the recompute is not sent.
            return
        state.pending_jobs += 1
        self._chunk_deferred.append(_ChunkTask(job, state))

    def _retiring(self, req_id: str) -> bool:
        """Whether a retirement of ``req_id`` has not settled."""
        return any(
            req_id in retirement.preempted or req_id in retirement.aborted
            for retirement in self._chunk_retirements
        )

    def _process_chunk_jobs(self) -> None:
        """Submit each deferred job whose request and peer are ready."""
        self._take_chunk_inbox()
        # Taken out first, so that a failure midway cannot submit a job again;
        # the jobs it drops leave their request to the lease.
        tasks, self._chunk_deferred = self._chunk_deferred, []
        for task in tasks:
            job, state = task.job, task.state
            if state.retired:
                continue
            # D decodes once the completion notification arrives, so it must
            # follow all of the request's data. It rides on the WRITE of the
            # last block, submitted once every earlier WRITE of the request
            # has completed; if one failed, the request is pushed whole when
            # it finishes.
            last = job.end_block == job.total_blocks
            if last and state.pending_jobs > 1:
                self._chunk_deferred.append(task)
                continue

            if state.reg_data is None:
                state.reg_data = self._pop_matching_registration(job.request_id)
            reg_data = state.reg_data
            if reg_data is None:
                self._chunk_deferred.append(task)
                continue
            handshake = task.handshake or self._ensure_handshake(
                reg_data["decode_engine_id"],
                reg_data["decode_host"],
                reg_data["decode_port"],
                reg_data["decode_tp_size"],
            )
            if handshake is not None and not handshake.done():
                task.handshake = handshake
                self._chunk_deferred.append(task)
                continue
            if handshake is not None and handshake.exception() is not None:
                # Not retried, as for a whole-request push.
                self._finish_chunk_job(task, failed=True)
                continue
            if not self._chunk_path_supported(job, reg_data):
                self._finish_chunk_job(task, failed=True)
                continue

            handles: list[TransferHandle] = []
            failed = False
            notify = last and not state.failed
            try:
                handles = self._submit_chunk_job(job, reg_data, notify=notify)
            except Exception:
                logger.exception(
                    "NIXL chunk push submission failed for %s", job.request_id
                )
                failed = True
            with self._sending_transfers_lock:
                if job.request_id in self._send_failures:
                    self._send_failures.discard(job.request_id)
                    failed = True
            if handles:
                state.failed |= failed
                state.notified = notify
                transfer_id = f"{job.request_id}:{self._next_chunk_transfer_id}"
                self._next_chunk_transfer_id += 1
                self._chunk_transfers[transfer_id] = handles
                self._chunk_transfer_tasks[transfer_id] = task
            else:
                self._finish_chunk_job(task, failed=failed)

    def _chunk_path_supported(
        self, job: NixlPushChunkJob, reg_data: dict[str, Any]
    ) -> bool:
        """Whether D's engine and registration let the job be written alone.

        Raises
        ------
        RuntimeError
            Before any handshake has set up the transfer topology.
        """
        remote = self._as_grouped_block_ids(reg_data["local_block_ids"])
        engine_id = reg_data["decode_engine_id"]
        if self.transfer_topo is None:
            raise RuntimeError(
                f"No transfer topology for decode engine {engine_id} after its "
                "handshake."
            )
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        if (
            remote_info.remote_tp_size != self.world_size
            or remote_info.remote_dcp_size != 1
        ):
            logger.warning_once(
                "NIXL chunked push needs a decode engine with the prefill TP "
                "size %d and DCP=1, but decode engine %s has TP=%d, DCP=%d. "
                "Falling back to whole-request push.",
                self.world_size,
                engine_id,
                remote_info.remote_tp_size,
                remote_info.remote_dcp_size,
            )
            return False
        # The slices in _chunk_block_ranges need D's blocks to fit the prompt.
        return job.total_blocks >= len(remote[0])

    def _submit_chunk_job(
        self, job: NixlPushChunkJob, reg_data: dict[str, Any], *, notify: bool
    ) -> list[TransferHandle]:
        """WRITE the job's blocks that fall in D's registered blocks.

        Parameters
        ----------
        notify
            Send the request's completion notification with the WRITE.

        Returns
        -------
        list[TransferHandle]
            The WRITE's handles; empty when no block of the job is in D's
            registration.
        """
        # Keep the engine alive while it receives pushes, as _do_start_push_kv
        # does.
        self._engine_last_active[reg_data["decode_engine_id"]] = time.perf_counter()
        remote_groups = self._as_grouped_block_ids(reg_data["local_block_ids"])
        ranges = self._chunk_block_ranges(job, len(remote_groups[0]))
        if ranges is None:
            return []
        local_range, remote_range = ranges
        local = (job.block_ids[local_range],)
        remote = (list(remote_groups[0][remote_range]),)
        physical_local = self._logical_to_kernel_block_ids(
            local, self._physical_blocks_per_logical_kv_block
        )
        push_meta = ReqMeta(
            local_block_ids=local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=remote,
                host="",
                port=0,
                engine_id=reg_data["decode_engine_id"],
                request_id=reg_data["request_id"],
            ),
        )
        return self._xfer_blocks_for_req(
            req_id=job.request_id,
            meta=push_meta,
            notif_msg=None if notify else b"",
            track_request=False,
        )

    def _finish_chunk_job(self, task: _ChunkTask, failed: bool = False) -> None:
        """Count a job as done on its request's record, failed or not."""
        task.state.pending_jobs -= 1
        task.state.failed |= failed

    def _set_chunk_finished(self, req_id: str, block_ids: BlockIds) -> bool:
        """Record that a request with chunk jobs finished.

        Returns
        -------
        bool
            False when the request has no record: it is pushed whole.
        """
        state = self._chunk_states.get(req_id)
        if state is None:
            return False
        state.finished_blocks = block_ids
        return True

    def _process_chunk_handoffs(self) -> None:
        """Hand finished requests off to D once their chunk WRITEs are done.

        A request with an unsettled retirement waits: its old WRITEs may still
        land in D's blocks.
        """
        finished = [
            (req_id, state)
            for req_id, state in self._chunk_states.items()
            if state.finished_blocks is not None
        ]
        for req_id, state in finished:
            if self._retiring(req_id):
                continue
            if state.reg_data is None:
                state.reg_data = self._pop_matching_registration(req_id)
            if state.reg_data is not None and state.pending_jobs > 0:
                continue
            self._drop_chunk_request(req_id)
            blocks = state.finished_blocks
            if state.notified and not state.failed:
                # The last chunk's WRITE carried the completion notification:
                # report the request like a whole-request push whose WRITEs
                # completed.
                with self._sending_transfers_lock:
                    self._sending_transfers[req_id] = []
            elif state.reg_data is None:
                # D has not registered yet: push the whole request on arrival.
                self._push_finished_blocks[req_id] = blocks
            else:
                # A chunk was not written. A failed notifying WRITE sent no
                # notification, so pushing the whole request is safe.
                self._do_start_push_kv(req_id, blocks, state.reg_data)

    def _poll_chunk_transfers(self) -> None:
        """Finish completed chunk WRITEs, settle retirements, release held jobs."""
        done, failed = self._pop_done_transfers(self._chunk_transfers)
        # A failed WRITE may still have sibling handles in flight.
        for transfer_id in failed:
            self._chunk_transfer_tasks[transfer_id].state.failed = True
        for transfer_id in done:
            self._finish_chunk_job(self._chunk_transfer_tasks.pop(transfer_id))
        for retirement in list(self._chunk_retirements):
            retired = retirement.preempted | retirement.aborted
            if any(
                task.job.request_id in retired
                for task in self._chunk_transfer_tasks.values()
            ):
                continue
            for req_id in retirement.preempted:
                state = self._chunk_states.get(req_id)
                # A notifying WRITE that failed did not hand the request off.
                if state is not None and state.notified and state.failed:
                    self._restart_chunk_request(req_id)
            self._chunk_retirements.remove(retirement)
            retirement.done.set()
        held, self._chunk_held = self._chunk_held, []
        for job in held:
            if self._retiring(job.request_id):
                self._chunk_held.append(job)
            else:
                self._admit_chunk_job(job)

    def _drop_chunk_request(self, req_id: str) -> None:
        """Drop the request's record; its remaining jobs are skipped."""
        if (state := self._chunk_states.pop(req_id, None)) is not None:
            state.retired = True

    def _restart_chunk_request(self, req_id: str) -> None:
        """Start a new record for the request that keeps D's registration.

        The recomputed prompt is then written again into the same D blocks.
        """
        reg_data = self._chunk_states[req_id].reg_data
        self._drop_chunk_request(req_id)
        self._chunk_states[req_id] = _ChunkRequestState(reg_data=reg_data)

    def _xfer_blocks_for_req(
        self,
        req_id: str,
        meta: ReqMeta,
        *,
        notif_msg: bytes | None = None,
        track_request: bool = True,
    ) -> list[TransferHandle]:
        """Issue WRITE transfers to one or more remote TP ranks.

        Parameters
        ----------
        notif_msg
            The notification D gets when a WRITE lands. None sends vLLM's
            completion notification ``"{D's request ID}:{world size}"``;
            ``b""`` sends none.
        track_request
            Add the handles to ``_sending_transfers``, so that their
            completion reports the request as sent. A chunk WRITE does not.

        Returns
        -------
        list[TransferHandle]
            The handles of the WRITEs submitted.

        Raises
        ------
        RuntimeError
            Without a remote or a transfer topology, or when an MLA model with
            heterogeneous TP writes from more than one rank.
        NotImplementedError
            When P's and D's cache-group layouts differ.
        """
        if meta.remote is None or self.transfer_topo is None:
            raise RuntimeError(
                f"A push of {req_id} needs a remote and a transfer topology; got "
                f"{meta.remote} and {self.transfer_topo}."
            )
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        # Expand D's logical IDs using the ratio learned during the NIXL
        # handshake. ``meta`` is built for this push, so mutating it is safe.
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        local_region_groups = self.region_group_ids
        remote_region_groups = self.dst_region_group_ids[engine_id]
        groups_differ = local_region_groups != remote_region_groups
        if groups_differ and not self._transfer_layer_group_ids:
            raise NotImplementedError(
                "NixlPushConnector does not support different producer and "
                "consumer cache-group layouts"
            )

        # MLA latent is replicated across D's TP ranks: the tp-mapping
        # collapses it to one rank (fine for reads), but push must WRITE every
        # D rank or the rest decode stale KV. For hybrid MLA+SSM the sharded
        # SSM state already targets every covered D rank, so only the
        # attention groups need widening; pure MLA writes to all handshaked
        # ranks (only the dst differs per rank).
        replicate_attn = self.use_mla and tp_ratio < 0
        if replicate_attn and not self._has_mamba:
            if len(plan.all_source_ranks) != 1:
                raise RuntimeError(
                    f"An MLA push expects one source rank; got {plan.all_source_ranks}."
                )
            write_ranks = sorted(self.dst_xfer_side_handles[engine_id])
        else:
            write_ranks = list(plan.all_source_ranks)

        num_groups = len(local_block_ids)

        def group_ids(block_ids: BlockIds, rank: int) -> BlockIds:
            return [
                list(block_ids[g])
                if (self._is_csa_linear and tp_ratio < 0)
                or (replicate_attn and _is_attention_spec(self._group_spec_types[g]))
                or rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ]

        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=group_ids(local_block_ids, rank),
                remote_block_ids=group_ids(remote_block_ids, rank),
            )
            for rank in write_ranks
        ]

        handles: list[TransferHandle] = []
        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _xfer_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            if tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1):
                # Multiple targets: write each rank its chunk of local memory.
                # Hybrid MLA+SSM also lands here: its split handles replicate
                # the attention descriptors and chunk only the SSM state.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                notif_msg=notif_msg,
            )
            if handle is not None:
                handles.append(handle)

        # Publish all the request's WRITE handles in one locked update: a
        # partial set would let ``_pop_done_transfers`` finish the request
        # early, then double-report it as the remaining writes land.
        if handles and track_request:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)
        return handles

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        notif_msg: bytes | None = None,
    ) -> TransferHandle | None:
        """Post a WRITE point-to-point transfer request.

        Parameters
        ----------
        notif_msg
            As for ``_xfer_blocks_for_req``.

        Returns
        -------
        TransferHandle or None
            The in-flight transfer handle, so that the caller can track all of
            a request's handles at once, or None if nothing was submitted.

        Raises
        ------
        RuntimeError
            Without a transfer topology, or when P's and D's block or
            descriptor counts differ.
        """
        if self.transfer_topo is None:
            raise RuntimeError(f"No transfer topology for a push of {request_id}.")
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )

        if notif_msg is None:
            notif_msg = f"{remote_request_id}:{self.world_size}".encode()

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return None

        # Prefix caching: D allocated only uncached blocks, so on a partial hit
        # it sends fewer than P's. End-trim P's blocks to that same suffix so
        # we WRITE only the uncomputed tail into D's slots. Runs on kernel
        # ids, post-expansion.
        remote_block_ids, local_block_ids = self._apply_prefix_caching(
            decode_block_ids=remote_block_ids,
            prefill_block_ids=local_block_ids,
            decode_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
            prefill_physical_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        if len(local_block_ids) != len(remote_block_ids):
            raise RuntimeError(
                f"push group-count mismatch for {request_id}: "
                f"{len(local_block_ids)} local vs {len(remote_block_ids)} remote "
                "groups"
            )
        for i in range(len(local_block_ids)):
            if len(local_block_ids[i]) != len(remote_block_ids[i]):
                raise RuntimeError(
                    f"push block-count mismatch for {request_id} group {i}: "
                    f"{len(local_block_ids[i])} local vs "
                    f"{len(remote_block_ids[i])} remote blocks"
                )

        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
            region_num_blocks=self.dst_region_num_blocks[dst_engine_id],
            region_group_ids=self.dst_region_group_ids[dst_engine_id],
            uses_region_group_mapping=self.dst_uses_region_group_mapping[dst_engine_id],
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=self.dst_region_num_blocks[self.engine_id],
            region_group_ids=self.region_group_ids,
            uses_region_group_mapping=self._uses_region_group_mapping,
        )
        if len(local_block_descs_ids) != len(remote_block_descs_ids):
            raise RuntimeError(
                f"push descriptor-count mismatch for {request_id}: "
                f"{len(local_block_descs_ids)} local vs "
                f"{len(remote_block_descs_ids)} remote"
            )

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_msg,
            )
            self.nixl_wrapper.transfer(handle)
            # Caller tracks the handle (atomically with the request's other
            # writes) so P can free blocks once all of them are done.
            return handle
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Push WRITE submission failed; releasing handle",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            # On the P side this WRITE failure is purely outbound; there is no
            # ``_recving_metadata`` entry to invalidate, so release the handle
            # and let the engine reschedule via the lease / watchdog.
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            if not self._handle_failed_transfer(request_id, handle):
                return handle
            return None

    @staticmethod
    def _chunk_block_ranges(
        job: NixlPushChunkJob, num_remote_blocks: int
    ) -> tuple[slice, slice] | None:
        """Map a job onto D's registered blocks, the last of the prompt.

        Returns
        -------
        tuple[slice, slice] or None
            Slices of the job's ``block_ids`` and of D's block list, or None
            when the job ends before D's first block.

        Examples
        --------
        D registered the last 3 blocks of a 5-block prompt, so a job of
        blocks 1 and 2 writes block 2 into D's first block:

        >>> job = NixlPushChunkJob("p", [11, 12], 1, 3, 5)
        >>> ChunkedPushWorker._chunk_block_ranges(job, 3)
        (slice(1, 2, None), slice(0, 1, None))
        """
        remote_start = job.total_blocks - num_remote_blocks
        start = max(job.start_block, remote_start)
        if start >= job.end_block:
            return None
        return (
            slice(start - job.start_block, job.end_block - job.start_block),
            slice(start - remote_start, job.end_block - remote_start),
        )
