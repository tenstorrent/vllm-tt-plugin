# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A NIXL push writer without NIXL, for unit tests of its passes.

``push_writer`` gives a worker built with ``__new__`` the vLLM push worker
state that a writer pass reads, one decode engine ``D_ENGINE`` with P's TP and
one KV region, so that vLLM's descriptor IDs equal the block IDs, and a
``FakeNixl`` agent. Tests run ``_push_writer_pass`` on their own thread and
finish WRITEs by hand.
"""

import dataclasses
import queue
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import TPMapping
from vllm.v1.kv_cache_interface import FullAttentionSpec

D_ENGINE = "d-engine"
P_ENGINE = "p-engine"


@dataclasses.dataclass
class Write:
    """One NIXL WRITE that P submitted.

    Attributes
    ----------
    handle
        NIXL's transfer handle.
    local, remote
        P's and D's descriptor IDs; with one KV region, the block IDs.
    notif
        The notification D gets when the WRITE lands; ``b""`` for none.
    """

    handle: int
    local: list[int]
    remote: list[int]
    notif: bytes


class nixlRemoteDisconnectError(Exception):  # noqa: N801
    """Stands in for the exception NIXL's bindings raise when the UCX endpoint to
    the remote agent has failed; the TT worker recognizes it by its name."""


class FakeNixl:
    """The NIXL agent calls of P's push writer.

    Attributes
    ----------
    name
        The agent's name, as its peer knows it.
    writes
        Each WRITE in submission order.
    states
        NIXL state per handle: ``"PROC"`` until a test sets ``"DONE"`` or
        ``"ERR"``.
    notifs
        What ``get_new_notifs`` returns next, by agent name.
    sent
        Each notification sent with ``send_notif``, until a test delivers it.
    disconnected
        Every ``send_notif`` raises ``nixlRemoteDisconnectError`` and delivers
        nothing, as NIXL does once the endpoint to a dead peer has failed.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.writes: list[Write] = []
        self.states: dict[int, str] = {}
        self.notifs: dict[str, list[bytes]] = {}
        self.sent: list[bytes] = []
        self.disconnected = False

    def make_prepped_xfer(
        self, operation, local_side, local_ids, remote_side, remote_ids, notif_msg
    ) -> int:
        handle = len(self.writes) + 1
        local = [int(i) for i in local_ids]
        remote = [int(i) for i in remote_ids]
        self.writes.append(Write(handle, local, remote, notif_msg))
        self.states[handle] = "PROC"
        return handle

    def transfer(self, handle: int) -> None:
        pass

    def check_xfer_state(self, handle: int) -> str:
        return self.states[handle]

    def get_xfer_telemetry(self, handle: int) -> None:
        return None

    def release_xfer_handle(self, handle: int) -> None:
        pass

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        if self.disconnected:
            raise nixlRemoteDisconnectError(agent_name)
        self.sent.append(notif_msg)

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        notifs, self.notifs = self.notifs, {}
        return notifs

    def finish(self, state: str = "DONE") -> None:
        """Move every WRITE still in flight to ``state``."""
        for handle, current in self.states.items():
            if current == "PROC":
                self.states[handle] = state


def registration(d_blocks: list[int], request_id: str = "p") -> dict:
    """D's registration of ``d_blocks`` for P's request ``request_id``."""
    return {
        "request_id": request_id,
        "decode_engine_id": D_ENGINE,
        "decode_host": "d-host",
        "decode_port": 1,
        "decode_tp_size": 1,
        "local_block_ids": (list(d_blocks),),
    }


def push_reg_notif(reg: dict) -> bytes:
    """``reg`` as D's writer thread sends it to P."""
    return PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg)


def push_writer(worker):
    """Give ``worker`` what P's writer pass reads; return it.

    The P-to-D handshake is done, D runs one NIXL agent and D's TP equals
    P's. ``worker.nixl_wrapper`` is a ``FakeNixl``.
    """
    worker._push_finished_blocks = {}
    worker._pending_d_registrations = {}
    worker._reg_send_inbox = queue.Queue()
    worker._finished_blocks_inbox = queue.Queue()
    worker._evict_finished_inbox = queue.Queue()
    worker._pending_completion_notifs = queue.Queue()
    worker._deferred_push_inbox = queue.Queue()
    worker._push_writer_wake = threading.Event()
    worker._push_writer_stop = threading.Event()
    worker._sending_transfers = defaultdict(list)
    worker._sending_transfers_lock = threading.Lock()
    worker._send_failures = set()
    worker._engine_last_active = {}
    worker._ensure_handshake = lambda *args, **kwargs: None
    # NIXL's add_remote_agent returns the peer's name as bytes.
    worker._remote_agents = {D_ENGINE: {(0, 0): f"{D_ENGINE}-agent".encode()}}
    worker._handshake_lock = threading.Lock()
    worker._log_failure = lambda **kwargs: None
    worker.xfer_stats = MagicMock()
    worker.nixl_wrapper = FakeNixl(f"{P_ENGINE}-agent")
    worker.engine_id = P_ENGINE
    worker.world_size = 1
    worker.transfer_topo = SimpleNamespace(
        get_engine_info=lambda engine_id, pp_rank=0: SimpleNamespace(
            remote_tp_size=1,
            remote_dcp_size=1,
            remote_block_size=16,
            remote_physical_blocks_per_logical=1,
        ),
        tp_ratio=lambda remote_tp_size: 1,
        block_size_ratio=lambda remote_block_size: 1,
    )
    worker.tp_mappings = {
        D_ENGINE: TPMapping(
            source_ranks_per_group=((0,),),
            all_source_ranks=(0,),
            rank_to_attention_slot={0: 0},
            rank_offset_factor=0,
        )
    }
    worker.dst_num_blocks = {D_ENGINE: 10_000, P_ENGINE: 10_000}
    worker.dst_region_num_blocks = {D_ENGINE: [10_000], P_ENGINE: [10_000]}
    worker.dst_region_group_ids = {D_ENGINE: [0], P_ENGINE: [0]}
    worker.dst_uses_region_group_mapping = {D_ENGINE: False, P_ENGINE: False}
    worker.dst_xfer_side_handles = {D_ENGINE: {0: 5000}}
    worker.src_xfer_handles_by_block_size = {16: 2000}
    worker.region_group_ids = [0]
    worker._uses_region_group_mapping = False
    worker._transfer_layer_group_ids = ()
    worker._has_mamba = False
    worker.use_mla = False
    worker._is_csa_linear = False
    worker._group_spec_types = (FullAttentionSpec,)
    worker.num_regions = 1
    worker._physical_blocks_per_logical_kv_block = 1
    return worker
