# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTKVStoreConnector: share prompt prefixes between the TT engines of one host through ttserve-kvstore.

A vLLM KVConnector v1 (with hybrid-allocator support) for models whose KV lives in paged ttnn buffers, written for
MiMo-V2.6-Flash on a Blackhole Galaxy row (one engine = one 8-chip row, global layers + sliding-window layers).
Design and measurements: ttserve docs/lmcache.md.

What is stored, per prompt prefix:

* global ("full attention") group: every full block (128 tokens with the MiMo pair split). One store entry per block
  under a chained hash of the tokens up to its end; the entry holds that block's page of every global layer, K and
  V, on every chip, as raw device bytes (BFP8 tiles), laid out plane-major (plane = chip x layer x K/V) in the pool,
  so a run of consecutive slots is one byte range per plane.
* sliding-window groups: only a snapshot of the last window (4 blocks of 64 tokens: vLLM keeps 255 positions) at the
  boundaries where a prefill chunk ended on a block boundary. A prefix can be loaded up to such a boundary only: the
  SWA layers' KV before the window cannot be recomputed from the global KV alone.

Load (scheduler: ``get_num_new_matched_tokens`` finds the deepest boundary B with every global block up to B and a
snapshot at B, pins them; worker: ``start_load_kv`` copies them into the blocks vLLM allocated, before the step's
forward). Save (worker: ``wait_for_save``, after the step's forward was enqueued; the copies run after it in the
command-queue order): the full blocks a prefill chunk completed and, when the chunk ended on a block boundary, a
snapshot there. Both are synchronous (no async load), so a load is part of the step that schedules the request.

The keys chain an identity (model, tt-metal build, plugin, layout, KV dtype) into every hash and the pool names
carry it, so an engine of a different build never reads these pages.

Configuration (``--kv-transfer-config``)::

    {"kv_connector": "TTKVStoreConnector", "kv_connector_module_path": "vllm_tt_plugin.kv_connector",
     "kv_role": "kv_both",
     "kv_connector_extra_config": {"socket": "/tmp/ttserve-kvstore.sock", "global_gib": 48, "swa_gib": 8,
                                   "min_tokens": 1024, "load": true, "save": true}}
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from array import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_tt_plugin.kvstore_client import KVStoreClient, KVStoreError, Pool

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

BFP8_TILE_BYTES = 1088  # one 32x32 BFLOAT8_B tile
# Set by the worker-side connector once it holds the device buffers. The TT engine runs its scheduler and its one
# worker in the same process, so the scheduler side promises loads and asks for saves only once this is set: a
# scheduler that promised a load the worker cannot do would leave the blocks unwritten.
_WORKER_READY = threading.Event()
SWA_WINDOW_BLOCKS = 4  # 4 x 64 = 256 >= vLLM's window of 255 positions (128 + chunk alignment - 1)


def _git_head(path: str | None) -> str:
    """Commit of a checkout (read from .git, no subprocess); '' if unknown."""
    if not path:
        return ""
    try:
        git = os.path.join(path, ".git")
        if os.path.isfile(git):  # worktree: "gitdir: <path>"
            git = open(git).read().split(":", 1)[1].strip()
        head = open(os.path.join(git, "HEAD")).read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            p = os.path.join(git, ref)
            if os.path.exists(p):
                return open(p).read().strip()
            for line in open(os.path.join(git, "packed-refs")):
                if line.strip().endswith(ref):
                    return line.split()[0]
        return head
    except OSError:
        return ""


@dataclass
class Layout:
    """Where a block's pages live on the device and in a pool slot. Derived from the KV cache config on both sides;
    the worker checks it against the real device buffers."""

    num_devices: int
    global_group: int
    global_layers: list[int]
    global_block_tokens: int
    swa_groups: list[int]
    swa_layers: list[int]  # every SWA layer, sorted
    swa_block_tokens: int
    page_bytes: tuple[int, int]  # (K, V): one block of one layer on one chip
    pages_per_block: tuple[int, int]

    @property
    def global_slot_bytes(self) -> int:
        return self.num_devices * len(self.global_layers) * sum(self.page_bytes)

    @property
    def swa_slot_bytes(self) -> int:
        return self.num_devices * len(self.swa_layers) * SWA_WINDOW_BLOCKS * sum(self.page_bytes)

    @staticmethod
    def from_config(kv_cache_config: "KVCacheConfig", num_devices: int) -> "Layout":
        g_global, swa_groups, layers_by_group, bt = None, [], {}, {}
        head = None
        for g, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            layers = sorted(int(n.split(".")[2]) for n in group.layer_names)
            layers_by_group[g] = layers
            bt[g] = spec.block_size
            if isinstance(spec, SlidingWindowSpec):
                swa_groups.append(g)
            elif isinstance(spec, FullAttentionSpec):
                if g_global is not None:
                    raise ValueError("more than one full-attention KV cache group")
                g_global = g
            else:
                raise ValueError(f"unsupported KV cache spec {type(spec).__name__}")
            head = (spec.head_size, getattr(spec, "head_size_v", spec.head_size))
        if g_global is None or not swa_groups:
            raise ValueError("TTKVStoreConnector expects one full-attention group and sliding-window groups")
        swa_bt = {bt[g] for g in swa_groups}
        if len(swa_bt) != 1:
            raise ValueError(f"SWA groups with different block sizes {swa_bt}")
        # Every physical page holds 64 positions of one layer on one chip (MiMo's buffers are [NB, 1, 64, dim]).
        tiles = tuple(64 * d // 1024 for d in head)
        return Layout(
            num_devices=num_devices,
            global_group=g_global,
            global_layers=layers_by_group[g_global],
            global_block_tokens=bt[g_global],
            swa_groups=swa_groups,
            swa_layers=sorted(l for g in swa_groups for l in layers_by_group[g]),
            swa_block_tokens=swa_bt.pop(),
            page_bytes=tuple(t * BFP8_TILE_BYTES for t in tiles),
            pages_per_block=tiles,
        )

    def describe(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class LoadSpec:
    req_id: str
    num_tokens: int  # tokens loaded
    global_slots: list[int]
    global_blocks: list[int]
    swa_slot: int
    # per SWA layer position (index into Layout.swa_layers): [(window index w, block id)]
    swa_blocks: dict[int, list[tuple[int, int]]]
    leases: list[int]
    # generations of the pools the slots belong to (KVStoreClient checked them at acquire time)
    global_gen: str = ""
    swa_gen: str = ""


@dataclass
class SaveSpec:
    req_id: str
    global_keys: list[bytes]
    global_blocks: list[int]
    snapshot_key: bytes | None
    swa_blocks: dict[int, list[tuple[int, int]]] | None


@dataclass
class TTKVMeta(KVConnectorMetadata):
    loads: list[LoadSpec] = field(default_factory=list)
    saves: list[SaveSpec] = field(default_factory=list)


@dataclass
class _ReqState:
    tokens: list[int]
    keys: list[bytes]
    block_ids: list[list[int]]
    saved_blocks: int = 0


class TTKVStoreConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self.socket = extra.get("socket", "/tmp/ttserve-kvstore.sock")
        self.global_bytes = int(float(extra.get("global_gib", 48)) * (1 << 30))
        self.swa_bytes = int(float(extra.get("swa_gib", 8)) * (1 << 30))
        self.min_tokens = int(extra.get("min_tokens", 1024))
        self.do_load = bool(extra.get("load", True))
        self.do_save = bool(extra.get("save", True))
        self.layout = Layout.from_config(kv_cache_config, int(extra.get("num_devices", 8)))
        ident = {
            "model": vllm_config.model_config.model,
            "tt_metal": _git_head(os.environ.get("TT_METAL_HOME")),
            "plugin": _git_head(os.path.join(os.path.dirname(__file__), "..", "..")),
            "kv_dtype": "bfp8_b",
            "pair_split": os.environ.get("MIMO_KV_PAIR_SPLIT", "1"),
            "layout": self.layout.describe(),
            "salt": extra.get("identity_salt", ""),
        }
        self.identity = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).digest()
        tag = self.identity.hex()[:24]
        self.pool_global, self.pool_swa = f"kv-{tag}-g", f"kv-{tag}-s"
        self.client = KVStoreClient(self.socket)
        self._req: dict[str, _ReqState] = {}
        self._pending: dict[str, tuple] = {}  # req_id -> lookup result held until update_state_after_alloc
        self._loads: list[LoadSpec] = []
        self._kv = None  # worker: per layer [K, V] ttnn tensors
        logger.info(
            "TTKVStoreConnector (%s): identity %s, pools %s / %s, global slot %d B, SWA slot %d B, layout %s",
            role.name, tag, self.pool_global, self.pool_swa, self.layout.global_slot_bytes,
            self.layout.swa_slot_bytes, self.layout.describe(),
        )

    # ------------------------------------------------------------------ shared
    def _open_pools(self) -> bool:
        """Both pools open on the client's current connection (it forgets them when the connection closes or the store
        reports a stale generation, so this reopens after a store restart)."""
        if self.client.pool(self.pool_global) is not None and self.client.pool(self.pool_swa) is not None:
            return True
        try:
            self.client.open_pool(self.pool_global, self.layout.global_slot_bytes, self.global_bytes)
            self.client.open_pool(self.pool_swa, self.layout.swa_slot_bytes, self.swa_bytes)
            return True
        except KVStoreError as e:
            logger.warning("TTKVStoreConnector: store unavailable (%s); running without it", e)
            return False

    def _keys(self, tokens: list[int], cache_salt: str | None) -> list[bytes]:
        """Chained key of every full global block: key i covers tokens [0, (i + 1) * block)."""
        bt = self.layout.global_block_tokens
        raw = array("I", tokens).tobytes()  # 4 bytes per token id
        step = 4 * bt
        h = hashlib.sha256(self.identity + (cache_salt or "").encode()).digest()
        out = []
        for i in range(len(tokens) // bt):
            h = hashlib.sha256(h + raw[i * step : (i + 1) * step]).digest()
            out.append(h)
        return out

    def _snapshot_key(self, keys: list[bytes], nblocks: int) -> bytes:
        return hashlib.sha256(b"swa-window" + keys[nblocks - 1]).digest()

    def _swa_window(self, block_ids: list[list[int]], boundary_tokens: int) -> dict[int, list[tuple[int, int]]]:
        """{SWA layer position: [(window index, block id)]} of the blocks holding [boundary - 256, boundary)."""
        L = self.layout
        last = boundary_tokens // L.swa_block_tokens  # first block past the boundary
        first = last - SWA_WINDOW_BLOCKS
        out: dict[int, list[tuple[int, int]]] = {}
        pos = {layer: i for i, layer in enumerate(L.swa_layers)}
        for g in L.swa_groups:
            ids = block_ids[g]
            win = []
            for j in range(max(0, first), last):
                if j >= len(ids) or ids[j] == 0:
                    raise ValueError(f"SWA group {g}: block {j} of the window before {boundary_tokens} is not allocated")
                win.append((j - first, ids[j]))
            for layer in self._group_layers(g):
                out[pos[layer]] = win
        return out

    def _group_layers(self, g: int) -> list[int]:
        return sorted(int(n.split(".")[2]) for n in self._kv_cache_config.kv_cache_groups[g].layer_names)

    # ------------------------------------------------------------------ scheduler side
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        self._release_pending(request.request_id)
        tokens = request.prompt_token_ids or []
        if len(tokens) < self.min_tokens:
            return 0, False
        st = self._state(request)  # keys with the request's cache salt, for the saves too
        if not self.do_load or not _WORKER_READY.is_set() or not self._open_pools():
            return 0, False
        bt = self.layout.global_block_tokens
        if num_computed_tokens % bt:
            return 0, False
        nmax = (len(tokens) - 1) // bt  # the last prompt token is always computed
        local = num_computed_tokens // bt
        if nmax <= local:
            return 0, False
        try:
            n = self.client.lookup(self.pool_global, st.keys[:nmax])
            cands = list(range(n, local, -1))
            have = self.client.contains(self.pool_swa, [self._snapshot_key(st.keys, b) for b in cands])
            b = next((c for c, h in zip(cands, have) if h), 0)
            if b == 0 or (b - local) * bt < self.min_tokens:
                return 0, False
            lease_g = lease_s = 0
            lease_g, slots_g = self.client.acquire(self.pool_global, st.keys[local:b])
            lease_s, slots_s = self.client.acquire(self.pool_swa, [self._snapshot_key(st.keys, b)])
            gens = (self.client.pool(self.pool_global).gen, self.client.pool(self.pool_swa).gen)
            if len(slots_g) != b - local or len(slots_s) != 1:  # evicted in between
                self._release(lease_g)
                self._release(lease_s)
                return 0, False
        except KVStoreError as e:
            logger.warning("TTKVStoreConnector: lookup failed (%s)", e)
            self._release(lease_g)
            self._release(lease_s)
            return 0, False
        self._pending[request.request_id] = (local, b, slots_g, slots_s[0], [lease_g, lease_s], gens)
        return b * bt - num_computed_tokens, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        pend = self._pending.pop(request.request_id, None)
        if pend is None:
            return
        local, b, slots_g, slot_s, leases, gens = pend
        if num_external_tokens == 0:
            for lease in leases:
                self._release(lease)
            return
        bt = self.layout.global_block_tokens
        ids = [list(x) for x in blocks.get_block_ids()]
        self._loads.append(
            LoadSpec(
                req_id=request.request_id,
                num_tokens=(b - local) * bt,
                global_slots=slots_g,
                global_blocks=ids[self.layout.global_group][local:b],
                swa_slot=slot_s,
                swa_blocks=self._swa_window(ids, b * bt),
                leases=leases,
                global_gen=gens[0],
                swa_gen=gens[1],
            )
        )
        self._state(request).saved_blocks = b  # already in the store

    def _state(self, request: "Request") -> _ReqState:
        st = self._req.get(request.request_id)
        if st is None:
            tokens = list(request.prompt_token_ids or [])
            st = _ReqState(tokens=tokens, keys=self._keys(tokens, getattr(request, "cache_salt", None)), block_ids=[])
            self._req[request.request_id] = st
        return st

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        meta = TTKVMeta(loads=self._loads)
        self._loads = []
        bt = self.layout.global_block_tokens
        starts: dict[str, int] = {}
        for r in scheduler_output.scheduled_new_reqs:
            st = self._req.get(r.req_id)
            if st is None:
                tokens = list(r.prompt_token_ids or [])
                st = self._req[r.req_id] = _ReqState(tokens=tokens, keys=self._keys(tokens, None), block_ids=[])
            st.block_ids = [list(x) for x in r.block_ids]
            starts[r.req_id] = r.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            st = self._req.get(rid)
            if st is None:
                continue
            new = cached.new_block_ids[i]
            if rid in cached.resumed_req_ids:
                st.block_ids = [list(x) for x in new] if new else []
            elif new:
                for g, ext in enumerate(new):
                    st.block_ids[g].extend(ext)
            starts[rid] = cached.num_computed_tokens[i]
        if self.do_save and _WORKER_READY.is_set() and self._open_pools():
            for rid, n in scheduler_output.num_scheduled_tokens.items():
                st = self._req.get(rid)
                start = starts.get(rid)
                if st is None or start is None or len(st.tokens) < self.min_tokens or start >= len(st.tokens):
                    continue
                end = start + n
                if end > len(st.tokens):
                    continue  # not a pure prefill chunk
                nb = end // bt
                save = None
                if nb > st.saved_blocks:
                    ids = st.block_ids[self.layout.global_group]
                    save = SaveSpec(rid, st.keys[st.saved_blocks : nb], ids[st.saved_blocks : nb], None, None)
                    st.saved_blocks = nb
                if end % bt == 0 and nb >= 1:
                    try:
                        swa = self._swa_window(st.block_ids, end)
                    except ValueError as e:
                        logger.debug("no SWA snapshot at %d for %s: %s", end, rid, e)
                        swa = None
                    if swa is not None:
                        save = save or SaveSpec(rid, [], [], None, None)
                        save.snapshot_key = self._snapshot_key(st.keys, nb)
                        save.swa_blocks = swa
                if save is not None:
                    meta.saves.append(save)
        for rid in scheduler_output.finished_req_ids:
            self._req.pop(rid, None)
            self._release_pending(rid)
        return meta

    def _release(self, lease: int) -> None:
        if not lease:
            return
        try:
            self.client.release(lease)
        except KVStoreError:
            pass

    def _release_pending(self, req_id: str) -> None:
        pend = self._pending.pop(req_id, None)
        if pend is not None:
            for lease in pend[4]:
                self._release(lease)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        self._req.pop(request.request_id, None)
        self._release_pending(request.request_id)
        return False, None

    def request_finished_all_groups(self, request: "Request", block_ids: tuple[list[int], ...]):
        return self.request_finished(request, [])

    # ------------------------------------------------------------------ worker side
    def register_kv_caches(self, kv_caches: dict[str, Any]):
        pass  # the TT runner calls register_tt_kv_caches with its ttnn buffers

    def register_tt_kv_caches(self, lane_caches: list) -> None:
        """``lane_caches``: the runner's KV caches, one list per lane of [K, V] per layer (ttnn tensors)."""
        if len(lane_caches) != 1:
            logger.warning("TTKVStoreConnector supports one lane per engine; %d lanes: disabled", len(lane_caches))
            return
        kv = lane_caches[0]
        L = self.layout
        k0, v0 = kv[L.global_layers[0]]
        ndev = k0.device().get_num_devices()
        got = []
        for t in (k0, v0):
            got.append((t.buffer_page_size(), t.buffer_num_pages()))
        nblocks = self._kv_cache_config.num_blocks
        ok = ndev == L.num_devices and all(
            ps == BFP8_TILE_BYTES and npg == nblocks * ppb for (ps, npg), ppb in zip(got, L.pages_per_block)
        )
        if not ok:
            logger.warning(
                "TTKVStoreConnector: device buffers %s on %d devices do not match the expected layout %s: disabled",
                got, ndev, L.describe(),
            )
            return
        self._kv = kv
        self._open_pools()
        _WORKER_READY.set()
        logger.info("TTKVStoreConnector: registered %d layers on %d devices", len(kv), ndev)

    def _global_regions(self, pool: Pool, slots: list[int], blocks: list[int]) -> dict[tuple[int, int], list[tuple]]:
        """{(layer, kv): [(device, byte offset, bytes, host address)]}: runs of consecutive slots and block ids are one
        region per plane (pool layout: plane-major, plane = (device, global layer position, kv)). ``pool`` is the
        mapping of the generation the slots were handed out in."""
        L = self.layout
        runs, i = [], 0
        while i < len(slots):
            j = i + 1
            while j < len(slots) and slots[j] == slots[j - 1] + 1 and blocks[j] == blocks[j - 1] + 1:
                j += 1
            runs.append((slots[i], blocks[i], j - i))
            i = j
        out: dict[tuple[int, int], list[tuple]] = {}
        plane_off = 0
        for dev in range(L.num_devices):
            for layer in L.global_layers:
                for kv in (0, 1):
                    pb = L.page_bytes[kv]
                    regs = out.setdefault((layer, kv), [])
                    for slot, blk, n in runs:
                        regs.append((dev, blk * pb, n * pb, pool.base + plane_off + slot * pb))
                    plane_off += pool.num_slots * pb
        return out

    def _swa_regions(self, pool: Pool, slot: int, swa_blocks: dict[int, list[tuple[int, int]]]) -> dict[tuple[int, int], list]:
        """Snapshot slot layout: [swa layer position][kv][device][window index], each a page."""
        L = self.layout
        base = pool.base + slot * pool.slot_bytes
        out: dict[tuple[int, int], list[tuple]] = {}
        off = 0
        for p, layer in enumerate(L.swa_layers):
            win = swa_blocks.get(p, [])
            for kv in (0, 1):
                pb = L.page_bytes[kv]
                regs = out.setdefault((layer, kv), [])
                for dev in range(L.num_devices):
                    i = 0
                    while i < len(win):
                        j = i + 1
                        while j < len(win) and win[j][0] == win[j - 1][0] + 1 and win[j][1] == win[j - 1][1] + 1:
                            j += 1
                        w, blk = win[i]
                        regs.append((dev, blk * pb, (j - i) * pb, base + off + (dev * SWA_WINDOW_BLOCKS + w) * pb))
                        i = j
                off += L.num_devices * SWA_WINDOW_BLOCKS * pb
        return out

    def _copy(self, regions: dict[tuple[int, int], list], to_device: bool) -> int:
        import ttnn

        items = [(k, r) for k, r in regions.items() if r]
        nbytes = 0
        for i, ((layer, kv), regs) in enumerate(items):
            last = i == len(items) - 1
            t = self._kv[layer][kv]
            if to_device:
                ttnn.copy_host_to_device_regions(t, regs, blocking=last)
            else:
                ttnn.copy_device_to_host_regions(t, regs, blocking=last)
            nbytes += sum(r[2] for r in regs)
        return nbytes

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if self._kv is None or not isinstance(meta, TTKVMeta) or not meta.loads:
            return
        if not self._open_pools():
            # The scheduler pinned these entries and vLLM counts their tokens as computed: running without them would
            # produce wrong output, so fail the step instead.
            raise RuntimeError("TTKVStoreConnector: promised loads but the store's pools cannot be opened")
        for ld in meta.loads:
            t0 = time.perf_counter()
            try:
                pg = self._mapping(self.pool_global, ld.global_gen)
                ps = self._mapping(self.pool_swa, ld.swa_gen)
                nb = self._copy(self._global_regions(pg, ld.global_slots, ld.global_blocks), to_device=True)
                nb += self._copy(self._swa_regions(ps, ld.swa_slot, ld.swa_blocks), to_device=True)
            finally:
                for lease in ld.leases:
                    self._release(lease)
            logger.info(
                "TTKVStoreConnector: loaded %d tokens of %s (%.1f MB) in %.1f ms",
                ld.num_tokens, ld.req_id, nb / 1e6, (time.perf_counter() - t0) * 1e3,
            )

    def _mapping(self, name: str, gen: str) -> Pool:
        """The worker's mapping of the pool generation the scheduler acquired a load's slots in. If the worker's
        mapping is of another generation it reopens the pool; if the store's current generation is still another (the
        store restarted since the scheduler pinned the slots), the slots are gone: the step fails rather than copying
        some other entry's bytes."""
        pool = self.client.pool(name)
        if pool is None or pool.gen != gen:
            try:
                size = self.layout.global_slot_bytes if name == self.pool_global else self.layout.swa_slot_bytes
                cap = self.global_bytes if name == self.pool_global else self.swa_bytes
                pool = self.client.open_pool(name, size, cap)
            except KVStoreError as e:
                raise RuntimeError(f"TTKVStoreConnector: pool {name} of a pinned load is unavailable: {e}") from e
        if pool.gen != gen:
            raise RuntimeError(
                f"TTKVStoreConnector: pool {name} is generation {pool.gen}, the load was pinned in {gen} "
                "(the store restarted): refusing to copy"
            )
        return pool

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        pass

    def wait_for_save(self):
        meta = self._get_connector_metadata()
        if self._kv is None or not isinstance(meta, TTKVMeta) or not meta.saves or not self._open_pools():
            return
        for sv in meta.saves:
            t0 = time.perf_counter()
            res_g = res_s = 0
            nb = 0
            try:
                if sv.global_keys:
                    # reserve sends the mapping's generation and checks the answer's: the slots are in this mapping
                    res_g, slots = self.client.reserve(self.pool_global, sv.global_keys)
                    pool = self.client.pool(self.pool_global)
                    pick = [(s, b) for s, b in zip(slots, sv.global_blocks) if s is not None]
                    if pick:
                        nb += self._copy(self._global_regions(pool, [s for s, _ in pick], [b for _, b in pick]), False)
                if sv.snapshot_key is not None:
                    res_s, slots = self.client.reserve(self.pool_swa, [sv.snapshot_key])
                    pool = self.client.pool(self.pool_swa)
                    if slots and slots[0] is not None:
                        nb += self._copy(self._swa_regions(pool, slots[0], sv.swa_blocks), False)
                # the copies above were blocking: the bytes are in place, publish
                self.client.commit(res_g)
                self.client.commit(res_s)
            except Exception as e:  # never fail the step for a cache
                logger.warning("TTKVStoreConnector: save of %s failed: %s", sv.req_id, e)
                for r in (res_g, res_s):
                    try:
                        self.client.abort(r)
                    except KVStoreError:
                        pass
                continue
            if nb:
                logger.info(
                    "TTKVStoreConnector: saved %d blocks%s of %s (%.1f MB) in %.1f ms",
                    len(sv.global_keys), " + SWA snapshot" if sv.snapshot_key else "", sv.req_id, nb / 1e6,
                    (time.perf_counter() - t0) * 1e3,
                )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    def shutdown(self):
        self.client.close()
