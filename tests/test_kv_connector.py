# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests of TTKVStoreConnector: layout, keys, the scheduler side's load / save planning against an
in-memory store, and the DMA regions it builds. No device and no ttserve-kvstore process."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

import vllm_tt_plugin.kv_connector as kvc
from vllm_tt_plugin.kvstore_client import Pool

PATTERN = [0, 1, 1, 1, 1] + [0, 1, 1, 1, 1, 1] * 7 + [0]  # MiMo: 9 global, 39 SWA layers
GLOBAL = [i for i, p in enumerate(PATTERN) if p == 0]
SWA = [i for i, p in enumerate(PATTERN) if p == 1]


def mimo_kv_config(num_blocks=1000):
    full = FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=192, head_size_v=128, dtype=torch.bfloat16)
    swa = SlidingWindowSpec(block_size=64, num_kv_heads=8, head_size=192, head_size_v=128, dtype=torch.bfloat16,
                            sliding_window=255)
    groups = [KVCacheGroupSpec([f"model.layers.{i}.self_attn" for i in GLOBAL], full)]
    for g in range(5):
        groups.append(KVCacheGroupSpec([f"model.layers.{i}.self_attn" for i in SWA[g * 9 : (g + 1) * 9]], swa))
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


class FakeStore:
    """The store's semantics in memory (prefix lookup, contains, pinned acquire, reserve / commit)."""

    def __init__(self):
        self.pools = {}
        self.entries = {}  # (pool, key) -> slot
        self.next_slot = {}
        self.leases = {}
        self.reservations = {}
        self._pools = {}

    def open_pool(self, name, slot_bytes, capacity_bytes):
        if name not in self._pools:
            self._pools[name] = Pool(name, f"/dev/shm/{name}", 64, slot_bytes, None, 1 << 40)
            self.next_slot[name] = 0
        return self._pools[name]

    def lookup(self, pool, keys):
        n = 0
        for k in keys:
            if (pool, k) not in self.entries:
                break
            n += 1
        return n

    def contains(self, pool, keys):
        return [(pool, k) in self.entries for k in keys]

    def acquire(self, pool, keys):
        n = self.lookup(pool, keys)
        lease = len(self.leases) + 1
        self.leases[lease] = keys[:n]
        return lease, [self.entries[(pool, k)] for k in keys[:n]]

    def release(self, lease):
        self.leases.pop(lease, None)

    def put(self, pool, keys):
        for k in keys:
            self.entries[(pool, k)] = self.next_slot[pool]
            self.next_slot[pool] += 1


@pytest.fixture
def connector(monkeypatch):
    def make(role=KVConnectorRole.SCHEDULER, min_tokens=256):
        cfg = SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector="TTKVStoreConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"min_tokens": min_tokens},
            ),
            model_config=SimpleNamespace(model="m"),
        )
        c = kvc.TTKVStoreConnector(cfg, role, mimo_kv_config())
        c.client = FakeStore()
        assert c._open_pools()
        return c

    kvc._WORKER_READY.set()
    yield make
    kvc._WORKER_READY.clear()


def test_layout_matches_mimo_pages():
    L = kvc.Layout.from_config(mimo_kv_config(), 8)
    assert L.global_layers == GLOBAL and L.swa_layers == SWA
    assert (L.global_block_tokens, L.swa_block_tokens) == (128, 64)
    assert L.page_bytes == (12 * 1088, 8 * 1088)
    assert L.global_slot_bytes == 8 * 9 * 21760 == 1_566_720  # 12.24 KB per token
    assert L.swa_slot_bytes == 8 * 39 * 4 * 21760


def test_keys_chain_and_salt(connector):
    c = connector()
    t = list(range(1000, 1000 + 640))
    keys = c._keys(t, None)
    assert len(keys) == 5
    assert c._keys(t[:256], None) == keys[:2], "a prefix has the same leading keys"
    assert c._keys(t, "s")[0] != keys[0], "a cache salt changes every key"
    t2 = list(t)
    t2[200] += 1
    k2 = c._keys(t2, None)
    assert k2[0] == keys[0] and k2[1] != keys[1] and k2[4] != keys[4], "a change propagates to every later block"


def request(rid, tokens, salt=None):
    return SimpleNamespace(request_id=rid, prompt_token_ids=tokens, cache_salt=salt)


class Blocks:
    def __init__(self, ids):
        self.ids = ids

    def get_block_ids(self):
        return self.ids


def test_load_goes_to_the_deepest_boundary_with_a_snapshot(connector):
    c = connector()
    tokens = list(range(5000, 5000 + 20 * 128 + 50))
    st = c._state(request("r", tokens))
    store = c.client
    store.put(c.pool_global, st.keys[:16])  # 16 global blocks stored
    store.put(c.pool_swa, [c._snapshot_key(st.keys, 8), c._snapshot_key(st.keys, 12)])
    n, is_async = c.get_num_new_matched_tokens(request("r", tokens), 0)
    assert (n, is_async) == (12 * 128, False), "16 blocks stored, but the deepest snapshot is at block 12"
    # vLLM allocated: the global group's 21 blocks, and for each SWA group blocks up to position 12 * 128
    ids = [list(range(100, 121))] + [list(range(1000 * g, 1000 * g + 25)) for g in range(1, 6)]
    c.update_state_after_alloc(request("r", tokens), Blocks(ids), n)
    meta = c.build_connector_meta(SimpleNamespace(
        scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[],
        resumed_req_ids=set(), num_computed_tokens=[]), num_scheduled_tokens={}, finished_req_ids=set()))
    (ld,) = meta.loads
    assert ld.num_tokens == 12 * 128
    assert ld.global_blocks == list(range(100, 112))
    assert ld.global_slots == list(range(0, 12))
    # SWA window before position 1536 = blocks 20..23 of 64 tokens, per SWA layer, from its own group
    first_layer_of_group2 = kvc.Layout.from_config(mimo_kv_config(), 8).swa_layers.index(SWA[9])
    assert ld.swa_blocks[first_layer_of_group2] == [(0, 2020), (1, 2021), (2, 2022), (3, 2023)]
    assert len(store.leases) == 2


def test_no_load_without_a_ready_worker(connector):
    c = connector()
    tokens = list(range(4096))
    st = c._state(request("r", tokens))
    c.client.put(c.pool_global, st.keys)
    c.client.put(c.pool_swa, [c._snapshot_key(st.keys, b) for b in range(1, 32)])
    kvc._WORKER_READY.clear()
    assert c.get_num_new_matched_tokens(request("r", tokens), 0) == (0, False)
    kvc._WORKER_READY.set()
    assert c.get_num_new_matched_tokens(request("r", tokens), 0) == (31 * 128, False)


def test_a_prefill_chunk_saves_its_blocks_and_a_window_snapshot(connector):
    c = connector()
    tokens = list(range(9000, 9000 + 5000))
    c.get_num_new_matched_tokens(request("r", tokens), 0)  # nothing stored: no load, but the keys are made
    ids = tuple([list(range(1, 33))] + [list(range(100 * g, 100 * g + 64)) for g in range(1, 6)])
    new = SimpleNamespace(req_id="r", prompt_token_ids=tokens, block_ids=ids, num_computed_tokens=0)
    so = SimpleNamespace(
        scheduled_new_reqs=[new],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set(),
                                              num_computed_tokens=[]),
        num_scheduled_tokens={"r": 4096},
        finished_req_ids=set(),
    )
    meta = c.build_connector_meta(so)
    (sv,) = meta.saves
    assert sv.global_blocks == list(range(1, 33)) and len(sv.global_keys) == 32
    assert sv.snapshot_key == c._snapshot_key(c._req["r"].keys, 32)
    assert sv.swa_blocks[0] == [(0, 160), (1, 161), (2, 162), (3, 163)]
    # the next chunk (the last 904 tokens, not block-aligned at its end) saves its 7 full blocks, no snapshot
    so2 = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r"], new_block_ids=[tuple([list(range(33, 41))] + [[]] * 5)],
                                              resumed_req_ids=set(), num_computed_tokens=[4096]),
        num_scheduled_tokens={"r": 904},
        finished_req_ids=set(),
    )
    (sv2,) = c.build_connector_meta(so2).saves
    assert sv2.global_blocks == list(range(33, 40)) and sv2.snapshot_key is None


def test_global_regions_coalesce_runs(connector):
    c = connector(role=KVConnectorRole.WORKER)
    regs = c._global_regions([3, 4, 5, 9], [10, 11, 12, 20])
    k = regs[(GLOBAL[0], 0)]
    assert len(regs) == 18 and len(k) == 8 * 2  # 8 chips x 2 runs
    pb = 12 * 1088
    pool = c.client._pools[c.pool_global]
    assert k[0] == (0, 10 * pb, 3 * pb, pool.base + 3 * pb)  # chip 0, layer 0, K: plane 0
    assert k[1] == (0, 20 * pb, pb, pool.base + 9 * pb)
