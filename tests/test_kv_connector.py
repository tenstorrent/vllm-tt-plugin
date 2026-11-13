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

        self.gen = "g1"

    def open_pool(self, name, slot_bytes, capacity_bytes):
        p = self._pools.get(name)
        if p is None or p.gen != self.gen:
            self._pools[name] = Pool(name, f"/dev/shm/{name}", self.gen, 64, slot_bytes, None, 1 << 40)
            self.next_slot.setdefault(name, 0)
        return self._pools[name]

    def pool(self, name):
        return self._pools.get(name)

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



def test_the_chunk_that_completes_a_prompt_snapshots_its_last_block_boundary(connector):
    """A prompt prefilled for another engine (prefill / decode disaggregation) can be continued from its last block
    boundary: the final, unaligned chunk saves a window snapshot there, from its own SWA blocks."""
    c = connector()
    tokens = list(range(5000))
    c.get_num_new_matched_tokens(request("r", tokens), 0)
    swa_ids = [list(range(100 * g, 100 * g + 79)) for g in range(1, 6)]  # 79 blocks of 64: positions up to 5056
    ids = tuple([list(range(1, 33))] + [x[:64] for x in swa_ids])
    new = SimpleNamespace(req_id="r", prompt_token_ids=tokens, block_ids=ids, num_computed_tokens=0)
    empty = SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set(), num_computed_tokens=[])
    c.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[new], scheduled_cached_reqs=empty,
                                           num_scheduled_tokens={"r": 4096}, finished_req_ids=set()))
    so2 = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r"], new_block_ids=[tuple([list(range(33, 41))] + [x[64:] for x in swa_ids])],
            resumed_req_ids=set(), num_computed_tokens=[4096]),
        num_scheduled_tokens={"r": 904},
        finished_req_ids=set(),
    )
    (sv2,) = c.build_connector_meta(so2).saves
    assert sv2.global_blocks == list(range(33, 40))  # 39 full blocks in all: tokens [0, 4992)
    assert sv2.snapshot_key == c._snapshot_key(c._req["r"].keys, 39)
    # the window [4736, 4992): SWA blocks 74..77 of group 1
    assert sv2.swa_blocks[0] == [(0, 174), (1, 175), (2, 176), (3, 177)]
    # off: no snapshot at the prompt end
    c2 = connector()
    c2.snapshot_prompt_end = False
    c2.get_num_new_matched_tokens(request("r", tokens), 0)
    c2.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[new], scheduled_cached_reqs=empty,
                                            num_scheduled_tokens={"r": 4096}, finished_req_ids=set()))
    (sv3,) = c2.build_connector_meta(so2).saves
    assert sv3.snapshot_key is None


def test_a_tail_after_a_loaded_boundary_snapshots_nothing_new(connector):
    """The decoding engine of a disaggregated prompt computes only the tail past the loaded boundary: its chunk starts
    at that boundary, so it asks for no snapshot (the store already has it)."""
    c = connector()
    tokens = list(range(5000))
    c.get_num_new_matched_tokens(request("r", tokens), 0)
    swa_ids = [list(range(100 * g, 100 * g + 79)) for g in range(1, 6)]
    ids = tuple([list(range(1, 41))] + swa_ids)
    new = SimpleNamespace(req_id="r", prompt_token_ids=tokens, block_ids=ids, num_computed_tokens=4992)
    empty = SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set(), num_computed_tokens=[])
    c._req["r"].saved_blocks = 39  # as after a load of 39 blocks
    meta = c.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[new], scheduled_cached_reqs=empty,
                                                  num_scheduled_tokens={"r": 8}, finished_req_ids=set()))
    assert meta.saves == []

def test_global_regions_coalesce_runs(connector):
    c = connector(role=KVConnectorRole.WORKER)
    regs = c._global_regions(c.client.pool(c.pool_global), [3, 4, 5, 9], [10, 11, 12, 20])
    k = regs[(GLOBAL[0], 0)]
    assert len(regs) == 18 and len(k) == 8 * 2  # 8 chips x 2 runs
    pb = 12 * 1088
    pool = c.client.pool(c.pool_global)
    assert k[0] == (0, 10 * pb, 3 * pb, pool.base + 3 * pb)  # chip 0, layer 0, K: plane 0
    assert k[1] == (0, 20 * pb, pb, pool.base + 9 * pb)


def test_a_load_pinned_in_another_pool_generation_is_refused(connector):
    """The scheduler pinned the slots in generation g1; the store restarted (g2): copying through either mapping
    would read some other entry's bytes, so the step fails."""
    c = connector(role=KVConnectorRole.WORKER)
    c._kv = [[object(), object()] for _ in range(48)]
    ld = kvc.LoadSpec("r", 128, [0], [5], 0, {}, [], global_gen="g1", swa_gen="g1")
    c.client.gen = "g2"
    c.client._pools.clear()  # the client forgot its pools when its connection to the old store closed
    c.bind_connector_metadata(kvc.TTKVMeta(loads=[ld]))
    with pytest.raises(RuntimeError, match="restarted"):
        c.start_load_kv(None)
    assert c.client.pool(c.pool_global).gen == "g2", "it reopened the pool and found a new generation"


def test_a_store_error_during_lookup_is_a_miss(connector):
    """The store restarting under an engine (its connection drops mid-lookup) must cost a miss, not the engine."""
    c = connector()

    def broken(*a, **k):
        raise kvc.KVStoreError("lookup: [Errno 32] Broken pipe")

    c.client.lookup = broken
    tokens = list(range(4096))
    assert c.get_num_new_matched_tokens(request("r", tokens), 0) == (0, False)


def test_a_failed_release_is_logged_not_swallowed(connector, caplog):
    c = connector()

    def fail(lease):
        raise kvc.KVStoreError("release: unknown reservation or lease 7")

    c.client.release = fail
    c._release(7)
    assert c.release_errors == 1


# --- KVStoreClient against a scripted store ---------------------------------------------------------------------

import json  # noqa: E402
import socket  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402

from vllm_tt_plugin.kvstore_client import KVStoreClient, KVStoreError, StalePool  # noqa: E402


class ScriptedStore:
    """A Unix socket server that answers each request with the next scripted response (a callable of the request)."""

    def __init__(self, tmp, script):
        self.path = f"{tmp}/s.sock"
        self.script = list(script)
        self.requests = []
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(4)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.script:
            conn, _ = self.srv.accept()
            f = conn.makefile("rwb")
            while self.script:
                line = f.readline()
                if not line:
                    break
                req = json.loads(line)
                self.requests.append(req)
                resp = self.script.pop(0)(req)
                if resp is None:  # drop the connection (a store restart)
                    break
                f.write((json.dumps(resp) + "\n").encode())
                f.flush()
            f.close()
            conn.shutdown(socket.SHUT_RDWR)
            conn.close()


def _pool_file(tmp, name, size=4096):
    path = f"{tmp}/{name}"
    with open(path, "wb") as f:
        f.truncate(size)
    return path


def test_client_checks_generations_and_forgets_pools_on_reconnect():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = _pool_file(tmp, "p.g1")
        p2 = _pool_file(tmp, "p.g2")
        opened = lambda path, gen: lambda r: {"ok": True, "path": path, "gen": gen, "num_slots": 1, "slot_bytes": 4096}
        store = ScriptedStore(tmp, [
            opened(p1, "g1"),
            lambda r: {"ok": True, "lease": 5, "slots": [0], "gen": "g2"},  # the store answers for another generation
            lambda r: {"ok": True},  # the client releases that lease
            lambda r: None,  # then the store "restarts": the connection drops
            opened(p2, "g2"),
            lambda r: {"ok": False, "error": "release: unknown reservation or lease 9"},
        ])
        c = KVStoreClient(store.path)
        pool = c.open_pool("p", 4096, 4096)
        assert (pool.gen, c.pool("p").gen) == ("g1", "g1")
        with pytest.raises(StalePool):
            c.acquire("p", [b"k" * 32])
        assert store.requests[1]["gen"] == "g1", "the mapping's generation is sent"
        assert store.requests[2] == {"op": "release", "lease": 5}, "the lease of the stale answer is given back"
        assert c.pool("p") is None, "a stale pool is forgotten"
        c._pools["p"] = pool
        with pytest.raises(KVStoreError):
            c.lookup("p", [b"k" * 32])  # the connection drops
        assert c.pool("p") is None, "every pool is forgotten when the connection to the store closes"
        assert c.open_pool("p", 4096, 4096).gen == "g2"
        with pytest.raises(KVStoreError):
            c.release(9)


def _item(identifier, offset, length):
    return SimpleNamespace(identifier=identifier, mm_position=SimpleNamespace(offset=offset, length=length))


def test_keys_carry_the_items_behind_identical_placeholder_tokens(connector):
    """Two images of one size are the same placeholder tokens: their blocks must not share keys."""
    c = connector()
    tokens = [7] * 100 + [151655] * 300 + [9] * 240  # 5 blocks of 128; the item covers [100, 400): blocks 0-3
    text = c._keys(tokens, None)
    a = c._keys(tokens, None, [_item("img-a", 100, 300)])
    b = c._keys(tokens, None, [_item("img-b", 100, 300)])
    assert len(text) == len(a) == len(b) == 5
    assert a == c._keys(tokens, None, [_item("img-a", 100, 300)])  # the same image: the same keys
    assert all(x != y for x, y in zip(a, b))  # every block from the first one the item touches differs
    assert all(x != y for x, y in zip(a, text))
    # an item that starts in a later block leaves the blocks before it as a text prompt's
    late = c._keys(tokens, None, [_item("img-a", 300, 100)])
    assert late[:2] == text[:2] and late[2] != text[2]
    # the same item at another offset is another prefix
    assert c._keys(tokens, None, [_item("img-a", 101, 300)])[0] != a[0]
    # no identifier: the request is not keyed at all
    assert c._keys(tokens, None, [_item(None, 100, 300)]) is None


def test_a_request_with_an_unidentified_item_is_neither_loaded_nor_saved(connector):
    c = connector()
    tokens = list(range(5000))
    req = SimpleNamespace(request_id="r", prompt_token_ids=tokens, cache_salt=None,
                          mm_features=[_item(None, 10, 50)])
    assert c.get_num_new_matched_tokens(req, 0) == (0, False)
    ids = tuple([list(range(1, 33))] + [list(range(100 * g, 100 * g + 64)) for g in range(1, 6)])
    new = SimpleNamespace(req_id="r", prompt_token_ids=tokens, block_ids=ids, num_computed_tokens=0)
    empty = SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set(), num_computed_tokens=[])
    meta = c.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[new], scheduled_cached_reqs=empty,
                                                  num_scheduled_tokens={"r": 4096}, finished_req_ids=set()))
    assert meta.saves == []


def test_a_failed_prefill_row_is_not_saved(connector, caplog):
    c = connector(role=KVConnectorRole.WORKER)
    c._kv = object()
    reserved = []
    c.client.reserve = lambda pool, keys: reserved.append(keys) or (1, [None] * len(keys))
    c.client.commit = lambda r: None
    saves = [kvc.SaveSpec("bad", [b"k1"], [5], None, None), kvc.SaveSpec("good", [b"k2"], [6], None, None)]
    c._get_connector_metadata = lambda: kvc.TTKVMeta(saves=saves)
    c.skip_save_req_ids = {"bad"}
    c.wait_for_save()
    assert reserved == [[b"k2"]]  # only the good request reached the store
    assert c.skip_save_req_ids == set()  # consumed: the next step saves again


def test_region_copies_pass_the_pool_mapping_when_ttnn_takes_it(monkeypatch):
    """``_copy`` hands ttnn the pool's mapping as ``host_buffer`` (bounds check, and kept alive for the non-blocking
    reads) on a build that takes it, blocks on every call when there is no mapping object, and calls an older build
    as before; a failed read waits for the reads already issued before the error reaches the caller."""
    import sys
    import types

    from vllm_tt_plugin import kv_connector as kc

    calls = []

    class Tensor:
        def device(self):
            return "mesh"

    def make_ttnn(with_host_buffer, fail_at=None):
        ttnn = types.ModuleType("ttnn")

        def copy(name):
            def fn(t, regs, blocking=True, **kwargs):
                if fail_at is not None and len(calls) == fail_at:
                    raise RuntimeError("bad region")
                calls.append((name, blocking, kwargs))

            fn.__doc__ = "copy(device_tensor, regions, blocking=True, cq_id=None" + (
                ", host_buffer=None)" if with_host_buffer else ")"
            )
            return fn

        ttnn.copy_device_to_host_regions = copy("read")
        ttnn.copy_host_to_device_regions = copy("write")
        ttnn.synchronize_device = lambda device: calls.append(("sync", device))
        return ttnn

    conn = object.__new__(kc.TTKVStoreConnector)
    conn._kv = {layer: (Tensor(), Tensor()) for layer in range(2)}
    regions = {(0, 0): [(0, 0, 64, 1000)], (0, 1): [], (1, 0): [(0, 0, 64, 2000)], (1, 1): [(0, 64, 64, 3000)]}
    mapping = object()
    pool = kc.Pool("p", "/x", "g", 4, 64, mapping, 1000)

    monkeypatch.setitem(sys.modules, "ttnn", make_ttnn(True))
    assert conn._copy(pool, regions, to_device=False) == 192
    assert calls == [("read", False, {"host_buffer": mapping})] * 2 + [("read", True, {"host_buffer": mapping})]
    calls.clear()
    conn._copy(pool, regions, to_device=True)
    assert [c[:2] for c in calls] == [("write", False), ("write", False), ("write", True)]
    assert all(c[2] == {"host_buffer": mapping} for c in calls)
    calls.clear()
    conn._copy(kc.Pool("p", "/x", "g", 4, 64, None, 1000), regions, to_device=False)  # no mapping object: blocking
    assert calls == [("read", True, {})] * 3
    calls.clear()

    monkeypatch.setitem(sys.modules, "ttnn", make_ttnn(False))  # a build from before host_buffer
    conn._copy(pool, regions, to_device=False)
    assert calls == [("read", False, {}), ("read", False, {}), ("read", True, {})]
    calls.clear()

    monkeypatch.setitem(sys.modules, "ttnn", make_ttnn(True, fail_at=1))
    with pytest.raises(RuntimeError, match="bad region"):
        conn._copy(pool, regions, to_device=False)
    assert calls == [("read", False, {"host_buffer": mapping}), ("sync", "mesh")]
