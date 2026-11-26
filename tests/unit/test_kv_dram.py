# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""KV in DRAM banks: the byte ranges NIXL moves for each layer.

``Dram`` stores each fake cache in simulated DRAM banks by ttnn's interleave
rule, independently of ``kv_dram``. The tests read every region back through
its NIXL address and compare it with the pages the rule put there.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ttnn

from tests.support.kv_cache import (
    BANK_SIZE,
    BANKS,
    TILE_BYTES,
    Cluster,
    FakeKVCache,
    fake_cluster,
)
from vllm_tt_plugin.kv_dram import DramRegion, kv_dram_layout
from vllm_tt_plugin.kv_layout import KVLayout, KVPart

BLOCKS = 4
LAYER = "model.layers.0.self_attn"


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Cluster:
    """Devices 0 to 3 on cards 10 to 13, each with 8 banks at distinct bases."""
    return fake_cluster(monkeypatch, tmp_path)


class Dram:
    """Each card's DRAM banks as 64 KiB of bytes, written the way ttnn does."""

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.banks: dict[tuple[int, int], np.ndarray] = {}

    def store(self, cache: FakeKVCache) -> None:
        """Write each chip's buffer page by page into its card's banks.

        Page ``i`` goes to bank ``i % 8``, ``i // 8`` pages past the buffer
        address plus the bank's base address (tt-metal ``Buffer::page_address``).
        """
        for chip, device_id in enumerate(cache.device_ids):
            card = self.cluster.cards[device_id]
            table = self.cluster.banks[device_id]
            for i, page in enumerate(cache.chips[chip].reshape(-1, TILE_BYTES)):
                bank = table[i % BANKS]
                at = cache.address + bank["base_addr"] + i // BANKS * TILE_BYTES
                memory = self.banks.setdefault(
                    (card, bank["bank_id"]), np.zeros(1 << 16, dtype=np.uint8)
                )
                memory[at : at + TILE_BYTES] = page

    def read(self, region: DramRegion, block: int) -> bytes:
        """Block ``block`` of ``region``, from where NIXL reads it."""
        bank, offset = region.addr >> 32, region.addr & 0xFFFFFFFF
        at = offset + block * region.block_stride
        return self.banks[region.card, bank][at : at + region.block_bytes].tobytes()


def _cache(heads: int, chips: int, address: int = 0x1000, seed: int = 0) -> FakeKVCache:
    """A ``[BLOCKS, heads, 64, 128]`` cache of random bytes: 8 pages a head."""
    cache = FakeKVCache(
        blocks=BLOCKS,
        heads=heads,
        chips=chips,
        block_size=64,
        head_dim=128,
        address=address,
    )
    rng = np.random.default_rng(seed)
    for buffer in cache.chips:
        buffer[:] = rng.integers(0, 256, buffer.size, dtype=np.uint8)
    return cache


def _split(cache: FakeKVCache) -> KVPart:
    """``cache`` as a part with one shard per chip."""
    shards = tuple((chip,) for chip in range(len(cache.chips)))
    return KVPart(cache, cache.block_bytes, shards)


def _pages_in_bank(block: np.ndarray, bank: int) -> bytes:
    """The pages of one block on one chip that ttnn puts in ``bank``, in order."""
    return block.reshape(-1, TILE_BYTES)[bank::BANKS].tobytes()


class TestRegions:
    def test_a_regions_block_is_that_blocks_pages_in_one_bank(self, cluster):
        """Regions go part by part, shard by shard, bank by bank.

        Mesh index 0 is device 2, so the shards are not in device ID order.
        """
        k = _cache(heads=2, chips=2, address=0x1000, seed=0)
        v = _cache(heads=2, chips=2, address=0x9000, seed=1)
        k.device_ids = v.device_ids = [2, 1]
        dram = Dram(cluster)
        dram.store(k)
        dram.store(v)

        layout = kv_dram_layout(KVLayout({LAYER: (_split(k), _split(v))}), BLOCKS)

        for block in range(BLOCKS):
            assert [dram.read(region, block) for region in layout.regions[LAYER]] == [
                _pages_in_bank(cache.blocks_on(chip)[block], bank)
                for cache in (k, v)
                for chip in range(2)
                for bank in range(BANKS)
            ]

    def test_one_chip_in_two_parts_reads_what_two_chips_hold(self, cluster):
        """A 1-chip server with ``parts=2`` lists, region for region, the bytes
        of a 2-chip server whose chips hold the first and the second half of
        the heads."""
        one = _cache(heads=4, chips=1, address=0x2000)
        two = _cache(heads=2, chips=2, address=0x5000)
        two.device_ids = [1, 2]
        halves = one.blocks_on(0).reshape(BLOCKS, 2, -1)
        for chip in range(2):
            two.blocks_on(chip)[:] = halves[:, chip]
        dram = Dram(cluster)
        dram.store(one)
        dram.store(two)

        sender = kv_dram_layout(KVLayout({LAYER: (_split(one),)}), BLOCKS, parts=2)
        receiver = kv_dram_layout(KVLayout({LAYER: (_split(two),)}), BLOCKS)

        for block in range(BLOCKS):
            held = [
                _pages_in_bank(halves[block, half], bank)
                for half in range(2)
                for bank in range(BANKS)
            ]
            assert [dram.read(r, block) for r in sender.regions[LAYER]] == held
            assert [dram.read(r, block) for r in receiver.regions[LAYER]] == held

    def test_every_bank_that_holds_a_region_is_listed_once(self, cluster):
        names = [LAYER, "model.layers.1.self_attn"]
        caches = [_cache(heads=2, chips=2) for _ in range(4)]
        for cache in caches:
            cache.device_ids = [3, 0]
        layout = KVLayout(
            {
                name: (_split(caches[2 * i]), _split(caches[2 * i + 1]))
                for i, name in enumerate(names)
            }
        )

        banks = kv_dram_layout(layout, BLOCKS).banks

        assert banks == [
            (bank << 32, BANK_SIZE, card) for card in (13, 10) for bank in range(BANKS)
        ]


class TestRefusals:
    """KV that NIXL cannot move as one range per bank is refused."""

    def test_a_shard_copied_to_several_chips_is_refused(self, cluster):
        cache = _cache(heads=2, chips=2)
        layout = KVLayout({LAYER: (KVPart(cache, cache.block_bytes, ((0, 1),)),)})
        with pytest.raises(ValueError, match=f"{LAYER}.*copied"):
            kv_dram_layout(layout, BLOCKS)

    def test_a_chip_without_all_8_dram_banks_is_refused(self, cluster):
        # P100a and Bin6 Galaxy chips use 7 banks.
        cluster.banks[1].pop()
        cache = _cache(heads=2, chips=2)
        with pytest.raises(ValueError, match=f"{LAYER}.*device 1 has 7 DRAM banks"):
            kv_dram_layout(KVLayout({LAYER: (_split(cache),)}), BLOCKS)

    def test_a_device_that_is_not_a_pcie_card_of_this_host_is_refused(self, cluster):
        del cluster.cards[1]
        cache = _cache(heads=2, chips=2)
        with pytest.raises(ValueError, match=f"{LAYER}.*device 1 is not a PCIe"):
            kv_dram_layout(KVLayout({LAYER: (_split(cache),)}), BLOCKS)

    def test_a_block_that_is_not_whole_pages_is_refused(self, cluster):
        cache = _cache(heads=2, chips=1)
        cache.buffer_aligned_page_size = lambda: 2048
        with pytest.raises(ValueError, match=f"{LAYER}.*17408.*2048"):
            kv_dram_layout(KVLayout({LAYER: (_split(cache),)}), BLOCKS)

    def test_a_block_whose_pages_do_not_divide_over_the_banks_is_refused(self, cluster):
        cache = FakeKVCache(blocks=BLOCKS, heads=1, chips=1, head_dim=128)
        with pytest.raises(ValueError, match=f"{LAYER}.*4 pages.*8 DRAM banks"):
            kv_dram_layout(KVLayout({LAYER: (_split(cache),)}), BLOCKS)

    @pytest.mark.parametrize(
        "bank_size",
        [
            pytest.param(0x100000, id="bank_size"),
            # A bank larger than the 0xFF000000 bytes NIXL maps of one.
            pytest.param(1 << 32, id="nixl_bank_window"),
        ],
    )
    def test_blocks_past_the_end_of_a_bank_are_refused(self, cluster, bank_size):
        end = min(bank_size, 0xFF000000)
        cache = _cache(heads=2, chips=1)
        for bank in cluster.banks[0]:
            bank["bank_size"] = bank_size
        layout = KVLayout({LAYER: (_split(cache),)})
        # Bank 7 has the highest base address: its range ends exactly at ``end``.
        per_bank = BLOCKS * cache.block_bytes // BANKS
        cache.address = end - cluster.banks[0][7]["base_addr"] - per_bank
        kv_dram_layout(layout, BLOCKS)

        cache.address += 1
        with pytest.raises(ValueError, match=f"{LAYER}.*bank 7 of device 0"):
            kv_dram_layout(layout, BLOCKS)

    @pytest.mark.parametrize(
        ("buffer_type", "memory_layout"),
        [
            pytest.param("L1", "INTERLEAVED", id="l1"),
            pytest.param("DRAM", "HEIGHT_SHARDED", id="dram_sharded"),
        ],
    )
    def test_a_tensor_that_is_not_dram_interleaved_is_refused(
        self, cluster, buffer_type, memory_layout
    ):
        cache = _cache(heads=2, chips=1)
        cache.memory_config = lambda: SimpleNamespace(
            buffer_type=getattr(ttnn.BufferType, buffer_type),
            memory_layout=getattr(ttnn.TensorMemoryLayout, memory_layout),
        )
        with pytest.raises(ValueError, match=f"{LAYER}.*not DRAM interleaved"):
            kv_dram_layout(KVLayout({LAYER: (_split(cache),)}), BLOCKS)

    @pytest.mark.parametrize("parts", [3, 1, 0])
    def test_parts_that_are_not_a_multiple_of_the_shards_are_refused(
        self, cluster, parts
    ):
        cache = _cache(heads=2, chips=2)
        with pytest.raises(ValueError, match=f"{LAYER}.*parts={parts}.*2 shards"):
            kv_dram_layout(KVLayout({LAYER: (_split(cache),)}), BLOCKS, parts=parts)

    def test_parts_that_split_a_page_in_a_bank_are_refused(self, cluster):
        # 16 pages a block, 2 in each bank: 4 parts would take half a page each.
        cache = _cache(heads=2, chips=1)
        with pytest.raises(ValueError, match=f"{LAYER}.*parts=4.*16 pages"):
            kv_dram_layout(KVLayout({LAYER: (_split(cache),)}), BLOCKS, parts=4)
