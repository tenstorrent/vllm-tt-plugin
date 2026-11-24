# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""KV layouts for connectors, and the copy they drive.

Fake device tensors hold each chip's buffer as bytes; the region copies are
replaced by their documented behaviour over those bytes.
"""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from vllm_tt_plugin import kv_layout, kv_transfer
from vllm_tt_plugin.kv_layout import KVPart, default_kv_layout, kv_transfer_layout

BLOCKS, BLOCK_SIZE, HEAD_DIM = 6, 32, 32
TILE_BYTES = 1088  # bfloat8_b


class FakeCache:
    """A paged ``[blocks, heads, block_size, head_dim]`` bfloat8_b cache on a mesh."""

    dtype = kv_layout.ttnn.bfloat8_b
    layout = kv_layout.ttnn.TILE_LAYOUT

    def __init__(self, heads: int, chips: int):
        self.shape = (BLOCKS, heads, BLOCK_SIZE, HEAD_DIM)
        self.block_bytes = heads * TILE_BYTES
        self.chips = [
            np.zeros(BLOCKS * self.block_bytes, dtype=np.uint8) for _ in range(chips)
        ]

    def device(self):
        return SimpleNamespace(get_num_devices=lambda: len(self.chips))


def _config(names: list[str], num_kv_heads: int) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=HEAD_DIM,
        dtype=torch.float32,
    )
    return KVCacheConfig(
        num_blocks=BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(names, spec)],
    )


class TestDefaultLayout:
    def test_heads_split_over_the_chips_are_one_shard_per_chip(self):
        k, v = FakeCache(heads=2, chips=4), FakeCache(heads=2, chips=4)
        layout = default_kv_layout([[k, v]], _config(["model.layers.0.attn"], 8))

        part_k, part_v = layout.layers["model.layers.0.attn"]
        assert part_k.shards == ((0,), (1,), (2,), (3,))
        assert part_k.block_bytes == 2 * TILE_BYTES
        assert layout.block_bytes("model.layers.0.attn") == 2 * 4 * 2 * TILE_BYTES

    def test_heads_copied_to_every_chip_are_one_shard_on_all(self):
        k, v = FakeCache(heads=8, chips=4), FakeCache(heads=8, chips=4)
        layout = default_kv_layout([[k, v]], _config(["model.layers.0.attn"], 8))

        part_k, _ = layout.layers["model.layers.0.attn"]
        assert part_k.shards == ((0, 1, 2, 3),)
        assert layout.block_bytes("model.layers.0.attn") == 2 * 8 * TILE_BYTES

    def test_a_partial_split_must_be_declared_by_the_model(self):
        # 8 heads on an 8-chip mesh split 4 ways and copied 2 ways.
        k, v = FakeCache(heads=2, chips=8), FakeCache(heads=2, chips=8)
        with pytest.raises(ValueError, match="get_kv_transfer_layout"):
            default_kv_layout([[k, v]], _config(["model.layers.0.attn"], 8))


class TestDeclaredLayout:
    def test_a_model_declares_extra_parts_and_layers(self):
        """An MLA latent copied to all chips, and an MTP drafter's own layer."""
        latent = FakeCache(heads=1, chips=2)
        drafter_k, drafter_v = FakeCache(heads=1, chips=2), FakeCache(heads=1, chips=2)
        names = ["model.layers.0.attn", "drafter.layers.0.attn"]

        class Model:
            def get_kv_transfer_layout(self, kv_caches, kv_cache_config):
                return {
                    names[0]: [(latent, latent.block_bytes, [[0, 1]])],
                    names[1]: [
                        (drafter_k, drafter_k.block_bytes, [[0], [1]]),
                        (drafter_v, drafter_v.block_bytes, [[0], [1]]),
                    ],
                }

        layout = kv_transfer_layout(Model(), None, _config(names, 2))

        assert layout.layers[names[0]] == (KVPart(latent, TILE_BYTES, ((0, 1),)),)
        assert layout.block_bytes(names[1]) == 4 * TILE_BYTES

    def test_a_declared_layout_covers_every_layer(self):
        class Model:
            def get_kv_transfer_layout(self, kv_caches, kv_cache_config):
                return {}

        with pytest.raises(ValueError, match="does not describe layers"):
            kv_transfer_layout(Model(), None, _config(["model.layers.0.attn"], 2))


@pytest.fixture
def fake_region_transfers(monkeypatch):
    def host_bytes(address, size):
        return np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(address))

    def to_host(cache, regions, blocking=True, cq_id=None):
        for chip, offset, size, address in regions:
            data = cache.chips[int(chip)][int(offset) : int(offset + size)]
            host_bytes(int(address), int(size))[:] = data

    def to_device(cache, regions, blocking=True, cq_id=None):
        for chip, offset, size, address in regions:
            data = host_bytes(int(address), int(size))
            cache.chips[int(chip)][int(offset) : int(offset + size)] = data

    monkeypatch.setattr(kv_transfer.ttnn, "copy_device_to_host_regions", to_host)
    monkeypatch.setattr(kv_transfer.ttnn, "copy_host_to_device_regions", to_device)


def test_a_copied_shard_is_read_once_and_written_to_every_chip(fake_region_transfers):
    """A latent copied to both chips crosses the host once and lands on both."""
    name = "model.layers.0.attn"
    config = _config([name], 1)
    sent, received = FakeCache(heads=1, chips=2), FakeCache(heads=1, chips=2)
    sent.chips[0][:] = np.arange(sent.chips[0].size, dtype=np.uint8)
    sent.chips[1][:] = 255  # a replica a correct copy never reads

    def server(cache):
        layout = kv_transfer_layout(
            SimpleNamespace(
                get_kv_transfer_layout=lambda *_: {
                    name: [(cache, cache.block_bytes, [[0, 1]])]
                }
            ),
            None,
            config,
        )
        description = kv_transfer.kv_cache_descriptions(config, layout)[name]
        buffer = {name: torch.zeros(description.shape, dtype=torch.uint8)}
        return buffer, kv_transfer.make_copy_blocks(layout)

    sender_buffer, sender_copy = server(sent)
    receiver_buffer, receiver_copy = server(received)
    sender_copy(None, sender_buffer, [4], [0], "d2h")
    receiver_buffer[name][2] = sender_buffer[name][0]
    receiver_copy(receiver_buffer, None, [2], [1], "h2d")

    block = slice(4 * sent.block_bytes, 5 * sent.block_bytes)
    target = slice(1 * sent.block_bytes, 2 * sent.block_bytes)
    for chip in received.chips:
        np.testing.assert_array_equal(chip[target], sent.chips[0][block])
