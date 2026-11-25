# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""KV transfer layouts: the default one and the one a model declares.

Fake device tensors hold each chip's buffer as bytes.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from vllm_tt_plugin import kv_layout
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
