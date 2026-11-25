# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The copy between a model's ttnn KV tensors and a connector's CPU buffer.

``FakeKVCache`` keeps each chip's buffer as bytes, and ``fake_region_copies``
stands in for ttnn's region copies over those bytes.
"""

import importlib.util

import numpy as np
import pytest
import torch
import ttnn
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from tests.support.kv_cache import FakeKVCache, fake_region_copies
from vllm_tt_plugin import kv_transfer
from vllm_tt_plugin.kv_layout import KVLayout, KVPart
from vllm_tt_plugin.kv_transfer import kv_cache_descriptions, make_copy_blocks

BLOCKS = 6
NAME = "model.layers.0.attn"


@pytest.fixture(autouse=True)
def blocking_calls(monkeypatch) -> list[bool]:
    return fake_region_copies(monkeypatch)


def _cache(seed: int) -> FakeKVCache:
    """One head on each of 2 chips, every byte drawn from ``seed``."""
    cache = FakeKVCache(blocks=BLOCKS, heads=1, chips=2)
    rng = np.random.default_rng(seed)
    for chip in cache.chips:
        chip[:] = rng.integers(0, 256, chip.size, dtype=np.uint8)
    return cache


def _buffer(layout: KVLayout, num_kv_heads: int) -> dict[str, torch.Tensor]:
    """A connector's CPU buffer for ``layout``, shaped like its description."""
    spec = FullAttentionSpec(
        block_size=32, num_kv_heads=num_kv_heads, head_size=32, dtype=torch.float32
    )
    config = KVCacheConfig(
        num_blocks=BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([NAME], spec)],
    )
    description = kv_cache_descriptions(config, layout)[NAME]
    return {NAME: torch.zeros(description.shape, dtype=torch.uint8)}


def test_a_host_block_is_each_part_shard_after_shard():
    """K, then V; within each, chip 0's heads, then chip 1's.

    These are the bytes a server with another chip count reads.
    """
    k, v = _cache(seed=0), _cache(seed=1)
    layout = KVLayout(
        {
            NAME: (
                KVPart(k, k.block_bytes, ((0,), (1,))),
                KVPart(v, v.block_bytes, ((0,), (1,))),
            )
        }
    )
    buffer = _buffer(layout, num_kv_heads=2)

    make_copy_blocks([layout])(None, buffer, [3, 4], [1, 0], "d2h")

    for device_block, host_block in ((3, 1), (4, 0)):
        expected = np.concatenate(
            [cache.blocks_on(chip)[device_block] for cache in (k, v) for chip in (0, 1)]
        )
        np.testing.assert_array_equal(
            buffer[NAME][host_block].reshape(-1).numpy(), expected
        )


def test_a_copied_shard_is_read_once_and_written_to_every_chip():
    """A latent copied to both chips crosses the host once and lands on both."""
    sent, received = _cache(seed=0), FakeKVCache(blocks=BLOCKS, heads=1, chips=2)
    sent.chips[1][:] = 255  # A replica a correct copy never reads.

    def copy_of(cache: FakeKVCache):
        layout = KVLayout({NAME: (KVPart(cache, cache.block_bytes, ((0, 1),)),)})
        return _buffer(layout, num_kv_heads=1), make_copy_blocks([layout])

    sender_buffer, sender_copy = copy_of(sent)
    receiver_buffer, receiver_copy = copy_of(received)
    sender_copy(None, sender_buffer, [4], [0], "d2h")
    receiver_buffer[NAME][2] = sender_buffer[NAME][0]
    receiver_copy(receiver_buffer, None, [2], [1], "h2d")

    for chip in (0, 1):
        np.testing.assert_array_equal(received.blocks_on(chip)[1], sent.blocks_on(0)[4])


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_a_copy_returns_once_its_last_transfer_landed(blocking_calls, direction):
    # The connector sends or reads its buffer right after the copy returns.
    k, v = _cache(seed=0), _cache(seed=1)
    layout = KVLayout(
        {NAME: tuple(KVPart(c, c.block_bytes, ((0,), (1,))) for c in (k, v))}
    )
    buffer = _buffer(layout, num_kv_heads=2)
    src, dst = (None, buffer) if direction == "d2h" else (buffer, None)

    make_copy_blocks([layout])(src, dst, [0, 1], [2, 3], direction)

    assert blocking_calls[-1] is True


def test_importing_the_module_needs_no_region_copies(monkeypatch):
    # A released ttnn lacks them; the plugin must still import with it.
    for name in ("copy_device_to_host_regions", "copy_host_to_device_regions"):
        monkeypatch.delattr(ttnn, name)
    spec = importlib.util.spec_from_file_location(
        "kv_transfer_copy", kv_transfer.__file__
    )
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
