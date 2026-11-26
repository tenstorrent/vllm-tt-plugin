# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A fake paged KV cache on a TT mesh.

``FakeKVCache`` keeps each chip's buffer as bytes, so KV layout tests can
compare bytes without a device. It reports the buffer facts a DRAM layout reads
(address, page size, device IDs, memory config) as an interleaved DRAM tensor.
"""

from types import SimpleNamespace

import numpy as np
import ttnn

TILE = 32
TILE_BYTES = 1088  # One 32x32 bfloat8_b tile: 1024 values and 64 exponents.


class FakeKVCache:
    """A paged ``[blocks, heads, block_size, head_dim]`` bfloat8_b cache.

    Attributes
    ----------
    shape
        Each chip's shape, as a ttnn mesh tensor reports it.
    block_bytes
        Bytes of one block on one chip.
    chips
        Each chip's buffer as ``uint8``, block after block, one page (tile) of
        ``TILE_BYTES`` after another.
    address
        The buffer address, the same on every chip, as ttnn allocates a mesh
        tensor.
    device_ids
        The ttnn device ID of each chip, in mesh order.
    """

    dtype = ttnn.bfloat8_b
    layout = ttnn.TILE_LAYOUT

    def __init__(
        self,
        *,
        blocks: int,
        heads: int,
        chips: int,
        block_size: int = TILE,
        head_dim: int = TILE,
        address: int = 0,
    ):
        self.shape = (blocks, heads, block_size, head_dim)
        self.block_bytes = (
            heads * (block_size // TILE) * (head_dim // TILE) * TILE_BYTES
        )
        self.chips = [
            np.zeros(blocks * self.block_bytes, dtype=np.uint8) for _ in range(chips)
        ]
        self.address = address
        self.device_ids = list(range(chips))

    def device(self) -> SimpleNamespace:
        """The mesh, which reports its chip count and device IDs."""
        return SimpleNamespace(
            get_num_devices=lambda: len(self.chips),
            get_device_ids=lambda: self.device_ids,
        )

    def buffer_address(self) -> int:
        """The buffer address, as on every chip of a ttnn mesh tensor."""
        return self.address

    def buffer_aligned_page_size(self) -> int:
        """Bytes of one page: one bfloat8_b tile."""
        return TILE_BYTES

    def memory_config(self) -> SimpleNamespace:
        """DRAM interleaved: page ``i`` in DRAM bank ``i % num_banks``."""
        return SimpleNamespace(
            buffer_type=ttnn.BufferType.DRAM,
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        )

    def blocks_on(self, chip: int) -> np.ndarray:
        """Chip ``chip``'s buffer as ``[blocks, block_bytes]``."""
        return self.chips[chip].reshape(-1, self.block_bytes)
