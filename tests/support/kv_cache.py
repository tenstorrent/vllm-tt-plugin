# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A fake paged KV cache on a TT mesh, and ttnn's region copies over it.

``FakeKVCache`` keeps each chip's buffer as bytes. ``fake_region_copies``
replaces ``ttnn.copy_device_to_host_regions`` and
``ttnn.copy_host_to_device_regions`` with their documented behavior over those
bytes, so KV layout and copy tests can compare bytes without a device.
"""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest
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
        Each chip's buffer as ``uint8``, block after block.
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
    ):
        self.shape = (blocks, heads, block_size, head_dim)
        self.block_bytes = (
            heads * (block_size // TILE) * (head_dim // TILE) * TILE_BYTES
        )
        self.chips = [
            np.zeros(blocks * self.block_bytes, dtype=np.uint8) for _ in range(chips)
        ]

    def device(self) -> SimpleNamespace:
        """The mesh, which only reports its chip count."""
        return SimpleNamespace(get_num_devices=lambda: len(self.chips))

    def blocks_on(self, chip: int) -> np.ndarray:
        """Chip ``chip``'s buffer as ``[blocks, block_bytes]``."""
        return self.chips[chip].reshape(-1, self.block_bytes)


def fake_region_copies(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Replace ttnn's region copies with their behavior over ``FakeKVCache``.

    A region is a row ``(chip, offset on the chip, bytes, host address)``. A
    released ttnn may lack both functions, so they are set, not replaced.

    Returns
    -------
    list[bool]
        The ``blocking`` argument of every call so far, in call order.
    """
    blocking_calls: list[bool] = []

    def host_bytes(address: int, size: int) -> np.ndarray:
        return np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(address))

    def to_host(cache, regions, blocking=True, cq_id=None):
        blocking_calls.append(blocking)
        for chip, offset, size, address in np.asarray(regions, dtype=np.uint64):
            data = cache.chips[int(chip)][int(offset) : int(offset + size)]
            host_bytes(int(address), int(size))[:] = data

    def to_device(cache, regions, blocking=True, cq_id=None):
        blocking_calls.append(blocking)
        for chip, offset, size, address in np.asarray(regions, dtype=np.uint64):
            data = host_bytes(int(address), int(size))
            cache.chips[int(chip)][int(offset) : int(offset + size)] = data

    monkeypatch.setattr(ttnn, "copy_device_to_host_regions", to_host, raising=False)
    monkeypatch.setattr(ttnn, "copy_host_to_device_regions", to_device, raising=False)
    return blocking_calls
