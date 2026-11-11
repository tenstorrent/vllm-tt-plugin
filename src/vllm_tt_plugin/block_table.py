# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Host block tables of the persistent input batch, one per KV cache group.

Adapted from ``BlockTable`` and ``MultiGroupBlockTable`` in vLLM v0.31.0
``vllm/v1/worker/block_table.py``. Only the host table is kept: the TT runner
slices it into the page table it sends to the device, so there is no device
copy, slot mapping, or context parallelism here.
"""

import math

import numpy as np
import torch
from vllm.utils.math_utils import cdiv

# vLLM's default row alignment for block tables, in tokens.
_TOKEN_ALIGNMENT = 128


def get_block_table_width(max_num_blocks: int, block_size: int) -> int:
    """Blocks per row, rounded up so a row covers a multiple of 128 tokens."""
    block_alignment = _TOKEN_ALIGNMENT // math.gcd(_TOKEN_ALIGNMENT, block_size)
    return cdiv(max_num_blocks, block_alignment) * block_alignment


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        kernel_block_size: int,
    ):
        """Args:
        block_size: Block size used for KV cache memory allocation.
        max_num_reqs: Rows in the table.
        max_num_blocks_per_req: Allocation blocks per row.
        kernel_block_size: Block size of the attention kernel. Equal to
            ``block_size`` unless allocation blocks are split into smaller
            kernel blocks.
        """
        if kernel_block_size == block_size:
            self.blocks_per_kv_block = 1
        else:
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )
            self.blocks_per_kv_block = block_size // kernel_block_size
        self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
            1, -1
        )

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block
        self.block_table_cpu = torch.zeros(
            (max_num_reqs, self.max_num_blocks_per_req), dtype=torch.int32
        )
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        if not block_ids:
            return
        if self.blocks_per_kv_block > 1:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table_np[row_idx, start : start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_np[tgt, :num_blocks] = self.block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks
        # A vacated row must not keep block ids that the KV cache manager may
        # free and hand to another request.
        self.block_table_np[src, :num_blocks] = 0
        self.num_blocks_per_row[src] = 0

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Expand each allocation block id into its kernel block ids.

        With 32-token allocation blocks and 16-token kernel blocks, allocation
        blocks ``[0, 1, 2]`` become kernel blocks ``[0, 1, 2, 3, 4, 5]``.
        """
        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )
        return kernel_block_ids.reshape(-1)

    def get_cpu_tensor(self) -> torch.Tensor:
        return self.block_table_cpu


class MultiGroupBlockTable:
    """The block tables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int],
    ) -> None:
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                get_block_table_width(num_blocks, block_size),
                kernel_block_size,
            )
            for block_size, kernel_block_size, num_blocks in zip(
                block_sizes, kernel_block_sizes, max_num_blocks
            )
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)
