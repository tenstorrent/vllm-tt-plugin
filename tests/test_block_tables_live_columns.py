# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""The block tables handed to the model carry no stale columns.

vLLM's ``BlockTable`` tracks only each row's live block count; condense and
move leave earlier occupants' block ids past that prefix. A model that writes
padded rows through those columns reaches another request's KV, so the plugin
zeroes them (0 is the null block, never allocated).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm  # noqa: F401

from vllm_tt_plugin.input_batch import InputBatch


class _FakeBlockTable:
    def __init__(self, table, live):
        self._table = torch.tensor(table, dtype=torch.int32)
        self.num_blocks_per_row = np.array(live, dtype=np.int32)

    def get_cpu_tensor(self):
        return self._table


def _batch(*tables):
    return SimpleNamespace(block_table=SimpleNamespace(block_tables=list(tables)))


def test_columns_past_the_live_prefix_are_zeroed_per_row_and_group():
    stale_a = _FakeBlockTable(
        [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]], live=[2, 4, 0]
    )
    stale_b = _FakeBlockTable(
        [[51, 52, 53, 54], [61, 62, 63, 64], [71, 72, 73, 74]], live=[1, 3, 4]
    )
    out = InputBatch.block_tables_for_rows(_batch(stale_a, stale_b), [0, 1, 2], width=4)
    assert out[0].tolist() == [[11, 12, 0, 0], [21, 22, 23, 24], [0, 0, 0, 0]]
    assert out[1].tolist() == [[51, 0, 0, 0], [61, 62, 63, 0], [71, 72, 73, 74]]
    # The plugin's own tensors are untouched.
    assert stale_a.get_cpu_tensor()[0].tolist() == [11, 12, 13, 14]


def test_row_selection_width_padding_and_truncation_still_hold():
    bt = _FakeBlockTable([[1, 2, 3], [4, 5, 6]], live=[3, 1])
    wide = InputBatch.block_tables_for_rows(_batch(bt), torch.tensor([1, 0]), width=5)
    assert wide[0].tolist() == [[4, 0, 0, 0, 0], [1, 2, 3, 0, 0]]
    narrow = InputBatch.block_tables_for_rows(_batch(bt), [0], width=2)
    assert narrow[0].tolist() == [[1, 2]]


def test_a_singleton_tensor_selector_keeps_its_row_dimension():
    bt = _FakeBlockTable([[1, 2, 3], [4, 5, 6]], live=[3, 1])
    out = InputBatch.block_tables_for_rows(_batch(bt), torch.tensor([1]), width=3)
    assert out[0].shape == (1, 3)
    assert out[0].tolist() == [[4, 0, 0]]


@pytest.mark.parametrize("tensor_selector", [False, True])
def test_fixed_width_snapshots_survive_growth_move_remove_and_reuse(tensor_selector):
    table = _FakeBlockTable(
        [[11, 12, 99, 98], [21, 97, 96, 95], [31, 32, 33, 94]], [2, 1, 3]
    )
    batch = _batch(table)

    def select(rows):
        selector = torch.tensor(rows, dtype=torch.long) if tensor_selector else rows
        return InputBatch.block_tables_for_rows(batch, selector, width=8192)[0]

    original = select([0, 1, 2])
    assert original.shape == (3, 8192)
    assert original[:, :4].tolist() == [[11, 12, 0, 0], [21, 0, 0, 0], [31, 32, 33, 0]]
    # Growth inside the backing width, followed by a row permutation.
    table._table[1, 1] = 22
    table.num_blocks_per_row[1] = 2
    moved = select([2, 1, 0])
    assert moved[:, :4].tolist() == [[31, 32, 33, 0], [21, 22, 0, 0], [11, 12, 0, 0]]
    # Removing all live blocks and reusing a row must not expose stale suffixes.
    table.num_blocks_per_row[0] = 0
    removed = select([0])
    assert not removed.any()
    table._table[0, 0] = 41
    table.num_blocks_per_row[0] = 1
    reused = select([0])
    assert reused[0, :4].tolist() == [41, 0, 0, 0]
    assert not reused[:, 4:].any()
    # A queued submission retains the original mapping after every mutation.
    assert original[:, :4].tolist() == [[11, 12, 0, 0], [21, 0, 0, 0], [31, 32, 33, 0]]
    assert moved[2, :4].tolist() == [11, 12, 0, 0]
    assert table._table[0].tolist() == [41, 12, 99, 98]


@pytest.mark.parametrize("rows", [[0], torch.tensor([0]), torch.tensor([True, False])])
def test_unpadded_selection_owns_storage_without_extra_clone(rows):
    table = _FakeBlockTable([[11, 12, 99, 98], [21, 97, 96, 95]], [2, 1])
    original = table._table.clone()
    out = InputBatch.block_tables_for_rows(_batch(table), rows, width=4)[0]
    assert out.data_ptr() != table._table.data_ptr()
    assert torch.equal(table._table, original)
    out.fill_(7)
    assert torch.equal(table._table, original)
