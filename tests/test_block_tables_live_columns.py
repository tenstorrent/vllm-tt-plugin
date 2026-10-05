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
