# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Runner-side handling of vLLM ``MambaSpec`` KV cache groups.

A model whose ``get_kv_cache_spec`` returns ``MambaSpec`` for its recurrent
layers gets one or more Mamba groups next to its attention groups. vLLM's
hybrid KV cache manager allocates the block ids of every group from one pool
and overlays the groups on the same memory: block id ``b`` names the same
bytes in every group, and a block id belongs to one group at a time. The TT
model keeps that overlay. It allocates one buffer per layer slot (the
``tensor_idx`` of ``allocate_kv_cache_per_layer``) and keeps each Mamba
layer's state inside the buffer of its slot, at the block ids that layer's
group owns.

The upstream GPU model runner, not the scheduler, decides which block of a
Mamba group's block-table row holds a request's state
(``mamba_get_block_table_tensor``) and, in mode ``align``, which state copies
must run before the forward (``preprocess_mamba``). This module ports both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_tt_plugin.input_batch import TTCachedRequestState

# ``CacheConfig.mamba_cache_mode`` values the TT runner implements.
SUPPORTED_MAMBA_CACHE_MODES = ("none", "align")


@dataclass
class MambaCacheGroups:
    """The ``MambaSpec`` groups of one ``KVCacheConfig``."""

    # Indices into ``KVCacheConfig.kv_cache_groups``, in group order.
    group_indices: tuple[int, ...]
    # ``CacheConfig.mamba_cache_mode``; every Mamba group's spec carries it.
    mamba_cache_mode: str
    # ``MambaSpec.block_size``, the same in every Mamba group.
    block_size: int
    # The first attention group; its block table is the ``page_table`` the
    # model receives. vLLM orders groups by spec, so group 0 can be a Mamba
    # group.
    attention_group_index: int
    # Mode ``align`` only: request id -> the block-table column that holds the
    # request's running state after its last scheduled step. Upstream keeps
    # the same map as ``GPUModelRunner.mamba_state_idx``.
    state_columns: dict[str, int] = field(default_factory=dict)

    def state_block_ids(
        self, block_table: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """The state block id of each row of one Mamba group's block table.

        Port of upstream ``mamba_get_block_table_tensor`` without speculative
        blocks. ``seq_lens`` counts each row's tokens after this step. In mode
        ``none`` a request holds one block, in column 0. In mode ``align`` row
        column ``p`` holds the state after ``(p + 1) * block_size`` tokens, so
        the running state is in column ``(seq_len - 1) // block_size``; a row
        with ``seq_len`` 0 reads column 0. Returns int32 ``[rows, 1]``.
        """
        if self.mamba_cache_mode == "none":
            return block_table[:, :1].to(torch.int32)
        columns = ((seq_lens.to(torch.int64) - 1) // self.block_size).clamp_(min=0)
        return torch.gather(block_table, 1, columns.unsqueeze(1)).to(torch.int32)

    def state_block_ids_per_layer(
        self,
        block_tables_per_group: Sequence[torch.Tensor],
        row_req_ids: Sequence[str],
        requests: Mapping[str, TTCachedRequestState],
        num_scheduled_tokens: Mapping[str, int],
        layer_to_group_idx: Sequence[int],
        num_rows: int,
    ) -> list[torch.Tensor | None]:
        """Per-layer state block ids for one forward, ``None`` for attention.

        Row ``i`` of ``block_tables_per_group`` belongs to ``row_req_ids[i]``;
        the rows after them are padding, with zero block tables. A padding row
        gets block 0, the null block vLLM never allocates. The result is padded
        to ``num_rows`` rows with block 0, the row count of
        ``page_tables_per_layer``.
        """
        seq_lens = torch.zeros(block_tables_per_group[0].shape[0], dtype=torch.int64)
        for row, req_id in enumerate(row_req_ids):
            seq_lens[row] = (
                requests[req_id].num_computed_tokens + num_scheduled_tokens[req_id]
            )
        by_group: dict[int, torch.Tensor] = {}
        for g in self.group_indices:
            ids = self.state_block_ids(block_tables_per_group[g], seq_lens)
            padded = torch.zeros((num_rows, 1), dtype=torch.int32)
            rows = min(ids.shape[0], num_rows)
            padded[:rows] = ids[:rows]
            by_group[g] = padded
        return [by_group.get(g) for g in layer_to_group_idx]

    def align_state_copies(
        self,
        scheduler_output: SchedulerOutput,
        requests: Mapping[str, TTCachedRequestState],
    ) -> list[tuple[int, int]]:
        """Mode ``align``: ``(src, dst)`` block copies to run before the forward.

        Port of upstream ``preprocess_mamba`` without speculative decoding.
        When a request's tokens cross into a new block-table column, its
        running state moves to the block of the new column: the model reads
        and writes only that block, so the block that holds the state at the
        previous boundary stays unchanged for prefix-cache hits. The copy
        covers every Mamba group, each with the block ids of its own row.
        """
        if self.mamba_cache_mode != "align":
            return []
        for req_id in (
            *scheduler_output.finished_req_ids,
            *(scheduler_output.preempted_req_ids or ()),
            *scheduler_output.scheduled_cached_reqs.resumed_req_ids,
        ):
            self.state_columns.pop(req_id, None)
        copies: list[tuple[int, int]] = []
        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            req_state = requests[req_id]
            num_computed = req_state.num_computed_tokens
            prev = self.state_columns.get(req_id)
            if prev is None:
                # A new or resumed request; -1 when nothing is computed yet.
                prev = (num_computed - 1) // self.block_size
            curr = cdiv(num_computed + num_scheduled, self.block_size) - 1
            self.state_columns[req_id] = curr
            if prev == -1 or prev == curr:
                continue
            for g in self.group_indices:
                block_ids = req_state.block_ids[g]
                copies.append((block_ids[prev], block_ids[curr]))
        return copies


def mamba_cache_groups(
    kv_cache_groups: Sequence[KVCacheGroupSpec], cache_config: Any
) -> MambaCacheGroups | None:
    """The Mamba groups of ``kv_cache_groups``, or ``None`` when there are none.

    Raises for a Mamba configuration the TT runner does not implement.
    """
    group_indices = tuple(
        i
        for i, group in enumerate(kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    )
    if not group_indices:
        return None
    attention_group_index = next(
        (
            i
            for i, group in enumerate(kv_cache_groups)
            if isinstance(group.kv_cache_spec, AttentionSpec)
        ),
        None,
    )
    if attention_group_index is None:
        raise ValueError(
            "The KV cache config has MambaSpec groups but no attention group. "
            "A TT model keeps Mamba state in the attention layer-slot buffers, "
            "so it needs at least one attention layer; groups: "
            f"{[type(g.kv_cache_spec).__name__ for g in kv_cache_groups]}"
        )
    mode = cache_config.mamba_cache_mode
    for i in group_indices:
        spec = kv_cache_groups[i].kv_cache_spec
        if spec.mamba_cache_mode != mode:
            raise ValueError(
                f"MambaSpec of KV cache group {i} has mamba_cache_mode="
                f"{spec.mamba_cache_mode!r}, but cache_config.mamba_cache_mode="
                f"{mode!r}; the model's get_kv_cache_spec must use the resolved "
                "cache mode"
            )
        if spec.num_speculative_blocks or spec.num_prefill_checkpoint_blocks:
            raise ValueError(
                f"MambaSpec of KV cache group {i} has num_speculative_blocks="
                f"{spec.num_speculative_blocks} and "
                f"num_prefill_checkpoint_blocks="
                f"{spec.num_prefill_checkpoint_blocks}; TT supports neither "
                "speculative nor prefill-checkpoint Mamba blocks yet"
            )
    if mode not in SUPPORTED_MAMBA_CACHE_MODES:
        raise ValueError(
            f"mamba_cache_mode={mode!r} is not supported on TT; supported: "
            f"{list(SUPPORTED_MAMBA_CACHE_MODES)}"
        )
    block_sizes = {kv_cache_groups[i].kv_cache_spec.block_size for i in group_indices}
    if len(block_sizes) != 1:
        raise ValueError(
            f"MambaSpec groups {list(group_indices)} have different block sizes "
            f"{sorted(block_sizes)}; upstream state selection assumes one"
        )
    return MambaCacheGroups(
        group_indices=group_indices,
        mamba_cache_mode=mode,
        block_size=block_sizes.pop(),
        attention_group_index=attention_group_index,
    )
