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

The TT runner implements ``mamba_cache_mode`` ``none`` and ``align`` without
speculative decoding. The upstream GPU model runner, not the scheduler, picks
the state blocks of each request from the group's block-table row
(``mamba_get_block_table_tensor``, and in mode ``align`` ``preprocess_mamba``);
this module does the same and hands the blocks to the model as
``mamba_state``.

In mode ``none`` a request holds one state block per Mamba group for its whole
life, in block-table column 0. In mode ``align`` the running state after ``n``
tokens is in column ``(n - 1) // block_size``. When a step's tokens cross into
a new column, upstream ``preprocess_mamba`` copies the running state into the
block of the new column and the kernels update that block in place, so the
block of the earlier column keeps the state of its boundary for the prefix
cache. The TT runner does not copy: it names the block that holds the start
state (``MambaGroupState.start_blocks``) and the block that takes the end state
(``MambaGroupState.windows``), and the model reads the one and writes the
other.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

from vllm_tt_plugin.config import is_tt_block_output_model, uses_tt_lane_coordinator

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

# ``CacheConfig.mamba_cache_mode`` values the TT runner implements.
SUPPORTED_MAMBA_CACHE_MODES = ("none", "align")


@dataclass(frozen=True, slots=True)
class MambaGroupState:
    """The recurrent-state inputs of one forward for one ``MambaSpec`` group.

    The tensors are CPU int32 with one row per submitted row. Padding rows
    hold block 0, the null block vLLM never allocates.
    """

    # The group's layers, ``KVCacheGroupSpec.layer_names``.
    layer_names: tuple[str, ...]
    # ``[rows, 1]``: the block that takes row ``i``'s end state in this group.
    windows: torch.Tensor
    # ``[rows]``: the block that holds row ``i``'s start state in this group,
    # 0 when the row starts at position 0 and has no earlier state. It is
    # ``windows[i, 0]`` unless the row's tokens cross into a new block-table
    # column in mode ``align``.
    start_blocks: torch.Tensor


@dataclass(frozen=True, slots=True)
class MambaStepState:
    """The recurrent-state inputs of one forward, the model's ``mamba_state``."""

    # One per ``MambaSpec`` KV cache group, in KV cache group order.
    groups: tuple[MambaGroupState, ...]


@dataclass(frozen=True)
class MambaCacheGroups:
    """The ``MambaSpec`` groups of one ``KVCacheConfig``."""

    # Indices into ``KVCacheConfig.kv_cache_groups``, in group order.
    group_indices: tuple[int, ...]
    # Each Mamba group's ``KVCacheGroupSpec.layer_names``, in the order of
    # ``group_indices``.
    layer_names: tuple[tuple[str, ...], ...]
    # The first attention group; its block table is the ``page_table`` the
    # model receives. vLLM orders groups by spec, so group 0 can be a Mamba
    # group.
    attention_group_index: int
    # ``CacheConfig.mamba_cache_mode``; every Mamba group's spec carries it.
    mamba_cache_mode: str
    # ``MambaSpec.block_size``, the same in every Mamba group.
    block_size: int

    def step_state(
        self,
        block_tables_per_group: Sequence[torch.Tensor],
        row_req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        num_rows: int,
    ) -> MambaStepState:
        """The model's ``mamba_state`` for one forward.

        Row ``i`` of every block table belongs to ``row_req_ids[i]``, which
        has computed ``num_computed_tokens[i]`` tokens before this step and
        computes ``num_scheduled_tokens[i]`` in it: the counts upstream
        ``preprocess_mamba`` and ``mamba_get_block_table_tensor`` select the
        blocks from. The result has ``num_rows`` rows, the row count of
        ``page_tables_per_layer``; rows past ``row_req_ids`` hold block 0.
        """
        live = len(row_req_ids)
        if live > num_rows:
            raise RuntimeError(
                f"{live} scheduled rows {list(row_req_ids)} exceed the "
                f"{num_rows} rows of the Mamba state inputs"
            )
        if not live == len(num_computed_tokens) == len(num_scheduled_tokens):
            raise RuntimeError(
                f"{live} scheduled rows {list(row_req_ids)} got "
                f"{len(num_computed_tokens)} computed-token counts and "
                f"{len(num_scheduled_tokens)} scheduled-token counts"
            )
        starts = np.asarray(num_computed_tokens, dtype=np.int64).reshape(live)
        ends = starts + np.asarray(num_scheduled_tokens, dtype=np.int64).reshape(live)
        if self.mamba_cache_mode == "none":
            # The request's one state block.
            start_columns = dest_columns = np.zeros(live, dtype=np.int64)
        else:
            # The running state after n tokens is in column (n - 1) //
            # block_size; a row at position 0 has none (column -1).
            start_columns = (starts - 1) // self.block_size
            dest_columns = (ends - 1) // self.block_size
        rows = np.arange(live)
        groups = []
        for g, names in zip(self.group_indices, self.layer_names):
            table = block_tables_per_group[g][:live].numpy()
            if live and int(dest_columns.max()) >= table.shape[1]:
                raise RuntimeError(
                    f"requests {list(row_req_ids)} need state block-table "
                    f"columns {dest_columns.tolist()} in KV cache group {g}, "
                    f"but the table has {table.shape[1]} columns"
                )
            dest = table[rows, dest_columns]
            start = np.where(starts > 0, table[rows, np.maximum(start_columns, 0)], 0)
            self._check_rows(g, row_req_ids, table, starts, start_columns, dest_columns)
            windows = torch.zeros((num_rows, 1), dtype=torch.int32)
            windows[:live, 0] = torch.from_numpy(dest.astype(np.int32))
            start_blocks = torch.zeros(num_rows, dtype=torch.int32)
            start_blocks[:live] = torch.from_numpy(start.astype(np.int32))
            groups.append(
                MambaGroupState(
                    layer_names=names, windows=windows, start_blocks=start_blocks
                )
            )
        return MambaStepState(groups=tuple(groups))

    @staticmethod
    def _check_rows(
        group: int,
        row_req_ids: Sequence[str],
        table: np.ndarray,
        starts: np.ndarray,
        start_columns: np.ndarray,
        dest_columns: np.ndarray,
    ) -> None:
        """Raise for a row whose blocks vLLM did not allocate as expected.

        vLLM gives every scheduled request its state block before the step,
        and block 0 is the null block that padding rows share. The block of
        the start state is the running block of the request's previous step,
        which vLLM frees only after this step. Mode ``align`` without prefill
        checkpoint blocks leaves block 0 in every column a step's tokens pass
        without ending there, so a block there would be one the model never
        writes and vLLM could cache.
        """
        for row, req_id in enumerate(row_req_ids):
            start_column = int(start_columns[row])
            dest_column = int(dest_columns[row])
            if not table[row, dest_column]:
                raise RuntimeError(
                    f"request {req_id} has no state block in KV cache group "
                    f"{group}; block-table column {dest_column} holds the null "
                    f"block 0: {table[row].tolist()}"
                )
            if starts[row] > 0 and not table[row, start_column]:
                raise RuntimeError(
                    f"request {req_id} starts at position {int(starts[row])}, "
                    f"but block-table column {start_column} of KV cache group "
                    f"{group} holds the null block 0 instead of its start "
                    f"state: {table[row].tolist()}"
                )
            passed = table[row, max(start_column + 1, 0) : dest_column]
            if passed.any():
                raise RuntimeError(
                    f"request {req_id} passes block-table columns "
                    f"{max(start_column + 1, 0)} to {dest_column - 1} of KV "
                    f"cache group {group} in this step, but vLLM allocated "
                    f"blocks there that the model does not write: "
                    f"{table[row].tolist()}"
                )

    def block_copies(
        self, copies: Sequence[KVCacheBlockCopy] | None
    ) -> list[tuple[int, int]]:
        """``SchedulerOutput.kv_cache_block_copies`` as ``(src, dst)`` pairs.

        vLLM queues copies for prefix-cache hits that end inside a block
        (copy-on-write of the shared block), which mode ``none`` never has.
        No block may be the destination of two copies or both a source and a
        destination, so applying the copies in list order and applying them
        all at once give the same result.
        """
        if not copies:
            return []
        pairs = [(int(c.src_block_id), int(c.dst_block_id)) for c in copies]
        if self.mamba_cache_mode == "none":
            raise RuntimeError(
                f"vLLM requested KV cache block copies (src, dst) {pairs} for a "
                "model with MambaSpec groups in mamba_cache_mode 'none', which "
                "runs without prefix caching"
            )
        destinations = [dst for _, dst in pairs]
        overlap = {src for src, _ in pairs} & set(destinations)
        if overlap or len(set(destinations)) != len(destinations):
            raise RuntimeError(
                f"KV cache block copies (src, dst) {pairs} write a block twice "
                f"or read a block another copy writes ({sorted(overlap)}); "
                "the copies would depend on the order the model applies them in"
            )
        return pairs


def check_mamba_launch(
    vllm_config: VllmConfig,
    specs: Iterable[MambaSpec],
    attention_block_sizes: Iterable[int],
) -> None:
    """Raise for a launch with ``MambaSpec`` layers that TT does not implement.

    ``specs`` are the model's ``MambaSpec`` layer specs, or the specs of the
    Mamba groups, and ``attention_block_sizes`` the block sizes of the other
    layers or groups. The worker calls this before it loads the model, and the
    runner again for the KV cache groups.
    """
    if uses_tt_lane_coordinator(vllm_config):
        raise ValueError(
            "MambaSpec KV cache groups are not supported with single-process "
            "lane-DP; launch with standard multi-process data parallelism"
        )
    if is_tt_block_output_model(vllm_config):
        raise ValueError(
            "Block-output models (output_tokens_per_step > 1) do not support "
            "MambaSpec KV cache groups"
        )
    if vllm_config.speculative_config is not None:
        raise ValueError(
            "Speculative decoding is not supported with MambaSpec KV cache "
            "groups on TT yet; launch without --speculative-config"
        )
    mode = vllm_config.cache_config.mamba_cache_mode
    if mode not in SUPPORTED_MAMBA_CACHE_MODES:
        raise ValueError(
            f"mamba_cache_mode={mode!r} is not supported with MambaSpec KV "
            f"cache groups on TT (supported: {list(SUPPORTED_MAMBA_CACHE_MODES)})"
        )
    if mode == "align":
        _check_align_launch(vllm_config)
    attention_block_sizes = set(attention_block_sizes)
    max_model_len = vllm_config.model_config.max_model_len
    for spec in specs:
        if spec.mamba_cache_mode != mode:
            raise ValueError(
                f"MambaSpec has mamba_cache_mode={spec.mamba_cache_mode!r}, "
                f"but cache_config.mamba_cache_mode={mode!r}; the model's "
                "get_kv_cache_spec must use the resolved cache mode"
            )
        if spec.num_speculative_blocks or spec.num_prefill_checkpoint_blocks:
            raise ValueError(
                "MambaSpec has num_speculative_blocks="
                f"{spec.num_speculative_blocks} and num_prefill_checkpoint_blocks="
                f"{spec.num_prefill_checkpoint_blocks}; TT supports neither "
                "speculative nor prefill-checkpoint state blocks yet, so both "
                "must be 0"
            )
        if mode == "none" and spec.block_size < max_model_len:
            # MambaManager allocates one block per block_size tokens in mode
            # none, so a smaller block size gives a request several state
            # blocks per group, which neither the pool sizing nor the state
            # inputs count.
            raise ValueError(
                f"MambaSpec block_size {spec.block_size} is below max_model_len "
                f"{max_model_len} in mamba_cache_mode='none'; declare block_size "
                "as cache_config.mamba_block_size, which vLLM resolves to "
                "max_model_len"
            )
        if mode == "align" and attention_block_sizes != {spec.block_size}:
            # The scheduler ends prefill chunks on multiples of
            # cache_config.block_size, the smallest group block size, and a
            # Mamba block-table column must cover the same tokens as an
            # attention one, as upstream's _align_hybrid_block_size arranges.
            raise ValueError(
                f"MambaSpec block_size {spec.block_size} differs from the "
                f"attention block sizes {sorted(attention_block_sizes)} in "
                "mamba_cache_mode='align'; declare one block_size for every "
                "KV cache spec"
            )


def _check_align_launch(vllm_config: VllmConfig) -> None:
    """Raise for a ``mamba_cache_mode`` ``align`` launch TT does not implement."""
    interval = getattr(
        vllm_config.cache_config, "prefix_cache_retention_interval", None
    )
    if interval != 0:
        # A nonzero interval also keeps the state of block boundaries that
        # decoding reaches; TT validates only the states vLLM keeps at
        # interval 0: prompt boundaries and shared-prefix junctions.
        raise ValueError(
            f"prefix_cache_retention_interval={interval!r} is not supported "
            "with MambaSpec KV cache groups on TT; only 0 is. Launch without "
            "--prefix-cache-retention-interval or with "
            "--prefix-cache-retention-interval 0"
        )
    if not vllm_config.scheduler_config.enable_chunked_prefill:
        # vLLM's _mamba_block_aligned_split ends prefill chunks on block
        # boundaries whether or not chunked prefill is enabled.
        raise ValueError(
            "mamba_cache_mode='align' prefills prompts in chunks that end on "
            "block boundaries, so a model with MambaSpec KV cache groups needs "
            "chunked prefill (model_capabilities['supports_chunked_prefill'] "
            "and --enable-chunked-prefill) to use prefix caching; launch with "
            "--no-enable-prefix-caching otherwise"
        )


def mamba_cache_groups(
    kv_cache_groups: Sequence[KVCacheGroupSpec], vllm_config: VllmConfig
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
    specs = [kv_cache_groups[i].kv_cache_spec for i in group_indices]
    check_mamba_launch(
        vllm_config,
        specs,
        (
            group.kv_cache_spec.block_size
            for i, group in enumerate(kv_cache_groups)
            if i not in group_indices
        ),
    )
    block_sizes = {spec.block_size for spec in specs}
    if len(block_sizes) != 1:
        raise ValueError(
            f"MambaSpec groups {list(group_indices)} have different block sizes "
            f"{sorted(block_sizes)}; upstream state selection assumes one"
        )
    return MambaCacheGroups(
        group_indices=group_indices,
        layer_names=tuple(tuple(kv_cache_groups[i].layer_names) for i in group_indices),
        attention_group_index=attention_group_index,
        mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
        block_size=block_sizes.pop(),
    )


__all__ = [
    "SUPPORTED_MAMBA_CACHE_MODES",
    "MambaCacheGroups",
    "MambaGroupState",
    "MambaStepState",
    "check_mamba_launch",
    "mamba_cache_groups",
]
