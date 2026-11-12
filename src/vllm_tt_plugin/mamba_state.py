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

With ``MambaSpec.num_prefill_checkpoint_blocks`` 1, a prefill chunk can also
pass the prompt's last cacheable position; the model then writes the state at
that position into a block vLLM reserved (``MambaGroupState.checkpoint_blocks``,
``MambaStepState.checkpoint_positions``) while it continues the prompt.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

from vllm_tt_plugin.config import is_tt_block_output_model, uses_tt_lane_coordinator
from vllm_tt_plugin.mamba_checkpoints import prefill_checkpoints

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
    # ``[rows, 1]`` when a row of this prefill writes a checkpoint, else
    # ``None``: the block that takes row ``i``'s state after
    # ``MambaStepState.checkpoint_positions[i, 0]`` tokens, 0 for no
    # checkpoint.
    checkpoint_blocks: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MambaStepState:
    """The recurrent-state inputs of one forward, the model's ``mamba_state``."""

    # One per ``MambaSpec`` KV cache group, in KV cache group order.
    groups: tuple[MambaGroupState, ...]
    # CPU int32 ``[rows, 1]`` when a row of this prefill writes a checkpoint,
    # else ``None``: the token count whose state row ``i`` writes into each
    # group's ``checkpoint_blocks[i, 0]``, strictly between the row's start
    # and end position; 0 for no checkpoint.
    checkpoint_positions: torch.Tensor | None = None


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
    # ``MambaSpec.prefill_checkpoint_alignment`` with
    # ``num_prefill_checkpoint_blocks`` 1, the same in every Mamba group;
    # ``None`` without prefill checkpoint blocks.
    prefill_checkpoint_alignment: int | None = None
    # ``cache_config.prefix_match_unit``, the token granularity of prefix
    # hits and of vLLM's prefill checkpoint positions; ``block_size`` when
    # ``None``.
    prefix_match_unit: int | None = None

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
        for g in self.group_indices:
            width = block_tables_per_group[g].shape[1]
            if live and int(dest_columns.max()) >= width:
                raise RuntimeError(
                    f"requests {list(row_req_ids)} need state block-table "
                    f"columns {dest_columns.tolist()} in KV cache group {g}, "
                    f"but the table has {width} columns"
                )
        rows = np.arange(live)
        # A row writes the checkpoint vLLM's rule names where vLLM reserved
        # its block. Every Mamba group reserves alike, so the first group's
        # row decides, and ``_check_rows`` raises where another group's
        # differs.
        checkpoints = self._checkpoint_columns(
            num_computed_tokens, num_scheduled_tokens
        )
        first_table = block_tables_per_group[self.group_indices[0]][:live].numpy()
        reserved = [
            c is not None and bool(first_table[row, c[1]])
            for row, c in enumerate(checkpoints)
        ]
        positions = np.array(
            [c[0] if r else 0 for c, r in zip(checkpoints, reserved)], dtype=np.int64
        ).reshape(live)
        checkpoint_columns = np.array(
            [c[1] if r else -1 for c, r in zip(checkpoints, reserved)], dtype=np.int64
        ).reshape(live)
        groups = []
        for g, names in zip(self.group_indices, self.layer_names):
            table = block_tables_per_group[g][:live].numpy()
            dest = table[rows, dest_columns]
            start = np.where(starts > 0, table[rows, np.maximum(start_columns, 0)], 0)
            checkpoint = np.where(
                checkpoint_columns >= 0,
                table[rows, np.maximum(checkpoint_columns, 0)],
                0,
            )
            self._check_rows(
                g,
                row_req_ids,
                table,
                starts,
                start_columns,
                dest_columns,
                checkpoint_columns,
            )
            windows = torch.zeros((num_rows, 1), dtype=torch.int32)
            windows[:live, 0] = torch.from_numpy(dest.astype(np.int32))
            start_blocks = torch.zeros(num_rows, dtype=torch.int32)
            start_blocks[:live] = torch.from_numpy(start.astype(np.int32))
            checkpoint_blocks = None
            if positions.any():
                checkpoint_blocks = torch.zeros((num_rows, 1), dtype=torch.int32)
                checkpoint_blocks[:live, 0] = torch.from_numpy(
                    checkpoint.astype(np.int32)
                )
            groups.append(
                MambaGroupState(
                    layer_names=names,
                    windows=windows,
                    start_blocks=start_blocks,
                    checkpoint_blocks=checkpoint_blocks,
                )
            )
        checkpoint_positions = None
        if positions.any():
            checkpoint_positions = torch.zeros((num_rows, 1), dtype=torch.int32)
            checkpoint_positions[:live, 0] = torch.from_numpy(
                positions.astype(np.int32)
            )
        return MambaStepState(
            groups=tuple(groups), checkpoint_positions=checkpoint_positions
        )

    def _checkpoint_columns(
        self,
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
    ) -> list[tuple[int, int] | None]:
        """Each row's ``(position, column)`` prefill checkpoint by vLLM's rule,
        or ``None``; all ``None`` without prefill checkpoint blocks."""
        if self.prefill_checkpoint_alignment is None:
            return [None] * len(num_computed_tokens)
        return prefill_checkpoints(
            [int(c) for c in num_computed_tokens],
            [int(s) for s in num_scheduled_tokens],
            mamba_block_size=self.block_size,
            prefix_match_unit=self.prefix_match_unit or self.block_size,
            checkpoint_alignment=self.prefill_checkpoint_alignment,
        )

    def _check_rows(
        self,
        group: int,
        row_req_ids: Sequence[str],
        table: np.ndarray,
        starts: np.ndarray,
        start_columns: np.ndarray,
        dest_columns: np.ndarray,
        checkpoint_columns: np.ndarray,
    ) -> None:
        """Raise for a row whose blocks vLLM did not allocate as expected.

        vLLM gives every scheduled request its state block before the step,
        and block 0 is the null block that padding rows share. The block of
        the start state is the running block of the request's previous step,
        which vLLM frees only after this step. In mode ``align`` vLLM leaves
        block 0 in every column a step's tokens pass without ending there,
        except the column of a prefill checkpoint (``checkpoint_columns``, -1
        for none), which every Mamba group reserves alike. A block in another
        passed column would be one the model never writes and vLLM could
        cache.
        """
        for row, req_id in enumerate(row_req_ids):
            start_column = int(start_columns[row])
            dest_column = int(dest_columns[row])
            checkpoint_column = int(checkpoint_columns[row])
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
            if checkpoint_column >= 0 and not table[row, checkpoint_column]:
                raise RuntimeError(
                    f"request {req_id} has a prefill checkpoint block in "
                    f"block-table column {checkpoint_column} of KV cache group "
                    f"{self.group_indices[0]} but not of group {group}: "
                    f"{table[row].tolist()}"
                )
            first = max(start_column + 1, 0)
            passed = [
                column
                for column in range(first, dest_column)
                if table[row, column] and column != checkpoint_column
            ]
            if passed:
                raise RuntimeError(
                    f"request {req_id} passes block-table columns {first} to "
                    f"{dest_column - 1} of KV cache group {group} in this step, "
                    f"but vLLM allocated blocks in columns {passed} that the "
                    "model does not write (vLLM's prefill checkpoint rule names "
                    f"column {checkpoint_column if checkpoint_column >= 0 else None}): "
                    f"{table[row].tolist()}"
                )

    def refuse_block_copies(self, copies: Sequence[KVCacheBlockCopy] | None) -> None:
        """Raise if vLLM queued ``SchedulerOutput.kv_cache_block_copies``.

        vLLM 0.31 queues copy-on-write copies only for a prefix-cache hit that
        ends inside a block. Mode ``none`` has no prefix caching. In mode
        ``align`` such a hit needs a ``prefix_match_unit`` below the block
        size, which ``TTScheduler`` rejects for a model with MambaSpec groups:
        its prefill chunk alignment is the KV block size
        (``get_tt_prefill_chunk_alignment``), and ``prefix_match_unit`` must
        be a multiple of the alignment. The TT model has no copy hook.
        """
        if copies:
            raise RuntimeError(
                "vLLM requested KV cache block copies (src, dst) "
                f"{[(int(c.src_block_id), int(c.dst_block_id)) for c in copies]} "
                "for a model with MambaSpec groups in mamba_cache_mode "
                f"{self.mamba_cache_mode!r}; TT copies no KV cache blocks, and "
                "vLLM queues them only for prefix-cache hits that end inside a "
                "block"
            )


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
    model_config = vllm_config.model_config
    if (
        getattr(model_config, "is_multimodal_model", False)
        and model_config.supports_multimodal_inputs
        and (mode != "none" or vllm_config.scheduler_config.enable_chunked_prefill)
    ):
        # The platform turns both off for such a launch
        # (_resolve_mamba_multimodal_launch) when vLLM resolved a Mamba cache
        # config for the architecture; this catches a launch it could not see.
        raise ValueError(
            "A model with MambaSpec KV cache groups cannot continue a multimodal "
            f"prompt yet, but this launch accepts multimodal input with "
            f"mamba_cache_mode={mode!r} and enable_chunked_prefill="
            f"{vllm_config.scheduler_config.enable_chunked_prefill}. Launch with "
            "--no-enable-prefix-caching --no-enable-chunked-prefill, or with "
            "--language-model-only"
        )
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
        if spec.num_speculative_blocks:
            raise ValueError(
                "MambaSpec has num_speculative_blocks="
                f"{spec.num_speculative_blocks}; TT supports no speculative "
                "state blocks yet, so it must be 0"
            )
        if spec.num_prefill_checkpoint_blocks and (
            mode != "align"
            or spec.num_prefill_checkpoint_blocks != 1
            or not spec.prefill_checkpoint_alignment
        ):
            # MambaManager reserves at most one checkpoint block per request,
            # only in mode align, and never without an alignment.
            raise ValueError(
                "MambaSpec has num_prefill_checkpoint_blocks="
                f"{spec.num_prefill_checkpoint_blocks} with "
                f"prefill_checkpoint_alignment={spec.prefill_checkpoint_alignment} "
                f"in mamba_cache_mode={mode!r}; prefill checkpoints need mode "
                "'align', exactly 1 block and a positive alignment"
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


def declares_mamba_spec(model_class: type, vllm_config: VllmConfig) -> bool:
    """Whether ``model_class.get_kv_cache_spec`` declares ``MambaSpec`` layers."""
    # The worker imports the model runner, which imports this module.
    from vllm_tt_plugin.worker import kv_cache_spec_from_model_class

    specs = kv_cache_spec_from_model_class(model_class, vllm_config)
    return specs is not None and any(isinstance(s, MambaSpec) for s in specs.values())


def mamba_attention_block_size(
    kv_cache_groups: Sequence[KVCacheGroupSpec],
) -> int | None:
    """The KV block size a prefill chunk of a model with ``MambaSpec`` groups
    must start on, or ``None`` without ``MambaSpec`` groups.

    That is the attention block size: the TT model writes a chunk's attention
    keys and values only from the start of a block. Several attention groups
    give the least common multiple of their block sizes.
    """
    if not any(isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_groups):
        return None
    return math.lcm(
        *(
            g.kv_cache_spec.block_size
            for g in kv_cache_groups
            if not isinstance(g.kv_cache_spec, MambaSpec)
        )
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
    block_size = block_sizes.pop()
    checkpoints = {
        (spec.num_prefill_checkpoint_blocks, spec.prefill_checkpoint_alignment)
        for spec in specs
    }
    if len(checkpoints) != 1:
        # vLLM's scheduler splits prefill with the first Mamba group's
        # alignment, and each group reserves its own checkpoint block.
        raise ValueError(
            f"MambaSpec groups {list(group_indices)} have different "
            "(num_prefill_checkpoint_blocks, prefill_checkpoint_alignment) "
            f"{sorted(checkpoints, key=str)}; declare one for every group"
        )
    num_checkpoint_blocks, checkpoint_alignment = checkpoints.pop()
    return MambaCacheGroups(
        group_indices=group_indices,
        layer_names=tuple(tuple(kv_cache_groups[i].layer_names) for i in group_indices),
        attention_group_index=attention_group_index,
        mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
        block_size=block_size,
        prefill_checkpoint_alignment=(
            checkpoint_alignment if num_checkpoint_blocks else None
        ),
        prefix_match_unit=getattr(vllm_config.cache_config, "prefix_match_unit", None),
    )


__all__ = [
    "SUPPORTED_MAMBA_CACHE_MODES",
    "MambaCacheGroups",
    "MambaGroupState",
    "MambaStepState",
    "check_mamba_launch",
    "declares_mamba_spec",
    "mamba_attention_block_size",
    "mamba_cache_groups",
]
