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

The TT runner implements ``mamba_cache_mode`` ``none`` and ``align``, with
speculative decoding or without it, under synchronous or asynchronous
scheduling. The upstream GPU model runner,
not the scheduler, picks the state blocks of each request from the group's
block-table row (``mamba_get_block_table_tensor``, and in mode ``align``
``preprocess_mamba``); this module does the same and hands the blocks to the
model as ``mamba_state``.

Each step writes into a **window** of ``1 + K`` consecutive block-table
columns, K being ``MambaSpec.num_speculative_blocks`` (0 without speculative
decoding): window column ``t`` takes the state after the step's query token
``t``, and a prefill or an ordinary decode puts its end state into window
column 0. The window begins at column 0 in mode ``none`` and, in mode
``align``, at the column of the state after all the step's tokens
(``window_start_column``). A step that committed ``a`` tokens leaves the
request's state in window column ``a - 1`` (``accepted_state_column``), where
the request's next step reads it. Upstream ``preprocess_mamba`` copies that
state into the next window's column 0 when the window moves, and so keeps the
state of a block boundary in the block of its own column for the prefix cache.
The TT runner does not copy: it names the block that holds the start state
(``MambaGroupState.start_blocks``) and the blocks that take the new states
(``MambaGroupState.windows``), and the model reads the one and writes the
others. Without speculative decoding the window is the running column, and
the state after ``n`` tokens is in column ``(n - 1) // block_size`` in mode
``align``.

With ``MambaSpec.num_prefill_checkpoint_blocks`` 1, a prefill chunk can also
pass the prompt's last cacheable position; the model then writes the state at
that position into a block vLLM reserved (``MambaGroupState.checkpoint_blocks``,
``MambaStepState.checkpoint_positions``) while it continues the prompt.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

from vllm_tt_plugin.config import is_tt_block_output_model, uses_tt_lane_coordinator
from vllm_tt_plugin.mamba_checkpoints import prefill_checkpoints

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.core.sched.output import SchedulerOutput

# ``CacheConfig.mamba_cache_mode`` values the TT runner implements.
SUPPORTED_MAMBA_CACHE_MODES = ("none", "align")


def window_start_column(
    mamba_cache_mode: str, block_size: int, num_computed: int, num_scheduled: int
) -> int:
    """The block-table column of a step's window column 0.

    The column upstream ``mamba_get_block_table_tensor`` starts the window at
    and ``preprocess_mamba`` records as the running column: column 0 in mode
    ``none``; in mode ``align`` the column that holds, by position, the state
    after all ``num_scheduled`` tokens of the step, drafts included.
    """
    if mamba_cache_mode == "none":
        return 0
    return (num_computed + num_scheduled - 1) // block_size


def accepted_state_column(window_column: int, accepted_count: int) -> int:
    """The block-table column of a request's state after a step whose window
    began at ``window_column`` and that committed ``accepted_count`` tokens.

    The step's committed tokens are its first ``accepted_count`` query tokens,
    and the step wrote the state after query token ``t`` into window column
    ``t``. A prefill and an ordinary decode commit a count of 1, their end
    state; a verify commits its column-0 token plus the drafts it accepted.
    """
    return window_column + accepted_count - 1


def restored_state_column(
    mamba_cache_mode: str, block_size: int, num_computed: int
) -> int:
    """The block-table column in which vLLM placed a request's state after
    ``num_computed`` tokens, or -1 when it placed none.

    That is where a new or resumed request's first step finds its start state:
    none at position 0, else the block of a prefix-cache hit, which in mode
    ``align`` vLLM puts into the hit's column. Without speculative decoding
    every later step finds its start state there too, because a step's window
    is then its running column.
    """
    if num_computed == 0:
        return -1
    if mamba_cache_mode == "none":
        return 0
    return (num_computed - 1) // block_size


def _holds_current_block(row: np.ndarray, column: int) -> bool:
    """Whether column ``column`` of a runner block-table row holds a block
    the request still holds there.

    The runner's row only grows: when ``MambaManager`` moves a speculative
    block to the end of the request's row, or frees a block and later gives it
    to the request again, the runner's row lists the block at its old column
    too. So a block that the row lists again in a later column is no longer
    the request's block at ``column``.
    """
    block = row[column]
    return bool(block) and not bool((row[column + 1 :] == block).any())


class MambaWindow(NamedTuple):
    """The window of a request's last step, recorded with speculative
    decoding for the request's next step."""

    # Block-table column of the step's window column 0.
    column: int
    # The last block-table column of any window the request has had. vLLM
    # can leave blocks of earlier windows in columns up to it that a later
    # step passes: ``MambaManager`` keeps unused speculative blocks in place
    # or moves them to the end of the row, and the runner's row still lists a
    # moved block at its old column too.
    end: int


@dataclass(frozen=True, slots=True)
class MambaGroupState:
    """The recurrent-state inputs of one forward for one ``MambaSpec`` group.

    The tensors are CPU int32 with one row per submitted row. Padding rows
    hold block 0, the null block vLLM never allocates.
    """

    # The group's layers, ``KVCacheGroupSpec.layer_names``.
    layer_names: tuple[str, ...]
    # ``[rows, 1 + K]``, K the group's ``MambaSpec.num_speculative_blocks``:
    # column ``t`` is the block that takes row ``i``'s state after its query
    # token ``t`` in this group. A prefill or an ordinary decode writes its
    # end state into column 0 only.
    windows: torch.Tensor
    # ``[rows]``: the block that holds row ``i``'s start state in this group,
    # 0 when the row starts at position 0 and has no earlier state. When it
    # is not one of the row's own ``windows`` blocks the model leaves it
    # unchanged: it can be a block vLLM keeps in the prefix cache.
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
    # Prefill only, else ``None``: CPU bool ``[rows]``, the step's
    # ``TTModelInput.intermediate_prefill_mask``. True for a row whose chunk
    # ends before the request's last token, so the step samples no token for
    # it, and for padding rows; False for a row that ends its prefill.
    intermediate_prefill_mask: torch.Tensor | None = None


@dataclass
class MambaCacheGroups:
    """The ``MambaSpec`` groups of one ``KVCacheConfig``, and the window of
    each request's last step."""

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
    # ``MambaSpec.num_speculative_blocks`` K, the same in every Mamba group
    # and equal to the launch's speculative tokens: a window has ``1 + K``
    # columns.
    num_speculative_blocks: int = 0
    # K > 0 only: request id -> the window of its last step, which upstream
    # keeps as ``mamba_state_idx``. ``step_state`` records it; ``forget``
    # drops the requests whose next step starts from vLLM's placement.
    windows: dict[str, MambaWindow] = field(default_factory=dict)

    def forget(self, scheduler_output: SchedulerOutput) -> None:
        """Drop the windows of finished, preempted and resumed requests, as
        upstream ``cleanup_mamba_state_idx`` does. A resumed request starts
        from position 0 or from a prefix-cache hit, on blocks vLLM gave it
        anew (``restored_state_column``)."""
        for req_id in (
            *scheduler_output.finished_req_ids,
            *(scheduler_output.preempted_req_ids or ()),
            *scheduler_output.scheduled_cached_reqs.resumed_req_ids,
        ):
            self.windows.pop(req_id, None)

    def step_state(
        self,
        block_tables_per_group: Sequence[torch.Tensor],
        row_req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        num_rows: int,
        *,
        accepted_counts: Sequence[int] | None = None,
        intermediate_prefill_mask: Sequence[bool] | None = None,
    ) -> MambaStepState:
        """The model's ``mamba_state`` for one forward. Call it once per step.

        Row ``i`` of every block table belongs to ``row_req_ids[i]``, which
        has computed ``num_computed_tokens[i]`` tokens before this step and
        computes ``num_scheduled_tokens[i]`` in it, drafts included: the counts
        upstream ``preprocess_mamba`` and ``mamba_get_block_table_tensor``
        select the blocks from. ``accepted_counts[i]`` is the number of tokens
        the request's previous step committed (``TTModelInput.accepted_counts``),
        ``None`` meaning 1 for every row. ``intermediate_prefill_mask`` is the
        prefill's ``TTModelInput.intermediate_prefill_mask``, ``None`` in
        decode. The result has ``num_rows`` rows, the row count of
        ``page_tables_per_layer``; rows past ``row_req_ids`` hold block 0.

        Under asynchronous scheduling vLLM schedules a step while the
        request's previous verify is still in flight and counts that
        verify's drafts as accepted in ``num_computed_tokens``.
        ``MambaManager`` allocates by that count, so the window follows it
        too, and the start column follows the recorded window of the
        previous step and its committed count, as without it.

        With speculative decoding this records each row's window
        (``windows``) for the request's next step.
        """
        live = len(row_req_ids)
        if live > num_rows:
            raise RuntimeError(
                f"{live} scheduled rows {list(row_req_ids)} exceed the "
                f"{num_rows} rows of the Mamba state inputs"
            )
        counts = [1] * live if accepted_counts is None else list(accepted_counts)
        if (
            not live
            == len(num_computed_tokens)
            == len(num_scheduled_tokens)
            == len(counts)
        ):
            raise RuntimeError(
                f"{live} scheduled rows {list(row_req_ids)} got "
                f"{len(num_computed_tokens)} computed-token counts, "
                f"{len(num_scheduled_tokens)} scheduled-token counts and "
                f"{len(counts)} accepted counts"
            )
        starts = [int(c) for c in num_computed_tokens]
        scheduled = [int(s) for s in num_scheduled_tokens]
        num_spec = self.num_speculative_blocks
        width = 1 + num_spec
        window_columns = np.array(
            [
                window_start_column(self.mamba_cache_mode, self.block_size, c, s)
                for c, s in zip(starts, scheduled)
            ],
            dtype=np.int64,
        ).reshape(live)
        previous = [self.windows.get(r) if num_spec else None for r in row_req_ids]
        start_columns = np.array(
            [
                self._start_column(req_id, start, int(count), window)
                for req_id, start, count, window in zip(
                    row_req_ids, starts, counts, previous
                )
            ],
            dtype=np.int64,
        ).reshape(live)
        held_ends = np.array(
            [-1 if w is None else w.end for w in previous], dtype=np.int64
        ).reshape(live)
        for g in self.group_indices:
            table_width = block_tables_per_group[g].shape[1]
            if live and int(window_columns.max()) + num_spec >= table_width:
                raise RuntimeError(
                    f"requests {list(row_req_ids)} need state block-table "
                    f"columns {window_columns.tolist()} to "
                    f"{(window_columns + num_spec).tolist()} in KV cache group "
                    f"{g}, but the table has {table_width} columns"
                )
        rows = np.arange(live)
        window_index = window_columns[:, None] + np.arange(width)
        # A row writes the checkpoint vLLM's rule names where vLLM reserved
        # its block. Every Mamba group reserves alike, so the first group's
        # row decides, and ``_check_rows`` raises where another group's
        # differs. A block the runner's row lists again in a later column is
        # a speculative block vLLM moved away, not a reservation.
        checkpoints = self._checkpoint_columns(starts, scheduled)
        first_table = block_tables_per_group[self.group_indices[0]][:live].numpy()
        reserved = [
            c is not None and _holds_current_block(first_table[row], c[1])
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
            self._check_rows(
                g,
                row_req_ids,
                table,
                starts,
                start_columns,
                window_columns,
                checkpoint_columns,
                held_ends,
            )
            dest = table[rows[:, None], window_index]
            start = np.where(
                start_columns >= 0, table[rows, np.maximum(start_columns, 0)], 0
            )
            checkpoint = np.where(
                checkpoint_columns >= 0,
                table[rows, np.maximum(checkpoint_columns, 0)],
                0,
            )
            windows = torch.zeros((num_rows, width), dtype=torch.int32)
            windows[:live] = torch.from_numpy(dest.astype(np.int32))
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
        mask = None
        if intermediate_prefill_mask is not None:
            values = [bool(m) for m in intermediate_prefill_mask]
            if len(values) != live:
                raise RuntimeError(
                    f"{live} scheduled rows {list(row_req_ids)} got an "
                    f"intermediate_prefill_mask of {len(values)} rows"
                )
            mask = torch.ones(num_rows, dtype=torch.bool)
            mask[:live] = torch.tensor(values, dtype=torch.bool)
        if num_spec:
            for req_id, column, window in zip(
                row_req_ids, window_columns.tolist(), previous
            ):
                end = column + num_spec
                self.windows[req_id] = MambaWindow(
                    column, end if window is None else max(window.end, end)
                )
        return MambaStepState(
            groups=tuple(groups),
            checkpoint_positions=checkpoint_positions,
            intermediate_prefill_mask=mask,
        )

    def _start_column(
        self, req_id: str, start: int, count: int, window: MambaWindow | None
    ) -> int:
        """The block-table column of a row's start state, -1 for none.

        ``window`` is the request's last recorded window, ``None`` for a
        request whose first step this is (and always without speculative
        decoding); ``count`` is the tokens that step committed.
        """
        if window is None:
            if count != 1:
                raise RuntimeError(
                    f"request {req_id} has no earlier window, but its previous "
                    f"step committed {count} tokens"
                )
            return restored_state_column(self.mamba_cache_mode, self.block_size, start)
        if start == 0:
            raise RuntimeError(
                f"request {req_id} starts at position 0, but its previous step's "
                f"window at block-table column {window.column} is recorded; "
                "vLLM resumes a request only through resumed_req_ids"
            )
        if not 1 <= count <= 1 + self.num_speculative_blocks:
            raise RuntimeError(
                f"request {req_id}'s previous step committed {count} tokens, "
                f"outside [1, {1 + self.num_speculative_blocks}]"
            )
        return accepted_state_column(window.column, count)

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
        starts: Sequence[int],
        start_columns: np.ndarray,
        window_columns: np.ndarray,
        checkpoint_columns: np.ndarray,
        held_ends: np.ndarray,
    ) -> None:
        """Raise for a row whose blocks vLLM did not allocate as expected.

        vLLM gives every scheduled request the blocks of its window before the
        step, and block 0 is the null block that padding rows share. The block
        of the start state is a block of the request's previous window or of
        a prefix-cache hit, which vLLM frees only after this step. In mode
        ``align`` vLLM leaves block 0 in every column a step's tokens pass
        before its window, except the column of a prefill checkpoint
        (``checkpoint_columns``, -1 for none), which every Mamba group
        reserves alike, and the columns of the request's earlier windows, up
        to ``held_ends`` (-1 for a request without one). A block in another
        passed column would be one the model never writes and vLLM could
        cache.
        """
        width = 1 + self.num_speculative_blocks
        for row, req_id in enumerate(row_req_ids):
            start_column = int(start_columns[row])
            window_column = int(window_columns[row])
            checkpoint_column = int(checkpoint_columns[row])
            for t in range(width):
                column = window_column + t
                if not table[row, column]:
                    raise RuntimeError(
                        f"request {req_id} has no state block in KV cache group "
                        f"{group}; block-table column {column} (window column "
                        f"{t}) holds the null block 0: {table[row].tolist()}"
                    )
                if not _holds_current_block(table[row], column):
                    raise RuntimeError(
                        f"request {req_id}'s window column {t} is block-table "
                        f"column {column} of KV cache group {group}, whose block "
                        f"{int(table[row, column])} vLLM moved to a later "
                        f"column: {table[row].tolist()}"
                    )
            if start_column >= 0 and not table[row, start_column]:
                raise RuntimeError(
                    f"request {req_id} starts at position {starts[row]}, "
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
            first = max(start_column, int(held_ends[row])) + 1
            passed = [
                column
                for column in range(first, window_column)
                if table[row, column] and column != checkpoint_column
            ]
            if passed:
                raise RuntimeError(
                    f"request {req_id} passes block-table columns {first} to "
                    f"{window_column - 1} of KV cache group {group} in this step, "
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
    speculative_config = vllm_config.speculative_config
    num_speculative_tokens = 0
    if speculative_config is not None:
        num_speculative_tokens = speculative_config.num_speculative_tokens
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
        if spec.num_speculative_blocks != num_speculative_tokens:
            # A verify writes the state after each of its 1 + K tokens into a
            # block of its own, and MambaManager allocates K blocks beyond the
            # running one.
            raise ValueError(
                "MambaSpec has num_speculative_blocks="
                f"{spec.num_speculative_blocks}, but the launch verifies "
                f"{num_speculative_tokens} speculative tokens per step; the "
                "model's get_kv_cache_spec must declare num_speculative_blocks "
                "equal to speculative_config.num_speculative_tokens (0 without "
                "--speculative-config), as upstream MambaBase.get_kv_cache_spec "
                "does"
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
        # check_mamba_launch made it the launch's draft length in every group.
        num_speculative_blocks=specs[0].num_speculative_blocks,
    )


__all__ = [
    "SUPPORTED_MAMBA_CACHE_MODES",
    "MambaCacheGroups",
    "MambaGroupState",
    "MambaStepState",
    "MambaWindow",
    "accepted_state_column",
    "check_mamba_launch",
    "declares_mamba_spec",
    "mamba_attention_block_size",
    "mamba_cache_groups",
    "restored_state_column",
    "window_start_column",
]
