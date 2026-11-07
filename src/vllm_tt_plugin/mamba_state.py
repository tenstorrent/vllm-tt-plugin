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

The upstream GPU model runner, not the scheduler, decides which blocks of a
Mamba group's block-table row a request's state uses
(``mamba_get_block_table_tensor``), which state copies run before the forward
(``preprocess_mamba``, mode ``align``) and which run after speculative tokens
are accepted (``postprocess_mamba_align_gpu``, mode ``align``). This module
ports all three.

With ``MambaSpec.num_speculative_blocks`` K > 0 a row's state window is
``1 + K`` block ids: the model writes the state after the row's query token
``t`` into window column ``t``, and the row's start state is in window column
``num_accepted - 1``. Upstream keeps the GDN conv state of all ``1 + K``
positions in window column 0 and shifts it on a copy; in the TT contract each
block stands for the full state (conv history and recurrent state) of one
position, so every copy here is a ``(src, dst)`` block copy from the block of
the accepted column, and ``copy_kv_cache_blocks`` copies that logical state,
also the part a model has not written into the source block yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_tt_plugin.input_batch import TTCachedRequestState

# ``CacheConfig.mamba_cache_mode`` values the TT runner implements.
SUPPORTED_MAMBA_CACHE_MODES = ("none", "align")


@dataclass(frozen=True, slots=True)
class MambaChainChoices:
    """A chained verify's state moves for one ``MambaSpec`` group, indexed by
    the count the pending verify accepted.

    Entry ``[i, c - 1]`` holds row ``i``'s values for a pending verify that
    accepted ``c`` tokens, ``1 <= c <= 1 + K``. All tensors are CPU int32.
    """

    # ``[rows, 1 + K, 2]``: the ``(src, dst)`` full-state block copy to run
    # before the step reads or writes any state; ``(0, 0)`` for no copy.
    copies: torch.Tensor
    # ``[rows, 1 + K]``: the block that holds the row's start state, read
    # after the copy.
    start_blocks: torch.Tensor
    # ``[rows, 1 + K, 1 + K]``: entry ``[i, c - 1, t]`` is the block that takes
    # the state after query token ``t``.
    windows: torch.Tensor


@dataclass(frozen=True, slots=True)
class MambaGroupState:
    """The recurrent-state inputs of one forward for one ``MambaSpec`` group.

    All tensors are CPU int32 with one row per submitted row; padding rows
    hold block 0, the null block vLLM never allocates.
    """

    # The group's layers, ``KVCacheGroupSpec.layer_names``.
    layer_names: tuple[str, ...]
    # ``[rows, 1 + K]``, K the group's ``MambaSpec.num_speculative_blocks``:
    # the step writes the state after query token ``t`` into column ``t``.
    windows: torch.Tensor
    # Prefill with prefill checkpoint blocks only: ``[rows, W]``, the blocks
    # that take the state after ``MambaStepState.checkpoint_positions`` tokens.
    checkpoint_blocks: torch.Tensor | None = None
    # A chained verify in mode ``align`` only.
    chain: MambaChainChoices | None = None


@dataclass(frozen=True, slots=True)
class MambaStepState:
    """The recurrent-state inputs of one forward, the model's ``mamba_state``."""

    # One per ``MambaSpec`` KV cache group, in KV cache group order.
    groups: tuple[MambaGroupState, ...]
    # K > 0 only: int32 ``[rows]``; the start state is in window column
    # ``count - 1``; -1 on a chained row, whose count only the device knows;
    # 1 on padding rows.
    accepted_counts: torch.Tensor | None = None
    # Prefill with prefill checkpoint blocks only: int32 ``[rows, W]``, the
    # tokens the state in each ``MambaGroupState.checkpoint_blocks`` entry
    # covers; 0 for an unused entry.
    checkpoint_positions: torch.Tensor | None = None


@dataclass
class MambaCacheGroups:
    """The ``MambaSpec`` groups of one ``KVCacheConfig``."""

    # Indices into ``KVCacheConfig.kv_cache_groups``, in group order.
    group_indices: tuple[int, ...]
    # Each Mamba group's ``KVCacheGroupSpec.layer_names``, in the order of
    # ``group_indices``.
    layer_names: tuple[tuple[str, ...], ...]
    # ``CacheConfig.mamba_cache_mode``; every Mamba group's spec carries it.
    mamba_cache_mode: str
    # ``MambaSpec.block_size``, the same in every Mamba group.
    block_size: int
    # The first attention group; its block table is the ``page_table`` the
    # model receives. vLLM orders groups by spec, so group 0 can be a Mamba
    # group.
    attention_group_index: int
    # ``MambaSpec.num_speculative_blocks`` K, the same in every Mamba group.
    num_speculative_blocks: int = 0
    # Mode ``align`` only: request id -> the block-table column that holds the
    # request's running state after its last scheduled step. Upstream keeps
    # the same map as ``GPUModelRunner.mamba_state_idx``.
    state_columns: dict[str, int] = field(default_factory=dict)
    # Requests whose state a copy moved into window column 0 after their last
    # commit, so their start state is there whatever they accepted. Upstream
    # records the same by setting ``num_accepted_tokens`` to 1.
    moved_to_window_start: set[str] = field(default_factory=set)
    # Request id -> the start-state count of the step being built.
    step_accepted_counts: dict[str, int] = field(default_factory=dict)
    # Request id -> the scheduled tokens of a chained step whose moves the
    # device runs; its bookkeeping catches up when the pending verify commits.
    deferred_chained_steps: dict[str, int] = field(default_factory=dict)
    # ``CacheConfig.prefix_cache_retention_interval``. vLLM keeps the state of
    # a block boundary that decoding reaches only where
    # ``MambaManager.reachable_block_mask`` keeps a periodic one: every block
    # boundary at ``None`` or an interval of one block, every multiple of the
    # interval above that, none at 0 (it then keeps a Mamba block only at a
    # prompt's replay boundary or a shared-prefix junction). A boundary copy
    # after acceptance at any other boundary has no reader.
    retention_interval: int | None = None
    # Every attention group has one KV precision
    # (``KVCacheConfig.has_mixed_precision_kv_cache`` is False), so only a
    # block a Mamba group owned can hold bytes attention reads as NaN or Inf.
    zeroes_only_state_blocks: bool = True
    # Block ids a Mamba group was given since the model last zeroed them.
    blocks_owned_since_zero: set[int] = field(default_factory=set)
    # Group index -> block size, for every group that is not a Mamba group.
    attention_block_sizes: dict[int, int] = field(default_factory=dict)
    # ``MambaSpec.num_prefill_checkpoint_blocks`` is 1: vLLM schedules a prompt
    # that fits one chunk as one chunk, and reserves a block inside the chunk
    # for the state at the prompt's last cacheable position
    # (``checkpoint_inputs``), instead of ending a chunk there.
    prefill_checkpoints: bool = False

    @property
    def kept_boundary_interval(self) -> int:
        """Tokens between two decode-reached block boundaries whose state vLLM
        keeps (``retention_interval``); 0 when it keeps none."""
        if self.retention_interval is None:
            return self.block_size
        if self.retention_interval == 0:
            return 0
        return max(self.retention_interval, self.block_size)

    def max_num_blocks(self, max_model_len: int) -> int:
        """Block-table row width of a Mamba group, as
        ``MambaSpec.max_num_blocks_per_req``: mode ``none`` holds ``1 + K``
        blocks, mode ``align`` indexes the row by position plus K."""
        if self.mamba_cache_mode == "none":
            return 1 + self.num_speculative_blocks
        return cdiv(max_model_len, self.block_size) + self.num_speculative_blocks

    def state_windows(
        self, block_table: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """The state window of each row of one Mamba group's block table.

        Port of upstream ``mamba_get_block_table_tensor``. ``seq_lens`` counts
        each row's tokens after this step, drafts included. The window starts
        at column 0 in mode ``none``; in mode ``align`` row column ``p`` holds
        the state after ``(p + 1) * block_size`` tokens, so the running state
        is in column ``(seq_len - 1) // block_size``, and a row with
        ``seq_len`` 0 starts at column 0. Returns int32 ``[rows, 1 + K]``.
        """
        width = 1 + self.num_speculative_blocks
        if self.mamba_cache_mode == "none":
            return block_table[:, :width].to(torch.int32)
        starts = ((seq_lens.to(torch.int64) - 1) // self.block_size).clamp_(min=0)
        columns = starts.unsqueeze(1) + torch.arange(width, dtype=torch.int64)
        return torch.gather(block_table, 1, columns).to(torch.int32)

    def state_block_ids(
        self, block_table: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Window column 0 of each row: int32 ``[rows, 1]``."""
        return self.state_windows(block_table, seq_lens)[:, :1]

    def step_inputs(
        self,
        block_tables_per_group: Sequence[torch.Tensor],
        row_req_ids: Sequence[str],
        num_computed: Mapping[str, int],
        num_scheduled_tokens: Mapping[str, int],
        num_rows: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
        """The state windows and start-state counts of one forward.

        Row ``i`` of ``block_tables_per_group`` belongs to ``row_req_ids[i]``;
        the rows after them are padding, with zero block tables. The result is
        padded to ``num_rows`` rows, the row count of ``page_tables_per_layer``.
        ``num_computed`` is as for ``begin_step``, which must have run for this
        step.

        Returns each Mamba group's windows, int32 ``[num_rows, 1 + K]`` in the
        order of ``group_indices``, and the counts, int32 ``[num_rows]``, or
        ``None`` when K is 0.
        """
        width = 1 + self.num_speculative_blocks
        live = min(len(row_req_ids), num_rows)
        req_ids = row_req_ids[:live]
        # NumPy, one array for all groups: a few small operations per group.
        stacked = np.zeros((len(self.group_indices), num_rows, width), dtype=np.int32)
        if self.mamba_cache_mode == "none":
            rows, columns = slice(None, live), slice(None, width)
        else:
            # The running state's column, as ``state_windows`` finds it.
            starts = [
                max(num_computed[r] + num_scheduled_tokens[r] - 1, 0) // self.block_size
                for r in req_ids
            ]
            rows = np.arange(live)[:, None]
            columns = np.array(starts, dtype=np.int64)[:, None] + np.arange(width)
        for i, g in enumerate(self.group_indices):
            stacked[i, :live] = block_tables_per_group[g].numpy()[rows, columns]
        windows = tuple(torch.from_numpy(stacked))
        if not self.num_speculative_blocks:
            return windows, None
        counts = torch.tensor(
            [self.step_accepted_counts.get(r, 1) for r in req_ids]
            + [1] * (num_rows - live),
            dtype=torch.int32,
        )
        return windows, counts

    def checkpoint_inputs(
        self,
        checkpoints: Mapping[str, Sequence[tuple[int, int]]],
        row_req_ids: Sequence[str],
        requests: Mapping[str, TTCachedRequestState],
        num_rows: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """The states a prefill chunk must leave inside itself, from the
        scheduler's ``(position, column)`` list per request
        (``get_tt_mamba_checkpoints``).

        Returns each Mamba group's blocks at those columns, int32
        ``[num_rows, W]`` in the order of ``group_indices``; and int32
        ``[num_rows, W]``: the tokens each block's state covers. ``W`` is the
        longest list, at least 1; unused entries hold block 0 and 0 tokens.
        vLLM can cache each block as the state after exactly that many tokens.
        """
        width = max((len(checkpoints.get(r, ())) for r in row_req_ids), default=0)
        width = max(width, 1)
        num_groups = len(self.group_indices)
        blocks = np.zeros((num_groups, num_rows, width), dtype=np.int32)
        positions = np.zeros((num_rows, width), dtype=np.int32)
        for row, req_id in enumerate(row_req_ids[:num_rows]):
            block_ids = requests[req_id].block_ids
            for entry, (position, column) in enumerate(checkpoints.get(req_id, ())):
                for i, g in enumerate(self.group_indices):
                    row_blocks = block_ids[g]
                    block = row_blocks[column] if column < len(row_blocks) else 0
                    if not block:
                        raise RuntimeError(
                            f"request {req_id} must write the state after "
                            f"{position} tokens into block-table column {column}, "
                            f"but group {g}'s row holds no block there: "
                            f"{list(row_blocks)}"
                        )
                    blocks[i, row, entry] = block
                positions[row, entry] = position
        return tuple(torch.from_numpy(blocks)), torch.from_numpy(positions)

    def step_state(
        self,
        windows: Sequence[torch.Tensor],
        accepted_counts: torch.Tensor | None,
        checkpoint_blocks: Sequence[torch.Tensor] | None = None,
        checkpoint_positions: torch.Tensor | None = None,
        chains: Sequence[MambaChainChoices] | None = None,
    ) -> MambaStepState:
        """The model's ``mamba_state`` from per-group tensors, each sequence in
        the order of ``group_indices``."""
        num_groups = len(self.group_indices)
        per_group = (
            windows,
            (None,) * num_groups if checkpoint_blocks is None else checkpoint_blocks,
            (None,) * num_groups if chains is None else chains,
        )
        lengths = [len(self.layer_names), *map(len, per_group)]
        if lengths != [num_groups] * len(lengths):
            raise RuntimeError(
                f"Mamba groups {list(self.group_indices)} need one entry each; "
                f"got {lengths} layer-name tuples, windows, checkpoint blocks "
                "and chain choices"
            )
        return MambaStepState(
            groups=tuple(
                MambaGroupState(names, *values)
                for names, *values in zip(self.layer_names, *per_group)
            ),
            accepted_counts=accepted_counts,
            checkpoint_positions=checkpoint_positions,
        )

    def blocks_to_zero(
        self,
        scheduler_output: SchedulerOutput,
        requests: Mapping[str, TTCachedRequestState],
    ) -> list[int]:
        """The blocks of ``new_block_ids_to_zero`` the model must zero.

        Run after the runner has applied the scheduler output to ``requests``.
        vLLM lists every block newly allocated to an attention group, because
        attention reads positions of a block that it has not written yet and
        masks them, and a NaN or Inf there still spoils the result. A listed
        block needs no zeroing when:

        - no Mamba group has owned it since it was last zeroed. Only recurrent
          state reads as NaN or Inf: the model allocates its KV buffers zeroed,
          as upstream ``vllm.v1.worker.utils.allocate_kv_cache`` does, and
          attention writes finite values;
        - or the prefill step that allocates it writes all of its positions.
          The model writes a step's keys and values before its attention reads
          them, so no later step can read a position of that block unwritten.

        Without ``zeroes_only_state_blocks``, every listed block.
        """
        listed = scheduler_output.new_block_ids_to_zero or []
        if not self.zeroes_only_state_blocks:
            return list(listed)
        owned = self.blocks_owned_since_zero
        written: set[int] = set()
        num_scheduled = scheduler_output.num_scheduled_tokens
        for new in scheduler_output.scheduled_new_reqs:
            for g in self.group_indices:
                owned.update(new.block_ids[g])
            self._add_written_blocks(
                written,
                new.block_ids,
                (0,) * len(new.block_ids),
                new.num_computed_tokens,
                num_scheduled.get(new.req_id, 0),
            )
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            block_ids = cached.new_block_ids[i]
            if block_ids is None:
                continue
            for g in self.group_indices:
                owned.update(block_ids[g])
            start = cached.num_computed_tokens[i]
            end = start + num_scheduled.get(req_id, 0)
            state = requests.get(req_id)
            # Past the prompt, asynchronous scheduling counts the drafts of a
            # step in flight as computed, so this step's positions can be lower.
            if state is None or end > state.num_prompt_tokens:
                continue
            first_columns = tuple(
                len(row) - len(ids) for row, ids in zip(state.block_ids, block_ids)
            )
            self._add_written_blocks(
                written, block_ids, first_columns, start, end - start
            )
        zero = [b for b in listed if b in owned and b not in written]
        # Each listed block holds attention data or zeroes after this step.
        owned.difference_update(listed)
        return zero

    def _add_written_blocks(
        self,
        written: set[int],
        block_ids: Sequence[Sequence[int]],
        first_columns: Sequence[int],
        start: int,
        num_tokens: int,
    ) -> None:
        """Add to ``written`` the attention blocks whose positions all lie in
        ``[start, start + num_tokens)``; ``block_ids[g]`` begins at block-table
        column ``first_columns[g]``."""
        end = start + num_tokens
        for g, block_size in self.attention_block_sizes.items():
            first = first_columns[g]
            for i, block in enumerate(block_ids[g]):
                column = first + i
                if start <= column * block_size and (column + 1) * block_size <= end:
                    written.add(block)

    def forget(self, scheduler_output: SchedulerOutput) -> None:
        """Drop the bookkeeping of finished, preempted and resumed requests."""
        cached = scheduler_output.scheduled_cached_reqs
        for req_id in (
            *scheduler_output.finished_req_ids,
            *(scheduler_output.preempted_req_ids or ()),
            *cached.resumed_req_ids,
        ):
            self.state_columns.pop(req_id, None)
            self.moved_to_window_start.discard(req_id)
            self.deferred_chained_steps.pop(req_id, None)

    def begin_step(
        self,
        scheduler_output: SchedulerOutput,
        requests: Mapping[str, TTCachedRequestState],
        num_computed: Mapping[str, int],
        accepted_counts: Mapping[str, int] | None = None,
    ) -> list[tuple[int, int]]:
        """Record each row's start-state count; return the copies due first.

        ``num_computed`` maps each scheduled request to the position of its
        first query token this step. Under asynchronous scheduling the
        scheduler's ``num_computed_tokens`` of a speculative row still counts
        the drafts of its step in flight as accepted, so the runner passes the
        positions it builds the step with. ``accepted_counts`` maps a request
        to the count the step's spec ``accepted_counts`` carries; absent means
        1. In mode ``none`` -1 (a chained row) passes through: the start state
        follows the pending verify's acceptance, which only the device knows.
        In mode ``align`` a chained verify carries its moves for every count
        instead (``plan_chained_step``), so -1 raises.

        Mode ``align`` ports upstream ``preprocess_mamba``: when a request's
        tokens cross into a new block-table column, its start state moves to
        the block of the new column, and the model then reads and writes that
        column only, so the block of the previous boundary keeps its state for
        prefix-cache hits. The source is the accepted column, ``prev +
        count - 1``, in every Mamba group with the block ids of its own row.
        """
        accepted_counts = accepted_counts or {}
        if self.mamba_cache_mode == "align":
            missing = scheduler_output.num_scheduled_tokens.keys() - num_computed.keys()
            if missing:
                raise RuntimeError(
                    f"scheduled requests {sorted(missing)} have no first query "
                    f"position; positions were given for {sorted(num_computed)}"
                )
        self.forget(scheduler_output)
        # A new or resumed request replays its history: its candidate states
        # are gone, as upstream's new input-batch row starts at count 1.
        cached = scheduler_output.scheduled_cached_reqs
        fresh = {new.req_id for new in scheduler_output.scheduled_new_reqs}
        fresh.update(cached.resumed_req_ids)
        self.step_accepted_counts = {}
        copies: list[tuple[int, int]] = []
        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            count = accepted_counts.get(req_id, 1)
            if count == PLACEHOLDER_TOKEN_ID and self.mamba_cache_mode == "align":
                raise RuntimeError(
                    f"chained request {req_id} at position "
                    f"{num_computed[req_id]} reached begin_step in "
                    "mamba_cache_mode 'align'; its moves depend on the pending "
                    "verify's accepted count and come from plan_chained_step"
                )
            if req_id in fresh or req_id in self.moved_to_window_start:
                count = 1
            if self.mamba_cache_mode == "align":
                moves, count = self._move_to_running_column(
                    req_id,
                    requests[req_id].block_ids,
                    num_computed[req_id],
                    num_scheduled,
                    count,
                )
                copies += moves
            self.step_accepted_counts[req_id] = count
        return copies

    def _move_to_running_column(
        self,
        req_id: str,
        block_ids: Sequence[Sequence[int]],
        computed: int,
        num_scheduled: int,
        count: int,
    ) -> tuple[list[tuple[int, int]], int]:
        """Mode ``align``: a step's move into its running column.

        Records the column and returns the copies, one per Mamba group, and the
        step's start-state count (1 after a move).
        """
        prev = self.state_columns.get(req_id)
        if prev is None:
            # A new or resumed request; -1 when nothing is computed yet.
            prev = (computed - 1) // self.block_size
        curr = cdiv(computed + num_scheduled, self.block_size) - 1
        self.state_columns[req_id] = curr
        if prev == -1 or prev == curr:
            return [], count
        self.moved_to_window_start.add(req_id)
        moves = [
            (block_ids[g][prev + count - 1], block_ids[g][curr])
            for g in self.group_indices
        ]
        return moves, 1

    def after_acceptance(
        self,
        committed: Mapping[str, tuple[int, int]],
        requests: Mapping[str, TTCachedRequestState],
    ) -> list[tuple[int, int]]:
        """Copies due after a speculative step committed its accepted tokens.

        ``committed`` maps each request to ``(position, count)``: the position
        of the step's first query token (the row's computed tokens before the
        step) and the tokens the step committed, bonus included.

        Mode ``align`` ports the decisions of upstream
        ``postprocess_mamba_fused_kernel``: when the committed tokens reach a
        block boundary past the row's running state, the state after exactly
        that boundary moves from its window column into the boundary's own
        block-table column, where prefix caching looks for it. If that column
        is the running column, the start state is now in window column 0.

        A request whose next step was chained with its moves on the device
        (``plan_chained_step``) gets no copies here, because that step ran
        them; its bookkeeping catches up with this commit and that step's move.
        """
        for req_id in committed:
            self.moved_to_window_start.discard(req_id)
        if self.mamba_cache_mode != "align":
            return []
        copies: list[tuple[int, int]] = []
        for req_id, (position, count) in committed.items():
            block_ids = requests[req_id].block_ids
            boundary = self._copy_boundary_state(req_id, block_ids, position, count)
            num_scheduled = self.deferred_chained_steps.pop(req_id, None)
            if num_scheduled is None:
                copies += boundary
                continue
            start_count = 1 if req_id in self.moved_to_window_start else count
            self._move_to_running_column(
                req_id, block_ids, position + count, num_scheduled, start_count
            )
        return copies

    def _copy_boundary_state(
        self,
        req_id: str,
        block_ids: Sequence[Sequence[int]],
        position: int,
        count: int,
    ) -> list[tuple[int, int]]:
        """Mode ``align``: the boundary copies after a verify committed
        ``count`` tokens from ``position``, one per Mamba group.

        None at a boundary whose state vLLM does not keep
        (``kept_boundary_interval``). The copy also moves the start state into
        window column 0 when the boundary ends the running column, but then
        the next step moves into a new column, and without the copy it reads
        the same state from window column ``count - 1``.
        """
        src = self.state_columns.get(req_id)
        if src is None:
            raise RuntimeError(
                f"request {req_id} committed {count} tokens at position "
                f"{position} but begin_step recorded no running column for it"
            )
        kept = self.kept_boundary_interval
        if not kept:
            return []
        running = position + 1
        aligned = (position + count) // self.block_size * self.block_size
        if aligned < running or aligned % kept:
            return []
        shift = aligned - running
        dst = aligned // self.block_size - 1
        if src == dst:
            self.moved_to_window_start.add(req_id)
            if shift == 0:
                return []
        return [
            (block_ids[g][src + shift], block_ids[g][dst]) for g in self.group_indices
        ]

    def plan_chained_step(
        self,
        row_req_ids: Sequence[str],
        requests: Mapping[str, TTCachedRequestState],
        num_scheduled_tokens: Mapping[str, int],
        num_rows: int,
    ) -> tuple[MambaChainChoices, ...]:
        """A chained verify's state moves for every count its pending verify
        can accept (mode ``align``), one ``MambaChainChoices`` per Mamba group
        in the order of ``group_indices``, padded to ``num_rows`` rows.

        Row ``i``'s pending verify starts at the row's committed length minus
        one (the verify before it has committed, ``chain_blocker``) and
        commits ``c`` tokens, ``1 <= c <= 1 + K``. For each ``c`` this runs,
        on the row's bookkeeping, what the host runs between two verifies, and
        then restores it: the pending verify's boundary copy
        (``after_acceptance``), and this step's move into its running column
        (``begin_step``). The model runs the boundary copy itself before the
        step, and reads the start state where the move would copy it from, so
        the move needs no copy: the step writes window column 0 anyway. The
        bookkeeping catches up when the pending verify commits.
        """
        width = 1 + self.num_speculative_blocks
        num_groups = len(self.group_indices)
        # Nested lists ``[group][row][count - 1]``, one tensor each at the end.
        copies = [
            [[(0, 0)] * width for _ in range(num_rows)] for _ in range(num_groups)
        ]
        windows = [
            [[[0] * width] * width for _ in range(num_rows)] for _ in range(num_groups)
        ]
        starts = [[[0] * width for _ in range(num_rows)] for _ in range(num_groups)]
        for row, req_id in enumerate(row_req_ids):
            if (
                req_id in self.deferred_chained_steps
                or req_id not in self.state_columns
            ):
                raise RuntimeError(
                    f"chained request {req_id} has no settled running column "
                    f"(column {self.state_columns.get(req_id)}, deferred step "
                    f"{self.deferred_chained_steps.get(req_id)}): the verify "
                    "before its pending one has not committed"
                )
            block_ids = requests[req_id].block_ids
            position = requests[req_id].num_tokens - 1
            num_scheduled = num_scheduled_tokens[req_id]
            column = self.state_columns[req_id]
            moved = req_id in self.moved_to_window_start
            for c in range(1, width + 1):
                self.moved_to_window_start.discard(req_id)
                boundary = self._copy_boundary_state(req_id, block_ids, position, c)
                start_count = 1 if req_id in self.moved_to_window_start else c
                moves, start_count = self._move_to_running_column(
                    req_id, block_ids, position + c, num_scheduled, start_count
                )
                first = (position + c + num_scheduled - 1) // self.block_size
                for i, g in enumerate(self.group_indices):
                    window = block_ids[g][first : first + width]
                    windows[i][row][c - 1] = window
                    if boundary:
                        copies[i][row][c - 1] = boundary[i]
                    starts[i][row][c - 1] = (
                        moves[i][0] if moves else window[start_count - 1]
                    )
                self.state_columns[req_id] = column
                if moved:
                    self.moved_to_window_start.add(req_id)
                else:
                    self.moved_to_window_start.discard(req_id)
            self.deferred_chained_steps[req_id] = num_scheduled
        return tuple(
            MambaChainChoices(copies=c, start_blocks=s, windows=w)
            for c, s, w in zip(
                torch.tensor(copies, dtype=torch.int32),
                torch.tensor(starts, dtype=torch.int32),
                torch.tensor(windows, dtype=torch.int32),
            )
        )

    def chain_blocker(
        self,
        requests: Mapping[str, TTCachedRequestState],
        pending_req_ids: Sequence[str],
        chained_req_ids: Sequence[str],
        num_pending_steps: int,
        step_width: int,
    ) -> str | None:
        """``"mamba_boundary"`` when a step may not be built over pending verifies.

        Mode ``align`` only. A pending verify's boundary copy
        (``after_acceptance``) takes its source from the verify's accepted
        count, so it is due when that verify commits on the host. A step built
        over pending verifies (a chained verify, or a prefill submitted behind
        one) runs before that commit, and the scheduler may have freed the
        copy's target block by the time the commit runs. So no pending row may
        be able to reach a block boundary until the commit: for each request,
        between its committed length and the most tokens its
        ``num_pending_steps`` pending verifies can compute (each step at most
        ``step_width``).

        ``pending_req_ids`` are the pending verifies' rows that are not
        finished, finishing rows included. A row of ``chained_req_ids`` is
        exempt: its step carries the moves for every count the one unresolved
        verify can accept (``plan_chained_step``; ``chain_evaluate`` chains
        over exactly one, and the steps before it commit before the build), and
        the host runs no boundary copy for it at the commit. A pending row that
        the step leaves out still counts.

        Only a boundary whose state vLLM keeps (``kept_boundary_interval``)
        has a copy, so only such a boundary blocks.
        """
        kept = self.kept_boundary_interval
        if self.mamba_cache_mode != "align" or not kept:
            return None
        chained = set(chained_req_ids)
        for req_id in pending_req_ids:
            req_state = requests.get(req_id)
            if req_state is None or req_id in chained:
                continue
            # The last committed token is the next step's input, not computed.
            first = req_state.num_tokens - 1
            last = first + step_width * num_pending_steps
            if first // kept != last // kept:
                return "mamba_boundary"
        return None


def mamba_cache_groups(
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    cache_config: Any,
    max_model_len: int,
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
    specs = [kv_cache_groups[i].kv_cache_spec for i in group_indices]
    for i, spec in zip(group_indices, specs):
        if spec.mamba_cache_mode != mode:
            raise ValueError(
                f"MambaSpec of KV cache group {i} has mamba_cache_mode="
                f"{spec.mamba_cache_mode!r}, but cache_config.mamba_cache_mode="
                f"{mode!r}; the model's get_kv_cache_spec must use the resolved "
                "cache mode"
            )
    if mode not in SUPPORTED_MAMBA_CACHE_MODES:
        raise ValueError(
            f"mamba_cache_mode={mode!r} is not supported on TT; supported: "
            f"{list(SUPPORTED_MAMBA_CACHE_MODES)}"
        )
    block_sizes = {spec.block_size for spec in specs}
    if len(block_sizes) != 1:
        raise ValueError(
            f"MambaSpec groups {list(group_indices)} have different block sizes "
            f"{sorted(block_sizes)}; upstream state selection assumes one"
        )
    block_size = block_sizes.pop()
    num_speculative_blocks = {spec.num_speculative_blocks for spec in specs}
    if len(num_speculative_blocks) != 1:
        raise ValueError(
            f"MambaSpec groups {list(group_indices)} have different "
            f"num_speculative_blocks {sorted(num_speculative_blocks)}; one state "
            "window width serves every group"
        )
    checkpoints = {
        (spec.num_prefill_checkpoint_blocks, spec.prefill_checkpoint_alignment)
        for spec in specs
    }
    if len(checkpoints) != 1:
        raise ValueError(
            f"MambaSpec groups {list(group_indices)} have different "
            "(num_prefill_checkpoint_blocks, prefill_checkpoint_alignment) "
            f"{sorted(checkpoints, key=str)}; vLLM places one checkpoint per "
            "request for every group"
        )
    num_checkpoint_blocks, checkpoint_alignment = checkpoints.pop()
    if num_checkpoint_blocks and (
        mode != "align" or num_checkpoint_blocks != 1 or checkpoint_alignment is None
    ):
        # vLLM's MambaManager reserves at most one checkpoint block per
        # request, only in mode align, and never without an alignment.
        raise ValueError(
            f"MambaSpec num_prefill_checkpoint_blocks={num_checkpoint_blocks} "
            f"with prefill_checkpoint_alignment={checkpoint_alignment} in "
            f"mamba_cache_mode={mode!r}; prefill checkpoints need mode 'align', "
            "exactly 1 block and an alignment"
        )
    if mode == "none" and block_size < max_model_len:
        # MambaManager allocates one block per block_size tokens, so a smaller
        # block size makes every request hold a state block per block_size
        # tokens in each group, which the TT pool sizing does not count.
        raise ValueError(
            f"MambaSpec block_size {block_size} is below max_model_len "
            f"{max_model_len} in mamba_cache_mode='none'; declare block_size "
            "max_model_len, as upstream resolves mamba_block_size"
        )
    if mode == "align":
        other_block_sizes = {
            g.kv_cache_spec.block_size
            for i, g in enumerate(kv_cache_groups)
            if i not in group_indices
        }
        if other_block_sizes != {block_size}:
            # The scheduler ends prefill chunks on multiples of the smallest
            # group block size, and column p must hold the state after exactly
            # (p + 1) * block_size tokens.
            raise ValueError(
                f"MambaSpec block_size {block_size} differs from the other KV "
                f"cache groups' block sizes {sorted(other_block_sizes)} in "
                "mamba_cache_mode='align'; all groups need one block_size"
            )
    return MambaCacheGroups(
        group_indices=group_indices,
        layer_names=tuple(tuple(kv_cache_groups[i].layer_names) for i in group_indices),
        mamba_cache_mode=mode,
        block_size=block_size,
        attention_group_index=attention_group_index,
        num_speculative_blocks=num_speculative_blocks.pop(),
        retention_interval=getattr(
            cache_config, "prefix_cache_retention_interval", None
        ),
        attention_block_sizes={
            i: group.kv_cache_spec.block_size
            for i, group in enumerate(kv_cache_groups)
            if i not in group_indices
        },
        prefill_checkpoints=bool(num_checkpoint_blocks),
    )


__all__ = [
    "SUPPORTED_MAMBA_CACHE_MODES",
    "MambaCacheGroups",
    "MambaChainChoices",
    "MambaGroupState",
    "MambaStepState",
    "mamba_cache_groups",
]
