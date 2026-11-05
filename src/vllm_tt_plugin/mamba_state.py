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
positions in window column 0 and shifts it on a copy; the TT contract keeps
the full state (conv history and recurrent state) of each position in its own
block, so every copy here is a whole-block ``(src, dst)`` copy from the block
of the accepted column.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_tt_plugin.input_batch import TTCachedRequestState

# ``CacheConfig.mamba_cache_mode`` values the TT runner implements.
SUPPORTED_MAMBA_CACHE_MODES = ("none", "align")


class MambaStepInputs(NamedTuple):
    """The recurrent-state inputs of one forward, one entry per layer.

    Attention layers get ``None``. Every tensor has ``num_rows`` rows; padding
    rows hold block 0, the null block vLLM never allocates, and count 1.
    """

    # int32 ``[rows, 1]``: window column 0, the block of the running state.
    state_block_ids_per_layer: list[torch.Tensor | None]
    # int32 ``[rows, 1 + K]``; ``None`` when K is 0.
    state_block_windows_per_layer: list[torch.Tensor | None] | None
    # int32 ``[rows]``: the start state is in window column ``count - 1``;
    # -1 on a chained row, whose count only the device knows; ``None`` when K
    # is 0.
    state_accepted_counts: torch.Tensor | None


class MambaChainPlan(NamedTuple):
    """A chained verify's state moves, indexed by the pending verify's count.

    One entry per layer, ``None`` for attention layers. Index ``[i, c - 1]``
    holds row ``i``'s values for a pending verify that committed ``c`` tokens.
    """

    # int32 ``[rows, 1 + K, 2]``: the ``(src, dst)`` block copy to run before
    # the step; ``(0, 0)`` (the null block) when there is none.
    copies_per_layer: list[torch.Tensor | None]
    # int32 ``[rows, 1 + K, 1 + K]``: the step's state window.
    windows_per_layer: list[torch.Tensor | None]
    # int32 ``[rows, 1 + K]``: the block holding the step's start state.
    start_blocks_per_layer: list[torch.Tensor | None]


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
    # The model declares ``supports_mamba_chained_state_moves``: a chained
    # verify carries its moves for every count the pending verify can accept
    # (``plan_chained_step``), and the model runs the one its count selects.
    chained_moves_on_device: bool = False
    # Request id -> the scheduled tokens of a chained step whose moves the
    # device runs; its bookkeeping catches up when the pending verify commits.
    deferred_chained_steps: dict[str, int] = field(default_factory=dict)
    # ``prefix_cache_retention_interval`` is not 0. At 0 vLLM caches a Mamba
    # block only at a prompt's replay boundary or a shared-prefix junction
    # (``MambaManager.reachable_block_mask``), never at a block boundary that
    # decoding reaches, so the boundary copy after acceptance has no reader.
    retains_decode_boundaries: bool = True

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
        layer_to_group_idx: Sequence[int],
        num_rows: int,
    ) -> MambaStepInputs:
        """The recurrent-state inputs of one forward.

        Row ``i`` of ``block_tables_per_group`` belongs to ``row_req_ids[i]``;
        the rows after them are padding, with zero block tables. The result is
        padded to ``num_rows`` rows, the row count of ``page_tables_per_layer``.
        ``num_computed`` is as for ``begin_step``, which must have run for this
        step.
        """
        seq_lens = torch.zeros(block_tables_per_group[0].shape[0], dtype=torch.int64)
        for row, req_id in enumerate(row_req_ids):
            seq_lens[row] = num_computed[req_id] + num_scheduled_tokens[req_id]
        width = 1 + self.num_speculative_blocks
        windows: dict[int, torch.Tensor] = {}
        for g in self.group_indices:
            ids = self.state_windows(block_tables_per_group[g], seq_lens)
            padded = torch.zeros((num_rows, width), dtype=torch.int32)
            rows = min(ids.shape[0], num_rows)
            padded[:rows] = ids[:rows]
            windows[g] = padded
        ids_per_layer = [
            windows[g][:, :1] if g in windows else None for g in layer_to_group_idx
        ]
        if not self.num_speculative_blocks:
            return MambaStepInputs(ids_per_layer, None, None)
        counts = torch.ones(num_rows, dtype=torch.int32)
        for row, req_id in enumerate(row_req_ids[:num_rows]):
            counts[row] = self.step_accepted_counts.get(req_id, 1)
        return MambaStepInputs(
            ids_per_layer, [windows.get(g) for g in layer_to_group_idx], counts
        )

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
        1, and -1 (a chained row) passes through: the start state follows the
        pending verify's acceptance, which only the device knows, and no move
        may depend on it (``chain_blocker``).

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
            if req_id in fresh:
                count = 1
            elif count != PLACEHOLDER_TOKEN_ID and req_id in self.moved_to_window_start:
                # A chained row keeps -1: the move happened before the pending
                # verify ran, and that verify's result is where the row starts.
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
        if count == PLACEHOLDER_TOKEN_ID:
            raise RuntimeError(
                f"chained request {req_id} moves its state from block-table "
                f"column {prev} to {curr} at position {computed}, but the copy "
                "source depends on the pending verify's accepted count, which "
                "only the device knows; chain_blocker must refuse this chain"
            )
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

        None without ``retains_decode_boundaries``. The copy also moves the
        start state into window column 0 when the boundary ends the running
        column, but then the next step moves into a new column, and without
        the copy it reads the same state from window column ``count - 1``.
        """
        src = self.state_columns.get(req_id)
        if src is None:
            raise RuntimeError(
                f"request {req_id} committed {count} tokens at position "
                f"{position} but begin_step recorded no running column for it"
            )
        if not self.retains_decode_boundaries:
            return []
        running = position + 1
        aligned = (position + count) // self.block_size * self.block_size
        if aligned < running:
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
        layer_to_group_idx: Sequence[int],
        num_rows: int,
    ) -> MambaChainPlan:
        """A chained verify's state moves for every count its pending verify
        can accept (mode ``align``, ``chained_moves_on_device``).

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
        copies = {
            g: torch.zeros((num_rows, width, 2), dtype=torch.int32)
            for g in self.group_indices
        }
        windows = {
            g: torch.zeros((num_rows, width, width), dtype=torch.int32)
            for g in self.group_indices
        }
        starts = {
            g: torch.zeros((num_rows, width), dtype=torch.int32)
            for g in self.group_indices
        }
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
                    window = torch.tensor(
                        block_ids[g][first : first + width], dtype=torch.int32
                    )
                    windows[g][row, c - 1] = window
                    if boundary:
                        copies[g][row, c - 1] = torch.tensor(boundary[i])
                    starts[g][row, c - 1] = (
                        moves[i][0] if moves else window[start_count - 1]
                    )
                self.state_columns[req_id] = column
                if moved:
                    self.moved_to_window_start.add(req_id)
                else:
                    self.moved_to_window_start.discard(req_id)
            self.deferred_chained_steps[req_id] = num_scheduled
        return MambaChainPlan(
            [copies.get(g) for g in layer_to_group_idx],
            [windows.get(g) for g in layer_to_group_idx],
            [starts.get(g) for g in layer_to_group_idx],
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

        Mode ``align`` only. A state move's source depends on a verify's
        accepted count, so it is due when that verify commits on the host: a
        boundary copy after acceptance (``after_acceptance``) and a move to a
        new block-table column before the next step (``begin_step``). A step
        built over pending verifies (a chained verify, or a prefill submitted
        behind one) runs before that commit, and the scheduler may have freed
        a target block by the time the commit runs. So no row may be able to
        reach a block boundary until the commit: for each request, between its
        committed length and the most tokens its ``num_pending_steps`` pending
        verifies, plus this step for ``chained_req_ids``, can compute (each
        step at most ``step_width``).

        ``pending_req_ids`` are the pending verifies' rows that are not
        finished, finishing rows included.

        With ``chained_moves_on_device`` a chained row is exempt: its step
        carries the moves for every count the one unresolved verify can accept
        (``plan_chained_step``; ``chain_evaluate`` chains over exactly one, and
        the steps before it commit before the build). A pending row that the
        step leaves out still counts, since its boundary copy waits for the
        host commit.

        Without ``retains_decode_boundaries`` no boundary copy is due, so only
        a chained row can block: when its step's running column could differ
        from the oldest pending verify's, that is, when a boundary lies within
        ``step_width * num_pending_steps`` tokens past the oldest pending
        verify's last query token.
        """
        if self.mamba_cache_mode != "align":
            return None
        chained = set(chained_req_ids)
        exempt = chained if self.chained_moves_on_device else ()
        if not self.retains_decode_boundaries:
            # No boundary copy is due, so a pending row needs nothing. A
            # chained row still needs the running column of every step from
            # the oldest pending one to the chained one to be the same: a move
            # into a new column reads from the column the count selects.
            for req_id in chained_req_ids:
                req_state = requests.get(req_id)
                if req_state is None or req_id in exempt:
                    continue
                first = req_state.num_tokens - 1
                last = first + step_width * (num_pending_steps + 1)
                if (first + step_width - 1) // self.block_size != (
                    last - 1
                ) // self.block_size:
                    return "mamba_boundary"
            return None
        for req_id in dict.fromkeys((*pending_req_ids, *chained_req_ids)):
            req_state = requests.get(req_id)
            if req_state is None or req_id in exempt:
                continue
            # The last committed token is the next step's input, not computed.
            first = req_state.num_tokens - 1
            steps = num_pending_steps + (req_id in chained)
            last = first + step_width * steps
            if first // self.block_size != last // self.block_size:
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
        if spec.num_prefill_checkpoint_blocks:
            raise ValueError(
                f"MambaSpec of KV cache group {i} has "
                f"num_prefill_checkpoint_blocks="
                f"{spec.num_prefill_checkpoint_blocks}; TT does not support "
                "prefill-checkpoint Mamba blocks yet"
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
        mamba_cache_mode=mode,
        block_size=block_size,
        attention_group_index=attention_group_index,
        num_speculative_blocks=num_speculative_blocks.pop(),
        retains_decode_boundaries=(
            getattr(cache_config, "prefix_cache_retention_interval", None) != 0
        ),
    )
