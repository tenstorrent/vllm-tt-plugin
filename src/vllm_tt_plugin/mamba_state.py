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
        cached = scheduler_output.scheduled_cached_reqs
        for req_id in (
            *scheduler_output.finished_req_ids,
            *(scheduler_output.preempted_req_ids or ()),
            *cached.resumed_req_ids,
        ):
            self.state_columns.pop(req_id, None)
            self.moved_to_window_start.discard(req_id)
        # A new or resumed request replays its history: its candidate states
        # are gone, as upstream's new input-batch row starts at count 1.
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
                req_state = requests[req_id]
                computed = num_computed[req_id]
                prev = self.state_columns.get(req_id)
                if prev is None:
                    # A new or resumed request; -1 when nothing is computed yet.
                    prev = (computed - 1) // self.block_size
                curr = cdiv(computed + num_scheduled, self.block_size) - 1
                self.state_columns[req_id] = curr
                if prev != -1 and prev != curr:
                    if count == PLACEHOLDER_TOKEN_ID:
                        raise RuntimeError(
                            f"chained request {req_id} moves its state from "
                            f"block-table column {prev} to {curr} at position "
                            f"{computed}, but the copy source depends on the "
                            "pending verify's accepted count, which only the "
                            "device knows; chain_blocker must refuse this chain"
                        )
                    for g in self.group_indices:
                        block_ids = req_state.block_ids[g]
                        copies.append((block_ids[prev + count - 1], block_ids[curr]))
                    count = 1
                    self.moved_to_window_start.add(req_id)
            self.step_accepted_counts[req_id] = count
        return copies

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
        """
        for req_id in committed:
            self.moved_to_window_start.discard(req_id)
        if self.mamba_cache_mode != "align":
            return []
        copies: list[tuple[int, int]] = []
        for req_id, (position, count) in committed.items():
            src = self.state_columns.get(req_id)
            if src is None:
                raise RuntimeError(
                    f"request {req_id} committed {count} tokens at position "
                    f"{position} but begin_step recorded no running column for it"
                )
            running = position + 1
            aligned = (position + count) // self.block_size * self.block_size
            if aligned < running:
                continue
            shift = aligned - running
            dst = aligned // self.block_size - 1
            if src == dst:
                self.moved_to_window_start.add(req_id)
                if shift == 0:
                    continue
            for g in self.group_indices:
                block_ids = requests[req_id].block_ids[g]
                copies.append((block_ids[src + shift], block_ids[dst]))
        return copies

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
        """
        if self.mamba_cache_mode != "align":
            return None
        chained = set(chained_req_ids)
        for req_id in dict.fromkeys((*pending_req_ids, *chained_req_ids)):
            req_state = requests.get(req_id)
            if req_state is None:
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
    )
