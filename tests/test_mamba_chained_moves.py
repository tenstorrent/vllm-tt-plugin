# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Mode-align state moves run by the device for chained verifies.

``MambaCacheGroups.plan_chained_step`` gives a verify that is built before the
verify before it commits (a chained verify) its window, start block and
boundary copy for every count the pending verify can accept. These tests run
one request through every sequence of accepted counts, at every offset of the
prompt in a block, twice:

- host order: each verify commits before the next one is built, and the host
  runs every move (``begin_step``, ``after_acceptance``);
- chained order: every verify after the first is built while the one before it
  is pending, and the "device" picks the ``MambaChainChoices`` entry by that
  verify's count.

A ledger replays every read, write and copy. Every read must find the state of
exactly the computed tokens, every column the request passed must keep the
state after its boundary, and the host bookkeeping must end where the host
order ends.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from vllm_tt_plugin.mamba_state import MambaCacheGroups
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

K = 3
WIDTH = K + 1
GDN = 1
REQ = "r"


def _groups(block_size: int, retention: int | None = None) -> MambaCacheGroups:
    return MambaCacheGroups(
        group_indices=(GDN,),
        layer_names=(("model.layers.1.linear_attn",),),
        mamba_cache_mode="align",
        block_size=block_size,
        attention_group_index=0,
        num_speculative_blocks=K,
        retention_interval=retention,
    )


def _scheduled(num_tokens: int, *, new: bool = False):
    return SimpleNamespace(
        num_scheduled_tokens={REQ: num_tokens},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        scheduled_new_reqs=[SimpleNamespace(req_id=REQ)] if new else [],
    )


class _Run:
    """One request: a prompt prefilled whole, then verifies of 1 + K tokens."""

    def __init__(self, block_size: int, prompt_len: int, retention: int | None = None):
        self.bs = block_size
        self.groups = _groups(block_size, retention)
        columns = 64
        self.row = list(range(100, 100 + columns))
        self.state = SimpleNamespace(block_ids=[[], self.row], num_tokens=0)
        self.requests = {REQ: self.state}
        self.blocks: dict[int, int] = {}
        # Prefill: the scheduler stops at block boundaries in mode align, so
        # every boundary before the prompt end holds its state.
        self.groups.begin_step(
            _scheduled(prompt_len, new=True), self.requests, {REQ: 0}
        )
        for column in range((prompt_len - 1) // block_size):
            self.blocks[self.row[column]] = (column + 1) * block_size
        self.first_column = (prompt_len - 1) // block_size
        self.blocks[self.row[self.first_column]] = prompt_len
        # One sampled token: the next verify starts at the prompt length.
        self.position = prompt_len
        self.state.num_tokens = prompt_len + 1

    def copy(self, pairs) -> None:
        for src, dst in pairs:
            if src == dst:
                continue
            assert src in self.blocks, f"copy from block {src}, which holds no state"
            self.blocks[dst] = self.blocks[src]

    def forward(self, start: int, window: list[int], position: int) -> None:
        found = self.blocks.get(start)
        assert found == position, (
            f"the verify at {position} read block {start} holding {found}"
        )
        for t, block in enumerate(window):
            self.blocks[block] = position + t + 1

    def boundaries_are_kept(self, columns: int) -> None:
        """Every column the request passed whose boundary vLLM keeps holds the
        state after that boundary."""
        kept = self.groups.kept_boundary_interval
        for column in range(self.first_column, columns):
            if kept and (column + 1) * self.bs % kept == 0:
                assert self.blocks.get(self.row[column]) == (column + 1) * self.bs, (
                    column
                )

    def window(self, position: int) -> list[int]:
        first = (position + WIDTH - 1) // self.bs
        return self.row[first : first + WIDTH]


def _host_order(block_size, prompt_len, counts, retention=None):
    run = _Run(block_size, prompt_len, retention=retention)
    position, count = run.position, 1
    for accepted in counts:
        run.copy(
            run.groups.begin_step(
                _scheduled(WIDTH), run.requests, {REQ: position}, {REQ: count}
            )
        )
        start_count = run.groups.step_accepted_counts[REQ]
        window = run.window(position)
        run.forward(window[start_count - 1], window, position)
        run.copy(run.groups.after_acceptance({REQ: (position, accepted)}, run.requests))
        position += accepted
        count = accepted
        run.state.num_tokens = position + 1
    return run, position


def _chained_order(block_size, prompt_len, counts, retention=None):
    """Step 0 is built by the host; step n > 0 is built while step n - 1 is
    pending, after step n - 2 committed, as ``chain_evaluate`` allows."""
    run = _Run(block_size, prompt_len, retention=retention)
    position = run.position
    run.copy(run.groups.begin_step(_scheduled(WIDTH), run.requests, {REQ: position}))
    window = run.window(position)
    run.forward(window[run.groups.step_accepted_counts[REQ] - 1], window, position)
    for n in range(1, len(counts)):
        # Build step n over pending step n - 1: the host knows step n - 2's
        # commit only, so the row's length is step n - 1's first position + 1.
        run.state.num_tokens = position + 1
        [chain] = run.groups.plan_chained_step([REQ], run.requests, {REQ: WIDTH}, 1)
        # The device: step n - 1 accepted counts[n - 1] tokens.
        c = counts[n - 1]
        run.copy([tuple(chain.copies[0, c - 1].tolist())])
        position += c
        run.forward(
            int(chain.start_blocks[0, c - 1]),
            chain.windows[0, c - 1].tolist(),
            position,
        )
        assert chain.windows[0, c - 1].tolist() == run.window(position)
        # The host commits step n - 1 after building step n: no copies, the
        # bookkeeping catches up.
        assert run.groups.after_acceptance({REQ: (position - c, c)}, run.requests) == []
    # The last step commits on the host after a drain.
    run.copy(run.groups.after_acceptance({REQ: (position, counts[-1])}, run.requests))
    return run, position + counts[-1]


def test_retention_0_issues_no_boundary_copy():
    run = _Run(8, 9, retention=0)
    run.groups.begin_step(_scheduled(WIDTH), run.requests, {REQ: run.position})
    # 9 + 4 crosses 16 with all four tokens accepted; dense retention copies.
    assert run.groups.after_acceptance({REQ: (12, 4)}, run.requests) == []
    assert run.groups.moved_to_window_start == set()
    dense = _Run(8, 9)
    dense.groups.begin_step(_scheduled(WIDTH), dense.requests, {REQ: dense.position})
    assert dense.groups.after_acceptance({REQ: (12, 4)}, dense.requests)


@pytest.mark.parametrize(
    ("retention", "kept"), [(0, 0), (None, 16), (16, 16), (32, 32)]
)
def test_retention_is_read_from_the_cache_config(retention, kept):
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    from vllm_tt_plugin.mamba_state import mamba_cache_groups

    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
            ),
        ),
        KVCacheGroupSpec(
            ["gdn"],
            MambaSpec(
                block_size=16,
                shapes=((4,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
    ]
    cache_config = SimpleNamespace(
        mamba_cache_mode="align", prefix_cache_retention_interval=retention
    )
    assert mamba_cache_groups(groups, cache_config, 64).kept_boundary_interval == kept


def test_sparse_retention_copies_only_at_kept_boundaries():
    """At retention 16 with block 8, vLLM keeps the decode-reached state at
    16 but not at 24, so only a verify that reaches 16 copies."""
    for retention, reaches, copies in [
        (16, 16, True),
        (16, 24, False),
        (None, 24, True),
    ]:
        # The verify runs from 2 tokens before the boundary and accepts all.
        run = _Run(8, reaches - 2, retention=retention)
        run.groups.begin_step(_scheduled(WIDTH), run.requests, {REQ: run.position})
        made = run.groups.after_acceptance({REQ: (run.position, WIDTH)}, run.requests)
        assert bool(made) is copies, (retention, reaches, made)


@pytest.mark.parametrize("retention_blocks", [None, 0, 1, 2])
@pytest.mark.parametrize("block_size", [4, 5, 8])
def test_every_count_sequence_matches_the_host_order(block_size, retention_blocks):
    """All 4**4 count sequences of four verifies, from every prompt offset in
    a block: the chained run reads what the host run reads, keeps every
    boundary state that vLLM keeps (dense, none, every block, every second
    block), and ends with the same bookkeeping."""
    retention = None if retention_blocks is None else retention_blocks * block_size
    for prompt_len in range(block_size + 1, 2 * block_size + 1):
        for counts in itertools.product(range(1, WIDTH + 1), repeat=4):
            host, host_end = _host_order(block_size, prompt_len, counts, retention)
            chained, chained_end = _chained_order(
                block_size, prompt_len, counts, retention
            )
            assert chained_end == host_end
            passed = (host_end - 1) // block_size
            host.boundaries_are_kept(passed)
            chained.boundaries_are_kept(passed)
            assert chained.groups.state_columns == host.groups.state_columns
            assert (
                chained.groups.moved_to_window_start
                == host.groups.moved_to_window_start
            )
            assert chained.groups.deferred_chained_steps == {}


def test_a_plan_needs_the_verify_before_the_pending_one_committed():
    run = _Run(8, 10)
    run.groups.begin_step(_scheduled(WIDTH), run.requests, {REQ: run.position})
    run.groups.plan_chained_step([REQ], run.requests, {REQ: WIDTH}, 1)
    with pytest.raises(RuntimeError, match="no settled running column"):
        run.groups.plan_chained_step([REQ], run.requests, {REQ: WIDTH}, 1)


def test_a_finished_request_drops_its_deferred_step():
    run = _Run(8, 10)
    run.groups.begin_step(_scheduled(WIDTH), run.requests, {REQ: run.position})
    run.groups.plan_chained_step([REQ], run.requests, {REQ: WIDTH}, 1)
    finished = _scheduled(0)
    finished.num_scheduled_tokens = {}
    finished.finished_req_ids = {REQ}
    run.groups.forget(finished)
    assert run.groups.deferred_chained_steps == {}
    assert REQ not in run.groups.state_columns


def test_begin_step_refuses_a_chained_row():
    """A chained verify's moves depend on the pending verify's count; in mode
    align they come from ``plan_chained_step``, never from ``begin_step``."""
    run = _Run(8, 10)
    with pytest.raises(RuntimeError, match="plan_chained_step"):
        run.groups.begin_step(
            _scheduled(WIDTH),
            run.requests,
            {REQ: run.position},
            {REQ: PLACEHOLDER_TOKEN_ID},
        )


# Two pending steps: the drain decision before the build still counts a
# resolved step that the build applies first.
@pytest.mark.parametrize("pending_steps", [1, 2])
def test_chained_rows_are_exempt(pending_steps):
    groups = _groups(16)
    requests = {"a": SimpleNamespace(num_tokens=14)}
    assert groups.chain_blocker(requests, ["a"], ["a"], pending_steps, WIDTH) is None
    # The same row as a pending row of a prefill built over the verify blocks.
    assert groups.chain_blocker(requests, ["a"], [], pending_steps, WIDTH)


def test_a_pending_row_left_out_of_the_chained_step_still_blocks():
    """A finishing row's boundary copy waits for the host commit."""
    groups = _groups(16)
    requests = {
        "a": SimpleNamespace(num_tokens=5),
        "finishing": SimpleNamespace(num_tokens=14),
    }
    assert groups.chain_blocker(requests, ["a", "finishing"], ["a"], 1, WIDTH)


def test_chain_blocker_ignores_pending_rows_at_retention_0():
    """A pending row 2 tokens before a boundary: with dense retention its
    boundary copy blocks a prefill over the pending verify (no chained rows);
    at retention 0 nothing is due."""
    requests = {"a": SimpleNamespace(num_tokens=14)}
    assert _groups(16).chain_blocker(requests, ["a"], [], 1, WIDTH)
    assert _groups(16, retention=0).chain_blocker(requests, ["a"], [], 1, WIDTH) is None


def test_chain_blocker_ignores_boundaries_whose_state_vllm_drops():
    """Retention 32 at block 16: a pending row 2 tokens before 48 has no
    boundary copy due (vLLM drops the state at 48), one 2 tokens before 64
    has."""
    groups = _groups(16, retention=32)
    assert (
        groups.chain_blocker({"a": SimpleNamespace(num_tokens=46)}, ["a"], [], 1, WIDTH)
        is None
    )
    assert groups.chain_blocker(
        {"a": SimpleNamespace(num_tokens=62)}, ["a"], [], 1, WIDTH
    )
