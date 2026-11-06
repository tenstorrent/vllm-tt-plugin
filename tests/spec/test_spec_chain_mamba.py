# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Chained fused verifies over GDN state in vLLM ``MambaSpec`` blocks.

A chained verify (``TTAsyncDecodeController.chain_enabled``) is built while
the verify before it is still in flight, so the host knows neither that
verify's accepted count nor its boundary copies when it builds the next step.
These tests run the chain engine of ``test_spec_chain_async`` with one
attention group and one ``MambaSpec`` group (K=3) and replay every state read
and write in a ledger:

- the model reads a row's start state from window column ``count - 1``, where a
  chained row's count (-1) is the device's own accepted count of the pending
  verify; in mode ``align`` a chained verify instead carries its moves for
  every count (``MambaGroupState.chain``), and the model applies the entry of
  the device's count;
- it writes the state after query token ``t`` into window column ``t``;
- ``copy_kv_cache_blocks`` moves whole states, in the order the runner hands
  them over.

Every read must find the state of exactly the tokens the row has computed, and
in mode ``align`` every block-table column the row has passed must end up
holding the state after exactly its boundary.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import NewRequestData

from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.mamba_state import MambaCacheGroups
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

from .test_spec_async import MAX_MODEL_LEN, MAX_NUM_REQS, PROMPT_LEN
from .test_spec_chain_async import (
    GREEDY,
    PREFILL_CAPS,
    Engine,
    _chain_model,
    _chain_off,
    _controller,
    _make_runner,
)

K = 3
WIDTH = K + 1
# Small blocks, so a run crosses many block boundaries.
MAMBA_BLOCK = 16
ATTENTION_GROUP, GDN_GROUP = 0, 1
GDN_LAYERS = ("model.layers.1.linear_attn",)


class _StateLedger:
    """What each GDN block holds: ``(request, tokens the state covers)``."""

    def __init__(self):
        self.blocks: dict[int, tuple[str, int]] = {}
        self.last_count: dict[str, int] = {}
        self.reads = 0
        self.chained_reads = 0

    def copy(self, copies):
        for src, dst in copies:
            if src not in self.blocks:
                raise AssertionError(f"copy from block {src}, which holds no state")
            self.blocks[dst] = self.blocks[src]

    def verify(self, req_id, window, count, p0, num_valid):
        if count == PLACEHOLDER_TOKEN_ID:
            count = self.last_count[req_id]
            self.chained_reads += 1
        if not 1 <= count <= len(window):
            raise AssertionError(f"{req_id}: start count {count} outside the window")
        self.verify_block(req_id, window[count - 1], window, p0, num_valid)

    def verify_block(self, req_id, block, window, p0, num_valid):
        if req_id not in self.last_count:
            # The prefill this harness skips left the state here.
            self.blocks[block] = (req_id, p0)
        found = self.blocks.get(block)
        if found != (req_id, p0):
            raise AssertionError(
                f"{req_id} at position {p0} read block {block} (window {window}) "
                f"holding {found}"
            )
        self.reads += 1
        for t in range(num_valid + 1):
            self.blocks[window[t]] = (req_id, p0 + t + 1)


class _MambaEngine(Engine):
    """``Engine`` whose requests own an attention and a GDN block-table row."""

    def __init__(self, runner, mode, **kw):
        super().__init__(runner, **kw)
        self.mode = mode
        self.block_ids: dict[str, list[int]] = {}
        self._next_block = 1

    def add(self, req_id, first_token, params=GREEDY, stop_after=None):
        width = runner_row_width(self.mode)
        gdn = list(range(self._next_block, self._next_block + width))
        self._next_block += width
        attention = list(
            range(1000 + len(self.block_ids) * 64, 1000 + len(self.block_ids) * 64 + 16)
        )
        self.block_ids[req_id] = gdn
        from .test_spec_async import _scheduler_output

        so = _scheduler_output(self.runner, decoding=tuple(self.live))
        so.scheduled_new_reqs = [
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
                mm_features=[],
                sampling_params=params,
                pooling_params=None,
                block_ids=(attention, gdn),
                num_computed_tokens=PROMPT_LEN,
                lora_request=None,
            )
        ]
        so.num_scheduled_tokens[req_id] = 1
        so.total_num_scheduled_tokens += 1
        self.runner._update_states(so)
        self._register(req_id, stop_after)

    def schedule(self):
        so = super().schedule()
        # vLLM schedules the 1 + K tokens of a speculative row.
        so.num_scheduled_tokens = {r: WIDTH for r in so.num_scheduled_tokens}
        so.total_num_scheduled_tokens = WIDTH * len(so.num_scheduled_tokens)
        return so


def runner_row_width(mode: str) -> int:
    if mode == "none":
        return 1 + K
    return cdiv(MAX_MODEL_LEN, MAMBA_BLOCK) + K


def _mamba_run(
    monkeypatch,
    mode,
    *,
    chain=True,
    steps=40,
    cycle=(3, 0, 1, 2),
    retains=True,
    applies_chain=True,
    **kw,
):
    """``applies_chain=False`` stands for a model that ignores
    ``MambaGroupState.chain`` and reads the ordinary window."""
    model = _chain_model(accept_cycle=cycle, **kw)
    runner = _make_runner(model)
    if not chain:
        _chain_off(runner)
    runner.input_batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=runner.input_batch.vocab_size,
        block_sizes=[16, MAMBA_BLOCK],
        kernel_block_sizes=[16, MAMBA_BLOCK],
        max_num_blocks=[MAX_MODEL_LEN // 16, runner_row_width(mode)],
    )
    runner._mamba_groups = MambaCacheGroups(
        group_indices=(GDN_GROUP,),
        layer_names=(GDN_LAYERS,),
        mamba_cache_mode=mode,
        block_size=MAMBA_BLOCK if mode == "align" else MAX_MODEL_LEN,
        attention_group_index=ATTENTION_GROUP,
        num_speculative_blocks=K,
        retains_decode_boundaries=retains,
    )
    runner._layer_to_group_idx = [ATTENTION_GROUP, GDN_GROUP]
    runner.tt_max_batch_size = MAX_NUM_REQS

    ledger = _StateLedger()
    calls: list[str] = []
    model.copy_kv_cache_blocks = lambda kv_cache, copies: (
        calls.append("copy"),
        ledger.copy(copies),
    )
    forward = model.decode_forward

    def decode_forward(*args, **kwargs):
        calls.append("forward")
        logged = len(model.chain_log)
        out = forward(*args, **kwargs)
        mamba_state = kwargs["mamba_state"]
        [gdn] = mamba_state.groups
        assert gdn.layer_names == GDN_LAYERS
        windows, counts = gdn.windows, mamba_state.accepted_counts
        chain_rows = kwargs.get("chain_rows")
        if len(model.chain_log) > logged:
            # A deferred fused verify: the device's own positions and counts.
            log = model.chain_log[-1]
            p0, num_valid = log["p0"], log["eff_nv"]
            accepted = model.verify_hidden._readout.accepted_counts
        else:
            # A synchronous verify: the next step carries host counts.
            p0, num_valid = kwargs["start_pos"][:, 0], kwargs["num_valid_drafts"]
            accepted = None
        chain = gdn.chain
        if chain is not None:
            calls.append("chain")
            if chain.copies[: len(runner._last_rows)].any():
                calls.append("chain_copy")
        for row, req_id in enumerate(runner._last_rows):
            if chain_rows is not None and bool(chain_rows[row]):
                assert int(counts[row]) == PLACEHOLDER_TOKEN_ID, (
                    f"chained row {req_id} carries a host count {int(counts[row])}"
                )
            if chain is not None and applies_chain:
                # The device picks the entry by the pending verify's count.
                entry = (row, ledger.last_count[req_id] - 1)
                ledger.chained_reads += 1
                src, dst = chain.copies[entry].tolist()
                if src != dst:
                    ledger.copy([(src, dst)])
                ledger.verify_block(
                    req_id,
                    int(chain.start_blocks[entry]),
                    chain.windows[entry].tolist(),
                    int(p0[row]),
                    int(num_valid[row]),
                )
            else:
                ledger.verify(
                    req_id,
                    windows[row].tolist(),
                    int(counts[row]),
                    int(p0[row]),
                    int(num_valid[row]),
                )
            if accepted is not None:
                ledger.last_count[req_id] = int(accepted[row])
            else:
                ledger.last_count.setdefault(req_id, 1)
        return out

    model.decode_forward = decode_forward
    real_build = runner.build_model_input

    def build_model_input(scheduler_output, grammar_output):
        model_input = real_build(scheduler_output, grammar_output)
        if model_input is not None:
            runner._last_rows = list(model_input.row_req_ids)
        return model_input

    runner.build_model_input = build_model_input
    engine = _MambaEngine(runner, mode)
    engine.add("a", 1)
    engine.add("b", 40)
    engine.add("c", 80)
    for _ in range(steps):
        engine.iterate()
    engine.finish()
    return SimpleNamespace(
        engine=engine,
        runner=runner,
        model=model,
        ledger=ledger,
        controller=_controller(runner),
        calls=calls,
    )


def _boundary_states_are_kept(run) -> None:
    """Mode align: every column a request has passed holds the state after
    exactly its boundary, as prefix caching expects."""
    groups = run.runner._mamba_groups
    for req_id, gdn in run.engine.block_ids.items():
        running = groups.state_columns[req_id]
        for column in range(running):
            assert run.ledger.blocks.get(gdn[column]) == (
                req_id,
                (column + 1) * MAMBA_BLOCK,
            ), (req_id, column)


@pytest.mark.parametrize("cycle", [(3, 0, 1, 2), (3, 3, 3, 3), (0, 0, 0, 0)])
def test_chained_align_steps_keep_the_chain_at_block_boundaries(monkeypatch, cycle):
    """In mode align a chained verify carries its moves for every count the
    pending verify can accept, so boundaries do not break the chain, the reads
    stay exact, and the boundary states are kept, though the host commits each
    verify after the next one ran."""
    run = _mamba_run(monkeypatch, "align", cycle=cycle)
    assert run.ledger.reads > 0
    assert run.ledger.chained_reads > 0, "no row was chained"
    assert "chain_copy" in run.calls
    assert run.controller._chain_breaks["mamba_boundary"] == 0
    _boundary_states_are_kept(run)
    # The "TT submissions:" line reports the share of fused verifies that
    # chained.
    controller = run.controller
    share = (
        100.0 * controller._chained_submissions / controller._fused_verify_submissions
    )
    assert (
        f"({share:.1f}% of {controller._fused_verify_submissions} fused"
        in controller._chain_report()
    )
    plain = _mamba_run(monkeypatch, "align", cycle=cycle, chain=False)
    for req_id, tokens in run.engine.outputs.items():
        shorter = min(len(tokens), len(plain.engine.outputs[req_id]))
        assert shorter > 20
        assert tokens[:shorter] == plain.engine.outputs[req_id][:shorter]


@pytest.mark.parametrize("cycle", [(3, 0, 1, 2), (3, 3, 3, 3), (0, 0, 0, 0)])
def test_retention_0_chains_without_boundary_copies(monkeypatch, cycle):
    """At retention 0 no boundary copy runs: every chained copy entry is
    ``(0, 0)``, the reads stay exact, and the streams match dense retention."""
    run = _mamba_run(monkeypatch, "align", cycle=cycle, retains=False)
    dense = _mamba_run(monkeypatch, "align", cycle=cycle)
    assert run.ledger.chained_reads > 0
    assert "chain" in run.calls
    assert "chain_copy" not in run.calls
    assert run.controller._chain_breaks["mamba_boundary"] == 0
    for req_id, tokens in run.engine.outputs.items():
        shorter = min(len(tokens), len(dense.engine.outputs[req_id]))
        assert shorter > 20
        assert tokens[:shorter] == dense.engine.outputs[req_id][:shorter]


def test_without_the_host_skip_a_late_boundary_copy_corrupts_the_state(monkeypatch):
    """The host must not also run a chained step's boundary copy when the
    pending verify commits: by then the chained step has overwritten the
    copy's source in its window, so the boundary column, which only a prefix
    hit reads, ends up holding a later state."""
    real = MambaCacheGroups.after_acceptance

    def after_acceptance(self, committed, requests):
        late: list[tuple[int, int]] = []
        deferred = set(self.deferred_chained_steps)
        copy_boundary = self._copy_boundary_state

        def capture(req_id, block_ids, position, count):
            copies = copy_boundary(req_id, block_ids, position, count)
            if req_id in deferred:
                late.extend(copies)
            return copies

        self._copy_boundary_state = capture
        try:
            return real(self, committed, requests) + late
        finally:
            del self._copy_boundary_state

    monkeypatch.setattr(MambaCacheGroups, "after_acceptance", after_acceptance)
    run = _mamba_run(monkeypatch, "align")
    with pytest.raises(AssertionError):
        _boundary_states_are_kept(run)


def test_mode_none_chains_across_block_boundaries(monkeypatch):
    """Mode none keeps the state in the 1 + K blocks of the row and never
    copies, so nothing breaks the chain, and no step carries chain choices."""
    run = _mamba_run(monkeypatch, "none")
    assert run.ledger.chained_reads > 0
    assert run.controller._chain_breaks["mamba_boundary"] == 0
    assert "copy" not in run.calls
    assert "chain" not in run.calls


def test_a_model_that_ignores_the_chain_choices_reads_a_stale_state(monkeypatch):
    """The negative control: a model that reads a chained row's start state
    from window column ``count - 1`` of ``MambaGroupState.windows``, as in mode
    none, and runs no chain copy, reads a state that was never written once a
    chained step crosses a block boundary."""
    with pytest.raises(AssertionError, match="read block"):
        _mamba_run(monkeypatch, "align", applies_chain=False)


def _pending_rows_reach_a_boundary(run, controller) -> bool:
    """Whether a row of the pending verify can reach a block boundary before
    the host applies it. A chained run can also hold a resolved verify that
    the host has not applied yet, so the row's committed length can lag by
    that verify too."""
    runner = run.runner
    unapplied = sum(not s.is_resolved() for s in runner._pending_async_steps) + len(
        runner._completed_decode_steps
    )
    for req_id in controller._last_fused_rows:
        first = runner.requests[req_id].num_tokens - 1
        if first // MAMBA_BLOCK != (first + WIDTH * unapplied) // MAMBA_BLOCK:
            return True
    return False


@pytest.mark.parametrize("retains", [True, False])
def test_a_prefill_waits_for_a_pending_verify_that_reaches_a_boundary(
    monkeypatch, retains
):
    """A prefill submitted behind a pending verify runs before that verify
    commits; with a row of it able to reach a block boundary, the boundary copy
    would follow the scheduler's free of the verify's blocks. Such a prefill
    drains the verify instead. At retention 0 there is no boundary copy, and
    the next decode drains the verify before it moves any state, so the
    prefill never waits."""
    run = _mamba_run(
        monkeypatch,
        "align",
        steps=0,
        model_capabilities=PREFILL_CAPS,
        retains=retains,
    )
    engine, controller = run.engine, run.controller
    from .test_spec_async import _scheduler_output

    seen = set()
    for _ in range(40):
        engine.iterate()
        if not controller._chain_candidate_pending():
            continue
        prefill = _scheduler_output(run.runner, new=[("p", 300)])
        prefill.scheduled_cached_reqs = dataclasses.replace(
            prefill.scheduled_cached_reqs, num_output_tokens=[]
        )
        expected = not _pending_rows_reach_a_boundary(run, controller)
        allowed = controller.prefill_overlap_allowed(prefill)
        assert allowed is (expected or not retains)
        seen.add(expected)
    engine.finish()
    # The accept cycle reaches both cases with dense retention; with
    # retention 0 the overlap is allowed whatever it reaches
    # (``test_chain_blocker_ignores_pending_rows_at_retention_0``).
    assert seen == {True, False} if retains else seen


# --- MambaCacheGroups.chain_blocker -------------------------------------------


def _groups(mode="align"):
    return MambaCacheGroups(
        group_indices=(1,),
        layer_names=(GDN_LAYERS,),
        mamba_cache_mode=mode,
        block_size=MAMBA_BLOCK,
        attention_group_index=0,
        num_speculative_blocks=K,
    )


def _requests(**lengths):
    return {r: SimpleNamespace(num_tokens=n) for r, n in lengths.items()}


@pytest.mark.parametrize(
    ("length", "pending", "blocked"),
    [
        # 9 computed tokens; one pending verify reaches 13.
        (10, 1, False),
        # Two pending verifies reach 17.
        (10, 2, True),
        # 13 computed tokens: 17 at most.
        (14, 1, True),
        # Exactly on a boundary: the computed state is already past it.
        (17, 1, False),
        (16, 1, True),
    ],
)
def test_chain_blocker_looks_one_step_width_ahead_per_pending_step(
    length, pending, blocked
):
    """A pending row that the step leaves out, as for a prefill built over the
    pending verifies; the same row is exempt as a chained row."""
    requests = _requests(a=length)
    reason = _groups().chain_blocker(requests, ["a"], [], pending, WIDTH)
    assert (reason == "mamba_boundary") is blocked
    assert _groups().chain_blocker(requests, ["a"], ["a"], pending, WIDTH) is None


def test_chain_blocker_counts_a_finishing_row_of_the_pending_verify():
    """A finishing row is not in the chained step, but its pending verify still
    commits, and its boundary copy must not follow the scheduler's free."""
    requests = _requests(a=5, finishing=14)
    assert _groups().chain_blocker(requests, ["a", "finishing"], ["a"], 1, WIDTH)
    assert _groups().chain_blocker(requests, ["a"], ["a"], 1, WIDTH) is None


def test_chain_blocker_is_off_in_mode_none():
    assert (
        _groups("none").chain_blocker(_requests(a=16), ["a"], ["a"], 1, WIDTH) is None
    )
