# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative decoding over Qwen3.8 GDN state in vLLM ``MambaSpec`` blocks.

The launch speculates with the model-owned drafter (``custom_class``) and K
drafts per step, so every GDN group's ``MambaSpec`` declares K speculative
blocks. ``tests.test_mamba_state_lifecycle._Harness`` drives a real
``TTScheduler`` with its ``KVCacheManager`` and a real ``TTModelRunner`` over
the Qwen3.8 TP=8 layout. Its drafter offers a scripted number of drafts per
row, and each verify accepts a scripted prefix of them through the runner's
own accept walk (``walk_spec_acceptance``) and commit
(``commit_spec_acceptance``), so the runner and the scheduler see the counts a
real verify produces.

Every step checks each row's ``mamba_state``:

- against vLLM's allocation: the window is the request's ``1 + K`` blocks at
  the columns upstream ``mamba_get_block_table_tensor`` selects, and vLLM
  still holds the start block for the request;
- against upstream: in mode ``align`` the start block is the block upstream
  ``preprocess_mamba`` would read (the source of its copy, or window column
  ``count - 1``), given the runner's accepted counts; in mode ``none`` it is
  block-table column ``count - 1``;
- against a state ledger: a verify writes the state after query token ``t``
  into window column ``t`` for each valid draft and anything into the other
  columns, a prefill or ordinary decode writes column 0 and anything into the
  rest, and every start state read must be the state of exactly the row's
  computed tokens. In mode ``align`` every block vLLM keeps in its prefix
  cache must hold the state of exactly the tokens its hash covers.
"""

from __future__ import annotations

import pytest
from vllm.v1.request import RequestStatus

from tests.test_mamba_state_lifecycle import (
    BLOCK_SIZE,
    EOS,
    MAX_NUM_SEQS,
    _Harness,
    _prompt,
)


def _all_counts(num_speculative_tokens: int) -> set[tuple[int, int]]:
    """Every ``(valid drafts, committed tokens)`` pair a verify can produce."""
    return {
        (valid, count)
        for valid in range(num_speculative_tokens + 1)
        for count in range(1, valid + 2)
    }


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_every_accepted_count_resumes_from_its_window_column(
    mode, num_speculative_tokens
):
    """A verify that commits ``a`` tokens leaves the request's state in window
    column ``a - 1``, and the next step reads it there.

    The drafter offers K, K - 1, ..., 0 drafts in turn, and each verify row
    commits 1 to ``v + 1`` tokens in turn for its ``v`` valid drafts: every
    accepted count from 1 (draft 0 rejected) to K + 1 (all drafts and the
    bonus), a rejection at every draft position, and rows with zero and with
    partial valid drafts. Prompts end around block boundaries, so verifies
    cross into new block-table columns.
    """
    h = _Harness(
        num_blocks=40,
        max_num_batched_tokens=4096,
        mode=mode,
        num_speculative_tokens=num_speculative_tokens,
    )
    for i, prompt_len in enumerate((1020, 2047, 300, 3000)):
        h.add(f"r{i}", prompt_len, max_tokens=80)

    h.run(max_steps=600)

    assert not h.scheduler.requests
    assert _all_counts(num_speculative_tokens) <= set(h.accepted)
    assert h.step_kinds["verify"] > h.step_kinds["prefill"]
    assert h.verify_crossings, "no verify crossed into a new column"
    if mode == "align":
        assert h.cached_states_checked


@pytest.mark.parametrize("narrow_decode", [True, False])
def test_a_window_that_crossed_a_boundary_moves_back_after_a_rejection(
    narrow_decode,
):
    """A verify's window begins at the column of its last draft's position,
    so a rejection can leave the request's state in a column past the one the
    next step's window begins at.

    Request a decodes from position 1021 with K = 3. The verify of positions
    1022 to 1025 writes into columns 1 to 4, and its first draft is rejected:
    the state after 1023 tokens is in column 1. The drafter then offers
    nothing, so the next step computes position 1023 alone and its window
    begins at column 0: it reads column 1's block and writes column 0's,
    which vLLM still holds. Without narrow decode that step is a verify with
    no valid drafts.
    """
    h = _Harness(
        num_blocks=24,
        mode="align",
        num_speculative_tokens=3,
        narrow_decode=narrow_decode,
        offers=(3, 0),
    )
    h.add("a", 1021, max_tokens=40)

    h.run()

    assert h.windows_moved_back == 1
    assert _all_counts(3) - {(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)} <= set(h.accepted)
    assert ("decode" in h.step_kinds) == narrow_decode


def test_eos_and_length_limits_trim_accepted_drafts():
    """A verify can commit more tokens than a request may still produce.

    Request ``eos`` gets EOS as its second draft once it has 10 output tokens,
    and the verify accepts all drafts: vLLM keeps the tokens up to EOS and
    finishes the request. Request ``short`` may produce 6 tokens, fewer than
    the prefill and two full verifies commit. Both finish with the trimmed
    output, their blocks go back to the pool, and request ``late`` reuses them.
    """
    h = _Harness(
        num_blocks=32,
        max_num_batched_tokens=4096,
        mode="align",
        num_speculative_tokens=3,
        offers=(3,),
    )
    h.add("eos", 900, max_tokens=200, ignore_eos=False)
    h.eos_after["eos"] = 10
    h.add("short", 1500, max_tokens=6)
    h.add("long", 2000, max_tokens=60)
    h.run(max_steps=200)
    h.add("late", 3000, max_tokens=30)
    h.run(max_steps=200)

    eos = h.requests["eos"]
    assert eos.output_token_ids[-1] == EOS
    assert eos.status == RequestStatus.FINISHED_STOPPED
    assert 10 < eos.num_output_tokens < 200
    short = h.requests["short"]
    assert short.num_output_tokens == 6
    assert short.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert set(h.ledger.gdn_blocks) & set(h.zeroed), (
        "no GDN block went back to attention"
    )


def test_prefix_hit_then_speculative_decode():
    """Request b repeats request a's 2500-token prompt: it hits the state
    after 2048 tokens that a's prefill wrote as a checkpoint, prefills the
    remaining 452 tokens and speculates. Requests c and d share a's first
    2048 tokens and diverge; all three read the kept block, which keeps its
    state while they verify across 3072."""
    h = _Harness(
        num_blocks=56,
        max_num_batched_tokens=4096,
        mode="align",
        checkpoint_alignment=64,
        num_speculative_tokens=3,
    )
    prompt = _prompt(1, 2500)
    h.add("a", prompt=prompt, max_tokens=8)
    h.run()
    h.add("b", prompt=prompt, max_tokens=60)
    for i, name in enumerate("cd"):
        suffix = _prompt(10 + i, 900 + 50 * i)
        h.add(name, prompt=prompt[:2048] + suffix, max_tokens=200)
    h.run(max_steps=800)

    assert h.checkpoints[0] == ("a", 2048)
    assert h.hits == {"b": [2048], "c": [2048], "d": [2048]}
    assert {block for blocks in h.hit_blocks.values() for block in blocks} == {
        h.checkpoint_blocks[0]
    }
    assert _all_counts(3) <= set(h.accepted)
    assert h.verify_crossings
    assert h.cached_states_checked


@pytest.mark.parametrize("max_num_batched_tokens", [1024, 2048, 4096])
def test_chunked_prefill_keeps_speculative_blocks_out_of_checkpoints(
    max_num_batched_tokens,
):
    """Between prefill chunks ``MambaManager`` moves the speculative blocks a
    chunk passes to the end of the request's row, or keeps one in place as the
    block of a prefill checkpoint. The runner's row still lists a moved block
    at its old column; that column is neither a checkpoint nor a block the
    chunk writes, and ``_Harness.check_blocks`` compares every checkpoint with
    the one vLLM reserved. Prompts of 3000 to 6000 tokens prefill in up to six
    chunks."""
    h = _Harness(
        num_blocks=64,
        max_num_batched_tokens=max_num_batched_tokens,
        mode="align",
        checkpoint_alignment=64,
        num_speculative_tokens=3,
    )
    for i, prompt_len in enumerate((3000, 4500, 6000)):
        h.add(f"r{i}", prompt_len, max_tokens=20)
    h.run(max_steps=400)

    assert max(len(spans) for spans in h.prefill_spans.values()) > 1
    assert bool(h.checkpoints) == (max_num_batched_tokens > BLOCK_SIZE)
    assert h.step_kinds["verify"]


@pytest.mark.parametrize("mode", ["none", "align"])
def test_preemption_resume_and_abort_while_speculating(mode):
    """A pool of 28 blocks admits two 1000-token requests with K = 3, which
    hold 4 GDN blocks per group each. Decoding past the 1024-token boundary
    needs another attention block per request, and in mode align new GDN
    blocks too, so vLLM preempts. Request r2 is aborted at step 20. A
    preempted request loses its blocks and its recorded window, and resumes
    as a prefill from position 0 or, in mode align, from a kept prefix; its
    first decode after that starts from the prefill's end state."""
    h = _Harness(
        num_blocks=28,
        max_num_batched_tokens=4096,
        mode=mode,
        num_speculative_tokens=3,
    )
    for i in range(MAX_NUM_SEQS):
        h.add(f"r{i}", 1000, max_tokens=60 + 10 * i)

    h.run(max_steps=2000, abort={20: "r2"})

    assert not h.scheduler.requests
    assert h.preempted - {"r2"}, "the pool was large enough to avoid preemption"
    assert h.preempted <= h.resumed_from_zero | {"r2"}
    assert any(len(rows) > 1 for rows in h.rows_seen.values()), "no row moved"
    assert h.requests["r2"].status == RequestStatus.FINISHED_ABORTED


def test_small_pool_reuses_speculative_blocks_across_groups():
    """Two waves of speculating requests share a pool that holds about one
    wave: a block that held a request's speculative states goes to another
    request's GDN window or to attention, which vLLM lists for zeroing."""
    h = _Harness(
        num_blocks=30,
        max_num_batched_tokens=4096,
        mode="align",
        num_speculative_tokens=3,
    )
    for i, prompt_len in enumerate((3000, 700, 1500, 300)):
        h.add(f"first-{i}", prompt_len, max_tokens=40)
    h.run(max_steps=600)
    for i, prompt_len in enumerate((900, 2500, 400)):
        h.add(f"second-{i}", prompt_len, max_tokens=40)
    h.run(max_steps=600)

    assert not h.scheduler.requests
    assert set(h.ledger.gdn_blocks) & set(h.zeroed), (
        "no GDN block went back to attention"
    )


def test_ngram_drafts_resume_from_their_window_column():
    """The state handling does not depend on who drafts: with the runner's
    n-gram proposer, which drafts from repeats in a request's own text, every
    accepted count resumes from its window column too."""
    h = _Harness(
        num_blocks=40,
        max_num_batched_tokens=4096,
        mode="align",
        num_speculative_tokens=3,
        spec_method="ngram",
    )
    for i, prompt_len in enumerate((1020, 1500)):
        h.add(f"r{i}", prompt=_prompt(i, 40) * (prompt_len // 40), max_tokens=80)

    h.run(max_steps=400)

    assert not h.scheduler.requests
    assert {(3, 1), (3, 2), (3, 3), (3, 4)} <= set(h.accepted)
    assert h.cached_states_checked
