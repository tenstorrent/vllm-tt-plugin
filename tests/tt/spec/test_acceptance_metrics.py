# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculation happened, and happened as configured.

A completed request proves nothing about drafting. The server serves the same
text whether it drafted five tokens a step or none, and a speculative launch
whose drafter silently proposed nothing looks exactly like a working one from
the outside. So every assertion here is a delta on vLLM's own counters across
the request, and the response is checked against what those counters imply.

What each counter has to show, per configuration:

``accept depth 0``
    Drafts offered on nearly every step, none accepted, every per-position
    count zero. One token committed per step.
``accept depth n``
    Exactly ``n`` positions accepted on every speculative step, and exactly
    the positions ``0..n-1``: a mean acceptance length of ``n+1``, which
    distinguishes "two of five every step" from "five of five two steps in
    five".
``accept every draft``
    Every position accepted on every step, and a mean acceptance length of
    ``1+K``.

The committed token ids are checked as well, because a counter says how many
tokens were accepted and not which ones.
"""

from __future__ import annotations

import pytest

from tests.tt.spec.spec_client import acceptance_delta

MAX_TOKENS = 96


def _run(spec_server, prompt, **overrides):
    overrides.setdefault("max_tokens", MAX_TOKENS)
    before = spec_server.metrics()
    completion = spec_server.complete(prompt, **overrides)
    after = spec_server.metrics()
    return completion, before, after


def test_a_completed_request_also_moved_the_draft_counters(
    spec_server, spec_config, ascending_prompt, record
):
    """The baseline claim: drafting occurred at all.

    This is the assertion whose absence made the first device experiment weak.
    A 200 with the right number of tokens is compatible with a server that
    never drafted, and with a drafter whose proposals were all thrown away
    before the verify.
    """
    prompt = ascending_prompt(MAX_TOKENS + 8)
    completion, before, after = _run(spec_server, prompt)
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        request=completion.request,
        response=completion.body,
        acceptance=delta.as_dict(),
    )

    assert completion.status == 200
    assert completion.completion_tokens == MAX_TOKENS
    assert delta.drafts > 0, "no step carried a draft, so nothing speculated"
    assert delta.draft_tokens == delta.drafts * spec_config.k, (
        "every speculative step offers the full draft length, so the offered "
        "count has to be the step count times K"
    )
    assert delta.steps > 0


def test_the_per_position_acceptance_is_exactly_the_configured_depth(
    spec_server, spec_config, ascending_prompt, record
):
    """Where acceptance stopped, position by position.

    A total acceptance count cannot tell a prefix of two accepted on every
    step from five accepted on two steps in five. The per-position counters
    can, and the configured depth predicts them exactly: positions below the
    depth accepted on every step, positions at or above it never.
    """
    prompt = ascending_prompt(MAX_TOKENS + 8)
    completion, before, after = _run(spec_server, prompt)
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        request=completion.request,
        response=completion.body,
        acceptance=delta.as_dict(),
    )

    assert completion.status == 200
    depth = (
        spec_config.k if spec_config.accept_depth is None else spec_config.accept_depth
    )
    expected = [delta.drafts if index < depth else 0 for index in range(spec_config.k)]
    assert delta.per_position == expected, (
        f"accept depth {depth} of {spec_config.k} predicts {expected} "
        f"acceptances by position, and the server reported {delta.per_position}"
    )
    assert delta.accepted == delta.drafts * depth
    assert delta.mean_acceptance_length == pytest.approx(depth + 1, abs=1e-6)


def test_the_committed_width_per_step_follows_the_acceptance(
    spec_server, spec_config, ascending_prompt, record
):
    """The response's length and the step count have to agree.

    Each speculative step commits the accepted drafts plus one token, so the
    tokens and the steps are two views of the same thing. A commit path that
    dropped or duplicated a token would leave them disagreeing while both
    looked plausible alone.
    """
    prompt = ascending_prompt(MAX_TOKENS + 8)
    completion, before, after = _run(spec_server, prompt)
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        request=completion.request,
        response=completion.body,
        acceptance=delta.as_dict(),
    )

    committed = completion.completion_tokens
    width = spec_config.committed_per_step
    # Every accepted draft is a token the response returned.
    assert committed >= delta.accepted, (
        f"the response carries {committed} tokens but the counters say "
        f"{delta.accepted} drafts were accepted, so tokens went missing"
    )
    # And no step commits more than its width, so the steps bound the tokens.
    assert committed <= delta.steps * width, (
        f"{committed} tokens across {delta.steps} steps is more than the "
        f"committed width of {width} allows, so a step committed twice"
    )
    # The counters' own view of the width, which is exact: every speculative
    # step accepted the same number of drafts and committed one more token.
    assert delta.mean_acceptance_length == pytest.approx(width, abs=1e-6)


def test_the_committed_ids_are_the_dummy_s_own_arithmetic(
    spec_server, spec_config, ascending_prompt, record
):
    """Which tokens, not only how many.

    The ``depth`` target commits an ascending run with the rejected position's
    correction skipping one value, so the sequence is predictable from the
    accept depth alone. The ``fixed`` target follows its own rule instead, and
    the losslessness test is what checks that one.
    """
    if spec_config.target != "depth":
        pytest.skip("the ascending-run arithmetic belongs to the 'depth' target")
    prompt = ascending_prompt(MAX_TOKENS + 8)
    completion, before, after = _run(spec_server, prompt)
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        request=completion.request,
        response=completion.body,
        acceptance=delta.as_dict(),
        committed_ids=completion.token_ids,
    )

    ids = completion.token_ids
    assert len(ids) == MAX_TOKENS
    # A prefill commits token 0, and every step after that counts up: by one
    # where a draft was accepted, and by two across the position that rejected
    # (the target's choice there is the draft plus one).
    assert ids[0] == 0
    steps = [step for step in zip(ids, ids[1:])]
    deltas = sorted({after_id - before_id for before_id, after_id in steps})
    if spec_config.accepts_every_draft:
        assert deltas == [1], f"an accept-all run counts up by one: saw {deltas}"
    else:
        assert deltas == [1, 2], (
            "a partially accepting run counts up by one inside an accepted "
            f"prefix and by two across the rejection: saw {deltas}"
        )


def test_a_request_that_asks_for_one_token_still_speculates_nothing_wrong(
    spec_server, spec_config, ascending_prompt, record
):
    """The degenerate length: one token, and no draft in flight yet.

    A request's first decode step has nothing drafted, because the drafter runs
    after a commit. So this request must complete with one token and move the
    draft counters not at all, which is also the one case where a zero draft
    count is correct rather than a failure.
    """
    prompt = ascending_prompt(32)
    completion, before, after = _run(spec_server, prompt, max_tokens=1)
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        request=completion.request,
        response=completion.body,
        acceptance=delta.as_dict(),
    )

    assert completion.status == 200
    assert completion.completion_tokens == 1
    assert delta.drafts == 0, (
        "a single-token request has no step that could carry a draft, so a "
        "draft counted here came from somewhere else"
    )
