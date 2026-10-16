# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The drafter offers for a lone request and declines for a batch.

`TT_SPEC_DRAFT_POLICY=solo` makes the model's drafter offer the full draft
length while one request is live and nothing while more are, which is the shape
a deployment has: speculation pays for a lone request and loses to batching for
a full one. A server launched that way has to do both, and "both responses came
back" is no evidence of either, because the text is the same whether the server
drafted or not.

So each test here is a delta on vLLM's own counters across the requests it
sent. `spec_decode_num_drafts_total` is the one that separates the two
behaviors: it counts the steps on which drafts were verified, so a solo request
moves it once per step and a batch barely moves it at all.

These tests need a server launched with that policy, which the `adaptive`
configuration of `run_spec_regression.sh` does. They skip elsewhere rather than
assert something the launch cannot produce.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tt.spec.spec_client import (
    acceptance_delta,
    assert_full_length_completion,
)

MAX_TOKENS = 64


@pytest.fixture(autouse=True)
def only_the_adaptive_launch(request):
    if str(request.config.getoption("--tt-spec-draft-policy")) != "solo":
        pytest.skip(
            "this server's drafter always drafts: launch with "
            "TT_SPEC_DRAFT_POLICY=solo and pass --tt-spec-draft-policy=solo"
        )


def test_a_lone_request_is_drafted_for_on_every_step(
    spec_server, spec_config, ascending_prompt, record
):
    """One request live, so the drafter offers and the verify accepts.

    The floor is what makes this an assertion rather than an observation: a
    request emitting `MAX_TOKENS` tokens while committing `1+K` per accepted
    step cannot have been drafted for fewer than a handful of steps, and a
    drafter that silently offered nothing would leave the counter at zero while
    the response looked identical.
    """
    before = spec_server.metrics()
    result = spec_server.complete(ascending_prompt(64), max_tokens=MAX_TOKENS)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(requests=[result.request], acceptance=delta.as_dict())

    assert_full_length_completion(result, MAX_TOKENS)
    assert delta.accepted > 0, "a lone request never had a draft accepted"
    # Close to the full width per accepted step: this policy declines by row,
    # and with one row live there is nothing to decline.
    assert delta.mean_acceptance_length > 1.0


def test_a_batch_is_not_drafted_for(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """Several requests live, so the drafter declines and the steps are plain.

    What this asserts is a ratio rather than a zero. The batch is not live for
    the whole run: the first request is alone until the others arrive, the last
    one is alone again once its peers finish, and the drafter decides at each
    commit, so the edges of the run legitimately draft. The middle must not,
    and a policy that ignored the batch would draft on nearly every step of it
    instead.
    """
    rows = min(4, max_batch_size)
    if rows < 2:
        pytest.skip("this server serves one row at a time")
    prompts = [ascending_prompt(64, start=100 * (index + 1)) for index in range(rows)]

    before = spec_server.metrics()

    def send(prompt):
        return spec_server.complete(prompt, max_tokens=MAX_TOKENS)

    with ThreadPoolExecutor(max_workers=rows) as pool:
        results = list(pool.map(send, prompts))
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        requests=[result.request for result in results],
        acceptance=delta.as_dict(),
        rows=rows,
    )

    for result in results:
        assert_full_length_completion(result, MAX_TOKENS)

    # Acceptance rather than the draft counters, because those cannot be read
    # the same way on both launches: under asynchronous scheduling
    # ``AsyncScheduler`` gives every scheduled request ``[-1] * K`` as its
    # lookahead reservation and vLLM counts that as drafts offered, whether or
    # not a real draft was ever verified. Nothing accepts without a real
    # draft, so the accepted count is the signal that means the same thing in
    # both modes.
    #
    # A ratio, not a zero: the first request is alone until its peers arrive
    # and the last is alone again once they finish, so the edges of the run
    # legitimately speculate. A policy ignoring the batch would accept on
    # nearly every step of it instead.
    assert delta.accepted < MAX_TOKENS * rows / 4, (
        f"{delta.accepted} token(s) were accepted from drafts across a "
        f"{rows}-row batch, which is not a policy that declines for a batch"
    )


def test_speculation_resumes_when_the_batch_empties(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """A peer arrives and leaves, and the lone request speculates again.

    The transition is what this is for. The drafter decides at the commit of
    the step that ran, so speculation stops one step after the peer arrives and
    restarts one step after it leaves, and neither edge may leave the surviving
    request stuck in plain decoding: that would look like a working server with
    no speculation for the rest of its life.
    """
    if max_batch_size < 2:
        pytest.skip("this server serves one row at a time")

    long_prompt = ascending_prompt(64, start=100)
    peer_prompt = ascending_prompt(64, start=900)

    # The lengths are chosen so the long request outlives its peer by a wide
    # margin. It is the faster of the two per step while it speculates,
    # committing 1+K tokens where the peer commits one, so a peer asking for a
    # comparable length would finish second and leave nothing to measure.
    long_tokens = MAX_TOKENS * 16
    peer_tokens = MAX_TOKENS

    with ThreadPoolExecutor(max_workers=2) as pool:
        long_future = pool.submit(
            spec_server.complete, long_prompt, max_tokens=long_tokens
        )
        # Short, and it has to be: this model's forward is host arithmetic,
        # so a step costs about a millisecond and a tenth of a second is a
        # hundred steps. A longer pause and the long request finishes before
        # its peer is even posted, leaving nothing to measure.
        time.sleep(0.02)
        peer_before = spec_server.metrics()
        peer = pool.submit(spec_server.complete, peer_prompt, max_tokens=peer_tokens)
        peer_result = peer.result()
        peer_after = spec_server.metrics()
        long_result = long_future.result()
    tail_after = spec_server.metrics()

    while_paired = acceptance_delta(peer_before, peer_after, spec_config.k)
    after_it_left = acceptance_delta(peer_after, tail_after, spec_config.k)
    record(
        requests=[long_result.request, peer_result.request],
        while_paired=while_paired.as_dict(),
        after_it_left=after_it_left.as_dict(),
    )

    assert long_result.completion_tokens > peer_tokens
    # Checked before the claim below, because an empty window would satisfy
    # "no drafts" for a reason that has nothing to do with the policy: the
    # long request would simply have finished first.
    assert after_it_left.steps > 0, (
        "the long request finished before its peer, so the window after the "
        "peer left holds no steps to draw a conclusion from"
    )
    assert after_it_left.accepted > 0, (
        "the request never speculated again after its peer finished"
    )
