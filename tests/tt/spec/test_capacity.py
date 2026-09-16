# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Preemption and replay, proved before anything is claimed about them.

Sending more requests than a server can hold does not establish that preemption
happened. The scheduler may queue them, admit them in turn, and never preempt
anything, and every response still arrives correct. So each test here reads
``vllm:num_preemptions_total`` across the work and **skips rather than passes**
when the counter did not move: a test that cannot reach its subject should say
so instead of reporting success.

Preemption matters more under speculation than without it. A preempted request
loses the KV blocks its candidate state was computed against, so the runner has
to drop its accepted count and its pending proposal; a resume that kept either
would continue from a candidate the model has since overwritten. That is
invisible in a completed response unless the output is checked against what the
request should have emitted.

Reaching preemption needs a server launched with little KV capacity, which is
what ``README.md`` calls the constrained configuration: a small
``--max_model_len`` and enough concurrent long requests that their blocks do
not fit at once.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tt.spec.spec_client import acceptance_delta


def _flood(spec_server, prompts, max_tokens):
    """Post every request at once and wait for all of them."""

    def send(prompt):
        return spec_server.complete(prompt, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        return list(pool.map(send, prompts))


def test_preempted_requests_still_emit_their_full_output(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """Every response complete, and the preemption counter moved.

    The assertion order matters: the counter is read first and the test skips
    if it did not move, so a pass here means the responses were produced
    across at least one preemption and replay rather than in spite of none.
    """
    rows = max(4, min(8, max_batch_size))
    max_tokens = 192
    prompts = [ascending_prompt(96, start=100 * (index + 1)) for index in range(rows)]

    before = spec_server.metrics()
    results = _flood(spec_server, prompts, max_tokens)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        requests=[result.request for result in results],
        responses=[result.body for result in results],
        acceptance=delta.as_dict(),
        preemptions=delta.preemptions,
    )

    if delta.preemptions == 0:
        pytest.skip(
            "no preemption occurred, so this run establishes nothing about it. "
            "Launch the constrained configuration from README.md: a smaller "
            "--max_model_len, or more concurrent requests asking for more "
            "tokens each"
        )

    for result in results:
        assert result.status == 200
        assert result.completion_tokens == max_tokens, (
            "a request that was preempted and resumed has to emit its whole "
            "output: a short response means the replay lost tokens"
        )
    assert delta.drafts > 0


def test_a_resumed_request_speculates_again_after_its_replay(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """Drafting continues across the preemption boundary.

    A resume re-prefills the prompt and every generated token, and the runner
    drops the request's accepted count because the candidate state it named is
    gone. If the resume left speculation off, or left a stale count behind, the
    counters after the flood would show drafting stalling rather than
    continuing.
    """
    rows = max(4, min(8, max_batch_size))
    prompts = [ascending_prompt(96, start=100 * (index + 1)) for index in range(rows)]

    before = spec_server.metrics()
    results = _flood(spec_server, prompts, 192)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        acceptance=delta.as_dict(),
        preemptions=delta.preemptions,
        responses=[result.body for result in results],
    )

    if delta.preemptions == 0:
        pytest.skip("no preemption occurred; see the constrained configuration")

    depth = (
        spec_config.k if spec_config.accept_depth is None else spec_config.accept_depth
    )
    if depth > 0:
        assert delta.accepted > 0, (
            "nothing was accepted across a run that included a preemption, so "
            "speculation did not resume with the requests"
        )
        # Drafting continued at the configured depth on both sides of the
        # boundary, which a stale accepted count would disturb.
        assert delta.per_position[:depth] == [delta.drafts] * depth
        assert delta.per_position[depth:] == [0] * (spec_config.k - depth)


def test_a_request_admitted_after_the_flood_is_unaffected(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """The server is still correct once the queue drains.

    A preemption that leaked a device state slot or a persistent-batch row
    would show up as the next request failing, speculating less, or answering
    short.
    """
    rows = max(4, min(8, max_batch_size))
    prompts = [ascending_prompt(96, start=100 * (index + 1)) for index in range(rows)]

    flood_before = spec_server.metrics()
    _flood(spec_server, prompts, 128)
    flood_after = spec_server.metrics()
    flood = acceptance_delta(flood_before, flood_after, spec_config.k)

    before = spec_server.metrics()
    completion = spec_server.complete(ascending_prompt(64, start=19), max_tokens=64)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        flood_preemptions=flood.preemptions,
        request=completion.request,
        response=completion.body,
        acceptance=delta.as_dict(),
    )

    if flood.preemptions == 0:
        pytest.skip("no preemption occurred; see the constrained configuration")

    assert completion.status == 200
    assert completion.completion_tokens == 64
    assert delta.drafts > 0
    depth = (
        spec_config.k if spec_config.accept_depth is None else spec_config.accept_depth
    )
    assert delta.accepted == delta.drafts * depth, (
        "the request after the flood did not accept at the configured depth, "
        "so something the preemptions left behind is still affecting it"
    )
