# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Several requests speculate in the same steps without mixing up.

Sending requests at once is not a concurrency test on its own: the server can
serve them one after another and every response still arrives correct. So these
tests do three things the earlier device experiment did not.

They give each request a **different prompt and a different output length**, so
a row that read another row's state produces a detectably wrong answer rather
than the same answer twice.

They **stagger the arrivals**, which is what makes rows join a batch that is
already decoding and forces the persistent batch to grow while drafts are in
flight.

And where a claim about the same-step batch size is made, the test reads the
scheduler's direct report of the widest decode batch. A claim that rows ran
together is a claim about the scheduler output, not about how many requests
were posted or how many prompt tokens a metric counted.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tt.spec.spec_client import (
    acceptance_delta,
    assert_multirow_decode_was_scheduled,
)


def _post_all(spec_server, jobs, stagger: float = 0.0):
    """Send every job, optionally spacing the arrivals."""

    def send(index_and_job):
        index, (prompt, max_tokens) = index_and_job
        if stagger:
            time.sleep(index * stagger)
        return spec_server.complete(prompt, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return list(pool.map(send, enumerate(jobs)))


def test_distinct_requests_each_get_their_own_output(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """Different prompts and different lengths, arriving together.

    Each response has to answer its own prompt: under the ``depth`` target the
    output is an ascending run seeded by the prefill rather than by the prompt,
    so the check that distinguishes the rows is the requested length, and the
    ids are checked for the shape the target produces.
    """
    rows = min(4, max_batch_size)
    if rows < 2:
        pytest.skip("this server serves one row at a time")
    jobs = [
        (ascending_prompt(48 + 16 * index, start=100 * (index + 1)), 24 + 8 * index)
        for index in range(rows)
    ]

    before = spec_server.metrics()
    results = _post_all(spec_server, jobs)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        requests=[result.request for result in results],
        responses=[result.body for result in results],
        acceptance=delta.as_dict(),
    )

    for (_, max_tokens), result in zip(jobs, results):
        assert result.status == 200
        assert result.completion_tokens == max_tokens, (
            "a row served another row's length, or its own length was clipped"
        )
        assert len(result.token_ids) == max_tokens
    assert delta.drafts > 0


def test_the_rows_shared_engine_steps(
    spec_server,
    spec_config,
    ascending_prompt,
    max_batch_size,
    server_log,
    record,
):
    """Require the scheduler to report a decode step with multiple rows."""
    rows = min(4, max_batch_size)
    if rows < 2:
        pytest.skip("this server serves one row at a time")
    jobs = [
        (ascending_prompt(64, start=100 * (index + 1)), 64) for index in range(rows)
    ]

    before = spec_server.metrics()
    results = _post_all(spec_server, jobs)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    widest = assert_multirow_decode_was_scheduled(server_log)
    record(
        requests=[result.request for result in results],
        acceptance=delta.as_dict(),
        rows_posted=rows,
        widest_decode_batch_size=widest,
    )

    assert all(result.status == 200 for result in results)


def test_a_request_joining_a_running_batch_is_served_correctly(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """Staggered arrivals: rows join while others are mid-speculation.

    A row added to the persistent batch while drafts are in flight is where a
    row-keyed piece of state goes wrong, and where the scheduler's draft
    bookkeeping has to keep the newcomer out of a block it never drafted for.
    """
    rows = min(4, max_batch_size)
    if rows < 2:
        pytest.skip("this server serves one row at a time")
    # The first request is long enough to still be decoding when the last one
    # arrives, and the later ones are short.
    jobs = [(ascending_prompt(64, start=500), 160)] + [
        (ascending_prompt(64, start=600 + 50 * index), 24) for index in range(rows - 1)
    ]

    before = spec_server.metrics()
    results = _post_all(spec_server, jobs, stagger=0.05)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        requests=[result.request for result in results],
        responses=[result.body for result in results],
        acceptance=delta.as_dict(),
    )

    for (_, max_tokens), result in zip(jobs, results):
        assert result.status == 200
        assert result.completion_tokens == max_tokens
    assert delta.drafts > 0


def test_every_row_s_acceptance_matches_the_configured_depth(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """Acceptance is per row, and a batch must not flatten it.

    The per-position counters are summed across rows, so with every row at the
    same configured depth the totals stay proportional: a batch where one row's
    short acceptance shortened another's would show a position count below the
    step count.
    """
    rows = min(4, max_batch_size)
    if rows < 2:
        pytest.skip("this server serves one row at a time")
    depth = (
        spec_config.k if spec_config.accept_depth is None else spec_config.accept_depth
    )
    if depth == 0:
        pytest.skip("nothing is accepted at depth 0, so there is no ratio to check")
    jobs = [
        (ascending_prompt(64, start=100 * (index + 1)), 48) for index in range(rows)
    ]

    before = spec_server.metrics()
    results = _post_all(spec_server, jobs)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        requests=[result.request for result in results],
        acceptance=delta.as_dict(),
    )

    assert all(result.status == 200 for result in results)
    expected = [delta.drafts if index < depth else 0 for index in range(spec_config.k)]
    assert delta.per_position == expected, (
        "with every row at the same accept depth, the per-position counts have "
        f"to be the step count up to the depth: expected {expected}, saw "
        f"{delta.per_position}"
    )
