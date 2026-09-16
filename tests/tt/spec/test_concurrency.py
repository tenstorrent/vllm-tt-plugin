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

And where a claim about the same-step batch size is made, they **derive it from
the scheduler's own instrumentation**: the tokens-per-step histogram bounds the
number of rows that shared a step from below, because one row commits at most
``1+K`` tokens in a step. A claim that four rows ran together is a claim about
that histogram, not about how many requests were posted.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tt.spec.spec_client import acceptance_delta, rows_in_the_widest_step


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
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """The batch-size claim, taken from the scheduler's histogram.

    One row commits at most ``1+K`` tokens in a step. A step that committed
    more than ``(rows - 1) * (1+K)`` tokens therefore had at least ``rows``
    rows in it. That is the strongest statement this instrumentation supports,
    and it is a bound rather than a count.
    """
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
    bound = rows_in_the_widest_step(before, after, spec_config.committed_per_step)
    record(
        requests=[result.request for result in results],
        acceptance=delta.as_dict(),
        rows_posted=rows,
        rows_in_the_widest_step_at_least=bound,
        buckets_before=before.iteration_token_buckets(),
        buckets_after=after.iteration_token_buckets(),
    )

    assert all(result.status == 200 for result in results)
    assert bound >= 2, (
        f"the tokens-per-step histogram shows no step wide enough for two rows "
        f"of at most {spec_config.committed_per_step} tokens, so these "
        f"{rows} requests were served one after another"
    )
    # And the step count is far below what serving them in turn would need.
    committed = sum(result.completion_tokens for result in results)
    assert delta.steps < committed, (
        "the engine took at least one step per committed token, which is what "
        "serving the requests in turn without speculation looks like"
    )


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
