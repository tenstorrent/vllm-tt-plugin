# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What an asynchronously scheduled speculating server must still emit.

`test_async_transitions.py` asks whether the asynchronous launch behaves as one:
the resolved scheduling mode, the overlap, the serialization of verify steps.
This file asks the other question, and asks it of every acceptance depth, every
draft length and every lifecycle event the server can be put through: is the
output still exactly what the model's own rule produces.

The `fixed` target is what makes that a question with one answer. It chooses by
`(token * 31 + position * 7 + 11) % vocab` for an ordinary decode and for a
verify alike, reading only the token and its position, so its output is a
property of the prompt tail and of nothing else: not of the accept depth, not
of how much any step committed, not of whether a step overlapped another, and
not of whether the launch schedules asynchronously at all. Every test here
therefore compares the whole response against `fixed_target_ids`, token for
token, rather than comparing a prefix or comparing two runs against each other.

What the accept depth decides is how much of each step's block the drafter gets
right, which the acceptance counters report and the output does not. So the
counters are read as well, and read the way an asynchronous launch requires:
`AsyncScheduler` gives every scheduled request `[-1] * K` as its lookahead
reservation and vLLM counts that as drafts offered, so
`spec_decode_num_drafts_total` and `num_draft_tokens_total` move on a launch
that never verified a real draft. `num_accepted_tokens_total` and the
per-position counters cannot, so those are what these tests assert on.

Skips rather than weakens: a configuration that is synchronous, or whose target
is not `fixed`, cannot answer these questions and says so.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tt.spec.dummy_arithmetic import fixed_target_ids
from tests.tt.spec.spec_client import (
    acceptance_delta,
    assert_full_length_completion,
)

MAX_TOKENS = 64


@pytest.fixture(autouse=True)
def only_the_asynchronous_fixed_launch(request, spec_config):
    if str(request.config.getoption("--tt-spec-async-scheduling")).lower() != "true":
        pytest.skip(
            "this server was launched with --no-async-scheduling: pass "
            "--tt-spec-async-scheduling=true for the launch that keeps it"
        )
    if spec_config.target != "fixed":
        pytest.skip(
            "these claims need the fixed target, whose output does not depend "
            "on what was drafted: pass --tt-spec-target=fixed"
        )


def _diagnose(prompt, ids):
    """Where a response leaves the rule, and what shape the departure has.

    A bare inequality is useless here: the lists are hundreds of tokens long,
    pytest truncates them, and every failure reads the same. What distinguishes
    the causes is the shape. One token inserted mid-sequence and the rest
    following the rule again means a frame was applied that should have been
    discarded, which is what a replayed prefill after a preemption or a reset
    produces. One token missing means a frame was dropped. Anything else means
    the sequence diverged outright, which is a wrong commit rather than a
    miscounted one.
    """
    rule = fixed_target_ids(prompt, len(ids) + 8)
    at = next((i for i, (got, want) in enumerate(zip(ids, rule)) if got != want), None)
    if at is None:
        return None
    without = ids[:at] + ids[at + 1 :]
    inserted = without == rule[: len(without)]
    with_extra = ids[:at] + [rule[at]] + ids[at:]
    dropped = with_extra == rule[: len(with_extra)]
    shape = (
        "one token inserted"
        if inserted
        else "one token dropped"
        if dropped
        else "diverged outright"
    )
    return (
        f"{shape} at index {at}: emitted {ids[at]}, rule says {rule[at]}; "
        f"previous emitted {ids[at - 1] if at else None}; "
        f"emitted[{at}:{at + 3}]={ids[at : at + 3]} "
        f"rule[{at}:{at + 3}]={rule[at : at + 3]}"
    )


def _assert_is_the_rule(prompt, result, expected_length=None):
    """The whole response, token for token, against the target's own rule."""
    if expected_length is not None:
        # Length, status and the termination reason first, from the shared
        # check: a short response explains a content mismatch on its own.
        assert_full_length_completion(result, expected_length)
    ids = result.token_ids
    assert ids, "the request returned no tokens"
    diagnosis = _diagnose(prompt, ids)
    assert diagnosis is None, (
        f"the response is not the sequence this target's rule produces: {diagnosis}"
    )
    return ids


def _run(spec_server, prompt, **overrides):
    before = spec_server.metrics()
    result = spec_server.complete(prompt, **overrides)
    after = spec_server.metrics()
    return result, before, after


# region Acceptance depth under asynchronous scheduling


def test_the_output_is_the_rule_whatever_the_acceptance(
    spec_server, spec_config, ascending_prompt, record
):
    """The claim that holds across every configuration in this file.

    Whatever the drafter offered and whatever the verify accepted, the emitted
    sequence is the one the rule produces. A deferred commit that wrote a token
    twice, dropped one, or published a prefix longer than it wrote would break
    this and would break nothing else that a response can show.
    """
    prompt = ascending_prompt(64, start=100)
    result, before, after = _run(spec_server, prompt, max_tokens=MAX_TOKENS)
    delta = acceptance_delta(before, after, spec_config.k)
    record(requests=[result.request], acceptance=delta.as_dict())

    ids = _assert_is_the_rule(prompt, result, expected_length=MAX_TOKENS)
    assert len(ids) == MAX_TOKENS


def test_zero_acceptance_commits_one_token_per_step(
    spec_server, spec_config, ascending_prompt, record
):
    """Every draft rejected, and the output unchanged by that.

    At an accept depth of zero the target disagrees with the drafter at the
    first candidate position of every step, so each step commits one token: the
    target's own choice there. That is the hardest case for the deferred accept
    walk, because the published prefix is one token while the block it came
    from is `1+K` wide.
    """
    if spec_config.accept_depth != 0:
        pytest.skip("this configuration accepts drafts: pass --tt-spec-accept-depth=0")
    prompt = ascending_prompt(64, start=200)

    result, before, after = _run(spec_server, prompt, max_tokens=MAX_TOKENS)
    delta = acceptance_delta(before, after, spec_config.k)
    record(requests=[result.request], acceptance=delta.as_dict())

    _assert_is_the_rule(prompt, result, expected_length=MAX_TOKENS)
    assert delta.accepted == 0, (
        f"{delta.accepted} token(s) were accepted at an accept depth of zero"
    )
    assert delta.per_position == [0] * spec_config.k
    assert delta.mean_acceptance_length == 1.0


def test_partial_acceptance_commits_exactly_the_depth(
    spec_server, spec_config, ascending_prompt, record
):
    """Some drafts accepted, the rest rejected, and the output unchanged.

    The positions are what distinguishes this from a mean: a depth of two means
    positions 0 and 1 accepted on every speculative step and positions 2 and up
    on none, which tells "two of five every step" apart from "five of five two
    steps in five".
    """
    depth = spec_config.accept_depth
    if depth is None or depth == 0 or depth >= spec_config.k:
        pytest.skip(
            "this claim needs a depth strictly between zero and K: pass "
            "--tt-spec-accept-depth with one"
        )
    prompt = ascending_prompt(64, start=300)

    result, before, after = _run(spec_server, prompt, max_tokens=MAX_TOKENS)
    delta = acceptance_delta(before, after, spec_config.k)
    record(requests=[result.request], acceptance=delta.as_dict())

    _assert_is_the_rule(prompt, result, expected_length=MAX_TOKENS)
    accepted_positions = delta.per_position[:depth]
    rejected_positions = delta.per_position[depth:]
    assert accepted_positions and min(accepted_positions) > 0
    assert max(accepted_positions) == min(accepted_positions), (
        "the accepted positions disagree, so some step accepted a shorter "
        f"prefix than the depth: {delta.per_position}"
    )
    assert rejected_positions == [0] * len(rejected_positions)
    assert delta.mean_acceptance_length == pytest.approx(depth + 1, abs=0.35)


# endregion Acceptance depth under asynchronous scheduling

# region Lifecycle under asynchronous scheduling


def test_preemption_replays_without_losing_a_token(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """A preempted request resumes and still emits its own whole sequence.

    Preemption frees the blocks an outstanding verify was computed against, and
    the resume replays the request from its saved history. On this path the
    commit of that verify can land after the preemption, which is the case the
    runner has to refuse: a late result must not write into a request whose
    device state is gone, and must not leave a gap in the one that comes back.
    """
    requests = max_batch_size
    if requests < 2:
        pytest.skip("this server serves one row at a time")
    prompts = [ascending_prompt(96, start=1000 * (i + 1)) for i in range(requests)]
    before = spec_server.metrics()

    def send(prompt):
        return spec_server.complete(prompt, max_tokens=MAX_TOKENS * 3)

    with ThreadPoolExecutor(max_workers=requests) as pool:
        results = list(pool.map(send, prompts))
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        requests=[result.request for result in results],
        acceptance=delta.as_dict(),
        preemptions=delta.preemptions,
    )

    if delta.preemptions == 0:
        pytest.skip(
            "no request was preempted, so this run cannot answer the question: "
            "launch with a smaller TT_SPEC_MAX_TOKENS_ALL_USERS"
        )
    for prompt, result in zip(prompts, results):
        _assert_is_the_rule(prompt, result, expected_length=MAX_TOKENS * 3)


def test_a_forced_prefix_reset_keeps_every_response_intact(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """The one case where a published frame must not reach request state.

    A wholesale prefix-cache reset preempts every running request, frees its
    blocks, and marks the frames already in flight as stale: vLLM replays each
    request from its saved history and discards that many published outputs. So
    the runner must publish those frames and not apply them, and a speculative
    frame is one output for a reservation of `1+K` tokens rather than one per
    token. Get that accounting wrong in either direction and a response loses a
    token or repeats one.
    """
    if not spec_server.prefix_cache_reset_is_available():
        pytest.skip(
            "this launch does not expose /reset_prefix_cache: set "
            "VLLM_SERVER_DEV_MODE=1 on the server"
        )
    requests = min(4, max_batch_size)
    if requests < 2:
        pytest.skip("this server serves one row at a time")
    prompts = [ascending_prompt(96, start=2000 * (i + 1)) for i in range(requests)]
    before = spec_server.metrics()
    resets: list[bool] = []

    def send(prompt):
        return spec_server.complete(prompt, max_tokens=MAX_TOKENS * 4)

    with ThreadPoolExecutor(max_workers=requests + 1) as pool:
        futures = [pool.submit(send, prompt) for prompt in prompts]
        for _ in range(3):
            time.sleep(0.01)
            resets.append(spec_server.reset_prefix_cache(reset_running_requests=True))
        results = [future.result() for future in futures]
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        requests=[result.request for result in results],
        acceptance=delta.as_dict(),
        resets=resets,
    )

    if not any(resets):
        pytest.skip(
            "every reset was refused while blocks were held, so no frame was "
            "ever marked stale and this run proves nothing"
        )
    for prompt, result in zip(prompts, results):
        _assert_is_the_rule(prompt, result, expected_length=MAX_TOKENS * 4)


def test_a_cancelled_row_is_reused_without_inheriting_anything(
    spec_server, ascending_prompt, record
):
    """A cancellation mid-decode, and the next request into that row.

    The cancelled request's last verify can complete after its row is gone and
    after another request has taken it, so the runner must drop that result
    rather than write it. The new request's own sequence is what says it did:
    one inherited token would put it off the rule from that point on.
    """
    abandoned_prompt = ascending_prompt(64, start=3000)
    # Long enough that the stream is still decoding when it is abandoned, and
    # inside the context this launch has: the capacity configurations run a
    # 512-token context deliberately, and a request asking past it is refused
    # by the server rather than cancelled by this test.
    room = spec_server.context_length() - len(abandoned_prompt) - 1
    chunks = spec_server.stream_and_abandon(
        abandoned_prompt, after=4, max_tokens=min(MAX_TOKENS * 8, room)
    )

    successor_prompt = ascending_prompt(64, start=4000)
    result, _, _ = _run(spec_server, successor_prompt, max_tokens=MAX_TOKENS)
    record(requests=[result.request], abandoned_after_chunks=chunks)

    assert chunks >= 1, "the abandoned stream never started"
    _assert_is_the_rule(successor_prompt, result, expected_length=MAX_TOKENS)


def test_prefills_arriving_during_decode_do_not_disturb_the_decoders(
    spec_server, ascending_prompt, max_batch_size, record
):
    """Staggered arrivals, so prefill steps land between decode steps.

    A prefill is not a decode: it interrupts the decode chain, changes the
    batch layout, and on this path drains whatever was outstanding. Each
    request still owes its own sequence afterwards, and a decoder that lost or
    repeated a token across one of those interruptions fails here.
    """
    arrivals = min(4, max_batch_size)
    if arrivals < 2:
        pytest.skip("this server serves one row at a time")
    prompts = [ascending_prompt(64, start=5000 * (i + 1)) for i in range(arrivals)]

    def send(index_and_prompt):
        index, prompt = index_and_prompt
        # Spaced so each arrival lands while the earlier requests are decoding
        # rather than all of them prefilling together.
        time.sleep(index * 0.05)
        return spec_server.complete(prompt, max_tokens=MAX_TOKENS * 2)

    with ThreadPoolExecutor(max_workers=arrivals) as pool:
        results = list(pool.map(send, enumerate(prompts)))
    record(requests=[result.request for result in results], arrivals=arrivals)

    for prompt, result in zip(prompts, results):
        _assert_is_the_rule(prompt, result, expected_length=MAX_TOKENS * 2)


# endregion Lifecycle under asynchronous scheduling

# region Boundaries


def test_an_output_limit_inside_an_accepted_block(
    spec_server, spec_config, ascending_prompt, record
):
    """A request that ends partway through a step's committed prefix.

    A speculative step commits up to `1+K` tokens at once, so a `max_tokens`
    that is not a multiple of that width has to cut inside one. The response
    must carry exactly the tokens asked for and stop: the rest of that step's
    prefix is discarded, and a runner that published the whole block would
    overrun the limit while one that dropped the block would fall short.

    Every offset across one block width is covered, because which offsets fall
    inside a block depends on the accept depth and the prefill's own token.
    """
    width = 1 + (
        spec_config.k if spec_config.accept_depth is None else spec_config.accept_depth
    )
    prompt = ascending_prompt(64, start=6000)
    lengths = []

    for offset in range(width + 1):
        limit = MAX_TOKENS + offset
        result, _, _ = _run(spec_server, prompt, max_tokens=limit)
        _assert_is_the_rule(prompt, result, expected_length=limit)
        lengths.append(len(result.token_ids))
    record(block_width=width, lengths=lengths)

    assert lengths == [MAX_TOKENS + offset for offset in range(width + 1)]


def test_distinct_histories_stay_distinct(
    spec_server, ascending_prompt, max_batch_size, record
):
    """Rows sharing steps, each answering its own prompt.

    Every row of a speculative step is verified in the same forward and
    committed in the same walk, so a row that read another row's count or
    another row's block would emit a sequence that belongs to its neighbour.
    Different prompts make that visible: each response has to be its own rule
    from its own tail, and no two may coincide.
    """
    rows = min(3, max_batch_size)
    if rows < 2:
        pytest.skip("this server serves one row at a time")
    # Different lengths, not different contents. This target chooses from the
    # token and its position, and the first token of every response is the
    # prefill's own 0 whatever the prompt was, so two prompts of the same
    # length produce the same sequence however different their tokens are.
    # Length is what moves the starting position and so the whole sequence.
    prompts = [ascending_prompt(48 + 16 * i, start=7000 * (i + 1)) for i in range(rows)]

    def send(prompt):
        return spec_server.complete(prompt, max_tokens=MAX_TOKENS)

    with ThreadPoolExecutor(max_workers=rows) as pool:
        results = list(pool.map(send, prompts))
    record(requests=[result.request for result in results])

    emitted = []
    for prompt, result in zip(prompts, results):
        emitted.append(_assert_is_the_rule(prompt, result, expected_length=MAX_TOKENS))
    assert len({tuple(ids) for ids in emitted}) == rows, (
        "two rows emitted the same sequence, so their histories were not kept apart"
    )


# endregion Boundaries
