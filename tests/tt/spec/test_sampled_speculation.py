# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Sampled requests speculate through the real stack, and stay lossless.

A launch whose model declares the ``logits`` accept mode rejection-samples
every verify row on the host, under the row's own temperature, top-k, top-p,
penalties and seed. The host suite holds that walk to independent references.
This file asks the same of the server: the engine core's draft handshake, the
scheduler's lookahead, the persistent batch's rows and both decode tails are
all on the path a sampled token takes here.

It needs the ``fixed`` target, whose distribution after a token at a position
is the same for a prefill, an ordinary decode and every verify column, and
does not depend on what was drafted: the rule's choice at 0.4 and three tokens
shared by every context at 0.3, 0.2 and 0.1. So each committed token can be
checked twice. Deterministically, it must be in its context's support, which
a commit of a rejected or padded draft is not. Statistically, the counts of
which support member was committed, pooled over every token of every request,
must match the expected counts computed from ``fixed_target_distribution``
with each request's own history, independently of the plugin.

The acceptance counters are read too, because a server that never verified a
draft would pass every distribution check. Under asynchronous scheduling vLLM
counts the lookahead reservation as drafts offered, so rejection is read off
the per-position acceptance counts instead: a launch that accepted every draft
accepts as often at the last position as at the first.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.tt.spec.dummy_arithmetic import (
    FIXED_SHARED_ALTERNATIVES,
    fixed_target_choice,
    fixed_target_distribution,
    fixed_target_greedy_ids,
    fixed_target_ids,
    fixed_target_support,
)
from tests.tt.spec.spec_client import (
    acceptance_delta,
    assert_full_length_completion,
    assert_rejections_happened,
)

CONCURRENCY = 8
# Upper critical values of chi-square for a tail of 1e-4, by degrees of
# freedom. A seeded check draws the same tokens on every run of a correct
# synchronous server, so a seeded failure there repeats and needs
# investigating. Under asynchronous scheduling a prefill-only step drops the
# model proposal of every request it hides, and arrival timing decides how
# often, so seeded counts vary between runs as the unseeded check's do.
CHI_SQUARE_CRITICAL = {1: 15.14, 2: 18.42, 3: 21.11}
_SUBMISSIONS = re.compile(
    r"TT submissions: \d+ ordinary decode, (\d+) verify, \d+ overlapped an "
    r"outstanding step, (\d+) verify in logits mode"
)
# Both are runner log lines; their emitters name this file.
_UNSPECULABLE = "argmax_ids mode included a sampled or penalized request"


@pytest.fixture(autouse=True)
def only_a_logits_launch_of_the_fixed_target(spec_config):
    if "logits" not in spec_config.accept_modes:
        pytest.skip(
            "these tests need a launch declaring the logits accept mode: set "
            "TT_SPEC_ACCEPT_MODES and pass --tt-spec-accept-modes"
        )
    if spec_config.target != "fixed":
        pytest.skip(
            "these tests need the fixed target, whose distribution does not "
            "depend on what was drafted: pass --tt-spec-target=fixed"
        )


@pytest.fixture
def asynchronous(request) -> bool:
    return str(request.config.getoption("--tt-spec-async-scheduling")).lower() == "true"


@pytest.fixture
def log_path(request) -> Path | None:
    path = request.config.getoption("--tt-spec-server-log")
    return Path(path) if path and Path(path).exists() else None


def _prompt(spec_config, ascending_prompt, index: int) -> list[int]:
    """A prompt per request, which the n-gram drafter can also find drafts in.

    For the model's own drafter any prompt serves. The n-gram drafter drafts
    only where the request's text repeats, and the shared tokens are what a
    sampled request emits again and again, so its prompt carries every ordered
    pair of them: once the output ends in two of them, the prompt holds a match.
    """
    if spec_config.drafter == "ngram":
        a, b, c = FIXED_SHARED_ALTERNATIVES
        cycle = [a, a, b, b, c, c, a, c, b, a]
        return ascending_prompt(24, start=1000 + index * 37) + cycle * 4
    return ascending_prompt(64, start=101 + index * 53)


def _concurrency(spec_config) -> int:
    """One request at a time under ``solo``, where a batch drafts nothing."""
    return 1 if spec_config.draft_policy == "solo" else CONCURRENCY


def _complete_all(spec_server, prompts, bodies, concurrency=CONCURRENCY):
    with ThreadPoolExecutor(concurrency) as pool:
        futures = [
            pool.submit(spec_server.complete, prompt, **body)
            for prompt, body in zip(prompts, bodies)
        ]
        return [future.result() for future in futures]


def _support_counts(prompt, ids, controls):
    """Observed and expected counts of each support member, for one response.

    Member 0 is the rule's choice and members 1..3 the shared tokens. A shared
    token equal to the rule's choice is counted as member 0, which is where
    ``fixed_target_distribution`` puts its merged share.
    """
    observed = [0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 0.0]
    token, position = prompt[-1], len(prompt) - 1
    history: list[int] = []
    for index, emitted in enumerate(ids):
        support = fixed_target_support(token, position)
        distribution = fixed_target_distribution(
            token, position, history=history, prompt=prompt, **controls
        )
        assert emitted in distribution, (
            f"token {index} ({emitted}) has probability zero after {token} at "
            f"position {position} under {controls}; the target allows "
            f"{sorted(distribution)}. A committed draft the target rules out "
            "means a wrong commit, not sampling noise"
        )
        observed[support.index(emitted)] += 1
        for member, candidate in enumerate(support):
            if support.index(candidate) == member:
                expected[member] += distribution.get(candidate, 0.0)
        history.append(emitted)
        token, position = emitted, position + 1
    return observed, expected


def _chi_square(observed, expected):
    """Pearson's statistic over the members expected at least five times."""
    kept = [(o, e) for o, e in zip(observed, expected) if e >= 5]
    pooled = [(o, e) for o, e in zip(observed, expected) if 0 < e < 5]
    absent = [o for o, e in zip(observed, expected) if e == 0]
    assert not any(absent), f"a member with probability zero was committed: {observed}"
    if pooled:
        kept.append((sum(o for o, _ in pooled), sum(e for _, e in pooled)))
    statistic = sum((o - e) ** 2 / e for o, e in kept)
    return statistic, len(kept) - 1


def _assert_no_greedy_fallback(log_path):
    if log_path is None:
        return
    text = log_path.read_text(errors="replace")
    # A log that never reported a submission is not this server's, or the
    # runner's log lines changed, and either way absence would prove nothing.
    assert _SUBMISSIONS.search(text), (
        f"{log_path} carries no TT submissions line, so the absence of the "
        "argmax fallback warning in it says nothing"
    )
    assert _UNSPECULABLE not in text, (
        "a sampled request was committed by an argmax_ids verify, which a "
        "launch serving logits must never do"
    )


def _submission_lines(log_path, start: int):
    lines = log_path.read_text(errors="replace").splitlines()[start:]
    return [
        (int(found.group(1)), int(found.group(2)))
        for line in lines
        if (found := _SUBMISSIONS.search(line))
    ]


# region The committed distribution


@pytest.mark.parametrize(
    "controls",
    [
        {"temperature": 1.0},
        {"temperature": 0.7, "top_k": 2},
        {"temperature": 1.2, "top_p": 0.75},
        {
            "temperature": 1.0,
            "presence_penalty": 0.4,
            "frequency_penalty": 0.3,
            "repetition_penalty": 1.3,
        },
    ],
    ids=["temperature", "top-k", "top-p", "penalties"],
)
def test_sampled_requests_speculate_and_commit_the_target_distribution(
    spec_server,
    spec_config,
    ascending_prompt,
    asynchronous,
    log_path,
    record,
    controls,
):
    """Many seeded sampled requests, pooled, against the expected counts.

    Seeded so a failing synchronous run repeats. The penalties case has a different
    distribution at every step, read from each request's own history, which is
    why the expected counts are summed per token rather than taken from a
    fixed table.
    """
    # 4096 committed tokens: enough to see an argmax commit on about one token
    # in fifteen at this threshold. Exactness is the host suite's to prove.
    requests = 128
    max_tokens = 32
    prompts = [_prompt(spec_config, ascending_prompt, i) for i in range(requests)]
    bodies = [
        {"max_tokens": max_tokens, "seed": 5000 + i, **controls}
        for i in range(requests)
    ]

    before = spec_server.metrics()
    results = _complete_all(spec_server, prompts, bodies, _concurrency(spec_config))
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)

    observed = [0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 0.0]
    for prompt, result in zip(prompts, results):
        assert_full_length_completion(result, max_tokens)
        o, e = _support_counts(prompt, result.token_ids, controls)
        observed = [a + b for a, b in zip(observed, o)]
        expected = [a + b for a, b in zip(expected, e)]
    statistic, df = _chi_square(observed, expected)
    record(
        controls=controls,
        observed=observed,
        expected=[round(value, 2) for value in expected],
        chi_square=statistic,
        degrees_of_freedom=df,
        acceptance=delta.as_dict(),
    )

    assert delta.drafts > 0, "nothing was drafted, so nothing was speculated"
    assert_rejections_happened(delta, asynchronous)
    assert statistic < CHI_SQUARE_CRITICAL[df], (
        f"the committed support members {observed} do not match the expected "
        f"{[round(value, 1) for value in expected]} (chi-square {statistic:.1f} "
        f"over {df} degrees of freedom)"
    )
    _assert_no_greedy_fallback(log_path)


def test_unseeded_sampled_requests_commit_the_target_distribution(
    spec_server, spec_config, ascending_prompt, record
):
    """The global stream rather than per-request generators, same claim."""
    requests, max_tokens = 64, 24
    prompts = [_prompt(spec_config, ascending_prompt, 200 + i) for i in range(requests)]
    results = _complete_all(
        spec_server,
        prompts,
        [{"max_tokens": max_tokens, "temperature": 1.0} for _ in range(requests)],
        _concurrency(spec_config),
    )
    observed = [0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 0.0]
    for prompt, result in zip(prompts, results):
        assert_full_length_completion(result, max_tokens)
        o, e = _support_counts(prompt, result.token_ids, {"temperature": 1.0})
        observed = [a + b for a, b in zip(observed, o)]
        expected = [a + b for a, b in zip(expected, e)]
    statistic, df = _chi_square(observed, expected)
    record(observed=observed, expected=expected, chi_square=statistic)

    assert statistic < CHI_SQUARE_CRITICAL[df], (observed, expected)


# endregion The committed distribution

# region Exact claims


# Output indices where the penalized greedy choice is made to differ from the
# rule. Index 0 is drawn at prefill and index 1 by the first verify, which has
# no drafts yet, so these all fall inside verifies that carry drafts.
_FLIPS = (4, 9, 15)


def _penalized_prompt(base: list[int], controls: dict) -> list[int]:
    """A prompt that makes the penalties change the greedy choice at ``_FLIPS``.

    For each index the rule's choice there is written into the prompt ahead of
    the tail, so the repetition penalty lowers it below the best shared token.
    The tail stays last, so the positions and every earlier choice are kept.
    """
    prompt = list(base)
    for slot, index in enumerate(_FLIPS, start=1):
        # The token emitted just before ``index`` and its position.
        before = fixed_target_greedy_ids(prompt, index, **controls)[-1]
        prompt[slot] = fixed_target_choice(before, len(prompt) - 1 + index)
    return prompt


def _flipped(prompt: list[int], ids: list[int]) -> list[int]:
    """The indices where ``ids`` leaves the rule's choice."""
    flips = []
    token, position = prompt[-1], len(prompt) - 1
    for index, emitted in enumerate(ids):
        if emitted != fixed_target_choice(token, position):
            flips.append(index)
        token, position = emitted, position + 1
    return flips


def test_a_greedy_penalized_request_speculates_to_its_exact_output(
    spec_server, spec_config, asynchronous, ascending_prompt, record
):
    """Temperature 0 with penalties is deterministic, so the claim is equality.

    The drafter proposes the unpenalized rule, so at each flipped index the
    draft is rejected and the correction has to be the penalized argmax, and
    everywhere else the accepted draft has to enter the next column's penalty
    history.
    """
    if spec_config.drafter != "model":
        pytest.skip("the n-gram drafter finds no repeat in the rule's greedy chain")
    controls = {
        "presence_penalty": 0.5,
        "frequency_penalty": 0.2,
        "repetition_penalty": 3.0,
    }
    prompt = _penalized_prompt(_prompt(spec_config, ascending_prompt, 7), controls)
    max_tokens = 96
    expected = fixed_target_greedy_ids(prompt, max_tokens, **controls)
    assert _flipped(prompt, expected) == list(_FLIPS), (
        "the penalties do not change this prompt's greedy output where the "
        "test needs them to, so it would not exercise a penalized rejection"
    )

    before = spec_server.metrics()
    result = spec_server.complete(
        prompt, max_tokens=max_tokens, temperature=0, **controls
    )
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(request=result.request, response=result.body, acceptance=delta.as_dict())

    assert_full_length_completion(result, max_tokens)
    assert result.token_ids == expected
    assert_rejections_happened(delta, asynchronous)


def test_a_seeded_sampled_request_repeats_alone_and_in_company(
    spec_server, spec_config, asynchronous, ascending_prompt, record
):
    """Same seed, same output, whoever else shares its verifies.

    A row's generator advances by its own draft counts only, and the model
    drafter under ``always`` offers every row its full draft length, so on a
    synchronous launch other requests in the batch cannot change what a seeded
    request reads. Under asynchronous scheduling a later arrival's prefill-only
    step hides the request and drops that step's proposal, so its draft counts
    depend on its company.
    """
    prompt = _prompt(spec_config, ascending_prompt, 11)
    body = {"max_tokens": 48, "temperature": 0.9, "top_p": 0.9, "seed": 1234}

    alone = spec_server.complete(prompt, **body)
    again = spec_server.complete(prompt, **body)
    assert_full_length_completion(alone, 48)
    assert again.token_ids == alone.token_ids

    if (
        spec_config.draft_policy != "always"
        or spec_config.drafter != "model"
        or asynchronous
    ):
        record(alone=alone.token_ids)
        pytest.skip(
            "under this drafter or asynchronous scheduling a request's draft "
            "counts depend on its company, so only the solo repeat is claimed"
        )
    others = [
        _prompt(spec_config, ascending_prompt, 300 + i) for i in range(CONCURRENCY - 1)
    ]
    results = _complete_all(
        spec_server,
        [prompt, *others],
        [body] + [{"max_tokens": 48, "temperature": 1.0} for _ in others],
    )
    record(alone=alone.token_ids, in_company=results[0].token_ids)

    assert results[0].token_ids == alone.token_ids


def test_greedy_and_sampled_requests_keep_their_own_semantics_in_one_batch(
    spec_server, spec_config, ascending_prompt, log_path, record
):
    """A greedy row in a sampled verify still emits exactly the rule."""
    greedy_prompts = [_prompt(spec_config, ascending_prompt, 400 + i) for i in range(4)]
    sampled_prompts = [
        _prompt(spec_config, ascending_prompt, 500 + i) for i in range(4)
    ]
    max_tokens = 64
    results = _complete_all(
        spec_server,
        [*greedy_prompts, *sampled_prompts],
        [{"max_tokens": max_tokens, "temperature": 0} for _ in greedy_prompts]
        + [{"max_tokens": max_tokens, "temperature": 1.0, "seed": i} for i in range(4)],
    )
    record(responses=[r.token_ids for r in results])

    for prompt, result in zip(greedy_prompts, results[:4]):
        assert_full_length_completion(result, max_tokens)
        assert result.token_ids == fixed_target_ids(prompt, max_tokens)
    for prompt, result in zip(sampled_prompts, results[4:]):
        assert_full_length_completion(result, max_tokens)
        _support_counts(prompt, result.token_ids, {"temperature": 1.0})
    _assert_no_greedy_fallback(log_path)


def test_a_stop_token_ends_a_sampled_request_where_it_was_drawn(
    spec_server, spec_config, ascending_prompt, record
):
    """The engine truncates at the first stop token, inside a committed block too.

    17 is a shared token at probability 0.3, so a sampled request draws it
    within a few tokens, and on most seeds after the first verify, where a
    step commits a block and the stop can sit anywhere inside it.
    """
    stop = FIXED_SHARED_ALTERNATIVES[0]
    prompts = [_prompt(spec_config, ascending_prompt, 700 + i) for i in range(16)]
    results = _complete_all(
        spec_server,
        prompts,
        [
            {
                "max_tokens": 200,
                "temperature": 1.0,
                "seed": 90 + i,
                "stop_token_ids": [stop],
            }
            for i in range(len(prompts))
        ],
        _concurrency(spec_config),
    )
    record(responses=[r.token_ids for r in results])

    late = 0
    for prompt, result in zip(prompts, results):
        assert result.status == 200
        assert result.finish_reason == "stop"
        ids = result.token_ids
        assert ids and ids[-1] == stop and stop not in ids[:-1], ids
        _support_counts(prompt, ids, {"temperature": 1.0})
        late += len(ids) > 2
    assert late > 0, "every stop came before the first verify could commit"


def test_a_greedy_request_on_a_logits_launch_still_follows_the_rule(
    spec_server, spec_config, ascending_prompt, record
):
    """Through ``argmax_ids`` where both modes are served, through ``logits``
    on a launch that serves only that."""
    if spec_config.drafter != "model":
        pytest.skip("the n-gram drafter finds no repeat in the rule's greedy chain")
    prompt = _prompt(spec_config, ascending_prompt, 13)
    before = spec_server.metrics()
    result = spec_server.complete(prompt, max_tokens=128, temperature=0)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(response=result.token_ids, acceptance=delta.as_dict())

    assert_full_length_completion(result, 128)
    assert result.token_ids == fixed_target_ids(prompt, 128)
    assert delta.accepted > 0


def test_sampled_requests_survive_preemption(
    spec_server, spec_config, ascending_prompt, record
):
    """Replay after preemption keeps every token in its context's support.

    A preempted request resumes with a prefill over its whole history, which
    then samples the next token under the same penalties and seed. Only a
    launch whose KV budget is smaller than the concurrent work preempts; on
    any other this skips rather than passes.
    """
    prompts = [ascending_prompt(96, start=900 + 211 * i) for i in range(8)]
    max_tokens = min(192, spec_server.context_length() - 96 - 8)
    before = spec_server.metrics()
    results = _complete_all(
        spec_server,
        prompts,
        [
            {
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "presence_penalty": 0.3,
                "seed": 70 + i,
            }
            for i in range(len(prompts))
        ],
    )
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(acceptance=delta.as_dict(), responses=[r.token_ids for r in results])

    if delta.preemptions == 0:
        pytest.skip("nothing was preempted, so replay was not exercised")
    for prompt, result in zip(prompts, results):
        assert_full_length_completion(result, max_tokens)
        _support_counts(prompt, result.token_ids, {"temperature": 1.0})


# endregion Exact claims

# region Which mode the verifies used


def test_verifies_ask_for_logits_only_when_a_row_needs_them(
    spec_server, spec_config, ascending_prompt, log_path, record
):
    """Read from the runner's own submission counters in the server log.

    The counters report every 128 submissions, so each phase is long enough
    to span two reports, and only reports written inside the phase are
    compared. On a launch serving both modes a greedy phase adds verifies and
    no logits verifies; a sampled phase adds logits verifies either way.
    """
    if log_path is None:
        pytest.skip("pass --tt-spec-server-log: the submission counters live there")
    if spec_config.drafter != "model":
        pytest.skip("the phases are sized for the model drafter's acceptance")
    if spec_server.context_length() < 1024:
        pytest.skip("the phases need a context of at least 1024 tokens")

    def phase(temperature, max_tokens, tokens_per_verify):
        # Two reports inside the phase can need 256 submissions, and after its
        # prefill token a lone request commits up to tokens_per_verify tokens
        # per submission.
        requests = max(3, math.ceil(256 * tokens_per_verify / (max_tokens - 1)))
        start = len(log_path.read_text(errors="replace").splitlines())
        for index in range(requests):
            prompt = _prompt(spec_config, ascending_prompt, 600 + index)
            spec_server.complete(
                prompt, max_tokens=max_tokens, temperature=temperature, seed=index
            )
        reports = _submission_lines(log_path, start)
        assert len(reports) >= 2, (
            f"only {len(reports)} submission report(s) were written during the "
            "phase, so its verifies cannot be counted"
        )
        (verify_0, logits_0), (verify_1, logits_1) = reports[0], reports[-1]
        return verify_1 - verify_0, logits_1 - logits_0

    # A greedy verify commits up to 1+K tokens, a sampled one under two on
    # average.
    greedy_verifies, greedy_logits = phase(0, 900, spec_config.k + 1)
    sampled_verifies, sampled_logits = phase(1.0, 300, 2)
    record(
        greedy=(greedy_verifies, greedy_logits),
        sampled=(sampled_verifies, sampled_logits),
    )

    assert greedy_verifies > 0 and sampled_verifies > 0
    assert sampled_logits == sampled_verifies
    if "argmax_ids" in spec_config.accept_modes:
        assert greedy_logits == 0, (
            "a greedy-only batch asked for logits although the model serves "
            "argmax_ids, which reads back a vocabulary's width more per row"
        )
    else:
        assert greedy_logits == greedy_verifies


# endregion Which mode the verifies used
