# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Grammars, token filters and logprobs speculate through the real stack.

A launch whose model declares the ``logits`` accept mode applies structured
output, ``allowed_token_ids``, ``bad_words`` and ``min_tokens`` at every
candidate column of a verify, and returns logprobs for every committed token.
The host suite holds the walk to vLLM's own sampler and to an independent
reference. This file asks the same of the server, where the scheduler's grammar
validation and bitmask layout, the engine's deferred sampling for structured
requests, the request's stop handling and the OpenAI logprobs encoding are all
on the path.

It needs the ``fixed`` target, as ``test_sampled_speculation.py`` does: after a
token at a position the target puts 0.4 on the rule's choice and 0.3, 0.2 and
0.1 on the shared tokens 17, 4099 and 65537, for every kind of step. Each
committed token is checked against what its controls allow, deterministically,
and the pooled counts against expected counts from
``fixed_target_distribution`` with the same tokens excluded.

The Llama 3.1 tokenizer the suite serves with decodes 17 as ``"2"``, 4099 as
``"pression"`` and 65537 as ``" Guang"``, and vLLM tokenizes the bad word
``"2pression"`` to ``[[17, 4099]]``. That makes a two-token bad word and two
regular grammars out of shared tokens alone, so every control here leaves the
target something to sample.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tests.tt.spec.dummy_arithmetic import (
    FIXED_SHARED_ALTERNATIVES,
    fixed_target_distribution,
    fixed_target_support,
)
from tests.tt.spec.spec_client import (
    acceptance_delta,
    assert_full_length_completion,
    assert_rejections_happened,
)
from tests.tt.spec.test_sampled_speculation import (
    CHI_SQUARE_CRITICAL,
    CONCURRENCY,
    _assert_no_greedy_fallback,
    _chi_square,
    _complete_all,
    _concurrency,
    _prompt,
)

TWO, PRESSION, GUANG = FIXED_SHARED_ALTERNATIVES
ALLOWED = [TWO, PRESSION, GUANG]
BAD_WORD = "2pression"
BAD_WORD_IDS = [TWO, PRESSION]
# Two grammars over different pairs of shared tokens, so a bitmask row that
# reached the wrong request lets through a token the other grammar forbids.
GRAMMARS = {
    "2-pression": (r"(2|pression)+", {TWO, PRESSION}),
    "guang-2": (r"( Guang|2)+", {GUANG, TWO}),
}


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


class _Excluded:
    """What one request's token filters rule out at each output position.

    Called with the output so far. The history a bad word is matched against is
    the output only, which is what vLLM's sampler hands its bad-words op.
    """

    def __init__(self, prompt, *, allowed=None, bad_words=(), min_tokens=None):
        self.prompt = prompt
        self.allowed = None if allowed is None else set(allowed)
        self.bad_words = [list(word) for word in bad_words]
        self.min_tokens = min_tokens

    def __call__(self, history: list[int]) -> set[int]:
        token = history[-1] if history else self.prompt[-1]
        position = len(self.prompt) - 1 + len(history)
        excluded: set[int] = set()
        if self.allowed is not None:
            excluded |= set(fixed_target_support(token, position)) - self.allowed
        for word in self.bad_words:
            prefix = word[:-1]
            if not prefix or history[len(history) - len(prefix) :] == prefix:
                excluded.add(word[-1])
        if self.min_tokens is not None:
            count, stops = self.min_tokens
            if len(history) < count:
                excluded |= set(stops)
        return excluded


def _counts(prompt, ids, excluded_at, controls):
    """Observed and expected support counts for one response.

    The bookkeeping of ``test_sampled_speculation._support_counts``, with what
    ``excluded_at`` rules out dropped from the reference before its
    temperature. A committed token the reference gives probability 0 fails
    here, not in the statistic.
    """
    observed = [0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 0.0]
    token, position = prompt[-1], len(prompt) - 1
    history: list[int] = []
    for index, emitted in enumerate(ids):
        support = fixed_target_support(token, position)
        excluded = excluded_at(history)
        distribution = fixed_target_distribution(
            token,
            position,
            history=history,
            prompt=prompt,
            excluded=excluded,
            **controls,
        )
        assert emitted in distribution, (
            f"token {index} ({emitted}) has probability zero after {token} at "
            f"position {position} with {sorted(excluded)} excluded; the target "
            f"allows {sorted(distribution)}"
        )
        observed[support.index(emitted)] += 1
        for member, candidate in enumerate(support):
            if support.index(candidate) == member:
                expected[member] += distribution.get(candidate, 0.0)
        history.append(emitted)
        token, position = emitted, position + 1
    return observed, expected


def _greedy(prompt, count, excluded_at):
    """The exact greedy output under ``excluded_at``."""
    ids: list[int] = []
    token, position = prompt[-1], len(prompt) - 1
    while len(ids) < count:
        distribution = fixed_target_distribution(
            token, position, temperature=0, excluded=excluded_at(ids)
        )
        token = next(iter(distribution))
        position += 1
        ids.append(token)
    return ids


def _pool(prompts, results, excluded_for, controls):
    observed = [0, 0, 0, 0]
    expected = [0.0, 0.0, 0.0, 0.0]
    for prompt, result in zip(prompts, results):
        o, e = _counts(prompt, result.token_ids, excluded_for(prompt), controls)
        observed = [a + b for a, b in zip(observed, o)]
        expected = [a + b for a, b in zip(expected, e)]
    return observed, expected


def _assert_distribution(observed, expected):
    statistic, df = _chi_square(observed, expected)
    assert statistic < CHI_SQUARE_CRITICAL[df], (
        f"the committed support members {observed} do not match the expected "
        f"{[round(value, 1) for value in expected]} (chi-square {statistic:.1f} "
        f"over {df} degrees of freedom)"
    )
    return statistic


def _bigrams(ids):
    return list(zip(ids, ids[1:]))


# region Token filters


@pytest.mark.parametrize("temperature", [1.0, 0.0], ids=["sampled", "greedy"])
def test_allowed_token_ids_speculate_and_commit_only_allowed_tokens(
    spec_server,
    spec_config,
    ascending_prompt,
    log_path,
    record,
    temperature,
):
    """Only the shared tokens are allowed, so the rule's choice never commits.

    The model drafter proposes the rule's choice, which the allowlist rules
    out at every column, so every such draft has probability 0 and is
    rejected; a draft the walk accepted shows up as a forbidden token.
    """
    requests, max_tokens = (64, 32) if temperature else (8, 48)
    prompts = [
        _prompt(spec_config, ascending_prompt, 1100 + i) for i in range(requests)
    ]
    bodies = [
        {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": 6100 + i,
            "allowed_token_ids": ALLOWED,
        }
        for i in range(requests)
    ]
    before = spec_server.metrics()
    results = _complete_all(spec_server, prompts, bodies, _concurrency(spec_config))
    delta = acceptance_delta(before, spec_server.metrics(), spec_config.k)

    for prompt, result in zip(prompts, results):
        assert_full_length_completion(result, max_tokens)
        assert set(result.token_ids) <= set(ALLOWED), result.token_ids
        if not temperature:
            excluded_at = _Excluded(prompt, allowed=ALLOWED)
            assert result.token_ids == _greedy(prompt, max_tokens, excluded_at)
    record(acceptance=delta.as_dict(), responses=[r.token_ids for r in results])

    assert delta.drafts > 0, "nothing was drafted, so nothing was speculated"
    if temperature:
        observed, expected = _pool(
            prompts,
            results,
            lambda prompt: _Excluded(prompt, allowed=ALLOWED),
            {"temperature": temperature},
        )
        statistic = _assert_distribution(observed, expected)
        record(observed=observed, expected=expected, chi_square=statistic)
    _assert_no_greedy_fallback(log_path)


def test_a_two_token_bad_word_is_never_committed(
    spec_server, spec_config, ascending_prompt, asynchronous, log_path, record
):
    """``"2pression"`` bans 4099 right after 17, in drafted columns too.

    The rule's drafts are accepted at probability 0.4, so verifies commit
    multi-token blocks and a 17 is often a draft or a correction inside one;
    the 4099 after it is then banned by a column whose history ends with a
    token that was not yet committed when the step began.
    """
    requests, max_tokens = 64, 32
    prompts = [
        _prompt(spec_config, ascending_prompt, 1300 + i) for i in range(requests)
    ]
    bodies = [
        {
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "seed": 6300 + i,
            "bad_words": [BAD_WORD],
        }
        for i in range(requests)
    ]
    before = spec_server.metrics()
    results = _complete_all(spec_server, prompts, bodies, _concurrency(spec_config))
    delta = acceptance_delta(before, spec_server.metrics(), spec_config.k)

    for result in results:
        assert_full_length_completion(result, max_tokens)
        assert tuple(BAD_WORD_IDS) not in _bigrams(result.token_ids), result.token_ids
    observed, expected = _pool(
        prompts,
        results,
        lambda prompt: _Excluded(prompt, bad_words=[BAD_WORD_IDS]),
        {"temperature": 1.0},
    )
    statistic = _assert_distribution(observed, expected)
    record(
        observed=observed,
        expected=expected,
        chi_square=statistic,
        acceptance=delta.as_dict(),
    )

    assert sum(r.token_ids.count(TWO) for r in results) > 100, (
        "too few 17s were committed for the ban after them to be exercised"
    )
    assert_rejections_happened(delta, asynchronous)
    _assert_no_greedy_fallback(log_path)


def test_min_tokens_holds_a_stop_token_back_until_it_is_reached(
    spec_server, spec_config, ascending_prompt, asynchronous, log_path, record
):
    """17 is a stop token at probability 0.3, masked for the first ten tokens.

    Without ``min_tokens`` most requests would stop within three tokens. With
    it, no 17 may appear before position 10, and the request stops at the
    first 17 after that. Verifies commit blocks that straddle position 10, so
    the column that may first commit 17 is often a drafted one.
    """
    requests, minimum, max_tokens = 48, 10, 40
    prompts = [
        _prompt(spec_config, ascending_prompt, 1500 + i) for i in range(requests)
    ]
    bodies = [
        {
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "seed": 6500 + i,
            "min_tokens": minimum,
            "stop_token_ids": [TWO],
        }
        for i in range(requests)
    ]
    before = spec_server.metrics()
    results = _complete_all(spec_server, prompts, bodies, _concurrency(spec_config))
    delta = acceptance_delta(before, spec_server.metrics(), spec_config.k)

    stopped = 0
    for result in results:
        assert result.status == 200, result.body
        ids = result.token_ids
        assert len(ids) >= minimum, ids
        assert TWO not in ids[:minimum], ids
        if result.finish_reason == "stop":
            stopped += 1
            assert ids[-1] == TWO and TWO not in ids[:-1], ids
        else:
            assert len(ids) == max_tokens and TWO not in ids, ids
    observed, expected = _pool(
        prompts,
        results,
        lambda prompt: _Excluded(prompt, min_tokens=(minimum, {TWO})),
        {"temperature": 1.0},
    )
    statistic = _assert_distribution(observed, expected)
    record(
        observed=observed,
        expected=expected,
        chi_square=statistic,
        stopped=stopped,
        acceptance=delta.as_dict(),
    )

    assert stopped > requests // 2, "too few requests stopped after min_tokens"
    assert_rejections_happened(delta, asynchronous)
    _assert_no_greedy_fallback(log_path)


# endregion Token filters

# region Logprobs


def _logprob_rows(result):
    """``(token id, its logprob, {top id: logprob})`` per committed token."""
    logprobs = result.body["choices"][0]["logprobs"]
    rows = []
    for token, value, top in zip(
        logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"]
    ):
        assert token.startswith("token_id:"), token
        rows.append(
            (
                int(token.removeprefix("token_id:")),
                value,
                {int(key.removeprefix("token_id:")): v for key, v in top.items()},
            )
        )
    return rows


@pytest.mark.parametrize(
    "controls",
    [
        {"temperature": 1.0},
        {"temperature": 0.0},
        {"temperature": 0.8, "presence_penalty": 0.6, "repetition_penalty": 1.4},
    ],
    ids=["sampled", "greedy", "penalized"],
)
def test_logprobs_report_the_target_s_raw_distribution(
    spec_server, spec_config, ascending_prompt, log_path, record, controls
):
    """Raw logprobs: the target's own 0.4, 0.3, 0.2 and 0.1, whatever was sampled.

    Every committed token, accepted draft or correction or bonus alike,
    reports the log of its raw probability and the support as its top
    entries. The penalties and temperature change what is committed, not what
    is reported for it, because vLLM's default ``raw_logprobs`` is taken
    before them.
    """
    requests, max_tokens = 16, 32
    prompts = [
        _prompt(spec_config, ascending_prompt, 1700 + i) for i in range(requests)
    ]
    bodies = [
        {
            "max_tokens": max_tokens,
            "seed": 6700 + i,
            "logprobs": 4,
            "return_tokens_as_token_ids": True,
            **controls,
        }
        for i in range(requests)
    ]
    before = spec_server.metrics()
    results = _complete_all(spec_server, prompts, bodies, _concurrency(spec_config))
    delta = acceptance_delta(before, spec_server.metrics(), spec_config.k)
    record(acceptance=delta.as_dict(), responses=[r.token_ids for r in results])

    for prompt, result in zip(prompts, results):
        assert_full_length_completion(result, max_tokens)
        rows = _logprob_rows(result)
        assert [row[0] for row in rows] == result.token_ids
        token, position = prompt[-1], len(prompt) - 1
        for emitted, value, top in rows:
            raw = fixed_target_distribution(token, position)
            assert value == pytest.approx(math.log(raw[emitted]), abs=1e-3)
            finite = {tid: v for tid, v in top.items() if v > -1000}
            assert finite.keys() == raw.keys(), (top, raw)
            for tid, v in finite.items():
                assert v == pytest.approx(math.log(raw[tid]), abs=1e-3)
            token, position = emitted, position + 1
        if controls["temperature"]:
            continue
        # Greedy is the rule's chain, which the logprobs ask must not move.
        assert result.token_ids == _greedy(prompt, max_tokens, lambda h: set())
    if spec_config.drafter == "model":
        # The n-gram drafter finds no repeat in the rule's greedy chain.
        assert delta.accepted > 0, (
            "no draft was accepted, so no drafted logprob was read"
        )
    _assert_no_greedy_fallback(log_path)


# endregion Logprobs

# region Structured output


def _spells(spec_server, result, pattern: str) -> bool:
    """Whether the committed tokens spell a string of ``pattern``'s language."""
    return re.fullmatch(pattern, spec_server.detokenize(result.token_ids)) is not None


@pytest.mark.parametrize("temperature", [1.0, 0.0], ids=["sampled", "greedy"])
def test_structured_output_speculates_and_follows_each_request_s_grammar(
    spec_server, spec_config, ascending_prompt, record, temperature
):
    """Two grammars in one batch, half the requests each.

    What each response's tokens spell must match its own grammar in full,
    which a bitmask row shifted onto the other request breaks: the grammars share 17 and
    differ in the other token. Among a response's 17s and its grammar's other
    token, 17 commits with probability 0.3 / (0.3 + 0.2) or 0.3 / (0.3 + 0.1),
    whatever else the grammar admits, which the pooled counts check.

    The model drafter proposes the rule's choice, which neither grammar
    allows, so on a synchronous launch the scheduler drops those drafts and on
    an asynchronous one the walk rejects them; only the n-gram drafter, which
    proposes from the output, gets drafts the grammar accepts.
    """
    requests, max_tokens = (32, 24) if temperature else (8, 24)
    names = list(GRAMMARS)
    prompts = [
        _prompt(spec_config, ascending_prompt, 1900 + i) for i in range(requests)
    ]
    bodies = [
        {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": 6900 + i,
            "structured_outputs": {"regex": GRAMMARS[names[i % 2]][0]},
        }
        for i in range(requests)
    ]
    before = spec_server.metrics()
    results = _complete_all(spec_server, prompts, bodies, _concurrency(spec_config))
    delta = acceptance_delta(before, spec_server.metrics(), spec_config.k)
    record(
        acceptance=delta.as_dict(),
        responses=[(r.token_ids, r.text) for r in results],
    )

    pairs = {name: [0, 0] for name in names}
    for index, result in enumerate(results):
        name = names[index % 2]
        pattern, tokens = GRAMMARS[name]
        assert_full_length_completion(result, max_tokens)
        assert _spells(spec_server, result, pattern), (name, result.token_ids)
        other = (tokens - {TWO}).pop()
        assert other in tokens and not (
            set(result.token_ids) & (set(FIXED_SHARED_ALTERNATIVES) - tokens)
        ), (name, result.token_ids)
        pairs[name][0] += result.token_ids.count(TWO)
        pairs[name][1] += result.token_ids.count(other)
    record(pairs=pairs)

    if spec_config.drafter == "ngram":
        assert delta.accepted > 0, "the n-gram drafter's grammar drafts never accepted"
    if not temperature:
        return
    shares = {"2-pression": 0.3 / 0.5, "guang-2": 0.3 / 0.4}
    for name, (twos, others) in pairs.items():
        total = twos + others
        assert total > 100, (name, pairs)
        expected_twos = shares[name] * total
        statistic = (twos - expected_twos) ** 2 / expected_twos + (
            others - (total - expected_twos)
        ) ** 2 / (total - expected_twos)
        assert statistic < CHI_SQUARE_CRITICAL[1], (name, twos, others, statistic)


def test_a_grammar_the_model_drafts_follow_accepts_its_drafts(
    spec_server, spec_config, ascending_prompt, record
):
    """``(.|\\n)*`` admits all but the special tokens, so the drafts are valid.

    The model drafter proposes the rule's choice, which this grammar admits,
    so a structured request accepts drafts. Under asynchronous scheduling
    that is the evidence the scheduler validated the drafts the runner
    actually verified: they reach it only through ``take_draft_token_ids`` on
    the engine's deferred-sampling path, and a structured row whose drafts did
    not reach it walks with none, so it could accept nothing. Each committed
    token must still be in its context's support.
    """
    if spec_config.drafter != "model":
        pytest.skip("only the model drafter proposes what this grammar admits")
    requests, max_tokens = 16, 32
    prompts = [
        _prompt(spec_config, ascending_prompt, 2300 + i) for i in range(requests)
    ]
    bodies = [
        {
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "seed": 7300 + i,
            "structured_outputs": {"regex": r"(.|\n)*"},
        }
        for i in range(requests)
    ]
    before = spec_server.metrics()
    results = _complete_all(spec_server, prompts, bodies, _concurrency(spec_config))
    delta = acceptance_delta(before, spec_server.metrics(), spec_config.k)
    record(acceptance=delta.as_dict(), responses=[r.token_ids for r in results])

    for prompt, result in zip(prompts, results):
        assert_full_length_completion(result, max_tokens)
        _counts(prompt, result.token_ids, lambda h: set(), {"temperature": 1.0})
    assert delta.accepted > 0, (
        "no draft of a structured request was accepted, so the scheduler never "
        "validated the drafts the runner verified"
    )


# endregion Structured output

# region Every control in one batch


def test_requests_with_different_controls_keep_their_own_in_one_verify(
    spec_server, spec_config, ascending_prompt, log_path, record
):
    """One request per control, concurrently, so the rows share verifies.

    Each row's filters, grammar and logprobs are captured per row, so a mix up
    between rows breaks a response's own check: a forbidden token, a banned
    pair, an early stop, a grammar violation or a logprob from another context.
    """
    prompts = [_prompt(spec_config, ascending_prompt, 2100 + i) for i in range(8)]
    pattern, tokens = GRAMMARS["guang-2"]
    bodies = [
        {"temperature": 0},
        {"temperature": 1.0, "seed": 1},
        {"temperature": 1.0, "seed": 2, "allowed_token_ids": ALLOWED},
        {"temperature": 1.0, "seed": 3, "bad_words": [BAD_WORD]},
        {"temperature": 1.0, "seed": 4, "min_tokens": 12, "stop_token_ids": [TWO]},
        {
            "temperature": 1.0,
            "seed": 5,
            "logprobs": 2,
            "return_tokens_as_token_ids": True,
        },
        {"temperature": 1.0, "seed": 6, "structured_outputs": {"regex": pattern}},
        {
            "temperature": 1.0,
            "seed": 7,
            "allowed_token_ids": ALLOWED,
            "bad_words": [BAD_WORD],
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        },
    ]
    max_tokens = 40
    for body in bodies:
        body["max_tokens"] = max_tokens
    results = _complete_all(spec_server, prompts, bodies, CONCURRENCY)
    record(responses=[r.token_ids for r in results])

    plain, sampled, allowed, bad, minimum, logprobs, structured, combined = results
    assert plain.token_ids == _greedy(prompts[0], max_tokens, lambda h: set())
    _counts(prompts[1], sampled.token_ids, lambda h: set(), {"temperature": 1.0})
    assert set(allowed.token_ids) <= set(ALLOWED)
    assert tuple(BAD_WORD_IDS) not in _bigrams(bad.token_ids)
    assert TWO not in minimum.token_ids[:12]
    for prompt, result in ((prompts[5], logprobs), (prompts[7], combined)):
        token, position = prompt[-1], len(prompt) - 1
        for emitted, value, _ in _logprob_rows(result):
            raw = fixed_target_distribution(token, position)
            assert value == pytest.approx(math.log(raw[emitted]), abs=1e-3)
            token, position = emitted, position + 1
    assert _spells(spec_server, structured, pattern), structured.token_ids
    assert set(combined.token_ids) <= set(ALLOWED)
    assert tuple(BAD_WORD_IDS) not in _bigrams(combined.token_ids)
    _assert_no_greedy_fallback(log_path)


# endregion Every control in one batch
