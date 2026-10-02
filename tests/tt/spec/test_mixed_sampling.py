# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A sampled request beside a speculating greedy one is never answered greedily.

The runner verifies in ``argmax_ids`` mode, and a verify commits the target's
argmax on every row of the step. So a sampled or penalized request must never be
part of a verify. Two launches enforce that two ways:

- With narrow decode (``TT_SPEC_DRAFT_POLICY=solo``), every step a sampled
  request is part of runs as the ordinary decode, and the drafts the greedy
  request holds for that step are dropped.
- Without it (``always``), every decode step is a verify, so a sampled request
  is refused when it arrives.

The first needs the ``fixed`` target, whose logits are a known distribution: the
rule's choice at 0.4 and three shared tokens at the rest. That makes three
things checkable from outside: the greedy request's output is the rule's
sequence exactly, every sampled token lies in the distribution's support, and
the join step really ran as an ordinary decode. The last is read off vLLM's
counters: the drafter under this target always proposes what the target
chooses, so a verify accepts every draft, and the only way a draft step can
accept nothing is the ordinary decode that dropped the drafts.

The ``mixed`` and ``async-mixed`` configurations of ``run_spec_regression.sh``
launch the first; every ``always`` configuration exercises the second.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.tt.spec.dummy_arithmetic import (
    fixed_target_choice,
    fixed_target_ids,
    fixed_target_support,
)
from tests.tt.spec.spec_client import acceptance_delta, assert_full_length_completion

# Long enough that the greedy request is still decoding when its sampled peer
# finishes, which the counter window below depends on. Each request is sent to
# a server whose forward is host arithmetic, so a step costs about a
# millisecond and the greedy request commits up to 1+K tokens per solo step.
GREEDY_TOKENS = 1024
SAMPLED_TOKENS = 16
TRIALS = 12
PROMPT_LEN = 64


def _sampled_tokens_outside_the_support(prompt: list[int], tokens: list[int]) -> list:
    """``(index, token)`` for every token the fixed target could not emit."""
    outside = []
    previous, position = prompt[-1], len(prompt) - 1
    for index, token in enumerate(tokens):
        if token not in fixed_target_support(previous, position):
            outside.append((index, token))
        previous, position = token, position + 1
    return outside


def _is_the_argmax(prompt: list[int], tokens: list[int], index: int) -> bool:
    previous = prompt[-1] if index == 0 else tokens[index - 1]
    return tokens[index] == fixed_target_choice(previous, len(prompt) - 1 + index)


@pytest.fixture
def narrow_fixed_launch(request, spec_config):
    if spec_config.draft_policy != "solo" or spec_config.target != "fixed":
        pytest.skip(
            "needs a launch with narrow decode and a known distribution: "
            "TT_SPEC_DRAFT_POLICY=solo and TT_SPEC_TARGET=fixed"
        )
    return str(request.config.getoption("--tt-spec-async-scheduling")) == "true"


def test_a_sampled_request_joining_a_speculating_one_is_sampled(
    spec_server,
    spec_config,
    ascending_prompt,
    max_batch_size,
    record,
    narrow_fixed_launch,
):
    """The join step runs as an ordinary decode, and both requests stay right.

    Each trial starts a long greedy request, lets it speculate alone, and sends
    a sampled peer. The greedy request holds drafts when the peer joins, so
    the first step the two share is the one that used to be a verify.
    """
    if max_batch_size < 2:
        pytest.skip("this server serves one row at a time")
    asynchronous = narrow_fixed_launch

    trials = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for trial in range(TRIALS):
            greedy_prompt = ascending_prompt(PROMPT_LEN, start=100 + 200 * trial)
            sampled_prompt = ascending_prompt(PROMPT_LEN, start=50000 + 200 * trial)
            greedy = pool.submit(
                spec_server.complete, greedy_prompt, max_tokens=GREEDY_TOKENS
            )
            # Long enough for the greedy request to be decoding alone, and
            # short enough that it has not finished: see GREEDY_TOKENS.
            time.sleep(0.02)
            before = spec_server.metrics()
            sampled = spec_server.complete(
                sampled_prompt, max_tokens=SAMPLED_TOKENS, temperature=1.0
            )
            after = spec_server.metrics()
            greedy_still_running = not greedy.done()
            greedy_result = greedy.result()
            window = acceptance_delta(before, after, spec_config.k)
            trials.append(
                {
                    "greedy_prompt": greedy_prompt,
                    "sampled_prompt": sampled_prompt,
                    "greedy": greedy_result,
                    "sampled": sampled,
                    "window": window,
                    "greedy_outlived_sampled": greedy_still_running,
                }
            )

    record(
        requests=[t["greedy"].request for t in trials]
        + [t["sampled"].request for t in trials],
        trials=[
            {
                "sampled_token_ids": t["sampled"].token_ids,
                "window": t["window"].as_dict(),
                "greedy_outlived_sampled": t["greedy_outlived_sampled"],
            }
            for t in trials
        ],
    )

    for t in trials:
        assert_full_length_completion(t["greedy"], GREEDY_TOKENS)
        assert_full_length_completion(t["sampled"], SAMPLED_TOKENS)
        # Dropping its drafts on the join step must not change the greedy
        # request's output: the fixed target's output is a function of the
        # prompt alone.
        assert t["greedy"].token_ids == fixed_target_ids(
            t["greedy_prompt"], GREEDY_TOKENS
        )
        outside = _sampled_tokens_outside_the_support(
            t["sampled_prompt"], t["sampled"].token_ids
        )
        assert not outside, (
            f"sampled tokens outside the fixed target's support: {outside}"
        )

    sampled = [token for t in trials for token in t["sampled"].token_ids]
    argmax = sum(
        _is_the_argmax(t["sampled_prompt"], t["sampled"].token_ids, index)
        for t in trials
        for index in range(SAMPLED_TOKENS)
    )
    # The rule's choice carries probability 0.4; a request committing the
    # argmax on most steps was not sampled.
    assert argmax < 0.75 * len(sampled), (
        f"{argmax} of {len(sampled)} sampled tokens are the argmax"
    )

    if asynchronous:
        # ``AsyncScheduler`` gives every scheduled request ``[-1] * K`` as its
        # lookahead reservation and vLLM counts that as a draft step, whether
        # or not a draft was verified, so the counters cannot single out the
        # join step here.
        return

    # The join steps. Each trial's window holds the sampled request's whole
    # life, and the greedy request's draft steps inside it accept every draft
    # except on a step that dropped them. A window whose greedy request had
    # already finished is left out, because a commit cut short at max_tokens
    # also accepts nothing.
    joins = [
        t
        for t in trials
        if t["greedy_outlived_sampled"]
        and t["window"].drafts > t["window"].per_position[0]
    ]
    assert joins, (
        "no trial shows a draft step that accepted nothing, so no step that "
        "the sampled request joined dropped the greedy request's drafts: "
        "either the join step was a verify, or the greedy request never held "
        "drafts when its peer arrived"
    )
    # On the join step and the one after it, a verify would have committed
    # the argmax for the sampled row. Sampled, both are the argmax with
    # probability 0.16, so all of them being so is not chance. Asserted only
    # over enough joins for that to hold: 0.16 ** 6 is below 2e-5.
    if len(joins) >= 6:
        both_argmax = sum(
            _is_the_argmax(t["sampled_prompt"], t["sampled"].token_ids, 1)
            and _is_the_argmax(t["sampled_prompt"], t["sampled"].token_ids, 2)
            for t in joins
        )
        assert both_argmax < len(joins), (
            f"in all {len(joins)} joins the sampled request committed the "
            "argmax on the join step and the step after it"
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 1.0},
        {"temperature": 0, "presence_penalty": 0.5},
        {"temperature": 0, "repetition_penalty": 1.1},
    ],
    ids=["temperature", "presence-penalty", "repetition-penalty"],
)
def test_a_sampled_request_is_refused_where_every_step_verifies(
    spec_server, spec_config, ascending_prompt, record, overrides
):
    """Without narrow decode, a sampled request would get greedy tokens.

    The ``always`` drafter declares a hidden feed through the runner and no
    narrow decode, so every decode step of the launch is a verify. A request
    that asked to sample would come back as greedy text with a 200, so the
    plugin refuses it with the reason.
    """
    if spec_config.draft_policy != "always":
        pytest.skip("needs a launch without narrow decode: TT_SPEC_DRAFT_POLICY=always")

    result = spec_server.complete(
        ascending_prompt(PROMPT_LEN), max_tokens=8, **overrides
    )
    record(requests=[result.request], status=result.status, body=result.body)

    assert result.status == 400, (
        f"a sampled request was served by a launch that verifies every step: "
        f"{result.status} {result.body}"
    )
    assert "supports_narrow_decode" in json.dumps(result.body)
