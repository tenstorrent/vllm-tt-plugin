# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The server emits the same tokens speculating as not, through the real stack.

The host suite settles this against a stand-in. This settles it through the
engine, the scheduler, the worker and HTTP, which is where the state that could
break it lives: the scheduler's draft bookkeeping, the persistent batch's rows,
and the commit path that writes a variable number of tokens per step.

It requires the ``fixed`` target. The ``depth`` target returns each draft
unchanged up to its accept depth, so its output is a function of what was
drafted and the two arms are meant to differ; asking it this question would
compare two different models. The ``fixed`` target chooses by a rule the drafts
never enter, so the sequence is a property of the rule and both arms have to
produce it.

The unspeculated arm is a second server, launched from the same model with no
``--speculative-config`` at all, whose URL is passed with
``--tt-reference-url``. Running the two arms against one server is impossible:
whether a launch speculates is fixed when the engine starts.
"""

from __future__ import annotations

import pytest

from tests.tt.spec.spec_client import SpecServer, acceptance_delta

MAX_TOKENS = 128


@pytest.fixture(scope="session")
def reference_server(request, tt_model_name, spec_config):
    """The unspeculated server, or a skip when none was launched."""
    url = request.config.getoption("--tt-reference-url")
    if not url:
        pytest.skip(
            "no --tt-reference-url: losslessness needs a second server launched "
            "from the same model without --speculative-config"
        )
    if spec_config.target != "fixed":
        pytest.skip(
            "losslessness needs TT_SPEC_TARGET=fixed, whose output does not "
            "depend on what was drafted"
        )
    return SpecServer(url, tt_model_name)


def test_the_speculated_output_equals_the_unspeculated_output(
    spec_server, reference_server, spec_config, ascending_prompt, record
):
    """Token for token, over the whole output, from one prompt.

    The two arms commit at different widths per step, so they take different
    numbers of steps to reach the same length. Comparing the full sequences at
    a fixed token count is what makes this a statement about the tokens rather
    than about the steps.
    """
    prompt = ascending_prompt(64, start=101)

    before = spec_server.metrics()
    speculated = spec_server.complete(prompt, max_tokens=MAX_TOKENS)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)

    plain = reference_server.complete(prompt, max_tokens=MAX_TOKENS)

    record(
        request=speculated.request,
        speculated=speculated.body,
        unspeculated=plain.body,
        acceptance=delta.as_dict(),
    )

    assert speculated.status == 200 and plain.status == 200
    assert speculated.completion_tokens == MAX_TOKENS
    assert plain.completion_tokens == MAX_TOKENS
    assert delta.drafts > 0, (
        "nothing was drafted, so this run says nothing about whether "
        "speculation preserves the output"
    )
    assert delta.accepted > 0, (
        "no draft was accepted, so the speculated arm committed one token per "
        "step and the comparison is with an unspeculated run of itself"
    )
    assert speculated.token_ids == plain.token_ids
    assert speculated.text == plain.text


def test_the_two_arms_reach_the_same_output_in_different_step_counts(
    spec_server, reference_server, spec_config, ascending_prompt, record
):
    """The equality above is not a coincidence of identical execution.

    If both arms took the same number of engine steps, the speculated arm
    committed one token per step and accepted nothing, and the comparison would
    hold trivially. The step counts have to differ for the equality to mean
    anything.
    """
    prompt = ascending_prompt(64, start=211)

    spec_before = spec_server.metrics()
    speculated = spec_server.complete(prompt, max_tokens=MAX_TOKENS)
    spec_after = spec_server.metrics()
    plain_before = reference_server.metrics()
    plain = reference_server.complete(prompt, max_tokens=MAX_TOKENS)
    plain_after = reference_server.metrics()

    spec_steps = acceptance_delta(spec_before, spec_after, spec_config.k).steps
    plain_steps = acceptance_delta(plain_before, plain_after, spec_config.k).steps
    record(
        request=speculated.request,
        speculated_steps=spec_steps,
        unspeculated_steps=plain_steps,
        equal=speculated.token_ids == plain.token_ids,
    )

    assert speculated.token_ids == plain.token_ids
    assert spec_steps < plain_steps, (
        f"the speculated arm took {spec_steps} steps and the plain arm "
        f"{plain_steps}: speculation committed no more per step than plain "
        "decoding, so the equality above is trivial"
    )


def test_a_deliberately_wrong_drafter_still_emits_the_same_output(
    spec_server, reference_server, spec_config, ascending_prompt, record
):
    """The rejection path, end to end.

    Under the ``fixed`` target the drafter bends every draft past its accept
    depth, so a partial-acceptance launch rejects on every step and commits the
    target's own choice at the position that rejected. That is where a wrong
    token could reach the output, and the sequence still has to match.
    """
    if spec_config.accepts_every_draft:
        pytest.skip(
            "this configuration accepts every draft, so no rejection is "
            "exercised; run it at an accept depth below K"
        )
    prompt = ascending_prompt(64, start=307)

    before = spec_server.metrics()
    speculated = spec_server.complete(prompt, max_tokens=MAX_TOKENS)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    plain = reference_server.complete(prompt, max_tokens=MAX_TOKENS)

    record(
        request=speculated.request,
        speculated=speculated.body,
        unspeculated=plain.body,
        acceptance=delta.as_dict(),
    )

    assert delta.accepted < delta.draft_tokens, (
        "every draft was accepted, so no rejection happened and this test "
        "covers nothing it claims to"
    )
    assert delta.accepted > 0
    assert speculated.token_ids == plain.token_ids
