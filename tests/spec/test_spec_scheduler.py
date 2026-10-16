# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The scheduler leaves real drafts on a request, never the async placeholder.

``TTScheduler`` inherits from ``AsyncScheduler`` in both execution modes,
because that is where ``num_output_placeholders`` lives. On a speculative
launch that inheritance carries one behaviour the TT runner cannot live with:
``AsyncScheduler._update_after_schedule`` assigns
``request.spec_token_ids = [-1] * num_spec_tokens`` to every scheduled request,
and upstream's GPU runner overwrites those placeholders from its own state in
``_prepare_input_ids`` before verifying anything.

The TT runner has no such step. It builds its candidate block from
``SchedulerOutput.scheduled_spec_decode_tokens`` and verifies whatever arrives,
so a surviving placeholder becomes a draft: the model is handed that same
column, returns it unchanged, the accept walk sees a match, and
``PLACEHOLDER_TOKEN_ID`` commits as an output token.

These tests drive the real ``AsyncScheduler._update_after_schedule`` with the
base ``Scheduler`` half stubbed out, because the placeholder assignment is the
behaviour under test and a hand-built ``SchedulerOutput`` would not exhibit it.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_tt_plugin.scheduler import TTScheduler, spec_lookahead_tokens
from vllm_tt_plugin.spec_admission import MODEL_OWNED_DRAFT_METHOD
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    DRAFTER_STATE_INTERNAL,
    SpecPlan,
)

NUM_SPEC_TOKENS = 5


def _request(req_id: str, is_prefill_chunk: bool = False) -> SimpleNamespace:
    """A request seen only through the fields the two methods touch."""
    return SimpleNamespace(
        req_id=req_id,
        is_prefill_chunk=is_prefill_chunk,
        use_structured_output=False,
        num_output_placeholders=0,
        spec_token_ids=[],
    )


def _scheduler(
    requests: dict[str, SimpleNamespace],
    num_spec_tokens: int,
    async_scheduling: bool = False,
):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.requests = requests
    scheduler.num_spec_tokens = num_spec_tokens
    # What decides who owns the drafts. Synchronously the scheduler delivers
    # them and the placeholders must go; asynchronously nothing delivers them
    # and the placeholders are the lookahead reservation.
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.use_v2_model_runner = False
    scheduler.pp_size = 1
    scheduler.current_step = 0
    scheduler._spec_token_placeholders = [-1] * num_spec_tokens
    # Read only on the block-output rail, which speculation cannot combine with.
    scheduler._is_block_output_model = False
    return scheduler


def _scheduler_output(req_ids, drafts):
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {req: 1 for req in req_ids}
    output.scheduled_spec_decode_tokens = dict(drafts)
    output.num_spec_tokens_to_schedule = NUM_SPEC_TOKENS
    return output


@pytest.fixture(autouse=True)
def _stub_the_base_scheduler(monkeypatch):
    """The base half reaches the KV cache manager, which needs a real engine."""
    monkeypatch.setattr(Scheduler, "_update_after_schedule", lambda self, out: None)


def test_no_scheduled_request_is_left_holding_placeholder_drafts():
    """The defect this guards: a ``-1`` draft commits as an output token."""
    requests = {"a": _request("a"), "b": _request("b")}
    scheduler = _scheduler(requests, NUM_SPEC_TOKENS)

    scheduler._update_after_schedule(
        _scheduler_output(["a", "b"], {"a": [11, 12, 13], "b": []})
    )

    assert requests["a"].spec_token_ids == []
    assert requests["b"].spec_token_ids == []


def test_the_placeholders_are_what_the_base_class_would_have_left():
    """Pins the upstream behaviour the override exists to correct.

    Without the override, every scheduled request holds ``num_spec_tokens``
    entries of ``-1``. If upstream stops doing this, the override becomes dead
    code rather than a silent correctness dependency, and this test says so.
    """
    requests = {"a": _request("a")}
    scheduler = _scheduler(requests, NUM_SPEC_TOKENS)

    AsyncScheduler._update_after_schedule(
        scheduler, _scheduler_output(["a"], {"a": [11, 12, 13]})
    )

    assert requests["a"].spec_token_ids == [-1] * NUM_SPEC_TOKENS


def test_a_spent_proposal_is_not_replayed():
    """Cleared, not restored to the ids just scheduled.

    A proposal is handed to the scheduler once, so a request whose row proposed
    nothing this step has to speculate on nothing next step. Restoring the ids
    that were just verified would have the scheduler offer them again.
    """
    requests = {"a": _request("a")}
    requests["a"].spec_token_ids = [11, 12, 13]
    scheduler = _scheduler(requests, NUM_SPEC_TOKENS)

    scheduler._update_after_schedule(_scheduler_output(["a"], {"a": [11, 12, 13]}))

    assert requests["a"].spec_token_ids == []


def test_an_asynchronous_launch_keeps_the_placeholders():
    """Asynchronously the placeholders are the reservation, not a proposal.

    Upstream stops routing drafts through the scheduler when asynchronous
    scheduling is on: ``EngineCore.post_step`` skips
    ``take_draft_token_ids`` because a step's drafts are not known before it
    runs. What ``AsyncScheduler`` leaves behind is then the only thing that
    budgets lookahead positions for the request, since the next schedule
    reserves ``1 + len(spec_token_ids)``. Clearing them there would leave the
    runner verifying a candidate block in a row budgeted for one token.

    ``TTModelRunner._drafts_to_verify`` is the other half: it reads a
    placeholder list as that reservation and verifies the proposal the runner
    holds, never the ``-1`` itself.
    """
    requests = {"a": _request("a"), "b": _request("b")}
    scheduler = _scheduler(requests, NUM_SPEC_TOKENS, async_scheduling=True)

    scheduler._update_after_schedule(
        _scheduler_output(["a", "b"], {"a": [11, 12, 13], "b": []})
    )

    assert requests["a"].spec_token_ids == [-1] * NUM_SPEC_TOKENS
    assert requests["b"].spec_token_ids == [-1] * NUM_SPEC_TOKENS


def test_a_launch_without_speculation_is_untouched():
    """``num_spec_tokens`` of 0 is every non-speculative launch."""
    requests = {"a": _request("a")}
    scheduler = _scheduler(requests, 0)
    output = _scheduler_output(["a"], {})
    output.num_spec_tokens_to_schedule = 0

    scheduler._update_after_schedule(output)

    assert requests["a"].spec_token_ids == []


def test_the_output_placeholder_accounting_is_preserved():
    """Only the draft ids are corrected; the reservation the base class made
    for this step's sampled and speculative tokens has to stand, because
    ``_update_request_with_output`` decrements it as the tokens arrive."""
    requests = {"a": _request("a")}
    scheduler = _scheduler(requests, NUM_SPEC_TOKENS)

    scheduler._update_after_schedule(_scheduler_output(["a"], {"a": [11, 12, 13]}))

    # One sampled token plus the three drafts actually scheduled.
    assert requests["a"].num_output_placeholders == 4


# region Lookahead for a model-owned drafter


def _plan(k: int) -> SpecPlan:
    return SpecPlan(
        effective_k=k,
        lanes_per_request=1,
        extra_bytes_per_seq=0,
        extra_bytes_per_token=0,
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
        drafter_state=DRAFTER_STATE_INTERNAL,
        supports_narrow_decode=False,
    )


def test_a_model_owned_drafter_reserves_the_anchor_and_every_draft():
    """The proposal that follows a commit writes K/V for the anchor plus K
    drafts past the committed position, so the scheduler has to have those
    K+1 slots allocated before the model runs."""
    assert (
        spec_lookahead_tokens(
            _plan(NUM_SPEC_TOKENS), NUM_SPEC_TOKENS, MODEL_OWNED_DRAFT_METHOD
        )
        == NUM_SPEC_TOKENS + 1
    )


def test_no_admitted_plan_reserves_nothing():
    """No admitted plan, or no drafts, leaves nothing to reserve for."""
    assert spec_lookahead_tokens(None, NUM_SPEC_TOKENS, MODEL_OWNED_DRAFT_METHOD) == 0
    assert (
        spec_lookahead_tokens(_plan(NUM_SPEC_TOKENS), 0, MODEL_OWNED_DRAFT_METHOD) == 0
    )


def test_an_ngram_launch_reserves_nothing():
    """ngram drafts come from the plugin and its target verifies them inside
    the step's own allocation; the plan it carries reserves nothing."""
    assert spec_lookahead_tokens(_plan(NUM_SPEC_TOKENS), NUM_SPEC_TOKENS, "ngram") == 0


# endregion Lookahead for a model-owned drafter
