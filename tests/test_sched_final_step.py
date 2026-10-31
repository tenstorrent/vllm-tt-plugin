# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TT_SCHED_FINAL_STEP: the cohort's first tokens from one decode step
(sched_final_step), on the stub of the base
scheduler's loop of test_sched_defer_final (its token budget read from
``max_num_scheduled_tokens``)."""

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

from tests.test_sched_defer_final import CHUNK, _req, _scheduled, _scheduler
from vllm_tt_plugin import sched_final_step
from vllm_tt_plugin.sched_defer_final import DeferFinalPolicy
from vllm_tt_plugin.sched_defer_final import check_agreement as check_defer
from vllm_tt_plugin.scheduler import KNOWN_SCHED_FLAGS, check_sched_flags


def test_flags_and_agreement():
    assert "TT_SCHED_FINAL_STEP" in KNOWN_SCHED_FLAGS
    check_sched_flags({"TT_SCHED_FINAL_STEP": "1", "TT_SCHED_DEFER_FINAL": "hold"})
    assert not sched_final_step.enabled({}) and sched_final_step.enabled(
        {"TT_SCHED_FINAL_STEP": "1"}
    )
    with pytest.raises(ValueError, match="TT_SCHED_FINAL_STEP"):
        sched_final_step.enabled({"TT_SCHED_FINAL_STEP": "yes"})
    sched_final_step.check_agreement({}, defer_enabled=False)  # off: nothing to check
    sched_final_step.check_agreement({"TT_SCHED_FINAL_STEP": "1"}, defer_enabled=True)
    with pytest.raises(ValueError, match="TT_SCHED_DEFER_FINAL"):
        sched_final_step.check_agreement(
            {"TT_SCHED_FINAL_STEP": "1"}, defer_enabled=False
        )
    with pytest.raises(ValueError, match="TT_V41_DEFER_FINAL"):
        sched_final_step.check_agreement(
            {"TT_SCHED_FINAL_STEP": "1", "TT_V41_DEFER_FINAL": "1"}, defer_enabled=True
        )
    # the deferred finals' own agreement: the generator's flag off is right under the
    # final step, wrong without it
    check_defer({"TT_SCHED_DEFER_FINAL": "hold", "TT_SCHED_FINAL_STEP": "1"})
    with pytest.raises(ValueError, match="TT_V41_DEFER_FINAL"):
        check_defer({"TT_SCHED_DEFER_FINAL": "hold"})
    policy = DeferFinalPolicy.from_env(
        {"TT_SCHED_DEFER_FINAL": "hold", "TT_SCHED_FINAL_STEP": "1"},
        chunk_tokens=CHUNK,
        cohort=32,
    )
    assert policy is not None and policy.final_step and policy.hold_admission
    assert "final_step=True" in policy.describe()


def test_the_rule():
    assert not sched_final_step.stepped(CHUNK, CHUNK) and sched_final_step.stepped(
        CHUNK + 1, CHUNK
    )
    assert sched_final_step.bulk_end(1028, CHUNK) == 1028
    assert sched_final_step.bulk_end(8196, CHUNK) == 8195
    assert sched_final_step.bulk_end(2 * CHUNK, CHUNK) == 2 * CHUNK - 1
    policy = DeferFinalPolicy(CHUNK, 32, final_step=True)
    assert (
        policy.prefix(_req("a", 8196)) == 8195 and policy.prefix(_req("b", 1028)) == 0
    )
    assert (
        policy.budget(_req("a", 8196, computed=0)) is None
    )  # a full chunk away: the stock budget
    assert (
        policy.budget(_req("a", 8196, computed=CHUNK)) == 3075
    )  # the last chunk, one token short
    assert policy.budget(_req("a", 2 * CHUNK, computed=CHUNK)) == CHUNK - 1
    assert policy.budget(_req("b", 1028)) is None
    assert DeferFinalPolicy(CHUNK, 32).budget(_req("a", 8196, computed=CHUNK)) is None


def test_is_final_step_classifies_rows():
    chunk = CHUNK
    assert sched_final_step.is_final_step([(8195, 8196, 1), (10239, 10240, 1)], chunk)
    assert not sched_final_step.is_final_step(
        [(5120, 8196, 3075)], chunk
    )  # a bulk's last chunk
    assert not sched_final_step.is_final_step(
        [(0, 1028, 1028)], chunk
    )  # a one-chunk prompt: stock
    assert not sched_final_step.is_final_step(
        [(1027, 1028, 1)], chunk
    )  # a one-chunk prompt's last token: stock
    assert not sched_final_step.is_final_step(
        [(8196, 8196, 1), (9000, 9000, 1)], chunk
    )  # decode rows
    with pytest.raises(AssertionError, match="final-step"):
        sched_final_step.is_final_step([(8195, 8196, 1), (5120, 8196, 3075)], chunk)


def _base_schedule(self):
    """The stub of test_sched_defer_final with the token budget read from
    ``max_num_scheduled_tokens``, and the
    running loop advancing every visible partial by one chunk (the base scheduler's loop
    over ``running``)."""
    rows = {}
    budget = getattr(self, "max_num_scheduled_tokens", CHUNK)
    partials = [r for r in self.running if r.is_prefill_chunk]
    if not partials and self.waiting and len(self.running) < self.max_num_running_reqs:
        partial = self.waiting.pop_request()
        self.running.append(partial)
        partials = [partial]
    if partials:
        for partial in partials:
            n = min(budget, partial.num_prompt_tokens - partial.num_computed_tokens)
            partial.num_computed_tokens += n
            partial.is_prefill_chunk = (
                partial.num_computed_tokens < partial.num_prompt_tokens
            )
            rows[partial.request_id] = n
    else:
        for r in self.running:
            if not r.is_prefill_chunk:
                r.num_computed_tokens += 1
                rows[r.request_id] = 1
    return _scheduled(rows) if rows else SchedulerOutput.make_empty()


@pytest.fixture
def base(monkeypatch):
    monkeypatch.setattr(AsyncScheduler, "schedule", _base_schedule)


def _sched(prompts, *, cohort=4, hold=True):
    scheduler = _scheduler(prompts, cohort=cohort, hold=hold)
    scheduler._defer = DeferFinalPolicy(
        CHUNK, cohort, hold_admission=hold, final_step=True
    )
    scheduler.max_num_scheduled_tokens = CHUNK
    return scheduler


def _trace(scheduler, steps):
    """Per step: ``P<id>:<start>-<end>`` for one prefill row, ``F<n>`` for n rows of one
    last prompt token each
    (the final step), ``D<n>`` for n decode rows, ``-`` for an empty step."""
    out = []
    for _ in range(steps):
        result = scheduler.schedule()
        if not result.num_scheduled_tokens:
            out.append("-")
            continue
        kinds = []
        for req_id, n in result.num_scheduled_tokens.items():
            r = scheduler.requests[req_id]
            if r.num_computed_tokens - n >= r.num_prompt_tokens:
                kinds.append("D")
            elif n == 1 and r.num_computed_tokens == r.num_prompt_tokens:
                kinds.append("F")
            else:
                kinds.append(
                    f"P{req_id}:{r.num_computed_tokens - n}-{r.num_computed_tokens}"
                )
        assert len(set(k[0] for k in kinds)) == 1, kinds
        out.append(f"{kinds[0]}{len(kinds)}" if kinds[0] in "DF" else kinds[0])
        assert scheduler.max_num_scheduled_tokens == CHUNK  # restored after every step
    return out


def test_bulks_end_one_token_short_and_the_cohort_steps_together(base):
    scheduler = _sched([CHUNK + 100] * 4, cohort=4)
    t = _trace(scheduler, 11)
    bulk = [
        f"Pw{i}:{s}-{e}" for i in range(4) for s, e in ((0, CHUNK), (CHUNK, CHUNK + 99))
    ]
    assert t == bulk + ["F4", "D4", "D4"], t
    assert scheduler._defer.closed_on == ["full"] and not scheduler._defer.draining


def test_a_prompt_of_whole_chunks_still_keeps_its_last_token(base):
    scheduler = _sched([2 * CHUNK] * 2, cohort=2)
    t = _trace(scheduler, 6)
    assert t == [
        f"Pw0:0-{CHUNK}",
        f"Pw0:{CHUNK}-{2 * CHUNK - 1}",
        f"Pw1:0-{CHUNK}",
        f"Pw1:{CHUNK}-{2 * CHUNK - 1}",
        "F2",
        "D2",
    ], t


def test_one_chunk_prompts_take_the_stock_path(base):
    scheduler = _sched([1028] * 3, cohort=3)
    t = _trace(scheduler, 5)
    assert t == ["Pw0:0-1028", "Pw1:0-1028", "Pw2:0-1028", "D3", "D3"], t


def test_the_finals_step_reports_every_final_to_the_release_policy(base):
    """The defer policy pops every final completed in the step (not only the head), so
    the release policy sees
    them all (its hold is moot here: the step's tokens come together anyway)."""
    scheduler = _sched([CHUNK + 100] * 3, cohort=3)
    _trace(scheduler, 6)
    out = scheduler.schedule()
    assert len(out.num_scheduled_tokens) == 3
    assert sorted(scheduler._defer.last_finals) == [
        ("w0", True),
        ("w1", True),
        ("w2", True),
    ]
    assert not scheduler._defer.draining


def test_a_queue_tail_cohort_steps_at_its_close(base):
    scheduler = _sched([CHUNK + 100] * 2, cohort=32)
    t = _trace(scheduler, 6)
    assert t[4:] == ["F2", "D2"], t
    assert scheduler._defer.closed_on == ["tail"]
