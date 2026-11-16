# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``supports_prefill_over_pending_spec``: the declaration check, and which
prefill step ``TTScheduler`` returns without a release step and
``TTModelRunner`` submits behind an unread verify.

``tests/test_mamba_spec_async.py`` drives such prefills through the
scheduler, the runner and a device model end to end.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.config import (
    get_tt_prefill_over_pending_spec,
    store_tt_prefill_over_pending_spec,
)
from vllm_tt_plugin.scheduler import (
    TTScheduler,
    prefill_qualifies_behind_verify,
    set_tt_forced_reset_discard_counts,
)
from vllm_tt_plugin.spec_admission import validate_prefill_over_pending_spec

KEY = "supports_prefill_over_pending_spec"


@pytest.mark.parametrize(
    ("capabilities", "match"),
    [
        ({KEY: 1}, "is a bool"),
        ({KEY: True}, "without supports_async_spec"),
        ({KEY: True, "supports_async_spec_decode": False}, "without supports_async"),
    ],
)
def test_the_declaration_needs_the_deferred_readback(capabilities, match):
    with pytest.raises(ValueError, match=match):
        validate_prefill_over_pending_spec(capabilities, "M")


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        (None, False),
        ({}, False),
        ({KEY: False}, False),
        ({KEY: True, "supports_async_spec_decode": True}, True),
    ],
)
def test_declarations_the_plugin_accepts(capabilities, expected):
    assert validate_prefill_over_pending_spec(capabilities, "M") is expected


def test_the_scheduler_reads_only_the_platform_stored_value():
    config = SimpleNamespace(additional_config={})
    assert not get_tt_prefill_over_pending_spec(config)
    store_tt_prefill_over_pending_spec(config, True)
    assert get_tt_prefill_over_pending_spec(config)
    # An operator-passed value that is not the platform's bool is not honored.
    config.additional_config["_tt_prefill_over_pending_spec"] = "yes"
    assert not get_tt_prefill_over_pending_spec(config)
    with pytest.raises(ValueError, match="boolean"):
        store_tt_prefill_over_pending_spec(config, 1)


class _Cached:
    def __init__(self, req_ids=(), context=(), resumed=()):
        self.req_ids = list(req_ids)
        self.resumed_req_ids = set(resumed)
        self._context = set(context)

    def is_context_phase(self, req_id):
        return req_id in self._context


def _step(new=(), cached=None, preempted=(), mm=(), spec=None, scheduled=None):
    cached = cached or _Cached()
    new_reqs = [
        SimpleNamespace(req_id=r, mm_features=[1] if r in mm else []) for r in new
    ]
    if scheduled is None:
        scheduled = [*new, *cached.req_ids]
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        preempted_req_ids=set(preempted),
        num_scheduled_tokens=dict.fromkeys(scheduled, 4),
        total_num_scheduled_tokens=4 * len(scheduled),
        scheduled_spec_decode_tokens=spec if spec is not None else {},
    )


@pytest.mark.parametrize(
    ("step", "qualifies"),
    [
        (_step(new=["J"]), True),
        (_step(new=["J", "L"]), True),
        (_step(cached=_Cached(["J"], context=["J"])), True),
        (_step(new=["J"], cached=_Cached(["R"], context=["R"], resumed=["R"])), False),
        (_step(new=["J"], preempted=["A"]), False),
        (_step(new=["J"], mm=["J"]), False),
        (_step(new=["J"], cached=_Cached(["A"])), False),
        (_step(), False),
    ],
    ids=[
        "new",
        "two-new",
        "chunk",
        "resumed",
        "preempted",
        "multimodal",
        "decode-row",
        "empty",
    ],
)
def test_which_prefill_step_runs_behind_a_verify(step, qualifies):
    assert prefill_qualifies_behind_verify(step) is qualifies


def test_a_prefill_with_forced_reset_counts_does_not_run_behind_a_verify():
    step = _step(new=["J"])
    set_tt_forced_reset_discard_counts(step, {"A": 1})
    assert not prefill_qualifies_behind_verify(step)


def _scheduler(supported=True, async_scheduling=True, pending_resets=None):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler._prefill_over_pending_spec = supported
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    scheduler._pending_forced_reset_discard_counts = pending_resets or {}
    return scheduler


VERIFY = _step(cached=_Cached(["A", "B"]), spec={"A": [-1] * 3, "B": [-1] * 3})


@pytest.mark.parametrize(
    ("scheduler", "held", "prefill", "behind"),
    [
        (_scheduler(), VERIFY, _step(new=["J"]), True),
        (_scheduler(supported=False), VERIFY, _step(new=["J"]), False),
        (_scheduler(async_scheduling=False), VERIFY, _step(new=["J"]), False),
        (_scheduler(pending_resets={"A": 1}), VERIFY, _step(new=["J"]), False),
        # The held step is not a speculative decode of every row.
        (
            _scheduler(),
            _step(cached=_Cached(["A", "B"]), spec={"A": [-1] * 3}),
            _step(new=["J"]),
            False,
        ),
        (_scheduler(), _step(new=["P"]), _step(new=["J"]), False),
        (_scheduler(), VERIFY, _step(new=["J"], mm=["J"]), False),
        (_scheduler(), VERIFY, _step(), False),
    ],
    ids=[
        "verify",
        "undeclared",
        "synchronous",
        "reset-pending",
        "partial-spec",
        "held-prefill",
        "multimodal",
        "nothing-to-prefill",
    ],
)
def test_the_scheduler_skips_the_release_step_only_behind_a_verify(
    scheduler, held, prefill, behind
):
    assert scheduler._prefill_runs_behind_verify(held, prefill) is behind
