# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Unit tests for the single-process lane-DP coordinator.

The coordinator is exercised over lightweight fake lane schedulers: a real
``TTScheduler`` needs a device KV cache config, but the coordinator only relies
on a small surface of each lane (``waiting`` / ``skipped_waiting`` / ``running``
length, forced-mode scheduling, and ``update_from_output``).
"""

import collections
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.request import RequestStatus

from vllm_tt_plugin.lane_scheduler import (
    TTLaneCoordinator,
    TTStepPlan,
    get_tt_step_plan,
    merge_lane_scheduler_outputs,
)
from vllm_tt_plugin.scheduler import (
    TTDecodeInterleavePolicy,
    TTSchedulingMode,
    get_tt_forced_reset_discard_counts,
    set_tt_forced_reset_discard_counts,
)


class FakeLane:
    """Minimal stand-in for a per-lane ``TTScheduler``.

    Prefill always schedules zero tokens here (simulating "no full prefill fits"
    under KV pressure); decode schedules one token per running request. Pending
    finished IDs are emitted on the first ``schedule`` call and then drained, so
    tests can assert the coordinator carries them across a prefill->decode
    fallback.
    """

    def __init__(
        self,
        waiting=0,
        running=0,
        skipped_waiting=0,
        partial_prefills=0,
        pending_finished=(),
        owns=(),
    ):
        self.waiting = [object()] * waiting
        self.skipped_waiting = [object()] * skipped_waiting
        self.running = [
            SimpleNamespace(is_prefill_chunk=False) for _ in range(running)
        ] + [
            SimpleNamespace(is_prefill_chunk=True, num_tokens=0, num_computed_tokens=0)
            for _ in range(partial_prefills)
        ]
        self.reset_calls: list[tuple[bool, bool]] = []
        self._pending_finished = set(pending_finished)
        self.requests: dict[str, SimpleNamespace] = {}
        self._mode = TTSchedulingMode.DEFAULT
        self.scheduled_modes: list[TTSchedulingMode] = []
        self.update_calls: list[SchedulerOutput] = []
        self._eco: dict[int, EngineCoreOutputs] = {}
        self._owned = {r.request_id: r for r in owns}
        self.finish_calls: list[tuple] = []

    def set_forced_mode(self, mode):
        self._mode = mode

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        self.reset_calls.append((reset_running_requests, reset_connector))
        return True

    def schedule(self):
        self.scheduled_modes.append(self._mode)
        finished = self._pending_finished
        self._pending_finished = set()
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = set(finished)
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        if self._mode == TTSchedulingMode.DECODE_ONLY and pure_decodes:
            out.num_scheduled_tokens = {f"dec-{id(self)}": len(pure_decodes)}
            out.total_num_scheduled_tokens = len(pure_decodes)
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        self.update_calls.append(scheduler_output)
        return self._eco

    def finish_requests(self, request_ids, finished_status):
        self.finish_calls.append((request_ids, finished_status))
        if request_ids is None:
            ids = list(self._owned)
        elif isinstance(request_ids, str):
            ids = [request_ids]
        else:
            ids = list(request_ids)
        return [self._owned.pop(rid) for rid in ids if rid in self._owned]


def _make_coordinator(lanes, *, per_lane_max=32, log_stats=False, **interleave_keys):
    coordinator = TTLaneCoordinator.__new__(TTLaneCoordinator)
    coordinator.lanes = lanes
    # The coordinator owns a single decode-interleave policy for the whole
    # step. Disabled unless a test passes bounds, so the mode negotiation
    # under test is the plain prefill-wins rule.
    coordinator._decode_interleave = TTDecodeInterleavePolicy(
        SimpleNamespace(
            additional_config={
                "tt": interleave_keys or {"decode_interleave_enabled": False}
            }
        )
    )
    coordinator.num_lanes = len(lanes)
    coordinator._per_lane_max = per_lane_max
    coordinator.log_stats = log_stats
    coordinator.structured_output_manager = None
    coordinator.connector = None
    coordinator._req_to_lane = {}
    coordinator._req_to_row = {}
    coordinator._uncached_tokens = {}
    coordinator._prefix_cache_routing = True
    coordinator._free_slots_by_lane = [
        list(range(per_lane_max)) for _ in range(len(lanes))
    ]
    return coordinator


def _scheduled_output(req_ids):
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = {req_id: 1 for req_id in req_ids}
    out.total_num_scheduled_tokens = len(req_ids)
    return out


def _preempting_output(preempted, scheduled=()):
    out = _scheduled_output(scheduled)
    out.preempted_req_ids = set(preempted)
    return out


def test_negotiate_prefill_when_any_lane_wants_prefill():
    # Lane 1 has a queued request and nothing running -> wants prefill.
    coordinator = _make_coordinator([FakeLane(running=2), FakeLane(waiting=1)])
    assert coordinator._negotiate_forced_mode() == TTSchedulingMode.PREFILL_ONLY


def test_negotiate_decode_when_no_lane_wants_prefill():
    coordinator = _make_coordinator([FakeLane(running=2), FakeLane(running=1)])
    assert coordinator._negotiate_forced_mode() == TTSchedulingMode.DECODE_ONLY


def test_negotiate_prefill_when_lane_has_only_grammar_blocked_request():
    # Lane 1's only pending work is a grammar-blocked structured-output request
    # held in skipped_waiting (waiting is empty). It must still force prefill so
    # the base scheduler can revisit and promote it after the grammar is ready.
    coordinator = _make_coordinator([FakeLane(running=2), FakeLane(skipped_waiting=1)])
    assert coordinator._negotiate_forced_mode() == TTSchedulingMode.PREFILL_ONLY


def test_negotiate_prefill_for_running_continuation_at_lane_capacity():
    # The lane's only work is a partial prefill occupying its one slot: nothing
    # waiting and no spare capacity, yet only a prefill step can advance it.
    coordinator = _make_coordinator([FakeLane(partial_prefills=1)], per_lane_max=1)

    assert coordinator._negotiate_forced_mode() == TTSchedulingMode.PREFILL_ONLY


def test_decode_interleave_overrides_a_prefill_intent_once_the_bound_is_hit():
    # One lane holds a multi-chunk prefill it cannot finish in one step; the
    # other lane is decoding. Without the interleave every step is prefill and
    # the decoding lane stalls for the whole prompt.
    lanes = [FakeLane(partial_prefills=1), FakeLane(running=2)]
    coordinator = _make_coordinator(
        lanes,
        decode_interleave_prefill_steps=2,
        decode_interleave_decode_steps=1,
    )

    # schedule() advances the counters in production; negotiation alone does
    # not, so drive them the way the step loop does.
    modes = []
    for _ in range(9):
        mode = coordinator._negotiate_forced_mode()
        coordinator._decode_interleave.record_step(
            is_decode=mode == TTSchedulingMode.DECODE_ONLY, prefill_pending=True
        )
        modes.append(mode)

    P, D = TTSchedulingMode.PREFILL_ONLY, TTSchedulingMode.DECODE_ONLY
    assert modes == [P, P, D, P, P, D, P, P, D]


def test_decode_interleave_needs_a_lane_with_a_genuine_running_decode():
    # Every running request across both lanes is a partial-prefill
    # continuation. A decode step samples nothing for those, so the interleave
    # would only produce empty steps.
    lanes = [FakeLane(partial_prefills=1), FakeLane(partial_prefills=1)]
    coordinator = _make_coordinator(lanes, decode_interleave_prefill_steps=1)

    for _ in range(4):
        mode = coordinator._negotiate_forced_mode()
        coordinator._decode_interleave.record_step(
            is_decode=mode == TTSchedulingMode.DECODE_ONLY, prefill_pending=True
        )
        assert mode == TTSchedulingMode.PREFILL_ONLY


def test_schedule_records_the_step_phase_for_the_interleave_policy():
    # The counters must follow the mode the step actually ran, which schedule()
    # settles only after the zero-token prefill fallback has had its say.
    # FakeLane always schedules zero prefill tokens, so this step falls back to
    # decode and must be recorded as a decode step.
    lanes = [FakeLane(waiting=1, running=1)]
    coordinator = _make_coordinator(lanes, decode_interleave_prefill_steps=1)

    coordinator.schedule()

    assert lanes[0].scheduled_modes == [
        TTSchedulingMode.PREFILL_ONLY,
        TTSchedulingMode.DECODE_ONLY,
    ]
    # The fallback decode consumed the allowance, so the next step prefills.
    assert not coordinator._decode_interleave.wants_decode_step(
        has_pending_prefill=True, has_running_decode=True
    )


def test_idle_step_propagates_finished_req_ids():
    # No lane has work, but one lane still has a finished request to report.
    lanes = [FakeLane(pending_finished={"done-0"}), FakeLane()]
    coordinator = _make_coordinator(lanes)

    output = coordinator.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert output.finished_req_ids == {"done-0"}
    # No lane wanted prefill, so the step is decode-only.
    assert get_tt_step_plan(output).is_decode is True
    # Every lane is scheduled (so each drains its own finished set).
    assert lanes[0].scheduled_modes == [TTSchedulingMode.DECODE_ONLY]
    assert lanes[1].scheduled_modes == [TTSchedulingMode.DECODE_ONLY]


def test_step_plan_reports_exact_lane_layout_changes():
    lane = FakeLane(running=1)
    coordinator = _make_coordinator([lane], per_lane_max=2)

    first_decode = coordinator.schedule()
    assert get_tt_step_plan(first_decode).decode_layout_changed

    # The same request keeps the same stable lane row.
    second_decode = coordinator.schedule()
    assert not get_tt_step_plan(second_decode).decode_layout_changed

    # A live request omitted from a prefill step still owns its row; unlike a
    # front-packed batch, unscheduled membership is not a layout transition.
    empty = SchedulerOutput.make_empty()
    prefill_plan = coordinator._build_step_plan([empty], empty, is_decode=False)
    assert not prefill_plan.decode_layout_changed


def test_decode_fallback_when_forced_prefill_schedules_nothing():
    # Lane 0 has running decodes (and a finished req to report); lane 1 has a
    # queued request that forces prefill. Prefill schedules nothing, so the
    # coordinator must fall back to decode to make progress.
    lane0 = FakeLane(running=2, pending_finished={"done-0"})
    lane1 = FakeLane(waiting=1)
    coordinator = _make_coordinator([lane0, lane1])

    output = coordinator.schedule()

    # Fell back to decode: lane 0's two decodes are scheduled.
    assert output.total_num_scheduled_tokens == 2
    assert get_tt_step_plan(output).is_decode is True
    # Finished IDs drained during the discarded prefill pass are carried over.
    assert output.finished_req_ids == {"done-0"}
    # Lane 0 was scheduled once for prefill, then again for the decode fallback.
    assert lane0.scheduled_modes == [
        TTSchedulingMode.PREFILL_ONLY,
        TTSchedulingMode.DECODE_ONLY,
    ]


def test_decode_fallback_carries_forced_reset_discard_counts():
    lane0 = FakeLane(running=1)
    lane1 = FakeLane(waiting=1)
    coordinator = _make_coordinator([lane0, lane1])
    lane_schedule = lane0.schedule

    def _schedule():
        output = lane_schedule()
        if lane0.scheduled_modes[-1] == TTSchedulingMode.PREFILL_ONLY:
            set_tt_forced_reset_discard_counts(output, {"stale": 2})
        return output

    lane0.schedule = _schedule

    output = coordinator.schedule()

    assert get_tt_forced_reset_discard_counts(output) == {"stale": 2}


def test_no_fallback_when_no_running_requests():
    # Forced prefill schedules nothing and there are no running decodes
    # anywhere: nothing to fall back to, so the step stays prefill (empty).
    lane0 = FakeLane(waiting=1)
    lane1 = FakeLane(waiting=1)
    coordinator = _make_coordinator([lane0, lane1])

    output = coordinator.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert get_tt_step_plan(output).is_decode is False
    # Only the prefill pass ran (no decode fallback).
    assert lane0.scheduled_modes == [TTSchedulingMode.PREFILL_ONLY]


def test_no_decode_fallback_when_only_continuations_are_running():
    # The only running request is a partial prefill. A decode step cannot
    # advance it, so falling back to decode would just burn a step.
    lane = FakeLane(partial_prefills=1)
    coordinator = _make_coordinator([lane])

    output = coordinator.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert get_tt_step_plan(output).is_decode is False
    assert lane.scheduled_modes == [TTSchedulingMode.PREFILL_ONLY]


def test_preemption_in_a_discarded_prefill_pass_is_carried_to_the_decode_step():
    # A prefill pass keeps partial prefills in ``running``, so its running loop
    # can preempt one -- and that preemption stands even when the pass's
    # schedule is thrown away for the decode fallback, because
    # ``_preempt_request`` already freed the KV. Dropping the report would
    # leave the row claimed by a request the lane has put back in its queue.
    lane = FakeLane(running=1, partial_prefills=1)
    coordinator = _make_coordinator([lane], per_lane_max=2)
    coordinator._req_to_lane = {"p": 0}
    coordinator._assign_slot("p", 0)
    lane_schedule = lane.schedule

    def _schedule():
        out = lane_schedule()
        if lane.scheduled_modes[-1] == TTSchedulingMode.PREFILL_ONLY:
            out.preempted_req_ids = {"p"}
        return out

    lane.schedule = _schedule

    output = coordinator.schedule()

    assert lane.scheduled_modes == [
        TTSchedulingMode.PREFILL_ONLY,
        TTSchedulingMode.DECODE_ONLY,
    ]
    assert output.preempted_req_ids == {"p"}
    assert "p" not in coordinator._req_to_row
    assert "p" not in get_tt_step_plan(output).req_id_to_row


def _structured_request(*, is_prefill_chunk):
    return SimpleNamespace(
        use_structured_output=True, is_prefill_chunk=is_prefill_chunk
    )


def _recording_structured_output_manager(recorded):
    def grammar_bitmask(requests, req_ids, spec_decode_tokens):
        recorded.append(list(req_ids))
        return "bitmask"

    return SimpleNamespace(grammar_bitmask=grammar_bitmask)


def test_grammar_bitmask_skips_intermediate_prefill_chunks():
    # An intermediate chunk samples no token, so giving it a bitmask row would
    # offset every later row against the batch the runner remaps onto.
    lane = FakeLane()
    lane.requests = {
        "chunk": _structured_request(is_prefill_chunk=True),
        "decode": _structured_request(is_prefill_chunk=False),
    }
    coordinator = _make_coordinator([lane])
    recorded: list[list[str]] = []
    coordinator.structured_output_manager = _recording_structured_output_manager(
        recorded
    )

    scheduler_output = _scheduled_output(["chunk", "decode"])
    # The base scheduler raises the flag for "decode": a structured request that
    # is past its prefill chunks.
    scheduler_output.has_structured_output_requests = True

    grammar_output = coordinator.get_grammar_bitmask(scheduler_output)

    assert recorded == [["decode"]]
    assert grammar_output.structured_output_request_ids == ["decode"]


def test_grammar_bitmask_is_none_when_only_prefill_chunks_are_scheduled():
    lane = FakeLane()
    lane.requests = {"chunk": _structured_request(is_prefill_chunk=True)}
    coordinator = _make_coordinator([lane])
    recorded: list[list[str]] = []
    coordinator.structured_output_manager = _recording_structured_output_manager(
        recorded
    )

    assert coordinator.get_grammar_bitmask(_scheduled_output(["chunk"])) is None
    assert recorded == []


def test_update_from_output_routes_and_merges_per_lane():
    lane0 = FakeLane()
    lane1 = FakeLane()
    lane0._eco = {0: EngineCoreOutputs(outputs=["a"])}
    lane1._eco = {0: EngineCoreOutputs(outputs=["b"], finished_requests={"x"})}
    coordinator = _make_coordinator([lane0, lane1])
    scheduler_output = coordinator.schedule()

    merged = coordinator.update_from_output(scheduler_output, model_runner_output=None)

    # Each lane received its own SchedulerOutput.
    assert lane0.update_calls == [scheduler_output._tt_step_state.lane_outputs[0]]
    assert lane1.update_calls == [scheduler_output._tt_step_state.lane_outputs[1]]
    # Per-client outputs concatenated and finished sets unioned.
    assert merged[0].outputs == ["a", "b"]
    assert merged[0].finished_requests == {"x"}
    # Stats disabled -> none attached.
    assert merged[0].scheduler_stats is None


def test_update_from_output_no_metadata_returns_empty():
    coordinator = _make_coordinator([FakeLane(), FakeLane()])
    assert coordinator.update_from_output(SchedulerOutput.make_empty(), None) == {}


# --------------------------------------------------------------------------
# merge_lane_scheduler_outputs (per-field stitching of non-empty lane outputs)
# --------------------------------------------------------------------------


def test_merge_combines_nonempty_lane_outputs():
    # Two lanes each scheduling a cached (decode) request, plus new requests,
    # preemption/invalid-spec reported by only one lane, and per-lane common
    # prefix block counts.
    lane0 = SchedulerOutput.make_empty()
    lane0.scheduled_new_reqs = ["new-a"]
    lane0.scheduled_cached_reqs = CachedRequestData(
        req_ids=["a"],
        resumed_req_ids={"a"},
        new_token_ids=[[11]],
        all_token_ids={"a": [1, 11]},
        new_block_ids=[([0],)],
        num_computed_tokens=[1],
        num_output_tokens=[1],
    )
    lane0.num_scheduled_tokens = {"a": 1}
    lane0.total_num_scheduled_tokens = 1
    lane0.num_common_prefix_blocks = [2, 1]
    lane0.finished_req_ids = {"fin-0"}
    lane0.free_encoder_mm_hashes = ["h0"]
    lane0.preempted_req_ids = {"pre-0"}
    lane0.num_invalid_spec_tokens = {"a": 3}
    set_tt_forced_reset_discard_counts(lane0, {"a": 1})

    lane1 = SchedulerOutput.make_empty()
    lane1.scheduled_new_reqs = ["new-b", "new-c"]
    lane1.scheduled_cached_reqs = CachedRequestData(
        req_ids=["b"],
        resumed_req_ids=set(),
        new_token_ids=[[22]],
        all_token_ids={"b": [2, 22]},
        new_block_ids=[([0],)],
        num_computed_tokens=[5],
        num_output_tokens=[2],
    )
    lane1.num_scheduled_tokens = {"b": 1}
    lane1.total_num_scheduled_tokens = 1
    lane1.num_common_prefix_blocks = [1, 3]
    lane1.finished_req_ids = {"fin-1"}
    lane1.free_encoder_mm_hashes = ["h1"]
    # lane1 reports no preemption and no invalid spec tokens.
    set_tt_forced_reset_discard_counts(lane1, {"b": 2})

    merged = merge_lane_scheduler_outputs([lane0, lane1])

    # New requests concatenated in lane order.
    assert merged.scheduled_new_reqs == ["new-a", "new-b", "new-c"]
    # Cached struct-of-arrays concatenated field-by-field in lane order.
    c = merged.scheduled_cached_reqs
    assert c.req_ids == ["a", "b"]
    assert c.new_token_ids == [[11], [22]]
    assert c.new_block_ids == [([0],), ([0],)]
    assert c.num_computed_tokens == [1, 5]
    assert c.num_output_tokens == [1, 2]
    assert c.all_token_ids == {"a": [1, 11], "b": [2, 22]}
    assert c.resumed_req_ids == {"a"}
    # Scheduled-token dicts merged; total recomputed from the merged dict.
    assert merged.num_scheduled_tokens == {"a": 1, "b": 1}
    assert merged.total_num_scheduled_tokens == 2
    # Elementwise max across the (equal-length) per-lane prefix-block lists.
    assert merged.num_common_prefix_blocks == [2, 3]
    # Finished unioned, free-encoder hashes extended.
    assert merged.finished_req_ids == {"fin-0", "fin-1"}
    assert merged.free_encoder_mm_hashes == ["h0", "h1"]
    # Optional fields reported by exactly one lane survive the merge.
    assert merged.preempted_req_ids == {"pre-0"}
    assert merged.num_invalid_spec_tokens == {"a": 3}
    assert get_tt_forced_reset_discard_counts(merged) == {"a": 1, "b": 2}


def test_merge_preserves_none_for_absent_optional_fields():
    # No lane reports preemption or invalid spec tokens: the merged output keeps
    # None (the base output's "absent" representation), not an empty set/dict.
    lane0 = SchedulerOutput.make_empty()
    lane0.num_scheduled_tokens = {"a": 1}
    lane0.total_num_scheduled_tokens = 1
    lane1 = SchedulerOutput.make_empty()

    merged = merge_lane_scheduler_outputs([lane0, lane1])

    assert merged.preempted_req_ids is None
    assert merged.num_invalid_spec_tokens is None


def test_merge_raises_on_mismatched_common_prefix_block_counts():
    # Lanes share one kv_cache_config, so their num_common_prefix_blocks lists
    # must be equal length (one entry per KV cache group). A mismatch is an
    # upstream bug; the merge must raise, not silently drop the extra groups.
    lane0 = SchedulerOutput.make_empty()
    lane0.num_common_prefix_blocks = [2, 1]
    lane1 = SchedulerOutput.make_empty()
    lane1.num_common_prefix_blocks = [1, 3, 5]

    try:
        merge_lane_scheduler_outputs([lane0, lane1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched group counts")


def test_schedule_attaches_runner_step_plan_with_stable_rows():
    lane0 = FakeLane()
    lane1 = FakeLane()
    coordinator = _make_coordinator([lane0, lane1], per_lane_max=4)
    lane0.schedule = lambda: _scheduled_output(["a", "b"])
    lane1.schedule = lambda: _scheduled_output(["c"])
    coordinator._req_to_lane = {"a": 0, "b": 0, "c": 1}
    coordinator._assign_slot("a", 0)
    coordinator._assign_slot("b", 0)
    coordinator._assign_slot("c", 1)

    output = coordinator.schedule()
    plan = get_tt_step_plan(output)

    assert isinstance(plan, TTStepPlan)
    assert plan.is_decode is True
    assert plan.scheduled_rows == (0, 1, 4)
    assert plan.scheduled_req_ids == ("a", "b", "c")
    assert plan.input_rows == tuple(range(8))
    assert plan.batch_size_per_dp == (4, 4)
    assert plan.prefill_empty_slots is None


def test_prefill_step_plan_exposes_empty_slots_without_lane_metadata():
    lane0 = FakeLane()
    lane1 = FakeLane(waiting=1)
    coordinator = _make_coordinator([lane0, lane1], per_lane_max=4)
    lane0.schedule = SchedulerOutput.make_empty
    lane1.schedule = lambda: _scheduled_output(["a"])
    coordinator._req_to_lane = {"a": 1}

    output = coordinator.schedule()
    plan = get_tt_step_plan(output)

    assert isinstance(plan, TTStepPlan)
    assert plan.is_decode is False
    assert plan.scheduled_rows == (4,)
    assert plan.scheduled_req_ids == ("a",)
    assert plan.input_rows == (4,)
    assert plan.batch_size_per_dp == (0, 1)
    assert plan.prefill_empty_slots == (4,)


def test_preempted_request_releases_its_row_to_its_lane():
    # Lane 0 is full (capacity 2) with "a" and "b", then preempts "a" under KV
    # pressure while decoding "b". The preempted request is neither finished nor
    # resumed, so without an explicit release it would hold row 0 until it
    # resumes -- yet its running slot is already back, so the lane can admit a
    # replacement. Under --scheduling-policy priority a higher-priority arrival
    # is admitted ahead of the preempted request and hits the empty free list.
    lane0 = FakeLane(running=2)
    coordinator = _make_coordinator([lane0, FakeLane()], per_lane_max=2)
    coordinator._req_to_lane = {"a": 0, "b": 0}
    assert coordinator._assign_slot("a", 0) == 0
    assert coordinator._assign_slot("b", 0) == 1

    lane0.schedule = lambda: _preempting_output(["a"], scheduled=["b"])
    coordinator.schedule()

    assert "a" not in coordinator._req_to_row
    assert coordinator._free_slots_by_lane[0] == [0]
    # The lane binding survives: the request resumes in the lane whose KV cache
    # manager and queues still hold it.
    assert coordinator._req_to_lane["a"] == 0

    # The freed row is handed to the newcomer instead of raising "no free slot".
    lane0.schedule = lambda: _scheduled_output(["c"])
    coordinator._req_to_lane["c"] = 0
    plan = get_tt_step_plan(coordinator.schedule())

    assert plan.req_id_to_row["c"] == 0
    assert "a" not in plan.req_id_to_row


def test_preempted_and_finished_in_one_step_frees_the_row_once():
    # A preempted request aborted in the same step is in both sets. The row must
    # come back exactly once, or the lane's free list would hand it out twice.
    lane0 = FakeLane(running=1)
    coordinator = _make_coordinator([lane0, FakeLane()], per_lane_max=2)
    coordinator._req_to_lane = {"a": 0}
    coordinator._assign_slot("a", 0)

    def _schedule():
        out = _preempting_output(["a"])
        out.finished_req_ids = {"a"}
        return out

    lane0.schedule = _schedule
    coordinator.schedule()

    assert coordinator._free_slots_by_lane[0] == [0, 1]
    assert coordinator._req_to_lane == {}


def test_per_lane_vllm_config_uses_per_lane_max_num_seqs():
    # Lanes must be constructed from a config whose max_num_seqs is the
    # *per-lane* cap, so the base scheduler derives max_num_running_reqs ==
    # per_lane at __init__ (Scheduler.__init__:
    # self.max_num_running_reqs = scheduler_config.max_num_seqs). Without this,
    # four lanes built from the global cap (32) would each believe they may run
    # the whole global batch, letting their combined running set reach 128 and
    # overflow the runner's merged persistent batch (req_index >= max_num_reqs).
    coordinator = TTLaneCoordinator.__new__(TTLaneCoordinator)
    coordinator._per_lane_max = 8
    global_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=32))

    per_lane_config = coordinator._build_per_lane_vllm_config(global_config)

    # The lanes' config carries the per-lane cap...
    assert per_lane_config.scheduler_config.max_num_seqs == 8
    # ...while the shared global config the coordinator/runner read for KV and
    # model sizing is left untouched (copied, not aliased/mutated).
    assert global_config.scheduler_config.max_num_seqs == 32
    assert per_lane_config.scheduler_config is not global_config.scheduler_config


def _req(rid, client_index=0):
    return SimpleNamespace(request_id=rid, client_index=client_index)


@pytest.mark.parametrize(
    "request_ids, expected_ids",
    [
        (None, ["a0", "a1", "b0"]),  # None -> every held request
        (["a1", "b0", "gone"], ["a1", "b0"]),  # spans lanes; unheld id no-ops
        ("a0", ["a0"]),  # single str id
    ],
)
def test_finish_requests_returns_only_owned_no_duplicates(request_ids, expected_ids):
    # Each lane returns only the requests it holds (a request lives in exactly one
    # lane), so the coordinator's concatenation is the aborted set with no
    # duplicates.
    a0, a1, b0 = _req("a0", 0), _req("a1", 1), _req("b0", 0)
    owned = {"a0": a0, "a1": a1, "b0": b0}
    coordinator = _make_coordinator([FakeLane(owns=[a0, a1]), FakeLane(owns=[b0])])

    aborted = coordinator.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    assert aborted == [owned[i] for i in expected_ids]  # exact objects, in lane order
    assert len({id(r) for r in aborted}) == len(aborted)  # no request returned twice


def test_finish_requests_result_is_routable_per_client():
    # The returned Requests must carry client_index so engine-core's
    # _send_abort_outputs can notify each client. True end-to-end notification is
    # covered device-side in tests/tt.
    lanes = [FakeLane(owns=[_req("a", 0), _req("b", 1)]), FakeLane(owns=[_req("c", 0)])]
    coordinator = _make_coordinator(lanes)

    aborted = coordinator.finish_requests(None, RequestStatus.FINISHED_ABORTED)

    by_client = collections.defaultdict(set)
    for r in aborted:
        by_client[r.client_index].add(r.request_id)
    assert by_client == {0: {"a", "c"}, 1: {"b"}}


# ---------------------------------------------------------------------------
# Prefix-cache lane routing
# ---------------------------------------------------------------------------


class FakeKVCacheManager:
    """Prefix cache of one lane: the tokens it holds per request hash."""

    def __init__(self, cached_tokens_by_hash=None, enable_caching=True):
        self.enable_caching = enable_caching
        self._cached = dict(cached_tokens_by_hash or {})
        self.coordinator = self
        self.lookups = 0

    def prefix_cache_lookup_enabled(self, request):
        return self.enable_caching and not request.skip_reading_prefix_cache

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        self.lookups += 1
        hit = min(self._cached.get(block_hashes, 0), max_cache_hit_length)
        return ((), hit, 0)


def _lane_with_cache(waiting=0, running=0, **cached):
    """A lane whose ``add_request`` queues the request, as a real lane does."""
    lane = FakeLane(waiting=waiting, running=running)
    lane.kv_cache_manager = FakeKVCacheManager(cached)
    lane.add_request = lane.waiting.append
    return lane


def _request(request_id, num_tokens=4096, prefix="", skip_reading_prefix_cache=False):
    # ``block_hashes`` stands in for the prompt's block-hash list; the fake
    # manager keys its cache on it.
    return SimpleNamespace(
        request_id=request_id,
        block_hashes=prefix or request_id,
        num_tokens=num_tokens,
        skip_reading_prefix_cache=skip_reading_prefix_cache,
    )


def _routing_coordinator(lanes, routing=True, per_lane_max=32):
    coordinator = _make_coordinator(lanes, per_lane_max=per_lane_max)
    coordinator._prefix_cache_routing = routing
    return coordinator


def _lane_of(coordinator, request):
    coordinator.add_request(request)
    return coordinator._req_to_lane[request.request_id]


def test_request_is_routed_to_the_lane_holding_its_cached_prefix():
    lanes = [_lane_with_cache(), _lane_with_cache(), _lane_with_cache(doc=3968)]
    coordinator = _routing_coordinator(lanes)
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 2
    assert [len(lane.waiting) for lane in lanes] == [0, 0, 1]


def test_the_lane_holding_the_longest_prefix_wins():
    lanes = [_lane_with_cache(doc=1024), _lane_with_cache(doc=3072), _lane_with_cache()]
    coordinator = _routing_coordinator(lanes)
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 1


def test_uncached_prompt_goes_to_the_least_loaded_lane():
    lanes = [
        _lane_with_cache(waiting=1),
        _lane_with_cache(running=2),
        _lane_with_cache(running=3),
    ]
    coordinator = _routing_coordinator(lanes)
    assert _lane_of(coordinator, _request("cold")) == 1
    assert coordinator._uncached_tokens["cold"] == 4096


def test_a_burst_of_followers_of_a_long_prefix_all_join_its_lane():
    # 20 questions on one cached 3968-token document: each queues only its 128
    # uncached tokens, far below the prefix they reuse, so none spills.
    lanes = [_lane_with_cache(doc=3968), _lane_with_cache(), _lane_with_cache()]
    coordinator = _routing_coordinator(lanes)
    picks = [_lane_of(coordinator, _request(f"q{i}", prefix="doc")) for i in range(20)]
    assert picks == [0] * 20
    assert coordinator._uncached_tokens["q19"] == 128


def test_followers_of_a_short_prefix_spill_once_the_queue_outweighs_the_hit():
    # A 256-token shared system prompt in 1024-token prompts: the first follower
    # queues 768 uncached tokens on lane 0, more than the 256 the next one would
    # save, so the next one takes the least-loaded lane.
    lanes = [_lane_with_cache(sys=256), _lane_with_cache(), _lane_with_cache()]
    coordinator = _routing_coordinator(lanes)
    picks = [
        _lane_of(coordinator, _request(f"q{i}", num_tokens=1024, prefix="sys"))
        for i in range(3)
    ]
    assert picks == [0, 1, 2]


def test_a_queue_of_uncached_work_ahead_sends_the_follower_elsewhere():
    lanes = [_lane_with_cache(doc=3968), _lane_with_cache()]
    coordinator = _routing_coordinator(lanes)
    # Two cold 8K prompts queued on lane 0 (routed there as the least-loaded).
    for i in range(2):
        lanes[0].waiting.append(_request(f"cold{i}", num_tokens=8192))
        coordinator._uncached_tokens[f"cold{i}"] = 8192
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 1


def test_a_full_cached_lane_loses_to_a_lane_with_a_free_slot():
    lanes = [_lane_with_cache(running=2, doc=3968), _lane_with_cache(running=1)]
    coordinator = _routing_coordinator(lanes, per_lane_max=2)
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 1
    # Both full: the cached lane still wins (the wait is the same either way).
    lanes = [_lane_with_cache(running=2, doc=3968), _lane_with_cache(running=2)]
    coordinator = _routing_coordinator(lanes, per_lane_max=2)
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 0


def test_routing_off_restores_least_loaded_routing_without_lookups():
    lanes = [_lane_with_cache(), _lane_with_cache(doc=3968)]
    coordinator = _routing_coordinator(lanes, routing=False)
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 0
    assert all(lane.kv_cache_manager.lookups == 0 for lane in lanes)


def test_requests_that_skip_the_prefix_cache_read_are_load_routed():
    lanes = [_lane_with_cache(), _lane_with_cache(doc=3968)]
    coordinator = _routing_coordinator(lanes)
    request = _request("q", prefix="doc", skip_reading_prefix_cache=True)
    assert _lane_of(coordinator, request) == 0


def test_finished_requests_drop_their_queued_prefill_record():
    finishing = FakeLane(pending_finished={"q"})
    finishing.kv_cache_manager = FakeKVCacheManager()
    coordinator = _routing_coordinator([finishing, _lane_with_cache()])
    coordinator._req_to_lane["q"] = 0
    coordinator._uncached_tokens["q"] = 128
    coordinator.schedule()
    assert "q" not in coordinator._uncached_tokens


def _partial_prefill(request_id, num_tokens, num_computed_tokens):
    """A running request still inside its prompt (a chunked prefill between steps)."""
    return SimpleNamespace(
        request_id=request_id,
        is_prefill_chunk=True,
        num_tokens=num_tokens,
        num_computed_tokens=num_computed_tokens,
    )


def test_equal_cache_hits_resolve_on_load_not_on_the_lower_lane_index():
    # Both lanes hold the same 3968-token prefix: neither saves anything over
    # the other, so the 512 tokens queued on lane 0 decide against it.
    lanes = [_lane_with_cache(doc=3968), _lane_with_cache(doc=3968)]
    coordinator = _routing_coordinator(lanes)
    lanes[0].waiting.append(_request("earlier", num_tokens=512))
    coordinator._uncached_tokens["earlier"] = 512
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 1
    assert coordinator._uncached_tokens["q"] == 128


def test_a_slightly_longer_hit_must_outweigh_the_queue_ahead():
    # Lane 0 holds 3968 of the prompt, lane 1 3840: lane 0 saves only the 128
    # tokens between them, less than the 512 queued ahead on it.
    lanes = [_lane_with_cache(doc=3968), _lane_with_cache(doc=3840)]
    coordinator = _routing_coordinator(lanes)
    lanes[0].waiting.append(_request("earlier", num_tokens=512))
    coordinator._uncached_tokens["earlier"] = 512
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 1
    assert coordinator._uncached_tokens["q"] == 4096 - 3840
    # A queue of 100 is within the 128 the longer hit saves: lane 0 wins.
    coordinator._uncached_tokens["earlier"] = 100
    assert _lane_of(coordinator, _request("q2", prefix="doc")) == 0


def test_a_running_partial_prefill_counts_as_work_ahead():
    # Lane 0 is 1024 tokens into an 8192-token prompt and holds a 1024-token
    # prefix of the newcomer; the 7168 tokens it still owes come first.
    lanes = [_lane_with_cache(doc=1024), _lane_with_cache()]
    lanes[0].running.append(_partial_prefill("long", 8192, 1024))
    coordinator = _routing_coordinator(lanes)
    assert coordinator._queued_prefill_tokens(0) == 7168
    assert _lane_of(coordinator, _request("q", num_tokens=2048, prefix="doc")) == 1
    # Once the partial prefill is through its prompt the hit draws the next one.
    lanes[0].running[-1].num_computed_tokens = 8192
    assert _lane_of(coordinator, _request("q2", num_tokens=2048, prefix="doc")) == 0


def test_a_prefix_cache_reset_drops_the_admission_time_estimates():
    lanes = [_lane_with_cache(doc=3968), _lane_with_cache()]
    coordinator = _routing_coordinator(lanes)
    victim = _request("victim", num_tokens=4096, prefix="doc")
    assert _lane_of(coordinator, victim) == 0
    assert coordinator._uncached_tokens["victim"] == 128
    victim.num_tokens = 4097  # one generated token before the reset
    assert coordinator.reset_prefix_cache(reset_running_requests=True) is True
    assert all(lane.reset_calls == [(True, False)] for lane in lanes)
    # The victim recomputes its whole prompt: 4097 tokens ahead, not 128.
    assert coordinator._uncached_tokens == {}
    assert coordinator._queued_prefill_tokens(0) == 4097
    # A newcomer with a 1024-token hit on lane 0 now finds 4097 ahead of it.
    lanes[0].kv_cache_manager._cached = {"sys": 1024}
    assert _lane_of(coordinator, _request("q", num_tokens=2048, prefix="sys")) == 1


def test_a_preempted_request_drops_its_admission_time_estimate():
    lane = FakeLane()
    lane.kv_cache_manager = FakeKVCacheManager()
    coordinator = _routing_coordinator([lane, _lane_with_cache()])
    coordinator._req_to_lane["victim"] = 0
    coordinator._uncached_tokens["victim"] = 128
    lane.schedule = lambda: _preempting_output(preempted=["victim"])
    coordinator.schedule()
    assert "victim" not in coordinator._uncached_tokens


def test_a_burst_above_the_lane_capacity_spills_once_its_slots_are_promised():
    # 64 followers of an 8064-token prefix on 32-slot lanes: the first 32 fill
    # lane 0's slots (queued requests count), the rest take the empty lane.
    lanes = [_lane_with_cache(doc=8064), _lane_with_cache()]
    coordinator = _routing_coordinator(lanes, per_lane_max=32)
    picks = [
        _lane_of(coordinator, _request(f"q{i}", num_tokens=8192, prefix="doc"))
        for i in range(64)
    ]
    assert picks == [0] * 32 + [1] * 32
    assert [len(lane.waiting) for lane in lanes] == [32, 32]


def test_equal_cached_prefixes_resolve_by_load_among_the_cached_lanes():
    # Lanes 0 and 1 both hold the 3968-token prefix, lane 2 holds nothing and is
    # the least loaded overall: the tie between the cached lanes goes to the less
    # loaded lane 1, not to the lower index.
    lanes = [
        _lane_with_cache(doc=3968),
        _lane_with_cache(running=1, doc=3968),
        _lane_with_cache(),
    ]
    coordinator = _routing_coordinator(lanes)
    for i in range(2):
        lanes[0].waiting.append(_request(f"ahead{i}", num_tokens=256))
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 1


def _skipped(request_id, status, *, grammar=None, num_tokens=8192):
    structured = (
        None
        if grammar is None
        else SimpleNamespace(grammar=grammar if grammar != "pending" else None)
    )
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=num_tokens,
        status=status,
        structured_output_request=structured,
    )


def test_ready_skipped_waiting_requests_count_as_prefill_work_ahead():
    # A grammar-blocked request whose grammar has compiled goes first at the next
    # prefill step: its 8192 tokens outweigh the 3968-token hit on lane 0.
    lanes = [_lane_with_cache(doc=3968), _lane_with_cache()]
    lanes[0].skipped_waiting.append(
        _skipped(
            "ready",
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            grammar=object(),
        )
    )
    coordinator = _routing_coordinator(lanes)
    assert coordinator._queued_prefill_tokens(0) == 8192
    assert _lane_of(coordinator, _request("q", prefix="doc")) == 1
    # Still compiling: not ahead of the newcomer, the hit draws it to lane 0.
    lanes[0].skipped_waiting[0] = _skipped(
        "pending",
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        grammar="pending",
    )
    assert coordinator._queued_prefill_tokens(0) == 0
    assert _lane_of(coordinator, _request("q2", prefix="doc")) == 0
    # q2 now waits on lane 0 with its 128 uncached tokens. Remote KVs that have
    # arrived count on top of it; a streaming-input request does not.
    lanes[0].finished_recving_kv_req_ids = {"kv"}
    lanes[0].skipped_waiting[0] = _skipped("kv", RequestStatus.WAITING_FOR_REMOTE_KVS)
    assert coordinator._queued_prefill_tokens(0) == 8192 + 128
    lanes[0].skipped_waiting[0] = _skipped(
        "stream", RequestStatus.WAITING_FOR_STREAMING_REQ
    )
    assert coordinator._queued_prefill_tokens(0) == 128


def test_lane_prefix_cache_routing_config_key():
    from vllm_tt_plugin.config import get_tt_lane_prefix_cache_routing

    assert (
        get_tt_lane_prefix_cache_routing(SimpleNamespace(additional_config={})) is True
    )
    off = SimpleNamespace(
        additional_config={"tt": {"lane_prefix_cache_routing": False}}
    )
    assert get_tt_lane_prefix_cache_routing(off) is False
    with pytest.raises(ValueError):
        get_tt_lane_prefix_cache_routing(
            SimpleNamespace(additional_config={"tt": {"lane_prefix_cache_routing": 1}})
        )
