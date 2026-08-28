# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only coverage for the explicit TT decode reload contract."""

import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import CachedRequestData

from vllm_tt_plugin.async_decode import (
    CompletedDecodeStep,
    DeferredDecodeOutput,
    SubmittedStepContext,
    TTAsyncDecodeController,
)
from vllm_tt_plugin.model_input import TTDecodeReloadPlan, TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner


def _controller(*, trace_mode="decode_only", supports_async=True):
    runner = SimpleNamespace(
        model=SimpleNamespace(
            model_capabilities={"supports_async_decode": supports_async}
        ),
        trace_mode=trace_mode,
    )
    return TTAsyncDecodeController(runner)


def _decode_input(*, device_sampling=True, layout_changed=False, page=0):
    return SimpleNamespace(
        perform_device_sampling=device_sampling,
        decode_layout_changed=layout_changed,
        block_tables_per_group=[torch.tensor([[page, 0]], dtype=torch.int32)],
    )


def _commit(controller, model_input):
    plan = controller.plan_decode_reload(model_input)
    controller.commit_decode_submission(model_input, plan)
    return plan


def _sampling_params(rows=1):
    return TTSamplingParams(
        temperature=torch.ones(rows),
        top_k=torch.full((rows,), 32),
        top_p=torch.ones(rows),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.full((rows,), -1),
        num_logprobs=torch.full((rows,), -2),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )


def _submission_input(*, device_sampling=True, layout_changed=True, page=0):
    page_table = torch.tensor([[page]], dtype=torch.int32)
    return SimpleNamespace(
        input_tokens=torch.zeros((1, 1), dtype=torch.int32),
        input_positions=torch.zeros((1,), dtype=torch.int32),
        block_tables=page_table,
        block_tables_per_group=[page_table],
        block_tables_per_layer=None,
        unpadded_batch_size=1,
        tt_sampling_params=_sampling_params(),
        perform_device_sampling=device_sampling,
        prompt_tokens=None,
        output_tokens=None,
        decode_layout_changed=layout_changed,
        slot_remap=torch.tensor([0], dtype=torch.int32),
    )


def _accepted_decode_hooks(runner):
    runner.note_decode_layout_consumed = lambda: None
    runner.note_decode_state_slots_settled = lambda: None
    return runner


def _cached_reqs(req_ids, *, context_phase=(), resumed=()):
    req_ids = list(req_ids)
    context_phase = set(context_phase)
    return CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=set(resumed),
        new_token_ids=[[] for _ in req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in req_ids],
        num_computed_tokens=[1 for _ in req_ids],
        num_output_tokens=[0 if req_id in context_phase else 1 for req_id in req_ids],
    )


def test_first_and_steady_device_decode_commands():
    controller = _controller()

    first = _commit(controller, _decode_input(page=1))
    steady = _commit(controller, _decode_input(page=1))

    assert first.reload_inputs
    assert first.reload_sampling_params
    assert first.reset_sampling_state
    assert not first.overlap_safe
    assert not steady.reload_inputs
    assert not steady.reload_page_table
    assert not steady.reload_sampling_params
    assert not steady.reset_sampling_state
    assert steady.overlap_safe


def test_reload_plan_rejects_incoherent_command_combinations():
    with pytest.raises(AssertionError, match="reset_sampling_state requires"):
        TTDecodeReloadPlan(
            reload_inputs=False,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=True,
        )

    with pytest.raises(AssertionError, match="reload_page_table"):
        TTDecodeReloadPlan(
            reload_inputs=True,
            reload_page_table=True,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )


def test_page_table_only_refresh_is_overlap_safe():
    controller = _controller()
    _commit(controller, _decode_input(page=1))

    plan = _commit(controller, _decode_input(page=2))

    assert not plan.reload_inputs
    assert plan.reload_page_table
    assert plan.overlap_safe


def test_submit_decode_delivers_page_table_only_refresh():
    calls = []

    class Model:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": True}

        def decode_forward(self, **kwargs):
            calls.append(kwargs)
            return torch.zeros((1, 1))

    runner = _accepted_decode_hooks(
        SimpleNamespace(
            model=Model(),
            trace_mode="decode_only",
            kv_caches=object(),
            request_specific_rope=False,
        )
    )
    controller = TTAsyncDecodeController(runner)

    controller.submit_decode(
        _submission_input(layout_changed=True, page=1), read_from_device=True
    )
    controller.submit_decode(
        _submission_input(layout_changed=False, page=2), read_from_device=True
    )

    assert calls[1]["reload_inputs"] is False
    assert calls[1]["reload_page_table"] is True
    assert calls[1]["reload_sampling_params"] is False
    assert calls[1]["reset_sampling_state"] is False


def test_v0_preserves_legacy_dp_greater_than_one_disable():
    runner = SimpleNamespace(
        model=SimpleNamespace(),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_size=4),
        async_decode_scheduling=True,
    )

    TTModelRunner._preserve_v0_async_decode_selection(runner)
    assert not runner.async_decode_scheduling


def test_v1_keeps_platform_async_for_dp_greater_than_one():
    runner = SimpleNamespace(
        model=SimpleNamespace(decode_input_update_contract=1),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_size=4),
        async_decode_scheduling=True,
    )

    TTModelRunner._preserve_v0_async_decode_selection(runner)
    assert runner.async_decode_scheduling


def test_v0_keeps_legacy_rank_local_async_decode():
    runner = SimpleNamespace(
        model=SimpleNamespace(),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        async_decode_scheduling=True,
    )

    TTModelRunner._preserve_v0_async_decode_selection(runner)
    assert runner.async_decode_scheduling


def test_disabled_upstream_async_stays_disabled_for_v1():
    runner = SimpleNamespace(
        model=SimpleNamespace(decode_input_update_contract=1),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=SimpleNamespace(data_parallel_size=4),
        async_decode_scheduling=False,
    )

    TTModelRunner._preserve_v0_async_decode_selection(runner)
    assert not runner.async_decode_scheduling


def test_v0_preserves_host_to_device_overlap_predicate():
    common_runner = dict(
        async_decode_scheduling=True,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        trace_mode="decode_only",
    )
    model_input = SimpleNamespace(
        prompt_lens=None,
        perform_device_sampling=True,
        decode_layout_changed=False,
        grammar_bitmask=[None],
        prompt_tokens=None,
        output_tokens=None,
        allowed_token_ids_mask_list=[None],
        bad_words_token_ids_list=[{}],
        max_num_logprobs=[None],
        block_tables_per_group=[torch.zeros((1, 1), dtype=torch.int32)],
    )

    v0 = TTAsyncDecodeController(
        SimpleNamespace(model=SimpleNamespace(), **common_runner)
    )
    v0._decode_chain_valid = True
    v0._previous_device_sampling = False
    assert v0.can_use_steady_decode_fast_path(model_input)

    v1 = TTAsyncDecodeController(
        SimpleNamespace(
            model=SimpleNamespace(
                decode_input_update_contract=1,
                model_capabilities={"supports_async_decode": True},
            ),
            **common_runner,
        )
    )
    v1._decode_chain_valid = True
    v1._previous_device_sampling = False
    assert not v1.can_use_steady_decode_fast_path(model_input)


def test_failed_deferred_readback_is_terminal():
    calls = 0

    class FailedReadback(DeferredDecodeOutput):
        def __init__(self):
            self._completion_event = threading.Event()
            self._init_deferred()

        def _get_output_impl(self):
            nonlocal calls
            calls += 1
            raise RuntimeError("readback failed")

    readback = FailedReadback()
    for _ in range(2):
        try:
            readback.ensure_finalized()
        except RuntimeError as exc:
            assert str(exc) == "readback failed"
        else:
            raise AssertionError("terminal readback failure was not replayed")

    assert calls == 1
    assert readback.is_resolved()


def test_state_slot_ownership_commits_only_after_accepted_decode():
    class Model:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": True}
        fail = True

        def decode_forward(self, **kwargs):
            if self.fail:
                raise RuntimeError("submission failed")
            return torch.zeros((1, 1))

    runner = SimpleNamespace(
        model=Model(),
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        tt_per_lane_max_num_seqs=2,
        _req_state_slot={"a": 1, "b": 0},
        _pending_state_slot_settle=None,
        _decode_layout_changed_since_last_decode=True,
    )
    runner.note_decode_layout_consumed = lambda: (
        TTModelRunner.note_decode_layout_consumed(runner)
    )
    runner.note_decode_state_slots_settled = lambda: (
        TTModelRunner.note_decode_state_slots_settled(runner)
    )
    controller = TTAsyncDecodeController(runner)

    remap = TTModelRunner._decode_state_slot_remap(runner, ["a", "b"])
    model_input = _submission_input(layout_changed=True)
    model_input.slot_remap = remap

    assert remap.tolist() == [1, 0]
    assert runner._req_state_slot == {"a": 1, "b": 0}
    assert runner._pending_state_slot_settle == {"a": 0, "b": 1}

    with pytest.raises(RuntimeError, match="submission failed"):
        controller.submit_decode(model_input, read_from_device=True)
    assert runner._req_state_slot == {"a": 1, "b": 0}
    assert runner._decode_layout_changed_since_last_decode

    runner.model.fail = False
    controller.submit_decode(model_input, read_from_device=True)
    assert runner._req_state_slot == {"a": 0, "b": 1}
    assert runner._pending_state_slot_settle is None
    assert not runner._decode_layout_changed_since_last_decode


def test_host_sampling_and_host_to_device_transition_reload():
    controller = _controller()

    first_host = _commit(controller, _decode_input(device_sampling=False))
    second_host = _commit(controller, _decode_input(device_sampling=False))
    device = _commit(controller, _decode_input(device_sampling=True))

    assert first_host.reload_inputs
    assert second_host.reload_inputs
    assert not second_host.reload_sampling_params
    assert device.reload_inputs
    assert device.reload_sampling_params
    assert device.reset_sampling_state


def test_prefill_layout_trace_and_capability_force_input_reload():
    controller = _controller()
    _commit(controller, _decode_input())
    controller.note_prefill_submitted()
    assert _commit(controller, _decode_input()).reload_inputs
    assert _commit(controller, _decode_input(layout_changed=True)).reload_inputs

    no_trace = _controller(trace_mode="none")
    _commit(no_trace, _decode_input())
    assert _commit(no_trace, _decode_input()).reload_inputs

    no_residency = _controller(supports_async=False)
    _commit(no_residency, _decode_input())
    assert _commit(no_residency, _decode_input()).reload_inputs


def test_transition_applies_drained_token_before_host_authoritative_reload():
    request_state = SimpleNamespace(output_token_ids=[])
    input_batch = SimpleNamespace(
        req_id_to_index={"request": 0},
        num_tokens=np.array([3], dtype=np.int32),
        token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
    )
    runner = SimpleNamespace(
        _output_tokens_per_step=1,
        requests={"request": request_state},
        input_batch=input_batch,
        model_config=SimpleNamespace(max_model_len=8),
        model=SimpleNamespace(
            decode_input_update_contract=1,
            model_capabilities={"supports_async_decode": True},
        ),
        trace_mode="decode_only",
    )
    runner._apply_sampled_tokens_to_state = lambda **kwargs: (
        TTModelRunner._apply_sampled_tokens_to_state(runner, **kwargs)
    )
    controller = TTAsyncDecodeController(runner)
    _commit(controller, _decode_input())
    completed = CompletedDecodeStep(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        logprobs=None,
        context=SubmittedStepContext(
            req_ids=["request"],
            req_id_to_index={"request": 0},
            submit_time_ns=1,
        ),
        completion_time_ns=2,
    )

    controller.apply_completed_decode_step(completed)
    controller.note_prefill_submitted()
    plan = controller.plan_decode_reload(_decode_input())

    next_position = int(input_batch.num_tokens[0]) - 1
    assert next_position == 3
    assert input_batch.token_ids_cpu[0, next_position] == 7
    assert request_state.output_token_ids == [7]
    assert plan.reload_inputs
    assert plan.reset_sampling_state


def test_scheduler_context_phase_distinguishes_chunked_prefill_from_decode():
    input_batch = SimpleNamespace(
        req_id_to_index={"request": 0},
        num_computed_tokens_cpu=np.array([2], dtype=np.int32),
        num_tokens=np.array([8], dtype=np.int32),
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(decode_input_update_contract=1),
        input_batch=input_batch,
        _decode_layout_changed_since_last_decode=False,
    )
    controller = TTAsyncDecodeController(runner)
    controller._decode_chain_valid = True
    controller._previous_device_sampling = True
    cached_reqs = _cached_reqs(["request"], context_phase=["request"])
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens={"request": 4},
    )

    assert not controller.steady_decode_scheduler_invariants_met(
        scheduler_output, decode_layout_changed=False
    )
    # A decode may schedule several speculative tokens; only the scheduler's
    # phase marker, not a token-count heuristic, classifies it.
    scheduler_output.scheduled_cached_reqs = _cached_reqs(["request"])
    runner.requests = {}
    scheduler_output.pending_structured_output_tokens = None
    input_batch.no_penalties = True
    input_batch.no_allowed_token_ids = True
    input_batch.sampling = SimpleNamespace(
        bad_words_token_ids={}, has_active_logitsprocs=lambda: False
    )
    input_batch.max_num_logprobs = None
    runner.model_config = SimpleNamespace(logits_processors=[])
    runner.check_perform_device_sampling = lambda **kwargs: True
    assert controller.steady_decode_scheduler_invariants_met(
        scheduler_output, decode_layout_changed=False
    )


def test_scheduler_prediction_rejects_every_front_packed_layout_transition():
    req_ids = ["a", "b"]
    input_batch = SimpleNamespace(
        req_id_to_index={"a": 0, "b": 1},
        no_penalties=True,
        no_allowed_token_ids=True,
        sampling=SimpleNamespace(
            bad_words_token_ids={}, has_active_logitsprocs=lambda: False
        ),
        max_num_logprobs=None,
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(decode_input_update_contract=1),
        input_batch=input_batch,
        requests={req_id: SimpleNamespace(sampling_params=None) for req_id in req_ids},
        model_config=SimpleNamespace(logits_processors=[]),
        _decode_layout_changed_since_last_decode=False,
        check_perform_device_sampling=lambda **kwargs: True,
    )
    controller = TTAsyncDecodeController(runner)
    controller._decode_chain_valid = True
    controller._previous_device_sampling = True

    def output(ids, *, context=(), resumed=(), scheduled_counts=None):
        ids = list(ids)
        return SimpleNamespace(
            num_scheduled_tokens=scheduled_counts or {req_id: 1 for req_id in ids},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=_cached_reqs(
                ids, context_phase=context, resumed=resumed
            ),
            pending_structured_output_tokens=False,
        )

    assert controller.steady_decode_scheduler_invariants_met(output(req_ids))
    assert not controller.steady_decode_scheduler_invariants_met(output(["a"]))
    assert not controller.steady_decode_scheduler_invariants_met(
        output(["a", "b", "c"])
    )
    assert not controller.steady_decode_scheduler_invariants_met(
        output(req_ids, resumed=["b"])
    )
    assert not controller.steady_decode_scheduler_invariants_met(
        output(req_ids, context=["b"])
    )
    # A speculative decode may schedule several tokens and remains decode work.
    assert controller.steady_decode_scheduler_invariants_met(
        output(req_ids, scheduled_counts={"a": 1, "b": 4})
    )


def test_contract_v1_receives_commands_without_legacy_reset():
    captured = {}

    class Model:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": True}

        def decode_forward(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 1))

    runner = _accepted_decode_hooks(
        SimpleNamespace(
            model=Model(),
            trace_mode="decode_only",
            kv_caches=object(),
            request_specific_rope=False,
        )
    )
    controller = TTAsyncDecodeController(runner)

    submission = controller.submit_decode(_submission_input(), read_from_device=True)

    assert submission.reload_plan is not None
    assert "reset_batch" not in captured
    assert captured["reload_inputs"] is True
    assert captured["reload_page_table"] is False
    assert captured["reload_sampling_params"] is True
    assert captured["reset_sampling_state"] is True
    assert captured["slot_remap"].tolist() == [0]


def test_contract_v1_host_sampling_still_receives_commands_and_remap():
    captured = {}

    class Model:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": False}

        def decode_forward(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 1))

    runner = _accepted_decode_hooks(
        SimpleNamespace(
            model=Model(),
            trace_mode="decode_only",
            kv_caches=object(),
            request_specific_rope=False,
        )
    )
    controller = TTAsyncDecodeController(runner)

    controller.submit_decode(
        _submission_input(device_sampling=False), read_from_device=True
    )

    assert captured["reload_inputs"] is True
    assert captured["reload_sampling_params"] is False
    assert captured["reset_sampling_state"] is False
    assert captured["slot_remap"].tolist() == [0]


def test_contract_v0_keeps_legacy_call_shape_and_warns_once(monkeypatch):
    calls = []
    warnings = []
    monkeypatch.setattr(
        "vllm_tt_plugin.async_decode.logger.warning",
        lambda *args: warnings.append(args),
    )

    class Model:
        model_capabilities = {"supports_async_decode": True}

        def decode_forward(self, **kwargs):
            calls.append(kwargs)
            return torch.zeros((1, 1))

    runner = _accepted_decode_hooks(
        SimpleNamespace(
            model=Model(),
            trace_mode="decode_only",
            kv_caches=object(),
            request_specific_rope=False,
        )
    )
    controller = TTAsyncDecodeController(runner)
    model_input = _submission_input()

    first = controller.submit_decode(model_input, read_from_device=True)
    second = controller.submit_decode(model_input, read_from_device=True)

    assert first.reload_plan is None
    assert second.reload_plan is None
    assert all(call["reset_batch"] is True for call in calls)
    assert all("reload_inputs" not in call for call in calls)
    # Slot-remap delivery in both sampling modes predates this contract in the
    # standalone plugin and therefore remains unchanged for version 0.
    assert all(call["slot_remap"].tolist() == [0] for call in calls)
    assert len(warnings) == 1


def _completed_step(token: int, runner_output=None) -> CompletedDecodeStep:
    return CompletedDecodeStep(
        sampled_token_ids=torch.tensor([[token]], dtype=torch.int32),
        logprobs=None,
        context=SubmittedStepContext(
            req_ids=["request"],
            req_id_to_index={"request": 0},
            submit_time_ns=1,
        ),
        completion_time_ns=2,
        runner_output=runner_output,
    )


def _async_apply_runner(request_state):
    runner = SimpleNamespace(
        _output_tokens_per_step=1,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        requests={"request": request_state},
        input_batch=SimpleNamespace(
            req_id_to_index={},
            num_tokens=np.zeros(1, dtype=np.int32),
            token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
        ),
        model_config=SimpleNamespace(max_model_len=8),
        _steady_decode_lock=threading.Lock(),
        _completed_decode_steps=deque(),
        _pending_async_steps=deque(),
        _pending_async_overlap_ok=deque(),
    )
    runner._apply_sampled_tokens_to_state = lambda **kwargs: (
        TTModelRunner._apply_sampled_tokens_to_state(runner, **kwargs)
    )
    return runner


def test_finished_async_result_updates_neither_runner_nor_published_output():
    request_state = SimpleNamespace(output_token_ids=[])
    runner_output = SimpleNamespace(
        req_id_to_index={"request": 0}, sampled_token_ids=[[7]]
    )
    runner = _async_apply_runner(request_state)
    controller = TTAsyncDecodeController(runner)
    completed = _completed_step(7, runner_output)

    controller.apply_completed_decode_step(
        completed,
        suppress_output_req_ids={"request"},
        skip_state_req_ids={"request"},
    )

    assert request_state.output_token_ids == []
    assert runner_output.sampled_token_ids == [[]]


@pytest.mark.parametrize("lifecycle", ["preempted", "resumed"])
def test_ordinary_preemption_or_resume_keeps_inflight_token(lifecycle):
    request_state = SimpleNamespace(output_token_ids=[])
    runner_output = SimpleNamespace(
        req_id_to_index={"request": 0}, sampled_token_ids=[[7]]
    )
    runner = _async_apply_runner(request_state)
    controller = TTAsyncDecodeController(runner)
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"request"} if lifecycle == "preempted" else set(),
        scheduled_cached_reqs=_cached_reqs(
            ["request"], resumed=["request"] if lifecycle == "resumed" else []
        ),
    )

    controller.apply_completed_decode_step(
        _completed_step(7, runner_output),
        suppress_output_req_ids=controller.suppressed_output_req_ids(scheduler_output),
    )

    assert request_state.output_token_ids == [7]
    assert runner_output.sampled_token_ids == [[7]]


def test_only_finished_result_ids_are_suppressed_from_scheduler_output():
    scheduler_output = SimpleNamespace(
        finished_req_ids={"finished"},
        preempted_req_ids={"preempted"},
        scheduled_cached_reqs=_cached_reqs(["resumed"], resumed=["resumed"]),
    )

    assert TTAsyncDecodeController.suppressed_output_req_ids(scheduler_output) == {
        "finished"
    }


def test_forced_reset_skips_only_newest_counted_runner_state_frames():
    request_state = SimpleNamespace(output_token_ids=[])
    runner = _async_apply_runner(request_state)
    controller = TTAsyncDecodeController(runner)
    outputs = [
        SimpleNamespace(req_id_to_index={"request": 0}, sampled_token_ids=[[token]])
        for token in (5, 6, 7)
    ]
    runner._completed_decode_steps.extend(
        _completed_step(token, output) for token, output in zip((5, 6, 7), outputs)
    )

    controller.apply_ready_completed_decode_steps(
        forced_reset_discard_counts={"request": 2}
    )

    assert request_state.output_token_ids == [5]
    assert [output.sampled_token_ids for output in outputs] == [
        [[5]],
        [[6]],
        [[7]],
    ]


def test_async_scheduled_final_prefill_uses_forced_reset_lifecycle():
    request_state = SimpleNamespace(output_token_ids=[])
    runner = _async_apply_runner(request_state)
    controller = TTAsyncDecodeController(runner)
    runner.async_decode = controller
    runner_output = SimpleNamespace(
        req_id_to_index={"request": 0}, sampled_token_ids=[[7]]
    )

    TTModelRunner._enqueue_deferred_state_apply(
        runner,
        torch.tensor([[7]], dtype=torch.int32),
        ["request"],
        runner_output,
    )

    # Sampling completed and its output is publishable, but the runner waits
    # for the next SchedulerOutput before committing host state.
    assert request_state.output_token_ids == []
    assert runner_output.sampled_token_ids == [[7]]

    controller.apply_ready_completed_decode_steps(
        forced_reset_discard_counts={"request": 1}
    )

    # The stale prefill frame stays published so AsyncScheduler consumes its
    # discard counter, while runner state remains at the committed prefix.
    assert request_state.output_token_ids == []
    assert runner_output.sampled_token_ids == [[7]]


def test_async_scheduled_final_prefill_applies_on_ordinary_next_step():
    request_state = SimpleNamespace(output_token_ids=[])
    runner = _async_apply_runner(request_state)
    controller = TTAsyncDecodeController(runner)
    runner.async_decode = controller
    runner_output = SimpleNamespace(
        req_id_to_index={"request": 0}, sampled_token_ids=[[7]]
    )
    TTModelRunner._enqueue_deferred_state_apply(
        runner,
        torch.tensor([[7]], dtype=torch.int32),
        ["request"],
        runner_output,
    )

    controller.apply_ready_completed_decode_steps()

    assert request_state.output_token_ids == [7]
    assert runner_output.sampled_token_ids == [[7]]


def test_forced_reset_discard_count_requires_all_stale_frames():
    request_state = SimpleNamespace(output_token_ids=[])
    runner = _async_apply_runner(request_state)
    controller = TTAsyncDecodeController(runner)
    runner._completed_decode_steps.append(
        _completed_step(
            7,
            SimpleNamespace(req_id_to_index={"request": 0}, sampled_token_ids=[[7]]),
        )
    )

    with pytest.raises(RuntimeError, match="missing completed frames"):
        controller.apply_ready_completed_decode_steps(
            forced_reset_discard_counts={"request": 2}
        )


def test_unscheduled_live_request_keeps_accepted_token_in_cached_state():
    request_state = SimpleNamespace(output_token_ids=[])
    runner = SimpleNamespace(
        _output_tokens_per_step=1,
        requests={"request": request_state},
        input_batch=SimpleNamespace(req_id_to_index={}),
        model_config=SimpleNamespace(max_model_len=8),
    )

    TTModelRunner._apply_sampled_tokens_to_state(
        runner,
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        req_ids=["request"],
    )

    assert request_state.output_token_ids == [7]
