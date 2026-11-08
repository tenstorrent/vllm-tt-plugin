# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A structured-output row stays on the fused verify while its reasoning is open.

vLLM applies no grammar to a request until its reasoning ends: every bitmask
row up to and including the first reasoning-end token is all ones. A commit is
the accepted drafts plus one final token, so a token can follow an end token
only when the end token is a draft. A structured row whose phase is open and
whose valid drafts hold no end id is therefore served like any other row. A row
whose phase has ended, one that never reasons (thinking off,
``include_reasoning=false``, ``enable_in_reasoning``) and one with an end id
among its drafts keeps the ``logits`` verify and the scheduler's bitmask.

The model is ``FakeSpecModel`` with ``serve_fused``; the runner is the real
step path of ``test_spec_fused.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.scheduler import (
    TT_SCHEDULER_OUTPUT_FIELDS,
    TTScheduler,
    carry_tt_scheduler_output_fields,
    get_tt_reasoning_start,
    reasoning_start_for_request,
    set_tt_reasoning_start,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
)
from vllm_tt_plugin.structured_output import ReasoningPhase

from ..test_stage_inputs import _new_request
from ..test_stage_inputs import _runner as _lifecycle_runner
from ..test_stage_inputs import _step as _lifecycle_step
from .fake_spec_model import FAKE_VOCAB_SIZE, make_fake_spec_model
from .test_spec_greedy_e2e import (
    DRAFT_LEN,
    PROMPT_LEN,
    _add_request,
    _model_drafter_runner,
    _scheduler_output_for,
)

# Qwen3's parser ends reasoning on either of two ids (</think>, <tool_call>).
THINK_END = 500
TOOL_CALL = 501
END_IDS = (THINK_END, TOOL_CALL)
OPEN = (False, END_IDS)
ENDED = (True, ())
FUSED_MODES = (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS, ACCEPT_MODE_FUSED_SAMPLE)
BITMASK_WORDS = 4


@pytest.fixture(autouse=True)
def _fused_env(monkeypatch):
    monkeypatch.delenv("VLLM_TT_SPEC_FUSED", raising=False)


def _runner():
    model = make_fake_spec_model(serve_fused=True, accept_modes=FUSED_MODES)()
    runner = _model_drafter_runner(model)
    runner._spec_accept_modes = tuple(model.accept_modes)
    runner._reasoning_starts = {}
    # What the sampling-route check was asked, for the tests of the route.
    runner.sampling_route_calls = []
    # Whether the check allows device sampling; only builds read the model
    # input of a step that samples on device, none runs it.
    runner.route_allows_device = False

    def check(**kwargs):
        runner.sampling_route_calls.append(kwargs)
        return runner.route_allows_device

    runner.check_perform_device_sampling = check
    return runner


def _add_structured(runner, req_id, start, *, first_token=1, output=()):
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
        mm_features=None,
        sampling_params=SamplingParams(
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(json_object=True),
        ),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=PROMPT_LEN,
        output_token_ids=list(output),
    )
    runner.requests[req_id] = request
    runner.input_batch.add_request(request)
    if start is not None:
        runner._reasoning_starts[req_id] = start


def _build(runner, *req_ids, drafts=None, structured=()):
    """The step's model input; the first half of ``_step`` in the other tests."""
    for req in req_ids:
        row = runner.input_batch.req_id_to_index[req]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    scheduler_output = _scheduler_output_for(runner, *req_ids, drafts=drafts)
    # What the scheduler sets on a step that schedules a structured request.
    scheduler_output.num_scheduled_tokens = {req: 1 for req in req_ids}
    return TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)


def _finish(runner, model_input, grammar_output=None):
    submission = TTAsyncDecodeController(runner).submit_decode(
        model_input, read_from_device=True
    )
    fwd = _SyncForward(
        tt_out=submission.tt_out,
        tt_log_probs=None,
        sampling_params=model_input.tt_sampling_params,
        model_input=model_input,
        batch_size_per_dp=[len(model_input.row_req_ids)],
        perform_device_sampling=False,
        is_decode=True,
        spec_hidden=submission.spec_hidden,
    )
    return runner._finish_front_packed_sync(grammar_output, fwd=fwd)


def _mode(runner, req_ids, drafts):
    """The accept mode the step asks for, from the drafts it was handed."""
    block = torch.full((len(req_ids), DRAFT_LEN), -1, dtype=torch.int32)
    valid = torch.zeros(len(req_ids), dtype=torch.int32)
    for row, req_id in enumerate(req_ids):
        for col, token in enumerate(drafts.get(req_id, ())):
            block[row, col] = token
        valid[row] = len(drafts.get(req_id, ()))
    return TTModelRunner._spec_mode_for_step(runner, list(req_ids), valid, block)


# region Mode choice


def test_an_open_phase_with_no_end_id_in_the_drafts_is_fused():
    runner = _runner()
    _add_structured(runner, "s", OPEN)

    assert _mode(runner, ["s"], {"s": [3, 4, 5]}) == ACCEPT_MODE_FUSED_SAMPLE
    assert _mode(runner, ["s"], {}) == ACCEPT_MODE_FUSED_SAMPLE


@pytest.mark.parametrize("end_id", END_IDS)
@pytest.mark.parametrize("position", [0, 1, 2])
def test_an_end_id_among_the_valid_drafts_asks_for_logits(end_id, position):
    runner = _runner()
    _add_structured(runner, "s", OPEN)
    drafts = [3, 4, 5]
    drafts[position] = end_id

    assert _mode(runner, ["s"], {"s": drafts}) == ACCEPT_MODE_LOGITS


def test_an_end_id_past_the_valid_drafts_is_not_a_draft():
    runner = _runner()
    _add_structured(runner, "s", OPEN)
    block = torch.tensor([[3, THINK_END, -1]], dtype=torch.int32)
    valid = torch.tensor([1], dtype=torch.int32)

    mode = TTModelRunner._spec_mode_for_step(runner, ["s"], valid, block)

    assert mode == ACCEPT_MODE_FUSED_SAMPLE


def test_an_ended_phase_asks_for_logits():
    runner = _runner()
    _add_structured(runner, "s", ENDED)

    assert _mode(runner, ["s"], {"s": [3, 4, 5]}) == ACCEPT_MODE_LOGITS


@pytest.mark.parametrize("end_id", END_IDS)
def test_a_committed_end_id_ends_the_phase(end_id):
    runner = _runner()
    _add_structured(runner, "s", OPEN)
    assert _mode(runner, ["s"], {"s": [3, 4, 5]}) == ACCEPT_MODE_FUSED_SAMPLE

    runner.requests["s"].output_token_ids.extend([9, 8, end_id])

    assert _mode(runner, ["s"], {"s": [3, 4, 5]}) == ACCEPT_MODE_LOGITS


def test_reasoning_off_from_the_start_asks_for_logits_at_token_zero():
    """Thinking off and ``include_reasoning=false`` both start ended."""
    runner = _runner()
    _add_structured(runner, "s", ENDED)

    assert runner.requests["s"].output_token_ids == []
    assert _mode(runner, ["s"], {}) == ACCEPT_MODE_LOGITS


def test_a_request_the_scheduler_sent_no_start_for_asks_for_logits():
    runner = _runner()
    _add_structured(runner, "s", None)

    assert _mode(runner, ["s"], {"s": [3, 4, 5]}) == ACCEPT_MODE_LOGITS


def test_an_open_phase_with_no_end_ids_asks_for_logits():
    """A parser the plugin cannot track keeps today's path."""
    runner = _runner()
    _add_structured(runner, "s", (False, ()))

    assert _mode(runner, ["s"], {"s": [3, 4, 5]}) == ACCEPT_MODE_LOGITS


def test_drafts_that_are_not_given_ask_for_logits():
    runner = _runner()
    _add_structured(runner, "s", OPEN)

    mode = TTModelRunner._spec_mode_for_step(runner, ["s"])

    assert mode == ACCEPT_MODE_LOGITS


def test_the_logits_step_is_per_step_so_a_normal_row_follows_the_grammar_row():
    runner = _runner()
    _add_structured(runner, "s", OPEN)
    _add_request(runner, "n", first_token=40)

    assert _mode(runner, ["s", "n"], {"s": [THINK_END, 4, 5]}) == ACCEPT_MODE_LOGITS
    assert _mode(runner, ["s", "n"], {"s": [3, 4, 5]}) == ACCEPT_MODE_FUSED_SAMPLE


def test_a_sampled_structured_row_in_an_open_phase_follows_the_support_rule():
    """Without a top-k support the sampled row cannot be fused, grammar or not."""
    runner = _runner()
    _add_structured(runner, "s", OPEN)
    runner.requests["s"].sampling_params.temperature = 1.0
    runner.input_batch.random_reqs.add("s")

    assert _mode(runner, ["s"], {"s": [3, 4, 5]}) == ACCEPT_MODE_LOGITS


def test_enable_in_reasoning_starts_every_request_ended():
    manager = SimpleNamespace(enable_in_reasoning=True)
    request = _request_stub(reasoning_ended=None)

    assert reasoning_start_for_request(manager, request) == ENDED


# endregion Mode choice

# region What the step does


def test_a_mixed_batch_keeps_the_normal_rows_on_the_fused_path_and_unchanged():
    """The structured row rides along; the normal row commits what it would alone."""
    runner = _runner()
    _add_structured(runner, "s", OPEN)
    _add_request(runner, "n", first_token=40)
    alone = _runner()
    _add_request(alone, "n", first_token=40)

    mixed_out = _finish(runner, _build(runner, "s", "n"))
    alone_out = _finish(alone, _build(alone, "n"))

    mixed_modes = [call["spec_mode"] for call in runner.model.verify_calls]
    assert mixed_modes == [ACCEPT_MODE_FUSED_SAMPLE]
    row = mixed_out.req_id_to_index["n"]
    assert list(mixed_out.sampled_token_ids[row]) == list(
        alone_out.sampled_token_ids[alone_out.req_id_to_index["n"]]
    )


def test_a_boundary_drafts_step_asks_every_row_for_logits_and_hands_over_the_drafts():
    runner = _runner()
    runner.async_decode_scheduling = True
    _add_structured(runner, "s", OPEN)
    _add_request(runner, "n", first_token=40)

    model_input = _build(runner, "s", "n", drafts={"s": [3, THINK_END, 5]})

    assert model_input.spec_mode == ACCEPT_MODE_LOGITS
    assert runner._grammar_draft_handoff == {"s": [3, THINK_END, 5]}


def test_a_fused_step_hands_the_scheduler_no_drafts_and_keeps_its_counts():
    """The handoff and ``_grammar_validated_drafts`` belong to the logits path."""
    runner = _runner()
    runner.async_decode_scheduling = True
    _add_structured(runner, "s", OPEN)

    model_input = _build(runner, "s", drafts={"s": [3, 4, 5]})

    assert model_input.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    assert runner._grammar_draft_handoff == {}
    assert model_input.num_valid_drafts[0].item() == 3


def test_grammar_does_not_push_a_fused_step_to_host_or_deferred_sampling():
    runner = _runner()
    runner.async_decode_scheduling = True
    runner.route_allows_device = True
    _add_structured(runner, "s", OPEN)
    runner.sampling_route_calls.clear()

    model_input = _build(runner, "s")

    assert model_input.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    assert runner.sampling_route_calls[-1]["has_structured_outputs"] is False
    assert model_input.perform_device_sampling is True
    assert model_input.defer_device_sampling is False


def test_grammar_still_counts_on_a_logits_step():
    runner = _runner()
    runner.route_allows_device = True
    _add_structured(runner, "s", ENDED)
    runner.sampling_route_calls.clear()

    model_input = _build(runner, "s")

    assert model_input.spec_mode == ACCEPT_MODE_LOGITS
    assert runner.sampling_route_calls[-1]["has_structured_outputs"] is True
    assert model_input.perform_device_sampling is False


def _grammar(rows_per_request: dict[str, int], *, constrained: dict[str, int]):
    """A scheduler bitmask: all ones, but one word cleared in each given request."""
    rows = []
    for req_id, count in rows_per_request.items():
        block = torch.full((count, BITMASK_WORDS), -1, dtype=torch.int32)
        if req_id in constrained:
            block[constrained[req_id], 0] = 0b1010
        rows.append(block)
    return GrammarOutput(list(rows_per_request), torch.cat(rows).numpy())


def test_a_fused_step_accepts_an_all_ones_bitmask_and_commits_as_without_it():
    runner = _runner()
    plain = _runner()
    _add_structured(runner, "s", OPEN)
    _add_structured(plain, "s", OPEN)
    drafts = {"s": [3, 4, 5]}
    model_input = _build(runner, "s", drafts=drafts)
    grammar = _grammar({"s": 1 + DRAFT_LEN}, constrained={})

    with_grammar = _finish(runner, model_input, grammar)
    without = _finish(plain, _build(plain, "s", drafts=drafts), None)

    assert list(with_grammar.sampled_token_ids[0]) == list(without.sampled_token_ids[0])
    assert runner.requests["s"].output_token_ids == plain.requests["s"].output_token_ids


@pytest.mark.parametrize("constrained_row", [0, 2, DRAFT_LEN])
def test_a_fused_step_refuses_a_constraining_bitmask(constrained_row):
    """The worker and the scheduler disagree about the phase: fail loudly."""
    runner = _runner()
    _add_structured(runner, "s", OPEN)
    model_input = _build(runner, "s", drafts={"s": [3, 4, 5]})
    grammar = _grammar({"s": 1 + DRAFT_LEN}, constrained={"s": constrained_row})

    with pytest.raises(RuntimeError, match="reasoning phase open"):
        _finish(runner, model_input, grammar)
    assert runner.requests["s"].output_token_ids == []


def test_the_window_that_ends_at_the_end_id_hands_the_next_step_the_grammar_start():
    """The end id is the last committed token; the next step is a logits step.

    The scheduler's bitmask for that step has the grammar's initial state in
    row 0, and it reaches the walk unchanged.
    """
    runner = _runner()
    _add_structured(runner, "s", OPEN)
    first = _build(runner, "s")
    assert first.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    _finish(runner, first)
    # The commit's last token is the end id (the commit applies it the same way).
    runner.requests["s"].output_token_ids.append(THINK_END)

    second = _build(runner, "s")

    assert second.spec_mode == ACCEPT_MODE_LOGITS
    grammar = _grammar({"s": 1}, constrained={"s": 0})
    applied = runner._apply_grammar_to_input(second, grammar, lane_total=None)
    mask = applied.grammar_bitmask[0]
    assert mask[0, 0, 0].item() == 0b1010, "row 0 carries the grammar's first mask"
    assert bool((mask[0, 1:] == -1).all())


# endregion What the step does

# region The phase tracker


def test_the_tracker_scans_only_new_tokens_and_latches():
    phase = ReasoningPhase.from_scheduler(OPEN)
    tokens = [1, 2, 3]
    phase.observe(tokens)
    assert phase.scanned == 3 and not phase.ended

    tokens.append(TOOL_CALL)
    phase.observe(tokens)
    assert phase.ended
    phase.observe([])
    assert phase.ended, "an ended phase stays ended"


def test_the_tracker_rescans_a_replayed_request():
    phase = ReasoningPhase.from_scheduler(OPEN)
    phase.observe([1, 2, 3, 4])

    phase.observe([1, THINK_END])

    assert phase.ended


def test_only_the_first_end_id_matters_to_the_window_rule():
    phase = ReasoningPhase.from_scheduler(OPEN)

    assert phase.fused_safe([1, 2, 3])
    assert not phase.fused_safe([1, TOOL_CALL, 3])


# endregion The phase tracker

# region The scheduler field


def _request_stub(*, reasoning_ended, prompt=(1, 2, 3), reasoner=None):
    structured = SimpleNamespace(reasoning_ended=reasoning_ended, reasoner=reasoner)
    return SimpleNamespace(
        structured_output_request=structured,
        prompt_token_ids=list(prompt),
        use_structured_output=True,
    )


class _Adapter(ParserEngineReasoningAdapter):
    """The adapter class vLLM tests, with the parser's answers fixed."""

    def __init__(self, end_ids, prompt_ends):
        self._end_ids = frozenset(end_ids)
        self._prompt_ends = prompt_ends

    @property
    def reasoning_end_token_ids(self):
        return self._end_ids

    def is_reasoning_end(self, input_ids):
        return self._prompt_ends


def _manager(reasoner, *, enable_in_reasoning=False):
    return SimpleNamespace(
        enable_in_reasoning=enable_in_reasoning, _get_reasoner=lambda request: reasoner
    )


def test_the_scheduler_start_follows_the_vllm_constraint_start_rules():
    thinking = _Adapter(END_IDS, prompt_ends=False)
    no_thinking = _Adapter(END_IDS, prompt_ends=True)
    cases = [
        # thinking on: open, and the end set travels with it
        (_manager(thinking), _request_stub(reasoning_ended=None), (False, END_IDS)),
        (_manager(thinking), _request_stub(reasoning_ended=False), (False, END_IDS)),
        # thinking off: the prompt already holds an end id
        (_manager(no_thinking), _request_stub(reasoning_ended=None), ENDED),
        # include_reasoning=false latches the flag before the first step
        (_manager(thinking), _request_stub(reasoning_ended=True), ENDED),
        # no reasoning parser configured: the grammar applies from token 0
        (_manager(None), _request_stub(reasoning_ended=None), ENDED),
        # the config flag
        (
            _manager(thinking, enable_in_reasoning=True),
            _request_stub(reasoning_ended=None),
            ENDED,
        ),
        # an adapter with no end ids cannot be tracked
        (
            _manager(_Adapter((), False)),
            _request_stub(reasoning_ended=None),
            (False, ()),
        ),
    ]
    for manager, request, expected in cases:
        assert reasoning_start_for_request(manager, request) == expected


def test_a_request_without_structured_output_has_no_start():
    request = SimpleNamespace(structured_output_request=None)

    assert reasoning_start_for_request(_manager(None), request) is None


def test_the_scheduler_attaches_a_start_for_each_scheduled_structured_request():
    structured = _request_stub(reasoning_ended=None)
    plain = SimpleNamespace(use_structured_output=False)
    scheduler = SimpleNamespace(
        requests={"s": structured, "p": plain},
        structured_output_manager=_manager(_Adapter(END_IDS, prompt_ends=False)),
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"s": 1, "p": 1}

    TTScheduler._attach_reasoning_start(scheduler, scheduler_output)

    assert get_tt_reasoning_start(scheduler_output) == {"s": (False, END_IDS)}


def test_the_start_is_in_the_inventory_and_survives_a_rebuild():
    assert "_tt_reasoning_start" in TT_SCHEDULER_OUTPUT_FIELDS
    source = SchedulerOutput.make_empty()
    set_tt_reasoning_start(source, {"s": (False, END_IDS)})
    rebuilt = SchedulerOutput.make_empty()
    assert get_tt_reasoning_start(rebuilt) == {}

    carry_tt_scheduler_output_fields(source, rebuilt)

    assert get_tt_reasoning_start(rebuilt) == {"s": (False, END_IDS)}


def test_the_field_reaches_the_worker_and_builds_the_phase():
    """Scheduler output -> carried copy -> ``_update_states`` -> phase."""
    runner = _lifecycle_runner(supports_stage_inputs=False)
    data = _new_request("a")
    data.sampling_params = SamplingParams(
        temperature=0.0, structured_outputs=StructuredOutputsParams(json_object=True)
    )
    plain = _new_request("b")
    source = _lifecycle_step(new=[data, plain], scheduled={"a": 4, "b": 4})
    set_tt_reasoning_start(source, {"a": (False, END_IDS)})
    arrived = _lifecycle_step(new=[data, plain], scheduled={"a": 4, "b": 4})
    carry_tt_scheduler_output_fields(source, arrived)

    runner._update_states(arrived)

    phase = TTModelRunner._reasoning_phase(runner, "a")
    assert phase is not None and not phase.ended
    assert phase.end_ids == frozenset(END_IDS)
    assert TTModelRunner._reasoning_phase(runner, "b") is None

    runner.requests["a"].output_token_ids.append(THINK_END)
    assert TTModelRunner._reasoning_phase(runner, "a").ended

    runner._update_states(_lifecycle_step(finished={"a"}))
    assert "a" not in runner._reasoning_phases
    assert "a" not in runner._reasoning_starts


# endregion The scheduler field


def test_the_fake_vocabulary_holds_the_end_ids():
    assert max(END_IDS) < FAKE_VOCAB_SIZE
