# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A request delivered in ``scheduled_new_reqs`` replaces what the runner holds.

vLLM's V1 scheduler output reports a request resumed after preemption in
``scheduled_cached_reqs`` with ``resumed_req_ids``, and the runner keeps the
request's ``CachedRequestState``, generated tokens included. The V2 scheduler
output delivers the same resume in ``scheduled_new_reqs``, and the generated
tokens appear only in ``NewRequestData.prefill_token_ids``. A request can also
come back as new while it still holds a persistent-batch row: a resumable
streaming-input request with its next input chunk, or a V2 resume in the step
that reports its preemption.

These tests drive the front-packed ``TTModelRunner._update_states`` and the
lane-DP ``TTLaneInputBatch.apply_step_plan`` through both output shapes.
"""

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from tests.test_lane_input_batch import _lane_batch, _plan, _step_output
from tests.test_stage_inputs import _resumed, _run_prefill, _runner, _step
from vllm_tt_plugin.input_batch import build_cached_request_state

PROMPT = [11, 12, 13, 14]
GENERATED = [21, 22, 23]
HISTORY = PROMPT + GENERATED


def _record(
    req_id,
    prompt,
    *,
    generated=None,
    num_computed_tokens=0,
    block_ids=([0],),
    seed=None,
) -> NewRequestData:
    """A ``scheduled_new_reqs`` entry. ``generated=None`` is the V1 shape, which
    leaves ``prefill_token_ids`` unset; a list is the V2 shape."""
    sampling_params = (
        SamplingParams(temperature=0.0)
        if seed is None
        else SamplingParams(temperature=1.0, seed=seed)
    )
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=sampling_params,
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
        prefill_token_ids=(None if generated is None else [*prompt, *generated]),
    )


def _rows_holding(batch, req_id) -> list[int]:
    return [row for row, rid in enumerate(batch.req_ids) if rid == req_id]


# --------------------------------------------------------------------------
# build_cached_request_state
# --------------------------------------------------------------------------


def test_the_v1_shape_starts_with_no_generated_tokens():
    state = build_cached_request_state(
        _record("a", PROMPT), supports_stage_inputs=False
    )
    assert state.output_token_ids == []
    assert state.num_tokens == len(PROMPT)


def test_the_v2_shape_seeds_the_generated_tokens_from_prefill_token_ids():
    state = build_cached_request_state(
        _record("a", PROMPT, generated=GENERATED), supports_stage_inputs=False
    )
    assert state.output_token_ids == GENERATED
    assert state.num_tokens == len(HISTORY)


def test_the_generated_tokens_are_copied_out_of_the_schedulers_list():
    """In a single-process engine ``prefill_token_ids`` can be the scheduler's own
    ``Request._all_token_ids``, which grows after this step."""
    record = _record("a", PROMPT, generated=GENERATED)
    state = build_cached_request_state(record, supports_stage_inputs=False)

    record.prefill_token_ids.append(99)

    assert state.output_token_ids == GENERATED


def test_prefill_token_ids_shorter_than_the_prompt_is_refused():
    record = _record("a", PROMPT, generated=[])
    record.prefill_token_ids = PROMPT[:2]

    with pytest.raises(ValueError, match="holds 2 tokens, fewer than its 4 prompt"):
        build_cached_request_state(record, supports_stage_inputs=False)


# --------------------------------------------------------------------------
# Front-packed: TTModelRunner._update_states
# --------------------------------------------------------------------------


def _preempted_runner():
    """A runner whose request "a" generated ``GENERATED`` and was then preempted."""
    runner = _runner(supports_stage_inputs=False)
    runner._update_states(
        _step(new=[_record("a", PROMPT)], scheduled={"a": len(PROMPT)})
    )
    # The runner appends every sampled token to the cached state.
    runner.requests["a"].output_token_ids.extend(GENERATED)
    runner._update_states(_step(preempted={"a"}))
    assert "a" not in runner.input_batch.req_id_to_index
    return runner


def _resume_step(shape, num_scheduled):
    if shape == "v1":
        return _step(cached=_resumed("a"), scheduled={"a": num_scheduled})
    return _step(
        new=[_record("a", PROMPT, generated=GENERATED, block_ids=([1],))],
        scheduled={"a": num_scheduled},
    )


@pytest.mark.parametrize("shape", ["v1", "v2"])
def test_a_resume_replays_the_prompt_and_the_generated_tokens(shape):
    runner = _preempted_runner()
    step = _resume_step(shape, len(HISTORY))

    runner._update_states(step)
    model_input = runner._prepare_model_inputs(step, None)

    batch = runner.input_batch
    row = batch.req_id_to_index["a"]
    assert runner.requests["a"].output_token_ids == GENERATED
    assert batch.num_tokens[row] == len(HISTORY)
    assert batch.num_prompt_tokens[row] == len(PROMPT)
    assert model_input.input_tokens[0, : len(HISTORY)].tolist() == HISTORY
    assert list(model_input.prompt_lens) == [len(HISTORY)]
    assert model_input.intermediate_prefill_mask.tolist() == [False]


@pytest.mark.parametrize("shape", ["v1", "v2"])
def test_a_resume_replayed_in_chunks_samples_only_after_the_last(shape):
    """Every chunk but the last stops short of ``num_tokens``, so it samples
    nothing. With the generated tokens missing, the first chunk would already
    reach ``num_tokens`` and sample against a partial replay."""
    runner = _preempted_runner()
    first = _resume_step(shape, 5)

    runner._update_states(first)
    model_input = runner._prepare_model_inputs(first, None)

    assert model_input.input_tokens[0, :5].tolist() == HISTORY[:5]
    assert model_input.intermediate_prefill_mask.tolist() == [True]


def test_a_request_scheduled_as_new_while_holding_a_row_is_rebuilt_in_that_row():
    """A streaming-input continuation: "a" ran in the previous step and comes
    back as new with its next input chunk, now unseeded."""
    runner = _runner(supports_stage_inputs=False)
    runner._update_states(
        _step(
            new=[_record("a", PROMPT, seed=7), _record("b", [5, 6])],
            scheduled={"a": len(PROMPT), "b": 2},
        )
    )
    batch = runner.input_batch
    assert batch.req_id_to_index["a"] in batch.sampling.generators

    continuation = _record("a", PROMPT + [31, 32], num_computed_tokens=len(PROMPT))
    runner._update_states(_step(new=[continuation], scheduled={"a": 2, "b": 1}))

    assert _rows_holding(batch, "a") == [batch.req_id_to_index["a"]]
    assert _rows_holding(batch, "b") == [batch.req_id_to_index["b"]]
    row = batch.req_id_to_index["a"]
    assert row not in batch.sampling.generators
    assert batch.num_tokens[row] == len(PROMPT) + 2


def test_a_resume_in_the_step_that_reports_the_preemption_holds_one_row():
    """V2 shape: "a" is in ``preempted_req_ids`` and in ``scheduled_new_reqs`` of
    one step, as after ``reset_prefix_cache(reset_running_requests=True)``. The
    model hears of the preemption once, and "a" ends on one row with its full
    history."""
    runner = _runner(supports_stage_inputs=False)
    _run_prefill(
        runner,
        _step(
            new=[_record("a", PROMPT), _record("b", [5, 6])],
            scheduled={"a": len(PROMPT), "b": 2},
        ),
    )
    slot = runner._req_state_slot["a"]
    runner.requests["a"].output_token_ids.extend(GENERATED)

    runner._update_states(
        _step(
            new=[_record("a", PROMPT, generated=GENERATED, block_ids=([1],))],
            scheduled={"a": len(HISTORY), "b": 1},
            preempted={"a"},
        )
    )

    batch = runner.input_batch
    assert runner.model.released == [slot]
    assert _rows_holding(batch, "a") == [batch.req_id_to_index["a"]]
    row = batch.req_id_to_index["a"]
    assert batch.token_ids_cpu[row, : len(HISTORY)].tolist() == HISTORY


# --------------------------------------------------------------------------
# Lane-DP: TTLaneInputBatch.apply_step_plan
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["v1", "v2"])
def test_lane_resume_rebuilds_the_same_row_in_either_output_shape(shape):
    b = _lane_batch(num_lanes=1, per_lane=2)
    requests: dict = {}
    b.apply_step_plan(
        _step_output(new_reqs=[_record("a", PROMPT), _record("b", [5, 6])]),
        _plan({"a": 0, "b": 1}, capacity=2),
        requests,
        {},
    )
    requests["a"].output_token_ids.extend(GENERATED)
    b.apply_step_plan(
        _step_output(preempted=["a"]),
        _plan({"b": 1}, is_decode=True, capacity=2),
        requests,
        {},
    )

    if shape == "v1":
        resume = _step_output(
            cached=CachedRequestData(
                req_ids=["a"],
                resumed_req_ids={"a"},
                new_token_ids=[[]],
                all_token_ids={},
                new_block_ids=[([7],)],
                num_computed_tokens=[0],
                num_output_tokens=[0],
            )
        )
    else:
        resume = _step_output(
            new_reqs=[_record("a", PROMPT, generated=GENERATED, block_ids=([7],))]
        )
    b.apply_step_plan(resume, _plan({"a": 0}, capacity=2), requests, {})

    assert b.req_id_to_index == {"a": 0, "b": 1}
    assert requests["a"].block_ids == ([7],)
    assert requests["a"].output_token_ids == GENERATED
    assert b.num_tokens[0] == len(HISTORY)
    assert b.token_ids_cpu[0, : len(HISTORY)].tolist() == HISTORY


def test_lane_request_scheduled_as_new_on_its_own_row_drops_its_old_row_state():
    """The coordinator keeps a row for a request that still holds one, so a
    streaming-input continuation is placed on the row it already occupies."""
    b = _lane_batch(num_lanes=1, per_lane=2)
    requests: dict = {}
    b.apply_step_plan(
        _step_output(new_reqs=[_record("a", PROMPT, seed=7)]),
        _plan({"a": 0}, capacity=2),
        requests,
        {},
    )
    assert 0 in b.sampling.generators

    continuation = _record("a", PROMPT + [31, 32], num_computed_tokens=len(PROMPT))
    b.apply_step_plan(
        _step_output(new_reqs=[continuation]),
        _plan({"a": 0}, capacity=2),
        requests,
        {},
    )

    assert b.req_id_to_index == {"a": 0}
    assert 0 not in b.sampling.generators
    assert b.num_tokens[0] == len(PROMPT) + 2
