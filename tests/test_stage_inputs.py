# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A pipeline stage's per-request inputs reach ``prefill_forward``.

vLLM-Omni carries a stage's request payload as ``additional_information`` on the
scheduled-new-request data. The runner keeps it on the cached request state and
passes it to every prefill of that request as ``stage_inputs``, one entry per
row, when the model declares ``supports_stage_inputs``.
"""

import inspect
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.config import (
    get_tt_supports_stage_inputs,
    validate_and_store_tt_supports_stage_inputs,
)
from vllm_tt_plugin.input_batch import InputBatch, build_cached_request_state
from vllm_tt_plugin.model_runner import TTModelRunner

VOCAB_SIZE = 64
BLOCK_SIZE = 16
MAX_MODEL_LEN = 64
MAX_NUM_REQS = 4
PROMPT_LEN = 4


class _StageModel:
    """Records every ``prefill_forward`` call's keywords and every release."""

    def __init__(self):
        self.prefills: list[dict] = []
        self.released: list[int] = []

    def prefill_forward(self, **kwargs):
        self.prefills.append(kwargs)
        return object()

    def release_request(self, slot):
        self.released.append(slot)


def _runner(*, supports_stage_inputs: bool) -> SimpleNamespace:
    """A front-packed runner fake; its lifecycle and prefill methods are real."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    runner = SimpleNamespace(
        model=_StageModel(),
        input_batch=batch,
        requests={},
        encoder_cache={},
        kv_caches=object(),
        trace_mode="none",
        request_specific_rope=False,
        async_decode=SimpleNamespace(note_prefill_submitted=lambda: None),
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=0,
        async_decode_scheduling=False,
        _spec_method=None,
        _spec_supports_narrow_decode=False,
        _spec_drafts_from_model=False,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _ngram_proposer=None,
        _req_state_slot={},
        vllm_config=SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
        ),
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        supports_stage_inputs=supports_stage_inputs,
        supports_stage_outputs=False,
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=MAX_MODEL_LEN
        ),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    for name, member in vars(TTModelRunner).items():
        if hasattr(runner, name):
            continue
        if isinstance(member, staticmethod):
            setattr(runner, name, member.__func__)
        elif isinstance(member, classmethod):
            setattr(runner, name, member.__func__.__get__(TTModelRunner))
        elif inspect.isfunction(member):
            setattr(runner, name, member.__get__(runner))
    return runner


def _new_request(req_id: str, stage_inputs=None) -> NewRequestData:
    """A request as the scheduler hands it to the runner, with
    ``additional_information`` when there is a payload. vLLM-Omni sends the payload
    serialized; a plain dict stands in here except where the decode is under test."""
    data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1] * PROMPT_LEN,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )
    if stage_inputs is not None:
        data.additional_information = stage_inputs
    return data


def _empty_cached() -> CachedRequestData:
    return CachedRequestData(
        req_ids=[],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )


def _resumed(req_id: str) -> CachedRequestData:
    """A preempted request coming back: fresh blocks, KV to rebuild from 0."""
    return CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids={req_id},
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[([1],)],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )


def _step(*, new=(), cached=None, scheduled=None, finished=(), preempted=()):
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = list(new)
    output.scheduled_cached_reqs = cached if cached is not None else _empty_cached()
    output.num_scheduled_tokens = dict(scheduled or {})
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    output.finished_req_ids = set(finished)
    output.preempted_req_ids = set(preempted)
    return output


def _run_prefill(runner, scheduler_output) -> dict:
    """Apply the step and submit its prefill; return the model's keywords."""
    runner._update_states(scheduler_output)
    model_input = runner._prepare_model_inputs(scheduler_output, None)
    assert model_input is not None
    runner.submit_prefill(model_input, [model_input.unpadded_batch_size])
    return runner.model.prefills[-1]


def test_stage_inputs_follow_the_request_until_it_finishes():
    """Unchanged at the first prefill, again after preemption, gone after finish."""
    payload = {"voice": "clone", "ref_code": [[1, 2, 3]]}
    runner = _runner(supports_stage_inputs=True)

    kwargs = _run_prefill(
        runner,
        _step(
            new=[_new_request("a", payload), _new_request("b")],
            scheduled={"a": PROMPT_LEN, "b": PROMPT_LEN},
        ),
    )
    by_row = dict(zip(runner.input_batch.req_ids, kwargs["stage_inputs"]))
    assert by_row["a"] is payload, "the payload reaches prefill unchanged"
    assert by_row["b"] is None, "a request without a payload gets None in its row"

    # Preemption frees the KV and the model state, not the cached request state.
    slot = runner._req_state_slot["a"]
    runner._update_states(_step(preempted={"a"}, scheduled={"b": 1}))
    assert runner.model.released == [slot], "the model released the request's state"
    assert runner.requests["a"].stage_inputs is payload

    kwargs = _run_prefill(
        runner, _step(cached=_resumed("a"), scheduled={"a": PROMPT_LEN})
    )
    assert kwargs["stage_inputs"] == [payload], "the resumed prefill gets it again"
    assert kwargs["stage_inputs"][0] is payload

    # vLLM reports an aborted request in ``finished_req_ids`` too.
    runner._update_states(_step(finished={"a"}))
    assert "a" not in runner.requests


def test_a_model_without_the_capability_gets_no_stage_inputs_keyword():
    runner = _runner(supports_stage_inputs=False)

    kwargs = _run_prefill(
        runner, _step(new=[_new_request("a")], scheduled={"a": PROMPT_LEN})
    )

    assert "stage_inputs" not in kwargs


def test_a_model_without_the_capability_serves_a_request_that_carries_a_payload():
    """vLLM-Omni puts engine bookkeeping on every request, wanted or not."""
    bookkeeping = {"global_request_id": ["g-1"], "omni_final_stage_id": [0]}
    runner = _runner(supports_stage_inputs=False)

    kwargs = _run_prefill(
        runner,
        _step(new=[_new_request("a", bookkeeping)], scheduled={"a": PROMPT_LEN}),
    )

    assert "stage_inputs" not in kwargs
    assert runner.requests["a"].stage_inputs is None


def test_the_builder_keeps_no_payload_without_the_capability():
    """The lane-DP call site passes ``supports_stage_inputs=False``."""
    state = build_cached_request_state(
        _new_request("a", {"voice": "x"}), supports_stage_inputs=False
    )

    assert state.stage_inputs is None


def test_a_serialized_vllm_omni_payload_reaches_prefill_as_a_dict():
    """vLLM-Omni serializes ``additional_information``; the model gets the dict."""
    serialization = pytest.importorskip("vllm_omni.engine.serialization")
    payload = {"task_type": ["Base"], "text": ["Hello."], "voice": {"name": ["v"]}}
    runner = _runner(supports_stage_inputs=True)

    kwargs = _run_prefill(
        runner,
        _step(
            new=[
                _new_request(
                    "a", serialization.serialize_additional_information(payload)
                )
            ],
            scheduled={"a": PROMPT_LEN},
        ),
    )

    assert kwargs["stage_inputs"] == [payload]


class _Model:
    pass


@pytest.mark.parametrize(
    "capabilities, expected",
    [(None, False), ({}, False), ({"supports_stage_inputs": True}, True)],
)
def test_the_resolved_capability_is_stored_on_the_config(capabilities, expected):
    vllm_config = SimpleNamespace(additional_config={})

    validate_and_store_tt_supports_stage_inputs(
        vllm_config, capabilities, _Model, False
    )

    assert get_tt_supports_stage_inputs(vllm_config) is expected


def test_a_non_boolean_declaration_is_refused():
    with pytest.raises(ValueError, match="as a boolean"):
        validate_and_store_tt_supports_stage_inputs(
            SimpleNamespace(additional_config={}),
            {"supports_stage_inputs": 1},
            _Model,
            False,
        )


def test_lane_mode_refuses_the_capability():
    with pytest.raises(ValueError, match="lane mode"):
        validate_and_store_tt_supports_stage_inputs(
            SimpleNamespace(additional_config={}),
            {"supports_stage_inputs": True},
            _Model,
            True,
        )
