# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Original prompt boundaries survive replay, row selection and submission."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput

from tests.test_lane_input_batch import (
    _add_to_lane,
    _lane_batch,
    _make_req,
    _without_speculation,
)
from tests.test_model_runner import _batch_with_one_request, _fake_runner, _prepare
from tests.test_stage_inputs import _new_request, _runner, _step
from vllm_tt_plugin import config
from vllm_tt_plugin.host_sampler import create_host_sampler
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_original_prompt_capability_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="supports_original_prompt_lens"):
        config.resolve_tt_supports_original_prompt_lens(
            {"supports_original_prompt_lens": value}
        )


@pytest.mark.parametrize(
    "capabilities,expected",
    [
        (None, False),
        ({}, False),
        ({"supports_original_prompt_lens": False}, False),
        ({"supports_original_prompt_lens": True}, True),
    ],
)
def test_original_prompt_capability_defaults_and_boolean(capabilities, expected):
    assert config.resolve_tt_supports_original_prompt_lens(capabilities) is expected


@pytest.mark.parametrize("capabilities", [{}, {"supports_original_prompt_lens": False}])
def test_legacy_prefill_signature_receives_no_new_keyword(capabilities):
    runner = _runner(supports_stage_inputs=False)
    received = []

    def strict_prefill(
        tokens, page_table, kv_cache, enable_trace, prompt_lens, start_pos, empty_slots
    ):
        received.append(prompt_lens.tolist())

    runner.model = SimpleNamespace(
        model_capabilities=capabilities, prefill_forward=strict_prefill
    )
    step = _step(new=[_new_request("a")], scheduled={"a": 4})
    runner._update_states(step)
    inputs = runner._prepare_model_inputs(step, None)
    runner.submit_prefill(inputs, [1])
    assert received == [[4]]


def test_front_packed_original_lengths_are_submission_owned_through_chunks():
    runner = _runner(supports_stage_inputs=False)
    runner.model.model_capabilities = {"supports_original_prompt_lens": True}
    first, second = _new_request("a"), _new_request("b")
    first.prompt_token_ids = [1, 2, 3, 4, 5]
    second.prompt_token_ids = [6, 7]
    step = _step(new=[first, second], scheduled={"a": 2, "b": 2})
    runner._update_states(step)
    inputs = runner._prepare_model_inputs(step, None)
    # The first request is intermediate: its original length exceeds chunk end.
    assert inputs.original_prompt_lens == [5, 2]
    assert inputs.intermediate_prefill_mask.tolist() == [True, False]
    assert not inputs.perform_device_sampling
    runner.input_batch.num_prompt_tokens[:2] = [2, 5]
    runner.submit_prefill(inputs, [2])
    assert runner.model.prefills[-1]["original_prompt_lens"] == [5, 2]
    assert "sampling_params" not in runner.model.prefills[-1]


@pytest.mark.parametrize(
    "presence,frequency,repetition",
    [
        (0.75, 0.0, 1.0),
        (0.0, 0.75, 1.0),
        (0.75, 0.5, 1.0),
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 1.0),
    ],
)
def test_fresh_and_replayed_tokens_have_correct_host_penalties(
    presence, frequency, repetition
):
    sampled = []
    boundaries = []
    for original in [3, 2]:
        batch, request = _batch_with_one_request(
            original,
            3 - original,
            0,
            SamplingParams(
                temperature=0.0,
                presence_penalty=presence,
                frequency_penalty=frequency,
                repetition_penalty=repetition,
            ),
        )
        runner = _fake_runner(batch, request)
        runner._is_block_output_model = False
        runner.vocab_size = 64
        runner.model = SimpleNamespace(model_capabilities={})  # host fix is not opt-in
        runner._check_host_logits = TTModelRunner._check_host_logits.__get__(runner)
        runner.host_sampler = create_host_sampler()
        inputs = _prepare(runner, ("r", 3, 0, 3 - original))
        boundaries.append(getattr(inputs, "original_prompt_lens", None))
        logits = torch.full((1, 1, 64), -10.0)
        logits[0, 0, 2], logits[0, 0, 3] = 2.0, 1.5
        tokens, _ = TTModelRunner._get_output_tokens(
            runner, logits, None, inputs.tt_sampling_params, inputs, [1], False, False
        )
        sampled.append(tokens[0].item())
    if repetition != 1.0:
        assert sampled == [3, 3]  # repetition sees the union, in both worlds
    elif presence or frequency:
        assert sampled == [2, 3]
    else:
        assert sampled == [2, 2]
    assert boundaries == [[3], [2]]


def test_replayed_boundary_reaches_opted_in_model_without_request_identity():
    for original in [3, 2]:
        batch, request = _batch_with_one_request(original, 3 - original, 0)
        runner = _fake_runner(batch, request)
        captured = []
        runner.model = SimpleNamespace(
            model_capabilities={"supports_original_prompt_lens": True},
            prefill_forward=lambda target=captured, **kwargs: target.append(kwargs),
        )
        runner.kv_caches = object()
        runner.trace_mode = "none"
        runner.request_specific_rope = False
        runner.async_decode = SimpleNamespace(note_prefill_submitted=lambda: None)
        inputs = _prepare(runner, ("r", 3, 0, 3 - original))
        TTModelRunner.submit_prefill(runner, inputs, [1])
        assert captured[0]["tokens"].tolist() == [[0, 1, 2]]
        assert captured[0]["original_prompt_lens"] == [original]


def test_lane_prefill_captures_selected_rows_and_preserves_existing_host_history():
    batch = _lane_batch(num_lanes=2, per_lane=2, with_custom=False)
    fresh = _make_req(
        "fresh", [0, 1, 2], [], dict(temperature=0.0, frequency_penalty=0.75)
    )
    resume = _make_req(
        "resume", [0, 1], [2], dict(temperature=0.0, frequency_penalty=0.75)
    )
    fresh.num_computed_tokens = resume.num_computed_tokens = 0
    fresh_row = _add_to_lane(batch, fresh, 0)
    resume_row = _add_to_lane(batch, resume, 1)
    batch.refresh_logitsprocs()
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"resume": 3, "fresh": 3}
    output.total_num_scheduled_tokens = 6
    rows = (resume_row, fresh_row)
    plan = TTStepPlan(
        is_decode=False,
        capacity=4,
        scheduled_req_ids=("resume", "fresh"),
        scheduled_rows=rows,
        input_rows=rows,
        req_id_to_row={"resume": resume_row, "fresh": fresh_row},
        batch_size_per_dp=(1, 1),
        prefill_empty_slots=rows,
    )
    runner = _without_speculation(
        SimpleNamespace(
            input_batch=batch,
            _output_tokens_per_step=1,
            max_num_blocks_per_req=16,
            requests={"fresh": fresh, "resume": resume},
            model_config=SimpleNamespace(is_multimodal_model=False),
            check_perform_device_sampling=lambda **kwargs: False,
            _block_tables_per_layer=lambda _: None,
        )
    )
    inputs = batch._build_prefill_input(runner, output, None, plan)
    assert inputs.original_prompt_lens == [2, 3]
    assert inputs.prefill_empty_slots == list(rows)
    metadata = batch.build_merged_sampling_metadata(list(rows), compact=True)
    assert metadata.output_token_ids == [[2], []]
    # Captured metadata survives changes to the persistent batch.
    batch.num_prompt_tokens[list(rows)] = [3, 2]
    assert inputs.original_prompt_lens == [2, 3]


def test_platform_rejects_invalid_original_prompt_capability(monkeypatch):
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    from tests.test_block_scheduler import (
        LOCAL_MODEL_CONFIG,
        _stub_model_resolution,
        _StubModel,
    )

    monkeypatch.setattr(
        _StubModel,
        "model_capabilities",
        {"supports_original_prompt_lens": 1},
        raising=False,
    )
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    with (
        _stub_model_resolution(),
        pytest.raises(ValueError, match="supports_original_prompt_lens"),
    ):
        VllmConfig(
            model_config=model_config,
            scheduler_config=SchedulerConfig(
                max_model_len=model_config.max_model_len,
                is_encoder_decoder=model_config.is_encoder_decoder,
            ),
            cache_config=CacheConfig(block_size=128),
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )


@pytest.mark.parametrize(
    "end,expected_prompt,expected_output",
    [
        (2, [1, 2], []),
        (3, [1, 2, 3], []),
        (4, [1, 2, 3], [4]),
        (5, [1, 2, 3], [4, 4]),
    ],
)
def test_replay_chunk_history_excludes_unscheduled_tokens(
    end, expected_prompt, expected_output
):
    from vllm_tt_plugin.model_input import prefill_penalty_history

    # 63 is an allocated but unscheduled future token, then TT padding.
    tokens = torch.tensor([[1, 2, 3, 4, 4, 63, -1]])
    prompt, output = prefill_penalty_history(tokens, [end], [3], 64)
    assert prompt[prompt != 64].tolist() == expected_prompt
    assert output == [expected_output]


def test_opted_in_model_requires_captured_boundary():
    runner = _runner(supports_stage_inputs=False)
    runner.model.model_capabilities = {"supports_original_prompt_lens": True}
    step = _step(new=[_new_request("a")], scheduled={"a": 4})
    runner._update_states(step)
    inputs = runner._prepare_model_inputs(step, None)
    from dataclasses import replace

    with pytest.raises(ValueError, match="original_prompt_lens"):
        runner.submit_prefill(replace(inputs, original_prompt_lens=None), [1])
    assert not runner.model.prefills
