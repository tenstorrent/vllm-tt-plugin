# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A prefill row the model could not serve fails its request, not the engine.

A model declaring ``model_capabilities["reports_prefill_errors"]`` returns
``{prefill row: reason}`` from ``take_prefill_errors()`` after each
``prefill_forward``. The runner maps the rows to request ids and puts them on the
step's output; ``TTScheduler.update_from_output`` drops the token sampled for such
a request and finishes it ``FINISHED_ERROR``, whose finish reason ``error`` the
OpenAI server answers with HTTP 500 (an error event, then ``[DONE]``, when
streaming). The other requests of the step, and the engine, carry on.
"""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.exceptions import GenerationError
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import TTLaneInputBatch, build_cached_request_state
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import (
    TTScheduler,
    get_tt_prefill_errors,
    set_tt_prefill_errors,
)

from .test_multimodal_items import _new_request, _run_prefill, _runner, _step

BLOCK_SIZE = 16
MAX_MODEL_LEN = 512
CHUNK = 128
EOS = 2
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"
ENCODER_FAILED = "a multimodal item of chunk [0, 32) failed to encode"


# region Runner


class _ReportingModel:
    """Answers every prefill, then reports ``errors`` once (and clears them)."""

    model_capabilities = {"reports_prefill_errors": True}

    def __init__(self, errors):
        self.errors = dict(errors)
        self.prefills: list[dict] = []

    def prefill_forward(self, **kwargs):
        self.prefills.append(kwargs)
        return torch.zeros(kwargs["tokens"].shape[0], 1, 8)

    def take_prefill_errors(self):
        errors, self.errors = self.errors, {}
        return errors


class _SilentModel(_ReportingModel):
    model_capabilities = {}

    def take_prefill_errors(self):
        raise AssertionError("asked a model without reports_prefill_errors")


def _front_packed(model) -> SimpleNamespace:
    runner = _runner(None)
    runner.model = model
    return runner


def test_a_front_packed_prefill_reports_its_failed_rows_by_request():
    runner = _front_packed(_ReportingModel({1: ENCODER_FAILED}))
    scheduled = _step(
        new=[_new_request("a"), _new_request("b"), _new_request("c")],
        scheduled={"a": 32, "b": 32, "c": 32},
    )

    runner._update_states(scheduled)
    model_input = runner._prepare_model_inputs(scheduled, None)
    runner.submit_prefill(model_input, [model_input.unpadded_batch_size])

    failed_req_id = model_input.row_req_ids[1]
    assert model_input.prefill_errors == {failed_req_id: ENCODER_FAILED}
    output = ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0})
    finish = TTModelRunner._with_prefill_errors(lambda _grammar: output, model_input)
    assert finish(None) is output
    assert get_tt_prefill_errors(output) == {failed_req_id: ENCODER_FAILED}


def test_a_step_without_failures_samples_as_before():
    runner = _front_packed(_ReportingModel({}))
    scheduled = _step(new=[_new_request("a")], scheduled={"a": 32})
    runner._update_states(scheduled)
    model_input = runner._prepare_model_inputs(scheduled, None)
    runner.submit_prefill(model_input, [model_input.unpadded_batch_size])

    def finish(_grammar):
        return None

    assert model_input.prefill_errors == {}
    assert TTModelRunner._with_prefill_errors(finish, model_input) is finish


def test_a_model_without_the_capability_is_not_asked():
    runner = _front_packed(_SilentModel({}))

    _run_prefill(runner, _step(new=[_new_request("a")], scheduled={"a": 32}))


@pytest.mark.parametrize("errors", [{3: "x"}, {-1: "x"}, {0: None}])
def test_a_report_naming_no_prefill_row_is_refused(errors):
    runner = _front_packed(_ReportingModel(errors))
    scheduled = _step(new=[_new_request("a")], scheduled={"a": 32})
    runner._update_states(scheduled)
    model_input = runner._prepare_model_inputs(scheduled, None)

    with pytest.raises(ValueError, match="take_prefill_errors"):
        runner.submit_prefill(model_input, [model_input.unpadded_batch_size])


@pytest.mark.parametrize("num_lanes", [1, 2], ids=["row-engine", "two-lanes"])
def test_a_lane_prefill_reports_its_failed_rows_by_request(num_lanes):
    """Lane prefill rows are the plan's slots in order; the last lane holds both."""
    per_lane = 2
    batch = TTLaneInputBatch(
        num_lanes=num_lanes,
        per_lane=per_lane,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * num_lanes * per_lane,
        vocab_size=64,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
        logitsprocs=None,
    )
    base = (num_lanes - 1) * per_lane
    rows = {"ok": base, "bad": base + 1}
    requests = {}
    for req_id, row in rows.items():
        requests[req_id] = build_cached_request_state(
            _new_request(req_id), supports_stage_inputs=False
        )
        batch.add_request_to_row(requests[req_id], row)
    counts = [0] * num_lanes
    counts[-1] = 2
    plan = TTStepPlan(
        is_decode=False,
        capacity=num_lanes * per_lane,
        scheduled_req_ids=("ok", "bad"),
        scheduled_rows=(rows["ok"], rows["bad"]),
        input_rows=(rows["ok"], rows["bad"]),
        req_id_to_row=rows,
        batch_size_per_dp=tuple(counts),
        prefill_empty_slots=(rows["ok"], rows["bad"]),
    )
    runner = SimpleNamespace(
        input_batch=batch,
        requests=requests,
        mm_modalities=None,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        _block_tables_per_layer=lambda _: None,
        check_perform_device_sampling=lambda **_: False,
        model_config=SimpleNamespace(is_multimodal_model=False),
        model=_ReportingModel({1: ENCODER_FAILED}),
        kv_caches=object(),
        trace_mode="none",
        request_specific_rope=False,
        tt_per_lane_max_num_seqs=per_lane,
        async_decode=SimpleNamespace(note_prefill_submitted=lambda: None),
    )
    model_input = batch.build_model_input(
        runner, _step(scheduled={"ok": 32, "bad": 32}), None, plan
    )

    TTModelRunner.submit_prefill(runner, model_input, list(plan.batch_size_per_dp))

    assert model_input.prefill_errors == {"bad": ENCODER_FAILED}


# endregion Runner

# region Scheduler


class _StubModel:
    model_capabilities = {"supports_chunked_prefill": True}


@contextmanager
def _stub_model_resolution():
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(_StubModel, None),
        ),
    ):
        yield


def _scheduler(*, async_scheduling: bool = False) -> TTScheduler:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=CHUNK,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
    config.scheduler_config.async_scheduling = async_scheduling
    num_blocks = 4 * MAX_MODEL_LEN // BLOCK_SIZE
    cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(request_id: str, prompt_length: int = 32) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=8)
    sampling_params.update_from_generation_config({}, eos_token_id=EOS)
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt_length,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _runner_output(tokens: dict[str, list[int]], errors=None) -> ModelRunnerOutput:
    """The model forces a failed row's first token to end-of-turn (EOS)."""
    req_ids = list(tokens)
    output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[tokens[req_id] for req_id in req_ids],
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    if errors:
        set_tt_prefill_errors(output, errors)
    return output


def _outputs_by_request(engine_core_outputs) -> dict:
    return {
        output.request_id: output
        for eco in engine_core_outputs.values()
        for output in eco.outputs
    }


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_a_failed_final_chunk_finishes_its_request_as_an_error(async_scheduling):
    scheduler = _scheduler(async_scheduling=async_scheduling)
    for req_id in ("ok", "bad"):
        scheduler.add_request(_request(req_id))
    prefill = scheduler.schedule()
    assert set(prefill.num_scheduled_tokens) == {"ok", "bad"}
    next_step = scheduler.schedule() if async_scheduling else None

    outputs = _outputs_by_request(
        scheduler.update_from_output(
            prefill,
            _runner_output({"ok": [7], "bad": [EOS]}, {"bad": ENCODER_FAILED}),
        )
    )

    assert outputs["bad"].finish_reason == FinishReason.ERROR
    assert str(outputs["bad"].finish_reason) == "error"
    assert outputs["bad"].new_token_ids == [], "the forced end-of-turn is dropped"
    assert outputs["ok"].new_token_ids == [7] and outputs["ok"].finish_reason is None
    assert "bad" not in scheduler.requests
    assert scheduler.requests["ok"].status == RequestStatus.RUNNING
    if next_step is not None:
        # The step scheduled before the failure was known still carries the
        # request; its output for it is ignored.
        assert "bad" in next_step.num_scheduled_tokens
        later = _outputs_by_request(
            scheduler.update_from_output(
                next_step, _runner_output({"ok": [8], "bad": [9]})
            )
        )
        assert "bad" not in later and later["ok"].new_token_ids == [8]
    following = scheduler.schedule()
    assert "bad" in following.finished_req_ids
    assert set(following.num_scheduled_tokens) == {"ok"}


def test_a_failed_intermediate_chunk_finishes_its_request_as_an_error():
    scheduler = _scheduler()
    scheduler.add_request(_request("long", prompt_length=2 * CHUNK + 16))
    first_chunk = scheduler.schedule()
    assert first_chunk.num_scheduled_tokens == {"long": CHUNK}

    outputs = _outputs_by_request(
        scheduler.update_from_output(
            first_chunk, _runner_output({"long": []}, {"long": ENCODER_FAILED})
        )
    )

    assert outputs["long"].finish_reason == FinishReason.ERROR
    assert scheduler.get_num_unfinished_requests() == 0
    following = scheduler.schedule()
    assert following.num_scheduled_tokens == {}
    assert "long" in following.finished_req_ids


def test_each_lane_fails_only_its_own_request_of_the_merged_output():
    """A lane coordinator hands every lane the same merged output."""
    lanes = [_scheduler(), _scheduler()]
    lanes[0].add_request(_request("ok"))
    lanes[1].add_request(_request("bad"))
    lane_steps = [lane.schedule() for lane in lanes]
    merged = _runner_output({"ok": [7], "bad": [EOS]}, {"bad": ENCODER_FAILED})

    first = _outputs_by_request(lanes[0].update_from_output(lane_steps[0], merged))
    second = _outputs_by_request(lanes[1].update_from_output(lane_steps[1], merged))

    assert first["ok"].new_token_ids == [7] and first["ok"].finish_reason is None
    assert second["bad"].finish_reason == FinishReason.ERROR
    assert merged.sampled_token_ids == [[7], [EOS]], "the merged output is not touched"


def test_an_error_finish_is_http_500_and_ends_a_stream_with_an_error_event():
    """vLLM's serving turns finish reason ``error`` into a 500; the plugin relies on
    it (``GenerateBaseServing._raise_if_error``, shared by chat and completions,
    whose stream generators then yield the error event before ``[DONE]``)."""
    serving = GenerateBaseServing.__new__(GenerateBaseServing)

    with pytest.raises(GenerationError) as raised:
        serving._raise_if_error(str(FinishReason.ERROR), "bad")

    assert raised.value.status_code == 500
    event = serving._convert_generation_error_to_streaming_response(raised.value)
    assert '"code": 500' in event


# endregion Scheduler
