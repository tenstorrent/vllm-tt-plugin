# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from vllm_tt_plugin.config import (
    get_tt_output_tokens_per_step,
    store_tt_output_tokens_per_step,
)
from vllm_tt_plugin.platform import (
    TTPlatform,
    _install_block_output_input_processor_patch,
)


@pytest.fixture(autouse=True)
def block_contract(monkeypatch):
    monkeypatch.setattr(TTPlatform, "output_tokens_per_step", 256)
    monkeypatch.setattr(
        TTPlatform,
        "block_model_config",
        SimpleNamespace(max_model_len=1024),
    )


def _validate(params: SamplingParams, prompt_len: int = 32) -> None:
    prompt = {"prompt_token_ids": [1] * prompt_len}
    TTPlatform.validate_request(prompt, params)


def _processor_harness(monkeypatch, *, output_size=256, max_model_len=1024):
    """Install the vLLM 0.24 validate -> clone -> default boundary."""
    import vllm.v1.engine.input_processor as input_processor

    def process_inputs(self, request_id, prompt, params, *args, **kwargs):
        self.process_inputs_calls.append(kwargs)
        TTPlatform.validate_request(prompt, params)
        cloned_params = params.clone()
        if cloned_params.max_tokens is None:
            seq_len = len(prompt.get("prompt_token_ids") or prompt["prompt_embeds"])
            cloned_params.max_tokens = self.model_config.max_model_len - seq_len
        return SimpleNamespace(
            sampling_params=cloned_params,
            resumable=kwargs.get("resumable", False),
        )

    monkeypatch.setattr(
        input_processor.InputProcessor,
        "process_inputs",
        process_inputs,
    )
    monkeypatch.delattr(
        input_processor,
        "_tt_original_process_inputs",
        raising=False,
    )
    _install_block_output_input_processor_patch()

    model_config = SimpleNamespace(max_model_len=max_model_len)
    config = SimpleNamespace(additional_config={}, model_config=model_config)
    store_tt_output_tokens_per_step(config, output_size)
    monkeypatch.setattr(TTPlatform, "output_tokens_per_step", output_size)
    monkeypatch.setattr(TTPlatform, "block_model_config", model_config)
    return input_processor.InputProcessor, SimpleNamespace(
        vllm_config=config,
        model_config=model_config,
        process_inputs_calls=[],
    )


def test_short_logical_request_reserves_one_physical_canvas():
    _validate(SamplingParams(max_tokens=16), prompt_len=768)


def test_rounded_physical_capacity_is_enforced():
    with pytest.raises(ValueError, match=r"requires 512 physical.*exceeding"):
        _validate(SamplingParams(max_tokens=257), prompt_len=513)


def test_auto_fitted_frontend_limit_is_read_live(monkeypatch):
    config = _config()
    _patch_model_resolution(monkeypatch)
    TTPlatform.check_and_update_config(config)
    platform = TTPlatform()

    assert platform.get_max_output_tokens(32) == 768

    # Mirrors vLLM 0.24 CoreEngineReadyResponse updating the same frontend
    # ModelConfig object after --max-model-len -1 auto-fit.
    config.model_config.max_model_len = 640

    assert platform.get_max_output_tokens(32) == 512
    with pytest.raises(ValueError, match=r"requires 256 physical.*exceeding"):
        _validate(SamplingParams(max_tokens=16), prompt_len=400)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.5),
        ("top_p", 0.9),
        ("top_k", 10),
        ("min_p", 0.1),
        ("seed", 42),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("repetition_penalty", 1.1),
    ],
)
def test_non_neutral_sampling_controls_are_accepted_and_neutralized(field, value):
    params = SamplingParams(max_tokens=16, **{field: value})

    _validate(params)

    expected = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "seed": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert getattr(params, field) == expected[field]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"n": 2}, "n"),
        ({"logprobs": 0}, "logprobs"),
        ({"bad_words": ["forbidden"]}, "bad_words"),
        (
            {"structured_outputs": StructuredOutputsParams(json_object=True)},
            "structured_outputs",
        ),
        ({"logit_bias": {2: -1.0}}, "logit_bias"),
        ({"allowed_token_ids": [2, 3]}, "allowed_token_ids"),
        ({"min_tokens": 1}, "min_tokens"),
        ({"extra_args": {"custom_sampler": True}}, "extra_args"),
    ],
)
def test_contract_changing_controls_are_rejected(kwargs, field):
    with pytest.raises(ValueError, match=field):
        _validate(SamplingParams(max_tokens=16, **kwargs))


def test_rejected_request_leaves_params_unmutated():
    params = SamplingParams(max_tokens=16, temperature=0.5, n=2)

    with pytest.raises(ValueError, match="n=2"):
        _validate(params)

    assert params.temperature == 0.5


def test_ar_request_validation_is_unchanged(monkeypatch):
    monkeypatch.setattr(TTPlatform, "output_tokens_per_step", 1)
    params = SamplingParams(max_tokens=16, temperature=0.5, n=2)

    _validate(params)

    assert params.temperature == 0.5
    assert params.n == 2


def _config(*, max_num_seqs=1, data_parallel_size=1, async_scheduling=False):
    return SimpleNamespace(
        additional_config={"tt": {"sample_on_device_mode": "all"}},
        model_config=SimpleNamespace(
            model="test-model",
            hf_config=SimpleNamespace(
                architectures=["FutureBlockModel"],
                model_type="gemma4",
            ),
            max_model_len=1024,
            original_max_model_len=None,
            max_logprobs=20,
            is_moe=False,
            generation_config="auto",
            logits_processors=None,
            get_sliding_window=lambda: None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=1024,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=128,
            async_scheduling=async_scheduling,
            scheduler_cls=None,
            verify_max_model_len=lambda _max_model_len: None,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        structured_outputs_config=SimpleNamespace(disable_any_whitespace=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=data_parallel_size,
            worker_cls="auto",
        ),
        diffusion_config=SimpleNamespace(canvas_length=256),
        speculative_config=None,
        lora_config=None,
    )


class BlockModel:
    model_capabilities = {
        "output_tokens_per_step": 256,
        "supports_sample_on_device": True,
        "supports_async_decode": False,
        "supports_prefix_caching": False,
    }


class ARModel:
    model_capabilities = {
        "output_tokens_per_step": 1,
        "supports_sample_on_device": True,
        "supports_async_decode": False,
        "supports_prefix_caching": False,
    }


def _patch_model_resolution(monkeypatch, model_class=BlockModel):
    monkeypatch.setattr(
        "vllm_tt_plugin.platform.register_tt_models", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
        lambda: ["TTFutureBlockModel"],
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda _model_config: (model_class, None),
    )


def test_startup_stores_block_capability_and_enforces_contract(monkeypatch):
    config = _config()
    _patch_model_resolution(monkeypatch)

    TTPlatform.check_and_update_config(config)

    assert get_tt_output_tokens_per_step(config) == 256
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.model_config.generation_config == "vllm"
    assert config.diffusion_config is None


@pytest.mark.parametrize(
    ("model_class", "expected_max_tokens", "expected_wrapped"),
    [
        (BlockModel, 768, True),
        (ARModel, 824, False),
    ],
)
def test_startup_installs_input_processor_patch_only_for_block_models(
    monkeypatch, model_class, expected_max_tokens, expected_wrapped
):
    import vllm.v1.engine.input_processor as input_processor

    def process_inputs(self, request_id, prompt, params, *args, **kwargs):
        TTPlatform.validate_request(prompt, params)
        cloned_params = params.clone()
        if cloned_params.max_tokens is None:
            cloned_params.max_tokens = self.model_config.max_model_len - len(
                prompt["prompt_token_ids"]
            )
        return SimpleNamespace(sampling_params=cloned_params)

    monkeypatch.setattr(
        input_processor.InputProcessor,
        "process_inputs",
        process_inputs,
    )
    monkeypatch.delattr(
        input_processor,
        "_tt_original_process_inputs",
        raising=False,
    )
    config = _config()
    _patch_model_resolution(monkeypatch, model_class)

    TTPlatform.check_and_update_config(config)

    assert (
        input_processor.InputProcessor.process_inputs is not process_inputs
    ) is expected_wrapped
    assert hasattr(input_processor, "_tt_original_process_inputs") is expected_wrapped

    params = SamplingParams(max_tokens=16)
    params.max_tokens = None
    processor = SimpleNamespace(
        vllm_config=config,
        model_config=config.model_config,
    )
    request = input_processor.InputProcessor.process_inputs(
        processor,
        "request",
        {"prompt_token_ids": [1] * 200},
        params,
    )

    assert request.sampling_params.max_tokens == expected_max_tokens
    assert params.max_tokens is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_num_seqs": 2}, "max-num-seqs 1"),
        ({"data_parallel_size": 2}, "data parallelism"),
        ({"async_scheduling": True}, "synchronous serving only"),
    ],
)
def test_startup_rejects_unsupported_block_modes(monkeypatch, overrides, message):
    config = _config(**overrides)
    _patch_model_resolution(monkeypatch)

    with pytest.raises(ValueError, match=message):
        TTPlatform.check_and_update_config(config)


@pytest.mark.parametrize(
    ("prompt_lens", "expected"),
    [
        ((200, 600), (768, 256)),
        ((600, 200), (256, 768)),
    ],
)
def test_input_processor_rounds_only_cloned_unresolved_default(
    monkeypatch, prompt_lens, expected
):
    processor_cls, processor = _processor_harness(monkeypatch)
    params = SamplingParams(max_tokens=16)
    params.max_tokens = None

    actual = tuple(
        processor_cls.process_inputs(
            processor,
            f"request-{prompt_len}",
            {"prompt_token_ids": [1] * prompt_len},
            params,
        ).sampling_params.max_tokens
        for prompt_len in prompt_lens
    )

    assert actual == expected
    assert params.max_tokens is None


def test_input_processor_preserves_explicit_max_tokens(monkeypatch):
    processor_cls, processor = _processor_harness(monkeypatch)
    params = SamplingParams(max_tokens=17)

    request = processor_cls.process_inputs(
        processor,
        "explicit",
        {"prompt_token_ids": [1] * 200},
        params,
    )

    assert request.sampling_params.max_tokens == 17
    assert params.max_tokens == 17


def test_input_processor_rejects_resumable_block_request_before_upstream(
    monkeypatch,
):
    processor_cls, processor = _processor_harness(monkeypatch)

    with pytest.raises(ValueError, match="do not support resumable streaming-input"):
        processor_cls.process_inputs(
            processor,
            "streaming-input",
            {"prompt_token_ids": [1] * 200},
            SamplingParams(max_tokens=256),
            resumable=True,
        )

    assert processor.process_inputs_calls == []


def test_input_processor_preserves_resumable_ar_request(monkeypatch):
    processor_cls, processor = _processor_harness(monkeypatch, output_size=1)

    request = processor_cls.process_inputs(
        processor,
        "streaming-input-ar",
        {"prompt_token_ids": [1] * 200},
        SamplingParams(max_tokens=16),
        resumable=True,
    )

    assert request.resumable is True
    assert processor.process_inputs_calls == [{"resumable": True}]


def test_input_processor_rejects_zero_canvas_default_for_embeds(monkeypatch):
    """Inputs without prompt_token_ids skip validate_request's canvas check;
    the wrapper must still reject a whole-canvas default that rounds to 0."""
    processor_cls, processor = _processor_harness(monkeypatch)
    params = SamplingParams(max_tokens=16)
    params.max_tokens = None

    with pytest.raises(ValueError, match="fewer than one physical"):
        processor_cls.process_inputs(
            processor,
            "embeds",
            {"prompt_token_ids": None, "prompt_embeds": [0] * 900},
            params,
        )

    assert params.max_tokens is None


def test_input_processor_rejects_when_no_canvas_fits(monkeypatch):
    processor_cls, processor = _processor_harness(monkeypatch)
    params = SamplingParams(max_tokens=16)
    params.max_tokens = None

    with pytest.raises(ValueError, match="physical 256-token output canvases"):
        processor_cls.process_inputs(
            processor,
            "too-long",
            {"prompt_token_ids": [1] * 1000},
            params,
        )

    assert params.max_tokens is None
