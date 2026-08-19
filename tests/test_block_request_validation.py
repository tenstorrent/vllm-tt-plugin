# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import asyncio
from functools import cached_property
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
    _install_block_output_streaming_input_patch,
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


def _upstream_process_inputs(
    self, request_id, prompt, params, *args, resumable=False, **kwargs
):
    """The vLLM 0.24 validate -> clone -> resolve-default boundary."""
    calls = getattr(self, "process_inputs_calls", None)
    if calls is not None:
        calls.append({"resumable": resumable})
    TTPlatform.validate_request(prompt, params)
    cloned_params = params.clone()
    if cloned_params.max_tokens is None:
        seq_len = len(prompt.get("prompt_token_ids") or prompt["prompt_embeds"])
        cloned_params.max_tokens = self.model_config.max_model_len - seq_len
    return SimpleNamespace(
        sampling_params=cloned_params,
        resumable=resumable,
        prompt_token_ids=prompt.get("prompt_token_ids"),
        prompt_embeds=prompt.get("prompt_embeds"),
    )


def _patch_upstream_boundary(monkeypatch):
    import vllm.v1.engine.input_processor as input_processor

    monkeypatch.setattr(
        input_processor.InputProcessor,
        "process_inputs",
        _upstream_process_inputs,
    )
    monkeypatch.delattr(
        input_processor,
        "_tt_original_process_inputs",
        raising=False,
    )
    return input_processor


def _processor_harness(monkeypatch, *, output_size=256, max_model_len=1024):
    """Install the vLLM 0.24 validate -> clone -> default boundary."""
    input_processor = _patch_upstream_boundary(monkeypatch)
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
    ("field", "value", "neutral"),
    [
        ("temperature", 0.5, 1.0),
        ("top_p", 0.9, 1.0),
        ("top_k", 10, 0),
        ("min_p", 0.1, 0.0),
        ("seed", 42, None),
        ("presence_penalty", 0.5, 0.0),
        ("frequency_penalty", 0.5, 0.0),
        ("repetition_penalty", 1.1, 1.0),
    ],
)
def test_non_neutral_sampling_controls_neutralized_on_clone_only(
    monkeypatch, field, value, neutral
):
    processor_cls, processor = _processor_harness(monkeypatch)
    params = SamplingParams(max_tokens=16, **{field: value})

    request = processor_cls.process_inputs(
        processor,
        f"sampling-{field}",
        {"prompt_token_ids": [1] * 200},
        params,
    )

    assert getattr(request.sampling_params, field) == neutral
    assert getattr(params, field) == value


def test_sampling_controls_are_accepted_by_validation():
    _validate(SamplingParams(max_tokens=16, temperature=0.5, seed=42))


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


class _ModelConfig(SimpleNamespace):
    @cached_property
    def is_diffusion(self) -> bool:
        return getattr(self.hf_config, "canvas_length", None) is not None


def _config(
    *,
    max_num_seqs=1,
    data_parallel_size=1,
    async_scheduling=False,
    distributed_executor_backend=None,
):
    return SimpleNamespace(
        additional_config={"tt": {"sample_on_device_mode": "all"}},
        model_config=_ModelConfig(
            model="test-model",
            hf_config=SimpleNamespace(
                architectures=["FutureBlockModel"],
                model_type="gemma4",
                canvas_length=256,
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
            distributed_executor_backend=distributed_executor_backend,
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

    @staticmethod
    def release_request(_slot):
        pass

    @staticmethod
    def release_persistent_capture():
        pass


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
    assert config.model_config.is_diffusion is True

    TTPlatform.check_and_update_config(config)

    assert get_tt_output_tokens_per_step(config) == 256
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.model_config.generation_config == "vllm"
    assert config.diffusion_config is None
    assert not hasattr(config.model_config.hf_config, "canvas_length")
    assert config.model_config.is_diffusion is False


def test_startup_rejects_mismatched_diffusion_canvas(monkeypatch):
    config = _config()
    config.model_config.hf_config.canvas_length = 128
    _patch_model_resolution(monkeypatch)

    with pytest.raises(ValueError, match=r"128 != 256"):
        TTPlatform.check_and_update_config(config)


def test_startup_requires_block_model_lifecycle_hooks(monkeypatch):
    class BlockModelWithoutLifecycle:
        model_capabilities = BlockModel.model_capabilities

    config = _config()
    _patch_model_resolution(monkeypatch, BlockModelWithoutLifecycle)

    with pytest.raises(
        ValueError,
        match=r"release_request, release_persistent_capture",
    ):
        TTPlatform.check_and_update_config(config)


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
    input_processor = _patch_upstream_boundary(monkeypatch)
    config = _config()
    _patch_model_resolution(monkeypatch, model_class)

    TTPlatform.check_and_update_config(config)

    assert (
        input_processor.InputProcessor.process_inputs is not _upstream_process_inputs
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
        ({"distributed_executor_backend": "mp"}, "uniproc executor"),
        ({"async_scheduling": True}, "synchronous serving only"),
    ],
)
def test_startup_rejects_unsupported_block_modes(monkeypatch, overrides, message):
    config = _config(**overrides)
    _patch_model_resolution(monkeypatch)

    with pytest.raises(ValueError, match=message):
        TTPlatform.check_and_update_config(config)


def test_startup_auto_disables_prefix_caching_for_block_models(monkeypatch):
    config = _config()
    config.cache_config.enable_prefix_caching = True
    _patch_model_resolution(monkeypatch)

    TTPlatform.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is False


def test_startup_rejects_block_model_declaring_prefix_caching(monkeypatch):
    class BlockModelClaimingPrefixCaching(BlockModel):
        model_capabilities = {
            **BlockModel.model_capabilities,
            "supports_prefix_caching": True,
        }

    config = _config()
    _patch_model_resolution(monkeypatch, BlockModelClaimingPrefixCaching)

    with pytest.raises(
        ValueError,
        match=r"supports_prefix_caching.*capability declaration",
    ):
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
        SamplingParams(max_tokens=16, temperature=0.5),
        resumable=True,
    )

    assert request.resumable is True
    assert processor.process_inputs_calls == [{"resumable": True}]
    # AR requests keep their sampling controls; neutralization is
    # block-output only.
    assert request.sampling_params.temperature == 0.5


@pytest.mark.parametrize("output_size", [1, 256], ids=["ar", "block"])
def test_prompt_embeds_are_rejected_for_every_tt_model(monkeypatch, output_size):
    monkeypatch.setattr(TTPlatform, "output_tokens_per_step", output_size)

    with pytest.raises(ValueError, match=r"prompt_embeds are not supported"):
        TTPlatform.validate_request(
            {"prompt_token_ids": None, "prompt_embeds": [0] * 513},
            SamplingParams(max_tokens=257),
        )


@pytest.mark.parametrize("output_size", [1, 256], ids=["ar", "block"])
def test_streaming_input_session_is_rejected_only_for_block_models(
    monkeypatch, output_size
):
    import vllm.v1.engine.async_llm as async_llm

    calls = []

    async def original(_self, *args, **kwargs):
        calls.append((args, kwargs))
        return "accepted"

    monkeypatch.setattr(async_llm.AsyncLLM, "_add_streaming_input_request", original)
    monkeypatch.delattr(
        async_llm,
        "_tt_original_add_streaming_input_request",
        raising=False,
    )
    _install_block_output_streaming_input_patch()

    config = SimpleNamespace(additional_config={})
    store_tt_output_tokens_per_step(config, output_size)
    engine = SimpleNamespace(vllm_config=config)

    if output_size > 1:
        with pytest.raises(
            ValueError, match="do not support resumable streaming-input"
        ):
            asyncio.run(async_llm.AsyncLLM._add_streaming_input_request(engine))
        assert calls == []
    else:
        result = asyncio.run(
            async_llm.AsyncLLM._add_streaming_input_request(engine, "request")
        )
        assert result == "accepted"
        assert calls == [(("request",), {})]


def test_explicit_request_uses_tile_aligned_physical_capacity(monkeypatch):
    monkeypatch.setattr(
        TTPlatform,
        "block_model_config",
        SimpleNamespace(max_model_len=1000),
    )

    with pytest.raises(
        ValueError,
        match=r"prompt length 737 \(aligned to 768\).*aligned max_model_len=992",
    ):
        _validate(SamplingParams(max_tokens=1), prompt_len=737)

    # The neighboring aligned prompt fits one complete 256-token canvas.
    _validate(SamplingParams(max_tokens=1), prompt_len=736)


@pytest.mark.parametrize(
    ("prompt", "error"),
    [
        pytest.param(
            {"prompt_token_ids": [1] * 1000},
            r"physical 256-token canvases.*requires 256 physical.*exceeding",
            id="token-ids-no-room",
        ),
        pytest.param(
            {"prompt_token_ids": None, "prompt_embeds": [0] * 900},
            r"prompt_embeds are not supported",
            id="embeds-partial-tile",
        ),
    ],
)
def test_input_processor_rejects_default_without_whole_canvas(
    monkeypatch, prompt, error
):
    processor_cls, processor = _processor_harness(monkeypatch)
    params = SamplingParams(max_tokens=16)
    params.max_tokens = None

    with pytest.raises(ValueError, match=error):
        processor_cls.process_inputs(processor, "no-canvas", prompt, params)

    assert params.max_tokens is None
