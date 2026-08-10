# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from vllm_tt_plugin.config import get_tt_output_tokens_per_step
from vllm_tt_plugin.platform import TTPlatform


@pytest.fixture(autouse=True)
def block_contract(monkeypatch):
    monkeypatch.setattr(TTPlatform, "output_tokens_per_step", 256)
    monkeypatch.setattr(TTPlatform, "block_model_max_len", 1024)


def _validate(params: SamplingParams, prompt_len: int = 32) -> None:
    prompt = {"prompt_token_ids": [1] * prompt_len}
    TTPlatform.validate_request(prompt, params)


def test_short_logical_request_reserves_one_physical_canvas():
    _validate(SamplingParams(max_tokens=16), prompt_len=768)


def test_rounded_physical_capacity_is_enforced():
    with pytest.raises(ValueError, match=r"requires 512 physical.*exceeding"):
        _validate(SamplingParams(max_tokens=257), prompt_len=513)


def test_platform_default_is_clamped_to_whole_canvases():
    platform = TTPlatform()

    assert platform.get_max_output_tokens(32) == 768
    assert platform.get_max_output_tokens(768) == 256
    with pytest.raises(ValueError, match="physical 256-token output canvases"):
        platform.get_max_output_tokens(769)


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
def test_non_neutral_sampling_controls_are_rejected_without_mutation(field, value):
    params = SamplingParams(max_tokens=16, **{field: value})

    with pytest.raises(ValueError, match=field):
        _validate(params)

    assert getattr(params, field) == value


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
