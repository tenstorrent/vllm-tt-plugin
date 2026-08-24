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
    _fit_block_output_max_tokens,
    _install_block_output_input_processor_patch,
    _install_block_output_streaming_input_patch,
)


def _set_platform_contract(monkeypatch, *, output_size=256, max_model_len=1024):
    model_config = SimpleNamespace(max_model_len=max_model_len)
    config = SimpleNamespace(additional_config={}, model_config=model_config)
    store_tt_output_tokens_per_step(config, output_size)
    monkeypatch.setattr(TTPlatform, "_tt_vllm_config", config)
    return config


@pytest.fixture(autouse=True)
def block_contract(monkeypatch):
    _set_platform_contract(monkeypatch)


def _validate(params: SamplingParams, prompt_len: int = 32) -> None:
    prompt = {"prompt_token_ids": [1] * prompt_len}
    TTPlatform.validate_request(prompt, params)


def _upstream_process_inputs(
    self, request_id, prompt, params, *args, resumable=False, **kwargs
):
    """The vLLM 0.25.1 validate -> clone -> resolve-default boundary."""
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
    """Install the vLLM 0.25.1 validate -> clone -> default boundary."""
    input_processor = _patch_upstream_boundary(monkeypatch)
    _install_block_output_input_processor_patch()

    config = _set_platform_contract(
        monkeypatch,
        output_size=output_size,
        max_model_len=max_model_len,
    )
    model_config = config.model_config
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

    # Mirrors vLLM 0.25.1 EngineCoreReadyResponse updating the same frontend
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
    _set_platform_contract(monkeypatch, output_size=1)
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
    max_model_len=1024,
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
            max_model_len=max_model_len,
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
        diffusion_config=None,
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


class _WeakrefableConfig(SimpleNamespace):
    """SimpleNamespace itself cannot be weak-referenced; VllmConfig can."""


def _weakrefable_config():
    return _WeakrefableConfig(**vars(_config()))


def _patch_model_resolution(monkeypatch, model_class=BlockModel):
    # Simulate a fresh process: the autouse block_contract fixture pre-seeds
    # the class handle, which the hook's one-engine-per-process guard would
    # otherwise mistake for a live engine.
    monkeypatch.setattr(TTPlatform, "_tt_vllm_config", None)
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
    assert TTPlatform._tt_vllm_config is config
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.model_config.generation_config == "vllm"
    assert config.diffusion_config is None
    assert not hasattr(config.model_config.hf_config, "canvas_length")
    assert config.model_config.is_diffusion is False


def test_admission_handle_releases_when_engine_config_is_collected(monkeypatch):
    # The one-engine-per-process guard must protect a LIVE engine only: once
    # the previous engine's config is garbage-collected, a new engine in the
    # same process is admitted instead of failing forever.
    import gc

    _patch_model_resolution(monkeypatch)
    first = _weakrefable_config()
    TTPlatform.check_and_update_config(first)
    with pytest.raises(ValueError, match=r"One TT engine per process"):
        TTPlatform.check_and_update_config(_weakrefable_config())

    del first
    gc.collect()

    TTPlatform.check_and_update_config(_weakrefable_config())


def test_cycle_pinned_dead_ar_config_does_not_block_a_block_engine(monkeypatch):
    # An AR engine shares the process freely, so its handle survives the entry
    # guard. When it dies leaving its config reachable only through a
    # reference cycle, the post-apply guard must collect before refusing to
    # start a block engine.
    _patch_model_resolution(monkeypatch, model_class=ARModel)
    ar_config = _weakrefable_config()
    ar_config.model_config.hf_config.canvas_length = None
    ar_config._self_cycle = ar_config
    TTPlatform.check_and_update_config(ar_config)

    del ar_config

    # Switch only the model resolution: _patch_model_resolution would reset
    # the admission handle and defeat the scenario.
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda _model_config: (BlockModel, None),
    )
    TTPlatform.check_and_update_config(_config())


def test_admission_guard_raise_does_not_pin_the_stale_config(monkeypatch):
    # The guard's own traceback must not become the reference that keeps a
    # dead engine's config alive: in a REPL, sys.last_traceback holds the
    # failed attempt, and a self-pinning raise would fail every retry.
    _patch_model_resolution(monkeypatch)
    first = _weakrefable_config()
    TTPlatform.check_and_update_config(first)

    with pytest.raises(ValueError, match=r"One TT engine per process") as excinfo:
        TTPlatform.check_and_update_config(_weakrefable_config())

    del first
    # excinfo still holds the guard's traceback, standing in for
    # sys.last_traceback; only the engine's own reference was dropped.
    TTPlatform.check_and_update_config(_weakrefable_config())
    assert excinfo.value is not None


def test_startup_rejects_explicit_diffusion_config(monkeypatch):
    # The general-plugins architecture rewrite keeps upstream's hook from
    # auto-creating this config, so any non-None value is explicit.
    config = _config()
    config.diffusion_config = SimpleNamespace(canvas_length=256)
    _patch_model_resolution(monkeypatch)

    with pytest.raises(ValueError, match=r"do not support --diffusion-config"):
        TTPlatform.check_and_update_config(config)


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


@pytest.mark.parametrize(
    "tt_config",
    [{}, {"sample_on_device_mode": "decode_only"}],
    ids=["unset", "decode-only"],
)
def test_startup_requires_device_sampling_for_block_models(monkeypatch, tt_config):
    """The block contract forces sample_on_device_mode='all'; anything else
    starts up cleanly and then dies on the first request when host sampling
    cannot construct a multi-token canvas."""
    config = _config()
    config.additional_config = {"tt": tt_config}
    _patch_model_resolution(monkeypatch)

    with pytest.raises(ValueError, match='sample_on_device_mode="all"'):
        TTPlatform.check_and_update_config(config)


def test_startup_rejects_logits_processors_for_block_models(monkeypatch):
    """Custom logits processors flip the step onto host sampling; without the
    gate the launch succeeds and the first request kills the engine."""
    config = _config()
    config.model_config.logits_processors = ["my.module.CustomProc"]
    _patch_model_resolution(monkeypatch)

    with pytest.raises(ValueError, match="logits-processors"):
        TTPlatform.check_and_update_config(config)


@pytest.mark.parametrize(
    ("tt_overrides", "warns"),
    [
        ({}, False),
        ({"trace_mode": "decode_only"}, True),
        ({"enable_model_warmup": False}, True),
    ],
    ids=["defaults", "trace-mode", "warmup-off"],
)
def test_startup_warns_when_block_capture_prerequisites_are_off(
    monkeypatch, tt_overrides, warns
):
    """A capture-based block model needs trace_mode='all' and warmup on;
    without them the failure would otherwise surface only on the first
    request, engine-fatally. Eager serving is legitimate, so this warns."""
    import vllm_tt_plugin.platform as platform_module

    config = _config()
    config.additional_config["tt"].update(tt_overrides)
    _patch_model_resolution(monkeypatch)
    warnings: list[str] = []
    original_warning = platform_module.logger.warning
    monkeypatch.setattr(
        platform_module.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args) or original_warning(msg, *args),
    )

    TTPlatform.check_and_update_config(config)

    matched = [w for w in warnings if "Block-output serving is validated" in w]
    assert bool(matched) is warns


def test_startup_rejects_the_rust_frontend_for_block_models(monkeypatch):
    # The Rust frontend serves HTTP outside Python, bypassing validate_request
    # and the InputProcessor patch entirely.
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")

    ar_config = _config()
    _patch_model_resolution(monkeypatch, ARModel)
    TTPlatform.check_and_update_config(ar_config)

    config = _config()
    _patch_model_resolution(monkeypatch)
    with pytest.raises(ValueError, match="Rust frontend"):
        TTPlatform.check_and_update_config(config)


def test_second_engine_config_is_rejected_when_a_block_model_is_involved(
    monkeypatch,
):
    """The platform keeps process-level admission state (one config handle);
    a second engine with a different contract would silently corrupt it. The
    same object re-running the hook (EngineCore re-runs __post_init__) passes."""
    _patch_model_resolution(monkeypatch)
    config = _config()
    TTPlatform.check_and_update_config(config)
    TTPlatform.check_and_update_config(config)  # same object: fine

    other = _config()
    with pytest.raises(ValueError, match="One TT engine per process"):
        TTPlatform.check_and_update_config(other)


def test_failed_same_config_reentry_restores_live_block_contract(monkeypatch):
    """A failed in-process re-run must not leave the admission handle at None.

    EngineCore / worker init_device re-enter this hook on the same config
    object, which skips the one-engine guard and used to wipe the handle
    before a later raise.
    """
    _patch_model_resolution(monkeypatch)
    config = _config()
    TTPlatform.check_and_update_config(config)
    assert TTPlatform._tt_vllm_config is config

    config.scheduler_config.max_num_seqs = 2
    with pytest.raises(ValueError, match="max-num-seqs 1"):
        TTPlatform.check_and_update_config(config)

    assert TTPlatform._tt_vllm_config is config
    _validate(SamplingParams(max_tokens=16), prompt_len=32)
    with pytest.raises(ValueError, match=r"requires 512 physical"):
        _validate(SamplingParams(max_tokens=257), prompt_len=513)


def test_failed_block_start_restores_prior_admission_handle(monkeypatch):
    _patch_model_resolution(monkeypatch, ARModel)
    ar_config = _config()
    TTPlatform.check_and_update_config(ar_config)
    assert TTPlatform._tt_vllm_config is ar_config

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda _model_config: (BlockModel, None),
    )
    with pytest.raises(ValueError, match="max-num-seqs 1"):
        TTPlatform.check_and_update_config(_config(max_num_seqs=2))

    assert TTPlatform._tt_vllm_config is ar_config


def test_failed_first_config_hook_leaves_admission_handle_unset(monkeypatch):
    _patch_model_resolution(monkeypatch)
    with pytest.raises(ValueError, match="max-num-seqs 1"):
        TTPlatform.check_and_update_config(_config(max_num_seqs=2))
    assert TTPlatform._tt_vllm_config is None


@pytest.mark.parametrize("max_model_len", [256, 280, 288, 512])
def test_startup_admits_exactly_the_lengths_a_request_can_use(
    monkeypatch, max_model_len
):
    """Startup and the per-request check must agree on the shortest usable length.

    Requiring only one canvas let ``max_model_len == output_tokens_per_step``
    start cleanly and then reject every request: a prompt is rounded up to a
    32-token tile, so the canvas needs a tile of headroom on top.
    """
    config = _config(max_model_len=max_model_len)
    _patch_model_resolution(monkeypatch)

    try:
        TTPlatform.check_and_update_config(config)
        starts = True
    except ValueError as exc:
        assert "must be at least 288" in str(exc)
        starts = False

    try:
        # The shortest possible prompt still occupies one whole tile.
        TTPlatform._resolve_block_output_max_tokens(1, None, 256, max_model_len)
        serves = True
    except ValueError:
        serves = False

    assert starts == serves


def test_fit_clamps_leftover_max_tokens_that_resolve_rejects():
    """Natural leftover max_tokens = max_model_len - prompt_len needs four
    256-token canvases after a 32-tile prompt, which overshoots 1024."""
    prompt_len = 100
    max_model_len = 1024
    output_size = 256
    leftover = max_model_len - prompt_len

    with pytest.raises(ValueError, match=r"requires 1024 physical"):
        TTPlatform._resolve_block_output_max_tokens(
            prompt_len, leftover, output_size, max_model_len
        )

    assert (
        _fit_block_output_max_tokens(prompt_len, leftover, output_size, max_model_len)
        == 768
    )
    assert (
        _fit_block_output_max_tokens(prompt_len, 16, output_size, max_model_len) == 16
    )
    assert (
        _fit_block_output_max_tokens(prompt_len, None, output_size, max_model_len)
        == 768
    )


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
    _set_platform_contract(monkeypatch, output_size=output_size)

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
    _set_platform_contract(monkeypatch, max_model_len=1000)

    with pytest.raises(
        ValueError,
        match=r"prompt length 737 \(aligned to 768\).*aligned max_model_len=992",
    ):
        _validate(SamplingParams(max_tokens=1), prompt_len=737)

    # The neighboring aligned prompt fits one complete 256-token canvas.
    _validate(SamplingParams(max_tokens=1), prompt_len=736)


def test_default_max_tokens_without_canvas_room_is_rejected_at_validation():
    # validate_request runs before the input-processor patch resolves the
    # whole-canvas default, so a prompt leaving no room for one canvas must
    # already reject there — and must not mutate the caller-owned params.
    params = SamplingParams(max_tokens=16)
    params.max_tokens = None

    with pytest.raises(
        ValueError,
        match=r"physical 256-token canvases.*requires 256 physical.*exceeding",
    ):
        _validate(params, prompt_len=1000)

    assert params.max_tokens is None
