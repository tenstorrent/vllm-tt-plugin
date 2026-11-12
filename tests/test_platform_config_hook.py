# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``TTPlatform.check_and_update_config`` run inside a real ``VllmConfig``."""

from unittest.mock import patch

import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.test_block_scheduler import LOCAL_MODEL_CONFIG, _stub_model_resolution


def test_the_tt_class_decides_multimodal_input_support(monkeypatch):
    # ModelConfig answers this question before the platform hook renames the
    # architecture to its TT class. Model an architecture whose upstream class
    # takes multimodal input while its TT class is text-only (Gemma4): the
    # answer must follow the TT class, or the renderer builds a multimodal
    # processor the TT class does not register.
    monkeypatch.setattr(
        ModelConfig,
        "_supports_multimodal_inputs",
        lambda self: not self.hf_config.architectures[0].startswith("TT"),
    )
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    assert model_config.supports_multimodal_inputs is True

    with _stub_model_resolution():
        config = VllmConfig(
            model_config=model_config,
            scheduler_config=SchedulerConfig(
                max_model_len=model_config.max_model_len,
                is_encoder_decoder=model_config.is_encoder_decoder,
            ),
            cache_config=CacheConfig(block_size=128),
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )

    assert config.model_config.hf_config.architectures == ["TTQwen2ForCausalLM"]
    assert config.model_config.supports_multimodal_inputs is False


def _launch(monkeypatch, *, mamba: bool, multimodal: bool) -> VllmConfig:
    """A launch with prefix caching and chunked prefill of a TT model class
    that declares both, for an architecture vLLM gives a Mamba cache config.

    The class's ``get_kv_cache_spec`` returns an attention layer and, with
    ``mamba``, a ``MambaSpec`` layer. ``multimodal`` makes the launch accept
    multimodal input.
    """

    class Model:
        model_capabilities = {
            "supports_prefix_caching": True,
            "supports_chunked_prefill": True,
        }

        @staticmethod
        def get_kv_cache_spec(_vllm_config):
            specs = {
                "model.layers.0.self_attn": FullAttentionSpec(
                    block_size=1024, num_kv_heads=2, head_size=8, dtype=torch.float16
                )
            }
            if mamba:
                specs["model.layers.1.linear_attn"] = MambaSpec(
                    block_size=1024,
                    shapes=((3, 8),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode="align",
                )
            return specs

    monkeypatch.setattr(
        ModelConfig, "is_multimodal_model", property(lambda self: multimodal)
    )
    monkeypatch.setattr(
        ModelConfig, "_supports_multimodal_inputs", lambda self: multimodal
    )
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    with (
        _stub_model_resolution(),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(Model, None),
        ),
    ):
        return VllmConfig(
            model_config=model_config,
            scheduler_config=SchedulerConfig(
                max_model_len=model_config.max_model_len,
                is_encoder_decoder=False,
                enable_chunked_prefill=True,
                max_num_batched_tokens=2048,
            ),
            cache_config=CacheConfig(
                block_size=1024,
                enable_prefix_caching=True,
                mamba_block_size=1024,
                mamba_cache_mode="align",
            ),
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )


def test_multimodal_launch_of_a_mamba_model_turns_prefix_caching_and_chunking_off(
    monkeypatch,
):
    """A MambaSpec model cannot continue a multimodal prompt yet, so a launch
    that accepts multimodal input runs in mamba_cache_mode none with whole
    prompts, as for a model that declares neither feature."""
    config = _launch(monkeypatch, mamba=True, multimodal=True)

    assert config.cache_config.enable_prefix_caching is False
    assert config.cache_config.mamba_cache_mode == "none"
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0


def test_text_only_launch_of_a_mamba_model_keeps_align_and_chunking(monkeypatch):
    config = _launch(monkeypatch, mamba=True, multimodal=False)

    assert config.cache_config.enable_prefix_caching is True
    assert config.cache_config.mamba_cache_mode == "align"
    assert config.scheduler_config.enable_chunked_prefill is True
    # VllmConfig.validate_block_size rejects the flag in mode align.
    assert config.scheduler_config.disable_chunked_mm_input is False


def test_multimodal_launch_of_a_model_without_mamba_spec_is_unchanged(monkeypatch):
    """A model class whose recurrent state is not declared through MambaSpec
    keeps both features and vLLM's multimodal chunking rule."""
    config = _launch(monkeypatch, mamba=False, multimodal=True)

    assert config.cache_config.enable_prefix_caching is True
    assert config.cache_config.mamba_cache_mode == "align"
    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.disable_chunked_mm_input is True
