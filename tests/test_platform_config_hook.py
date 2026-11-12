# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``TTPlatform.check_and_update_config`` run inside a real ``VllmConfig``."""

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)

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
