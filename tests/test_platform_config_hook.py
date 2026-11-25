# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``TTPlatform.check_and_update_config`` run inside a real ``VllmConfig``."""

import pytest
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.kv_transfer import KVTransferConfig

from tests.test_block_scheduler import (
    LOCAL_MODEL_CONFIG,
    _ResumingStubModel,
    _stub_model_resolution,
)


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


def _kv_transfer_launch(kv_transfer_config: KVTransferConfig) -> VllmConfig:
    """A launch with ``--kv-transfer-config`` of a model that resumes prefills."""
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    with _stub_model_resolution(_ResumingStubModel):
        return VllmConfig(
            model_config=model_config,
            scheduler_config=SchedulerConfig(
                max_model_len=model_config.max_model_len,
                is_encoder_decoder=model_config.is_encoder_decoder,
            ),
            cache_config=CacheConfig(block_size=128),
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            kv_transfer_config=kv_transfer_config,
        )


@pytest.mark.parametrize(
    ("connector", "patched"),
    [
        ("NixlConnector", True),
        ("NixlPushConnector", True),
        ("TTNixlPushConnector", True),
        ("ExampleConnector", False),
    ],
)
def test_only_a_nixl_launch_makes_the_nixl_worker_rank_0_of_1(connector, patched):
    from vllm.distributed import parallel_state
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    # Plugin load registers TTNixlPushConnector by name; this test names its module.
    module_path = "vllm_tt_plugin.nixl_staging" if connector.startswith("TT") else None
    _kv_transfer_launch(
        KVTransferConfig(
            kv_connector=connector,
            kv_role="kv_both",
            kv_connector_module_path=module_path,
        )
    )

    for name in (
        "get_tensor_model_parallel_rank",
        "get_tensor_model_parallel_world_size",
    ):
        unchanged = getattr(base_worker, name) is getattr(parallel_state, name)
        assert unchanged is not patched
