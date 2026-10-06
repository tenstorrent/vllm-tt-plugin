# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import vllm.config  # noqa: F401  # finish vLLM init before the plugin package

import vllm_tt_plugin.platform as tt_platform

KOLIBRI_TARGET = (
    "models.autoports.aleph_alpha_kolibri_1_bf16.tt.generator_vllm:KolibriForCausalLM"
)


def test_kolibri1_config_type_is_loadable_by_autoconfig():
    from transformers import AutoConfig, PretrainedConfig

    tt_platform._register_kolibri1_hf_config()
    # Registering again must not raise (general plugins run in every process).
    tt_platform._register_kolibri1_hf_config()

    config = AutoConfig.for_model(
        "kolibri1", num_hidden_layers=50, sliding_window=513, layer_types=["x"]
    )
    assert isinstance(config, PretrainedConfig)
    assert config.model_type == "kolibri1"
    # Checkpoint fields pass through verbatim, which is all the TT adapter needs.
    assert (config.num_hidden_layers, config.sliding_window, config.layer_types) == (
        50,
        513,
        ["x"],
    )


def test_kolibri1_architectures_resolve_to_the_tt_autoport():
    from vllm.model_executor.models.registry import ModelRegistry

    tt_platform.register_tt_models()

    for arch in ("Kolibri1ForCausalLM", "TTKolibri1ForCausalLM"):
        entry = ModelRegistry.models[arch]
        assert f"{entry.module_name}:{entry.class_name}" == KOLIBRI_TARGET
