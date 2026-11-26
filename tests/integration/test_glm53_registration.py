# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GLM-5.3-Flash: the TT-prefixed Glm5Next names resolve to the stack adapter."""

import vllm.config  # noqa: F401  # finish vLLM init before the plugin package

import vllm_tt_plugin.platform as tt_platform

GLM53_TARGET = "models.demos.glm53_flash.tt.generator_vllm_stack:GLM53StackForCausalLM"


def test_glm53_flash_tt_architectures_resolve_to_the_tt_adapter(monkeypatch):
    from vllm.model_executor.models.registry import ModelRegistry

    monkeypatch.delenv("TT_MODEL_CLASS_OVERRIDES", raising=False)
    monkeypatch.delenv("TT_VLLM_BUILTIN_MODELS", raising=False)
    tt_platform.register_tt_models()

    for arch in ("TTGlm5NextForConditionalGeneration", "TTGlm5NextForCausalLM"):
        entry = ModelRegistry.models[arch]
        assert f"{entry.module_name}:{entry.class_name}" == GLM53_TARGET
    # The checkpoint's bare name stays upstream's: vLLM resolves it before the
    # platform prefixes the architecture, and the prefix selects the adapter.
    for arch in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"):
        entry = ModelRegistry.models[arch]
        assert f"{entry.module_name}:{entry.class_name}" != GLM53_TARGET
