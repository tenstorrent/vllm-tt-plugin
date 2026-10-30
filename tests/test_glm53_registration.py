# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""GLM-5.3-Flash: built-in registration under the bare and TT-prefixed names."""

from vllm_tt_plugin.platform import register_tt_models


def test_glm53_flash_registered_bare_and_tt_prefixed(monkeypatch):
    from vllm.model_executor.models.registry import ModelRegistry

    monkeypatch.delenv("TT_MODEL_CLASS_OVERRIDES", raising=False)
    monkeypatch.delenv("TT_VLLM_BUILTIN_MODELS", raising=False)
    register_tt_models()
    archs = set(ModelRegistry.get_supported_archs())
    for arch in (
        "TTGlm5NextForConditionalGeneration",
        "Glm5NextForConditionalGeneration",
        "TTGlm5NextForCausalLM",
        "Glm5NextForCausalLM",
    ):
        assert arch in archs, arch
