# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import os
from types import SimpleNamespace

import pytest
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package,

# whose bare import re-enters vllm.platforms mid-initialization
import vllm_tt_plugin.platform as tt_platform

# Captured at import, before any test can monkeypatch it away: worker.py runs
# the real registration at module scope during collection, so this is the value
# that actually decided what got registered.
_EXTRA_MODELS_DIR_AT_IMPORT = os.environ.get("EXTRA_MODELS_DIR")


def test_diffusion_gemma_uses_tt_architecture_before_upstream_config_hooks(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.config import VllmConfig
    from vllm.config import model as model_config_module
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    hf_config = SimpleNamespace(
        architectures=[
            "DiffusionGemmaForBlockDiffusion",
            "DiffusionGemmaForCausalLM",
            "UnrelatedForCausalLM",
        ]
    )
    monkeypatch.setattr(
        model_config_module, "get_config", lambda *args, **kwargs: hf_config
    )

    tt_platform._install_diffusion_gemma_architecture_patch()
    patched_get_config = model_config_module.get_config
    tt_platform._install_diffusion_gemma_architecture_patch()

    assert model_config_module.get_config is patched_get_config
    assert patched_get_config().architectures == [
        "TTDiffusionGemmaForBlockDiffusion",
        "TTDiffusionGemmaForCausalLM",
        "UnrelatedForCausalLM",
    ]
    assert "DiffusionGemmaForBlockDiffusion" in MODELS_CONFIG_MAP
    assert "TTDiffusionGemmaForBlockDiffusion" not in MODELS_CONFIG_MAP

    upstream_calls = []
    upstream_hook = MODELS_CONFIG_MAP["DiffusionGemmaForBlockDiffusion"]
    monkeypatch.setattr(
        upstream_hook,
        "verify_and_update_config",
        classmethod(lambda cls, config: upstream_calls.append(config)),
    )
    monkeypatch.setattr(
        ModelRegistry,
        "_normalize_arch",
        staticmethod(lambda architecture, model_config: architecture),
    )
    model_config = SimpleNamespace(
        architecture=patched_get_config().architectures[0],
        config_updated=False,
        is_hybrid=False,
        convert_type="none",
    )
    vllm_config = SimpleNamespace(model_config=model_config, diffusion_config=None)

    VllmConfig.try_verify_and_update_config(vllm_config)

    assert upstream_calls == []
    assert model_config.config_updated is True
    assert vllm_config.diffusion_config is None


def _capture_registry_calls(monkeypatch, module, *, test_models_cli=False):
    events = []
    monkeypatch.setattr(
        module,
        "_should_pre_register_tt_test_models_from_cli",
        lambda: test_models_cli,
    )
    monkeypatch.setattr(
        module,
        "register_tt_models",
        lambda register_test_models=False: events.append(
            ("register", register_test_models)
        ),
    )
    monkeypatch.setattr(
        module,
        "_install_diffusion_gemma_architecture_patch",
        lambda: events.append("patch"),
    )
    return events


def test_pre_register_reinstalls_the_architecture_patch(
    monkeypatch: pytest.MonkeyPatch,
):
    # VLLM_PLUGINS=tt keeps the platform but drops the general-plugins entry
    # point, so this hook must reinstall the (idempotent) rewrite.
    monkeypatch.setattr(tt_platform, "_pin_v1_model_runner", lambda: None)
    monkeypatch.setattr(
        tt_platform, "_install_tt_harmony_truncation_patch", lambda: None
    )
    events = _capture_registry_calls(monkeypatch, tt_platform, test_models_cli=True)

    tt_platform.TTPlatform.pre_register_and_update()

    assert events == [("register", True), "patch"]


def test_general_plugin_entry_point_installs_architecture_patch(
    monkeypatch: pytest.MonkeyPatch,
):
    # Engine-core/worker/registry subprocesses never run pre_register_and_
    # update; the rewrite must travel with registration.
    from vllm_tt_plugin import model_registry

    events = _capture_registry_calls(monkeypatch, model_registry)

    model_registry.register_tt_models_from_plugin()

    assert events == [("register", False), "patch"]


def test_general_plugin_entry_point_is_inert_without_ttnn(
    monkeypatch: pytest.MonkeyPatch,
):
    # Without ttnn, rewriting the upstream-owned architecture would break
    # upstream DiffusionGemma serving on a non-TT box.
    import sys

    from vllm_tt_plugin import model_registry

    events = _capture_registry_calls(monkeypatch, model_registry)
    monkeypatch.setitem(sys.modules, "ttnn", None)

    model_registry.register_tt_models_from_plugin()

    assert events == []


def test_gemma4_parsers_are_owned_by_upstream_vllm():
    from vllm.reasoning import ReasoningParserManager
    from vllm.reasoning.gemma4_engine_reasoning_parser import (
        Gemma4ParserReasoningAdapter,
    )
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser

    assert (
        ReasoningParserManager.get_reasoning_parser("gemma4")
        is Gemma4ParserReasoningAdapter
    )
    assert ToolParserManager.get_tool_parser("gemma4") is Gemma4EngineToolParser
    with pytest.raises(KeyError):
        ReasoningParserManager.get_reasoning_parser("diffusion_gemma")
    with pytest.raises(KeyError):
        ToolParserManager.get_tool_parser("diffusion_gemma")


def test_gemma4_bare_architectures_are_owned_by_upstream_vllm(monkeypatch):
    """vLLM 0.25.1 registers all three Gemma4 architectures itself, and
    ``_register_model_if_missing`` never overrides an existing entry, so
    registering the bare names was a no-op that read as the opposite. The TT
    class is reached through the ``TT``-prefix rewrite instead.

    Also pins the upstream ownership the comment now claims: if a future vLLM
    drops these, the assertion below fails and the registration story has to be
    revisited rather than silently regressing to a fallback resolution.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    # worker.py runs the real registration at module import, so it has already
    # happened once under the shell's environment by collection time. Clear the
    # gates and re-invoke, so anything not already registered is deterministic:
    # TT_VLLM_BUILTIN_MODELS=0 returns before the Gemma4 aliases, and a
    # TT_*_VER gate raises on a value it does not recognize.
    for var in ["TT_VLLM_BUILTIN_MODELS"] + [
        k for k in os.environ if k.startswith("TT_") and k.endswith("_VER")
    ]:
        monkeypatch.delenv(var, raising=False)

    # EXTRA_MODELS_DIR cannot be undone here: the first registration wins, so a
    # bundle that claimed a TT Gemma4 name during collection already owns it.
    if _EXTRA_MODELS_DIR_AT_IMPORT:
        pytest.skip(
            "EXTRA_MODELS_DIR was set at collection time, so the TT rows may be "
            "bundle-owned and this test cannot attribute them"
        )

    tt_platform.register_tt_models()
    registry = ModelRegistry.models
    supported = ModelRegistry.get_supported_archs()

    # Positive identity per arch, not "is not the TT bridge" -- a
    # transformers-backend fallback would satisfy that too. Coupled to upstream
    # layout on purpose: if these move, platform.py's comment needs rechecking.
    for bare, module in (
        ("Gemma4ForCausalLM", "gemma4"),
        ("Gemma4ForConditionalGeneration", "gemma4_mm"),
        ("Gemma4UnifiedForConditionalGeneration", "gemma4_unified"),
    ):
        assert bare in supported, (
            f"{bare} is expected to be an upstream vLLM registration"
        )
        # Exact, not substring: "...models.gemma4" is a prefix of gemma4_mm,
        # gemma4_unified and gemma4_mtp.
        assert registry[bare].module_name == f"vllm.model_executor.models.{module}", (
            f"{bare} no longer resolves to upstream's {module}: {registry[bare]!r}"
        )

    # The rewritten names are the ones the plugin owns.
    for tt_arch in (
        "TTGemma4ForCausalLM",
        "TTGemma4ForConditionalGeneration",
        "TTGemma4UnifiedForConditionalGeneration",
    ):
        assert tt_arch in supported, (
            f"{tt_arch} was not registered; the builtin model map did not run"
        )
        # Exact, for the same reason as the upstream rows above.
        assert registry[tt_arch].module_name == "models.demos.gemma4.tt.generator_vllm"
        assert registry[tt_arch].class_name == "Gemma4ForCausalLM"


def test_gemma4_conditional_archs_are_multimodal_upstream():
    """Pins what the ``TT`` prefix does *not* buy, which the old comment got
    wrong: ``ModelConfig`` inspects the bare architecture and caches its model
    info before ``check_and_update_config`` -- the hook that does the rewrite --
    so for two of the three names the cached info is upstream's multimodal class
    and ``multimodal_config`` is already built. (The DiffusionGemma
    ``get_config`` patch pre-empts this because it is installed earlier still,
    on import of the ``vllm.general_plugins`` entry point, rather than from a
    config hook.) The prefix only selects the class the loader instantiates.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    expected = {
        "Gemma4ForCausalLM": False,
        "Gemma4ForConditionalGeneration": True,
        "Gemma4UnifiedForConditionalGeneration": True,
    }
    # Single-arch inspection: the public entry point wants a ModelConfig, and
    # building one would load a checkpoint.
    #
    # It imports the class, so the multimodal archs need the full processor
    # dependency set (torchvision). Registration is asserted either way -- that
    # is import-free -- and the multimodal flag is skipped rather than failed
    # where those deps are absent, as on the host-only CI job. The registration
    # test above pins the same claim import-free, via module identity.
    supported = ModelRegistry.get_supported_archs()
    actual = {}
    for arch in expected:
        assert arch in supported, (
            f"{arch} is expected to be an upstream vLLM registration"
        )
        info = ModelRegistry._try_inspect_model_cls(arch)
        if info is None:
            pytest.skip(
                f"upstream inspection of {arch} could not import it; this "
                "environment lacks the multimodal processor dependencies"
            )
        actual[arch] = info.supports_multimodal

    assert actual == expected
