# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Built-in registration of the text architectures served by tt-transformers."""

import importlib
import sys

import pytest
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package,

# whose bare import re-enters vllm.platforms mid-initialization
import vllm_tt_plugin.platform as tt_platform

TT_TRANSFORMERS_TARGETS = {
    "TTLlamaForCausalLM": "tt_transformers.vllm_registry:LlamaForCausalLM",
    "TTQwen2ForCausalLM": "tt_transformers.vllm_registry:Qwen2ForCausalLM",
    "TTQwen3ForCausalLM": "tt_transformers.vllm_registry:Qwen3ForCausalLM",
    "TTMistralForCausalLM": "tt_transformers.vllm_registry:MistralForCausalLM",
    "TTPhi3ForCausalLM": "tt_transformers.vllm_registry:Phi3ForCausalLM",
}

SELECTOR_ENVS = (
    "TT_LLAMA_TEXT_VER",
    "TT_QWEN3_TEXT_VER",
    "TT_MODEL_CLASS_OVERRIDES",
    "TT_VLLM_BUILTIN_MODELS",
    "EXTRA_MODELS_DIR",
)


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """An empty, recording ModelRegistry with every selector unset."""
    from vllm.model_executor.models.registry import ModelRegistry

    for name in SELECTOR_ENVS:
        monkeypatch.delenv(name, raising=False)
    table: dict[str, str] = {}
    monkeypatch.setattr(
        ModelRegistry, "get_supported_archs", staticmethod(lambda: list(table))
    )
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        staticmethod(lambda arch, target: table.__setitem__(arch, target)),
    )
    return table


def _text_targets(table: dict[str, str]) -> dict[str, str]:
    return {arch: table.get(arch) for arch in TT_TRANSFORMERS_TARGETS}


def test_default_registers_tt_transformers_for_the_text_architectures(
    registered: dict[str, str],
):
    tt_platform.register_tt_models()

    assert _text_targets(registered) == TT_TRANSFORMERS_TARGETS


def test_tt_transformers_selector_value_is_the_default(
    registered: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    monkeypatch.setenv("TT_QWEN3_TEXT_VER", "tt_transformers")

    tt_platform.register_tt_models()

    assert _text_targets(registered) == TT_TRANSFORMERS_TARGETS


@pytest.mark.parametrize(
    ("env", "value", "arch", "target"),
    [
        (
            "TT_LLAMA_TEXT_VER",
            "llama3_70b_galaxy",
            "TTLlamaForCausalLM",
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM",
        ),
        (
            "TT_LLAMA_TEXT_VER",
            "llama2_70b",
            "TTLlamaForCausalLM",
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM",
        ),
        (
            "TT_LLAMA_TEXT_VER",
            "llama31_8b_qb2",
            "TTLlamaForCausalLM",
            "models.demos.llama31_8b_qb2.tt.generator_vllm:LlamaForCausalLM",
        ),
        (
            "TT_QWEN3_TEXT_VER",
            "qwen3_32b_galaxy",
            "TTQwen3ForCausalLM",
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:QwenForCausalLM",
        ),
    ],
)
def test_demo_selector_values_keep_their_targets(
    registered: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    env: str,
    value: str,
    arch: str,
    target: str,
):
    monkeypatch.setenv(env, value)

    tt_platform.register_tt_models()

    assert registered[arch] == target
    others = {k: v for k, v in TT_TRANSFORMERS_TARGETS.items() if k != arch}
    assert {k: registered[k] for k in others} == others


@pytest.mark.parametrize(
    ("llama", "qwen3"),
    [
        (None, None),
        ("tt_transformers", "tt_transformers"),
        ("llama3_70b_galaxy", "qwen3_32b_galaxy"),
        ("llama2_70b", "tt_transformers"),
        ("llama31_8b_qb2", "qwen3_32b_galaxy"),
    ],
)
def test_no_selector_reaches_tt_metal_tt_transformers_for_text(
    registered: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    llama: str | None,
    qwen3: str | None,
):
    """The text architectures served by the standalone package have no value,
    default or fallback that registers tt-metal's in-tree implementation."""
    for env, value in (("TT_LLAMA_TEXT_VER", llama), ("TT_QWEN3_TEXT_VER", qwen3)):
        if value is not None:
            monkeypatch.setenv(env, value)

    tt_platform.register_tt_models()

    for arch in TT_TRANSFORMERS_TARGETS:
        assert not registered[arch].startswith("models.tt_transformers."), arch


@pytest.mark.parametrize(
    ("env", "listed"),
    [
        (
            "TT_LLAMA_TEXT_VER",
            "tt_transformers, llama3_70b_galaxy, llama2_70b, llama31_8b_qb2",
        ),
        ("TT_QWEN3_TEXT_VER", "tt_transformers, qwen3_32b_galaxy"),
    ],
)
def test_invalid_selector_lists_the_valid_values(
    registered: dict[str, str], monkeypatch: pytest.MonkeyPatch, env: str, listed: str
):
    monkeypatch.setenv(env, "tt_transformers_v2")

    with pytest.raises(ValueError, match=rf"pick one of \[{listed}\]"):
        tt_platform.register_tt_models()


def test_override_wins_over_the_tt_transformers_default(
    registered: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    target = "tt_transformers.models.llama3_8b.vllm_generator:Llama3Generator"
    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", f"TTLlamaForCausalLM={target}")

    tt_platform.register_tt_models()

    assert registered["TTLlamaForCausalLM"] == target
    assert (
        registered["TTQwen2ForCausalLM"]
        == TT_TRANSFORMERS_TARGETS["TTQwen2ForCausalLM"]
    )


def test_builtin_switch_off_registers_none_of_them(
    registered: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TT_VLLM_BUILTIN_MODELS", "0")

    tt_platform.register_tt_models()

    assert not set(TT_TRANSFORMERS_TARGETS) & set(registered)


def test_registration_imports_nothing_from_tt_transformers(
    registered: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    """The targets are lazy strings: an environment without tt-transformers
    must still start, and only a row on these architectures fails, at load."""
    for name in [m for m in sys.modules if m.split(".")[0] == "tt_transformers"]:
        monkeypatch.delitem(sys.modules, name)

    tt_platform.register_tt_models()

    assert all(isinstance(registered[arch], str) for arch in TT_TRANSFORMERS_TARGETS)
    assert not [m for m in sys.modules if m.split(".")[0] == "tt_transformers"]


@pytest.mark.parametrize("arch", sorted(TT_TRANSFORMERS_TARGETS))
def test_target_resolves_when_tt_transformers_is_installed(arch: str):
    """Pins the dotted paths from this side: a rename in tt-transformers fails
    here before it fails at serve time."""
    pytest.importorskip("tt_transformers.vllm_registry")
    module_name, _, class_name = TT_TRANSFORMERS_TARGETS[arch].partition(":")

    model_class = getattr(importlib.import_module(module_name), class_name)

    assert callable(model_class.initialize_vllm_model)
    assert callable(model_class.get_max_tokens_all_users)
    assert isinstance(model_class.model_capabilities, dict)
    assert "fabric_config" not in model_class.model_capabilities


@pytest.mark.parametrize("arch", sorted(TT_TRANSFORMERS_TARGETS))
def test_class_capabilities_drive_the_config_hook(
    arch: str, monkeypatch: pytest.MonkeyPatch, vllm_config
):
    """The registered class is read before the model exists: it must keep
    prefix caching, async decode and on-device sampling, and turn chunked
    prefill off, for every architecture it serves."""
    pytest.importorskip("tt_transformers.vllm_registry")
    from vllm_tt_plugin.platform import TTPlatform

    module_name, _, class_name = TT_TRANSFORMERS_TARGETS[arch].partition(":")
    model_class = getattr(importlib.import_module(module_name), class_name)
    monkeypatch.setattr(
        "vllm_tt_plugin.platform.register_tt_models", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
        lambda: [arch],
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda _model_config: (model_class, None),
    )
    vllm_config.model_config.hf_config.architectures = [arch.removeprefix("TT")]
    vllm_config.scheduler_config.enable_chunked_prefill = True
    vllm_config.scheduler_config.async_scheduling = True
    vllm_config.cache_config.enable_prefix_caching = True
    vllm_config.additional_config = {"tt": {"sample_on_device_mode": "all"}}

    TTPlatform.check_and_update_config(vllm_config)

    assert vllm_config.scheduler_config.enable_chunked_prefill is False
    assert vllm_config.cache_config.enable_prefix_caching is True
    assert vllm_config.scheduler_config.async_scheduling is True
    assert TTPlatform.sample_on_device_mode == "all"
