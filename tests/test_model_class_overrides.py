# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package,

# whose bare import re-enters vllm.platforms mid-initialization
import vllm_tt_plugin.platform as tt_platform


def test_overrides_env_unset_is_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TT_MODEL_CLASS_OVERRIDES", raising=False)
    assert tt_platform._tt_model_class_overrides() == {}


def test_overrides_parse_multiple_entries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "TT_MODEL_CLASS_OVERRIDES",
        "ArchA=pkg.mod:ClassA , ArchB=other.mod:ClassB,",
    )
    assert tt_platform._tt_model_class_overrides() == {
        "ArchA": "pkg.mod:ClassA",
        "ArchB": "other.mod:ClassB",
    }


@pytest.mark.parametrize(
    "raw",
    [
        "ArchOnlyNoTarget",
        "Arch=missing.colon.Class",
        "=pkg.mod:Class",
    ],
    ids=["no-equals", "no-colon", "empty-arch"],
)
def test_overrides_malformed_entry_raises(monkeypatch: pytest.MonkeyPatch, raw: str):
    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", raw)
    with pytest.raises(ValueError, match="TT_MODEL_CLASS_OVERRIDES"):
        tt_platform._tt_model_class_overrides()


def test_register_tt_models_applies_overrides_first(
    monkeypatch: pytest.MonkeyPatch,
):
    """The override target is registered before the built-in target for the
    same architecture, and if-missing precedence keeps it authoritative: on
    the old code the built-in class always won."""
    from vllm.model_executor.models.registry import ModelRegistry

    registered: dict[str, str] = {}
    monkeypatch.setattr(
        ModelRegistry,
        "get_supported_archs",
        staticmethod(lambda: list(registered)),
    )
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        staticmethod(lambda arch, target: registered.setdefault(arch, target)),
    )
    monkeypatch.setenv(
        "TT_MODEL_CLASS_OVERRIDES",
        "Gemma4ForCausalLM=models.demos.gemma4.tt.generator_vllm:Gemma4MTPForCausalLM",
    )

    tt_platform.register_tt_models()

    assert (
        registered["Gemma4ForCausalLM"]
        == "models.demos.gemma4.tt.generator_vllm:Gemma4MTPForCausalLM"
    )
    # Architectures without an override keep their built-in targets.
    assert (
        registered["Gemma4ForConditionalGeneration"]
        == "models.demos.gemma4.tt.generator_vllm:Gemma4ForCausalLM"
    )
