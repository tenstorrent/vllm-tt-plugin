# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import json
import sys
from pathlib import Path

import pytest
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package

import vllm_tt_plugin.platform as tt_platform


def _write_bundle(base: Path, name: str, metadata: dict) -> Path:
    folder = base / name
    folder.mkdir()
    (folder / "vllm_metadata.json").write_text(json.dumps(metadata))
    return folder


class _FakeRegistry:
    def __init__(self, initial: dict[str, str] | None = None):
        self.models: dict[str, str] = dict(initial or {})

    def get_supported_archs(self) -> list[str]:
        return list(self.models)

    def register_model(self, arch: str, target: str) -> None:
        self.models[arch] = target


def test_extra_dir_default_registers_tt_prefixed_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    folder = _write_bundle(
        tmp_path,
        "demo",
        {"arch": "DemoForCausalLM", "main_class": "demo_adapter:DemoModel"},
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))

    reg = _FakeRegistry()
    count = tt_platform._register_models_from_extra_dir(reg)

    assert count == 1
    assert reg.models == {"TTDemoForCausalLM": "demo_adapter:DemoModel"}
    assert "DemoForCausalLM" not in reg.models
    assert str(folder) in sys.path


def test_extra_dir_register_bare_arch_and_override_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_bundle(
        tmp_path,
        "demo",
        {
            "arch": "DemoBareForCausalLM",
            "main_class": "demo_adapter:DemoModel",
            "register_bare_arch": True,
        },
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))

    # Pre-existing operator override on the bare name must stay authoritative.
    reg = _FakeRegistry({"DemoBareForCausalLM": "override_mod:OverrideModel"})
    count = tt_platform._register_models_from_extra_dir(reg)

    assert count == 1
    assert reg.models["TTDemoBareForCausalLM"] == "demo_adapter:DemoModel"
    assert reg.models["DemoBareForCausalLM"] == "override_mod:OverrideModel"

    # When nothing overrides it, both TT-prefixed and bare names are registered.
    reg_clean = _FakeRegistry()
    tt_platform._register_models_from_extra_dir(reg_clean)
    assert reg_clean.models == {
        "TTDemoBareForCausalLM": "demo_adapter:DemoModel",
        "DemoBareForCausalLM": "demo_adapter:DemoModel",
    }


def test_extra_dir_registers_string_hf_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from transformers import AutoConfig, PretrainedConfig

    model_type = "extra_bundle_plain_test_model"
    _write_bundle(
        tmp_path,
        "plain_cfg",
        {
            "arch": "PlainCfgForCausalLM",
            "main_class": "plain_adapter:PlainModel",
            "hf_config": model_type,
        },
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))

    reg = _FakeRegistry()
    assert tt_platform._register_models_from_extra_dir(reg) == 1
    # Re-running must be idempotent across worker/server processes.
    assert tt_platform._register_models_from_extra_dir(reg) == 1

    cfg = AutoConfig.for_model(model_type, num_hidden_layers=12, custom_flag=True)
    assert isinstance(cfg, PretrainedConfig)
    assert cfg.model_type == model_type
    assert (cfg.num_hidden_layers, cfg.custom_flag) == (12, True)


def test_extra_dir_registers_custom_hf_config_class_from_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from transformers import AutoConfig, PretrainedConfig

    model_type = "extra_bundle_custom_cls_test_model"
    folder = _write_bundle(
        tmp_path,
        "custom_cfg",
        {
            "arch": "CustomCfgForCausalLM",
            "main_class": "custom_adapter:CustomModel",
            "hf_config": {
                "model_type": model_type,
                "config_class": "bundle_custom_cfg_mod:CustomBundleConfig",
            },
        },
    )
    (folder / "bundle_custom_cfg_mod.py").write_text(
        "from transformers import PretrainedConfig\n"
        "class CustomBundleConfig(PretrainedConfig):\n"
        f"    model_type = {model_type!r}\n"
        "    def __init__(self, extra_scale=4, **kwargs):\n"
        "        self.extra_scale = extra_scale\n"
        "        super().__init__(**kwargs)\n"
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "bundle_custom_cfg_mod", raising=False)

    reg = _FakeRegistry()
    assert tt_platform._register_models_from_extra_dir(reg) == 1

    cfg = AutoConfig.for_model(model_type)
    assert isinstance(cfg, PretrainedConfig)
    assert cfg.model_type == model_type
    assert cfg.extra_scale == 4


def test_extra_dir_skips_invalid_hf_config_without_dropping_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_bundle(
        tmp_path,
        "bad_cfg",
        {
            "arch": "BadCfgForCausalLM",
            "main_class": "bad_adapter:BadModel",
            "hf_config": {"model_type": "bad_cfg_type", "config_class": "no_colon"},
        },
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))

    reg = _FakeRegistry()
    assert tt_platform._register_models_from_extra_dir(reg) == 1
    assert reg.models == {"TTBadCfgForCausalLM": "bad_adapter:BadModel"}
