# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host tests for the transparent DP -> single-process lane conversion.

Single-execute models are served on one device mesh, so ``--data_parallel_size
N`` is folded into ``N`` in-process TT lanes by
``platform._convert_dp_to_lanes``. This covers the Galaxy generators
(llama3_70b_galaxy, qwen3_32b_galaxy), GPT-OSS under user-row sharding, and
model classes declaring ``model_capabilities["folds_dp_into_lanes"]``.
"""

from types import SimpleNamespace

import pytest

# ``vllm`` must finish importing before any plugin module, otherwise vllm's own
# platform resolution re-enters the plugin against a partially initialized
# ``vllm.platforms``. No other import here pulls ``vllm`` in first.
import vllm  # noqa: F401  isort: skip

from vllm_tt_plugin import config as tt_config
from vllm_tt_plugin import platform as tt_platform


def _vllm_config(*, data_parallel_size, max_num_seqs):
    return SimpleNamespace(
        additional_config={"tt": {}},
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_size_local=data_parallel_size,
            data_parallel_rank=0,
            data_parallel_rank_local=None,
            data_parallel_index=0,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def _model_class(name):
    """A stand-in model class whose ``__name__`` drives GPT-OSS detection."""
    return type(name, (), {})


def test_galaxy_dp_converted_to_lanes(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)

    tt_platform._convert_dp_to_lanes(config)

    # Multi-process DP collapsed to a single in-process engine with 4 lanes.
    assert config.parallel_config.data_parallel_size == 1
    assert tt_config.get_tt_data_parallel_size(config) == 4
    assert tt_config.uses_tt_lane_coordinator(config)
    # Global capacity scaled so per-lane capacity stays the requested 8.
    assert config.scheduler_config.max_num_seqs == 32
    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 8


def test_qwen3_galaxy_dp_converted_to_lanes(monkeypatch):
    monkeypatch.setenv("TT_QWEN3_TEXT_VER", "qwen3_32b_galaxy")
    config = _vllm_config(data_parallel_size=2, max_num_seqs=16)

    tt_platform._convert_dp_to_lanes(config)

    assert config.parallel_config.data_parallel_size == 1
    assert tt_config.get_tt_data_parallel_size(config) == 2
    assert config.scheduler_config.max_num_seqs == 32


def test_collapse_resets_derived_parallel_fields(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)
    config.parallel_config.data_parallel_size_local = 4
    config.parallel_config.data_parallel_external_lb = True

    tt_platform._convert_dp_to_lanes(config)

    pc = config.parallel_config
    assert pc.data_parallel_size_local == 1
    assert pc.data_parallel_rank == 0
    # Local rank must collapse to 0, not None: the TT plugin gates mesh open,
    # model load, and KV-cache allocation on ``data_parallel_rank_local == 0``,
    # which is the value a genuine single-process run resolves to.
    assert pc.data_parallel_rank_local == 0
    assert pc.data_parallel_index == 0
    assert pc.data_parallel_external_lb is False
    assert pc.data_parallel_hybrid_lb is False


def test_no_conversion_for_non_single_execute_model(monkeypatch):
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)

    tt_platform._convert_dp_to_lanes(config)

    # Left as standard multi-process DP, untouched.
    assert config.parallel_config.data_parallel_size == 4
    assert config.scheduler_config.max_num_seqs == 8
    assert tt_config._RESOLVED_LANE_COUNT_KEY not in config.additional_config


def test_conversion_is_noop_without_dp(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    config = _vllm_config(data_parallel_size=1, max_num_seqs=8)

    tt_platform._convert_dp_to_lanes(config)

    assert config.parallel_config.data_parallel_size == 1
    assert config.scheduler_config.max_num_seqs == 8
    assert tt_config._RESOLVED_LANE_COUNT_KEY not in config.additional_config


def test_conversion_is_idempotent(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)

    tt_platform._convert_dp_to_lanes(config)
    tt_platform._convert_dp_to_lanes(config)

    # Second call short-circuits (data_parallel_size already 1); no double scale.
    assert config.parallel_config.data_parallel_size == 1
    assert config.scheduler_config.max_num_seqs == 32
    assert tt_config.get_tt_data_parallel_size(config) == 4


def test_gpt_oss_dp_converted_to_lanes(monkeypatch):
    # GPT-OSS drives DP inside one process, so DP folds into lanes
    # regardless of mesh shape or batch size.
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=4, max_num_seqs=32)

    tt_platform._convert_dp_to_lanes(config, _model_class("GptOssForCausalLM"))

    assert config.parallel_config.data_parallel_size == 1
    assert tt_config.get_tt_data_parallel_size(config) == 4
    assert tt_config.uses_tt_lane_coordinator(config)
    # Per-lane capacity stays the requested 32; global scaled to 128.
    assert config.scheduler_config.max_num_seqs == 128
    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 32


def test_non_gpt_oss_model_not_converted(monkeypatch):
    # A non-GPT-OSS, non-Galaxy model keeps standard multi-process DP.
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=4, max_num_seqs=32)

    tt_platform._convert_dp_to_lanes(config, _model_class("LlamaForCausalLM"))

    assert config.parallel_config.data_parallel_size == 4
    assert tt_config._RESOLVED_LANE_COUNT_KEY not in config.additional_config


def _declaring_model_class(name, folds):
    return type(name, (), {"model_capabilities": {"folds_dp_into_lanes": folds}})


def test_declared_capability_dp_converted_to_lanes(monkeypatch):
    # A model class that drives every DP replica from one process declares it,
    # so the fold needs no Galaxy env var and no class-name match.
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=4, max_num_seqs=4)

    tt_platform._convert_dp_to_lanes(
        config, _declaring_model_class("MiMoV2ForCausalLM", True)
    )

    assert config.parallel_config.data_parallel_size == 1
    assert tt_config.get_tt_data_parallel_size(config) == 4
    assert tt_config.uses_tt_lane_coordinator(config)
    assert config.scheduler_config.max_num_seqs == 16
    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 4


def test_declared_false_keeps_standard_dp(monkeypatch):
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=4, max_num_seqs=4)

    tt_platform._convert_dp_to_lanes(
        config, _declaring_model_class("MiMoV2ForCausalLM", False)
    )

    assert config.parallel_config.data_parallel_size == 4
    assert tt_config._RESOLVED_LANE_COUNT_KEY not in config.additional_config


def test_non_bool_declaration_is_rejected(monkeypatch):
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=4, max_num_seqs=4)

    with pytest.raises(ValueError, match="folds_dp_into_lanes"):
        tt_platform._convert_dp_to_lanes(
            config, _declaring_model_class("MiMoV2ForCausalLM", "yes")
        )


def test_declared_capability_single_engine_runs_one_lane(monkeypatch):
    # At --data-parallel-size 1 a declaring model still takes the lane path,
    # with one lane, so several engines can each serve one submesh.
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=1, max_num_seqs=8)
    model = _declaring_model_class("MiMoV2ForCausalLM", True)

    tt_platform._convert_dp_to_lanes(config, model)
    tt_platform._convert_dp_to_lanes(config, model)

    assert config.parallel_config.data_parallel_size == 1
    assert tt_config.get_tt_data_parallel_size(config) == 1
    assert tt_config.uses_tt_lane_coordinator(config)
    assert config.scheduler_config.max_num_seqs == 8
    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 8


def test_multi_lane_reentry_does_not_become_one_lane(monkeypatch):
    # After a 4-lane fold data_parallel_size is 1; the re-run of the hook in
    # the worker must keep the 4 lanes.
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=4, max_num_seqs=4)
    model = _declaring_model_class("MiMoV2ForCausalLM", True)

    tt_platform._convert_dp_to_lanes(config, model)
    tt_platform._convert_dp_to_lanes(config, model)

    assert tt_config.get_tt_data_parallel_size(config) == 4
    assert tt_config._SINGLE_LANE_KEY not in config.additional_config
    assert config.scheduler_config.max_num_seqs == 16


@pytest.mark.parametrize("galaxy_env", [False, True], ids=["gpt_oss", "galaxy"])
def test_undeclared_single_engine_keeps_single_engine_path(monkeypatch, galaxy_env):
    # GPT-OSS and the Galaxy generators fold DP without declaring it: at
    # --data-parallel-size 1 they keep the single-engine path.
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    if galaxy_env:
        monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    else:
        monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=1, max_num_seqs=8)

    tt_platform._convert_dp_to_lanes(
        config, None if galaxy_env else _model_class("GptOssForCausalLM")
    )

    assert not tt_config.uses_tt_lane_coordinator(config)
    assert tt_config._RESOLVED_LANE_COUNT_KEY not in config.additional_config
