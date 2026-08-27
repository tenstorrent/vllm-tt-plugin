# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Which models keep token-chunked prefill."""

from types import SimpleNamespace

# vLLM's own bootstrap resolves the platform plugin, which imports this module.
# Letting the plugin module trigger that bootstrap deadlocks the cycle on a
# half-built module, so let vLLM finish importing itself first.
import vllm  # noqa: F401

from vllm_tt_plugin.platform import _apply_chunked_prefill_policy


class _FakeModel:
    """Stand-in for the resolved TT model class; the policy only reads its name."""


def _vllm_config(
    *,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int = 2048,
    max_model_len: int = 16384,
    long_prefill_token_threshold: int = 512,
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            long_prefill_token_threshold=long_prefill_token_threshold,
            disable_chunked_mm_input=False,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def _apply(config, capabilities):
    _apply_chunked_prefill_policy(config, capabilities, _FakeModel)


def test_declared_support_keeps_chunked_prefill():
    config = _vllm_config()

    _apply(config, {"supports_chunked_prefill": True})

    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.disable_chunked_mm_input is True


def test_declared_support_leaves_the_scheduler_budget_alone():
    # The resume offsets the scheduler hands out are corrected by the tt-metal
    # generator, which derives the alignment from the model's own program config
    # per padded suffix length. No single number the plugin could round to would
    # satisfy that, so it rounds nothing and passes the operator's budget through.
    config = _vllm_config(
        max_num_batched_tokens=3000, long_prefill_token_threshold=1000
    )

    _apply(config, {"supports_chunked_prefill": True})

    assert config.scheduler_config.max_num_batched_tokens == 3000
    assert config.scheduler_config.long_prefill_token_threshold == 1000


def test_undeclared_model_loses_chunked_prefill_and_gets_a_full_prompt_budget():
    config = _vllm_config()

    _apply(config, {"supports_prefix_caching": True})

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.max_num_batched_tokens == 16384
    assert config.scheduler_config.long_prefill_token_threshold == 0


def test_model_without_any_capabilities_loses_chunked_prefill():
    config = _vllm_config()

    _apply(config, None)

    assert config.scheduler_config.enable_chunked_prefill is False


def test_unsplit_prefill_leaves_chunked_mm_input_enabled():
    # vLLM raises outright when this is set and one mm item is larger than
    # max_num_batched_tokens, e.g. a VL model pinned to a short max_model_len.
    # With prefill never split the flag is inert, so it must stay off.
    config = _vllm_config(max_num_batched_tokens=2048, max_model_len=2048)

    _apply(config, {"supports_prefix_caching": True})

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.disable_chunked_mm_input is False


def test_undeclared_model_zeroes_the_long_prefill_threshold():
    # The base scheduler applies this cap before it consults
    # enable_chunked_prefill, so leaving it set would still split a prefill.
    config = _vllm_config(enable_chunked_prefill=False)

    _apply(config, None)

    assert config.scheduler_config.long_prefill_token_threshold == 0
    # Chunked prefill was already off, so the token budget is left alone.
    assert config.scheduler_config.max_num_batched_tokens == 2048


def test_token_budget_is_left_alone_when_it_already_covers_the_model_len():
    config = _vllm_config(max_num_batched_tokens=32768)

    _apply(config, None)

    assert config.scheduler_config.max_num_batched_tokens == 32768


def test_declared_support_with_the_flag_off_falls_back_to_the_unsplit_policy():
    config = _vllm_config(enable_chunked_prefill=False)

    _apply(config, {"supports_chunked_prefill": True})

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.scheduler_config.disable_chunked_mm_input is False
