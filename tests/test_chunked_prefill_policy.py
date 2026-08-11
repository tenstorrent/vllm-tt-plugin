# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Which model types keep token-chunked prefill, and what is forced off."""

from types import SimpleNamespace

from vllm_tt_plugin.platform import _apply_chunked_prefill_policy


def _vllm_config(
    *,
    model_type: str,
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
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            max_model_len=max_model_len,
        ),
    )


def test_gemma4_keeps_chunked_prefill_and_its_token_budget():
    config = _vllm_config(model_type="gemma4")

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.max_num_batched_tokens == 2048
    assert config.scheduler_config.long_prefill_token_threshold == 512


def test_unified_gemma4_checkpoint_also_keeps_chunked_prefill():
    config = _vllm_config(model_type="gemma4_unified")

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is True


def test_other_model_type_loses_chunked_prefill_and_gets_a_full_prompt_budget():
    config = _vllm_config(model_type="llama")

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.max_num_batched_tokens == 16384


def test_other_model_type_zeroes_the_long_prefill_threshold():
    # The base scheduler applies this cap before it consults
    # enable_chunked_prefill, so leaving it set would still split a prefill.
    config = _vllm_config(model_type="llama", enable_chunked_prefill=False)

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.long_prefill_token_threshold == 0
    # Chunked prefill was already off, so the token budget is left alone.
    assert config.scheduler_config.max_num_batched_tokens == 2048


def test_token_budget_is_left_alone_when_it_already_covers_the_model_len():
    config = _vllm_config(model_type="llama", max_num_batched_tokens=32768)

    _apply_chunked_prefill_policy(config)

    assert config.scheduler_config.max_num_batched_tokens == 32768
