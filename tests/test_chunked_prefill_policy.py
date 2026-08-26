# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Which models keep token-chunked prefill, and how its token budget is aligned."""

from types import SimpleNamespace

import pytest

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

    _apply(
        config,
        {"supports_chunked_prefill": True, "chunked_prefill_token_alignment": 256},
    )

    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.disable_chunked_mm_input is True


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


def test_declared_support_without_an_alignment_is_a_config_error():
    config = _vllm_config()

    with pytest.raises(ValueError, match="chunked_prefill_token_alignment"):
        _apply(config, {"supports_chunked_prefill": True})


@pytest.mark.parametrize("alignment", [0, -256, 256.0, True, "256"])
def test_a_non_positive_int_alignment_is_a_config_error(alignment):
    config = _vllm_config()

    with pytest.raises(ValueError, match="chunked_prefill_token_alignment"):
        _apply(
            config,
            {
                "supports_chunked_prefill": True,
                "chunked_prefill_token_alignment": alignment,
            },
        )


def test_a_bad_alignment_is_rejected_even_when_the_feature_is_off():
    # The declaration is wrong whatever this run does with it, and a model author
    # must not learn about it only from the one deployment that turns it on.
    config = _vllm_config(enable_chunked_prefill=False)

    with pytest.raises(ValueError, match="chunked_prefill_token_alignment"):
        _apply(config, {"supports_chunked_prefill": True})


def test_aligned_budget_and_threshold_are_left_alone():
    config = _vllm_config(
        max_num_batched_tokens=8192, long_prefill_token_threshold=2048
    )

    _apply(
        config,
        {"supports_chunked_prefill": True, "chunked_prefill_token_alignment": 256},
    )

    assert config.scheduler_config.long_prefill_token_threshold == 2048
    assert config.scheduler_config.max_num_batched_tokens == 8192


def test_an_unaligned_threshold_is_rounded_down_and_the_budget_follows():
    config = _vllm_config(
        max_num_batched_tokens=4096, long_prefill_token_threshold=1000
    )

    _apply(
        config,
        {"supports_chunked_prefill": True, "chunked_prefill_token_alignment": 256},
    )

    # 1000 -> 768 (3 x 256), and the budget drops to the largest multiple of the
    # threshold it still covers. That reduces ragged remainders, it does not
    # remove them: the budget is shared, so a request admitted into what is left
    # of a step still gets an arbitrary count.
    assert config.scheduler_config.long_prefill_token_threshold == 768
    assert config.scheduler_config.max_num_batched_tokens == 3840


def test_an_unset_threshold_takes_the_whole_aligned_budget():
    config = _vllm_config(max_num_batched_tokens=3000, long_prefill_token_threshold=0)

    _apply(
        config,
        {"supports_chunked_prefill": True, "chunked_prefill_token_alignment": 256},
    )

    assert config.scheduler_config.long_prefill_token_threshold == 2816
    assert config.scheduler_config.max_num_batched_tokens == 2816


def test_a_threshold_above_the_budget_is_capped_to_it():
    config = _vllm_config(
        max_num_batched_tokens=2048, long_prefill_token_threshold=9999
    )

    _apply(
        config,
        {"supports_chunked_prefill": True, "chunked_prefill_token_alignment": 256},
    )

    assert config.scheduler_config.long_prefill_token_threshold == 2048
    assert config.scheduler_config.max_num_batched_tokens == 2048


def test_a_threshold_below_the_alignment_is_raised_to_it():
    config = _vllm_config(max_num_batched_tokens=2048, long_prefill_token_threshold=100)

    _apply(
        config,
        {"supports_chunked_prefill": True, "chunked_prefill_token_alignment": 256},
    )

    # Flooring 100 would zero the threshold, which turns the cap off entirely.
    assert config.scheduler_config.long_prefill_token_threshold == 256
    assert config.scheduler_config.max_num_batched_tokens == 2048


def test_a_budget_below_the_alignment_is_a_config_error():
    config = _vllm_config(max_num_batched_tokens=128)

    with pytest.raises(ValueError, match="smaller than this model's"):
        _apply(
            config,
            {"supports_chunked_prefill": True, "chunked_prefill_token_alignment": 256},
        )


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

    _apply(
        config,
        {"supports_chunked_prefill": True, "chunked_prefill_token_alignment": 256},
    )

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.scheduler_config.disable_chunked_mm_input is False
