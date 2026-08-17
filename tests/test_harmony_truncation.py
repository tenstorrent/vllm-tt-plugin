# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""GPT-OSS harmony prompts must truncate from the right.

The override has to land on the call that actually builds the engine's
tokenizer. ``tokenizer_args_from_config`` looks like the natural hook but its
only caller keeps the renderer mode and discards the tokenizer kwargs, so
patching it is a silent no-op. These tests pin the working hook.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest

from vllm_tt_plugin.platform import _install_tt_harmony_truncation_patch

_SENTINEL = "_tt_original_cached_tokenizer_from_config"


@pytest.fixture
def recorder():
    """Install the patch over a stub, then restore both registries."""
    tokenizer_registry = importlib.import_module("vllm.tokenizers.registry")
    renderer_registry = importlib.import_module("vllm.renderers.registry")

    saved_tokenizer = tokenizer_registry.cached_tokenizer_from_config
    saved_renderer = renderer_registry.cached_tokenizer_from_config
    saved_sentinel = getattr(tokenizer_registry, _SENTINEL, None)

    calls: list[dict] = []

    def stub(model_config, **kwargs):
        calls.append(kwargs)
        return "tokenizer"

    tokenizer_registry.cached_tokenizer_from_config = stub
    renderer_registry.cached_tokenizer_from_config = stub
    # The patch is idempotent by design, so clear its marker to install over
    # the stub regardless of what earlier tests configured.
    if hasattr(tokenizer_registry, _SENTINEL):
        delattr(tokenizer_registry, _SENTINEL)

    _install_tt_harmony_truncation_patch()

    yield SimpleNamespace(
        calls=calls,
        tokenizer_registry=tokenizer_registry,
        renderer_registry=renderer_registry,
    )

    tokenizer_registry.cached_tokenizer_from_config = saved_tokenizer
    renderer_registry.cached_tokenizer_from_config = saved_renderer
    if saved_sentinel is None:
        if hasattr(tokenizer_registry, _SENTINEL):
            delattr(tokenizer_registry, _SENTINEL)
    else:
        setattr(tokenizer_registry, _SENTINEL, saved_sentinel)


def _model_config(*, tokenizer, runner_type="generate"):
    return SimpleNamespace(tokenizer=tokenizer, runner_type=runner_type)


def test_gpt_oss_generate_gets_right_truncation(recorder):
    recorder.tokenizer_registry.cached_tokenizer_from_config(
        _model_config(tokenizer="openai/gpt-oss-120b")
    )

    assert recorder.calls == [{"truncation_side": "right"}]


def test_the_patch_reaches_the_renderer_tokenizer_call(recorder):
    """``vllm.renderers.registry`` binds the name at import, so it needs its own
    replacement; this is the call the serving path actually makes."""
    recorder.renderer_registry.cached_tokenizer_from_config(
        _model_config(tokenizer="openai/gpt-oss-120b")
    )

    assert recorder.calls == [{"truncation_side": "right"}]


def test_an_explicit_truncation_side_wins(recorder):
    recorder.tokenizer_registry.cached_tokenizer_from_config(
        _model_config(tokenizer="openai/gpt-oss-120b"), truncation_side="left"
    )

    assert recorder.calls == [{"truncation_side": "left"}]


@pytest.mark.parametrize(
    "model_config",
    [
        _model_config(tokenizer="meta-llama/Llama-3.1-8B"),
        _model_config(tokenizer="openai/gpt-oss-120b", runner_type="pooling"),
    ],
    ids=["other-model", "pooling-runner"],
)
def test_other_configs_are_left_alone(recorder, model_config):
    recorder.tokenizer_registry.cached_tokenizer_from_config(model_config)

    assert recorder.calls == [{}]


def test_upstream_still_builds_its_tokenizer_through_this_hook():
    """Fail loudly if upstream stops routing tokenizer construction here.

    That would make the override a silent no-op again, which is the failure this
    module exists to catch.
    """
    renderer_registry = importlib.import_module("vllm.renderers.registry")

    source = inspect.getsource(renderer_registry.renderer_from_config)

    assert "cached_tokenizer_from_config(" in source
