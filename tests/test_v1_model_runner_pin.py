# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The TT backend runs only vLLM's V1 model runner.

Under V2 the scheduler folds preemption-resumed requests into
``scheduled_new_reqs`` and stops populating ``CachedRequestData.all_token_ids``,
which ``build_cached_request_state`` does not read, so a resume would silently
drop every generated token. Upstream defaults V2 on for dense architectures, so
these tests guard both halves of the pin: that the plugin writes it, and that
upstream still honors the knob the plugin writes.
"""

import os

import pytest

from vllm_tt_plugin.platform import _V2_MODEL_RUNNER_ENV, _pin_v1_model_runner


@pytest.fixture(autouse=True)
def restore_v2_env():
    saved = os.environ.get(_V2_MODEL_RUNNER_ENV)
    yield
    if saved is None:
        os.environ.pop(_V2_MODEL_RUNNER_ENV, None)
    else:
        os.environ[_V2_MODEL_RUNNER_ENV] = saved


def test_pin_sets_the_env_var_when_unset():
    os.environ.pop(_V2_MODEL_RUNNER_ENV, None)

    _pin_v1_model_runner()

    assert os.environ[_V2_MODEL_RUNNER_ENV] == "0"


def test_pin_is_idempotent():
    os.environ[_V2_MODEL_RUNNER_ENV] = "0"

    _pin_v1_model_runner()

    assert os.environ[_V2_MODEL_RUNNER_ENV] == "0"


@pytest.mark.parametrize("requested", ["1", "true", "00"])
def test_pin_refuses_an_explicit_v2_opt_in(requested):
    os.environ[_V2_MODEL_RUNNER_ENV] = requested

    with pytest.raises(ValueError, match="V2 model runner"):
        _pin_v1_model_runner()


def test_upstream_still_honors_the_v2_env_knob():
    """Fail loudly if upstream renames the knob or changes how it parses it.

    Either change would silently un-pin the backend, which is the whole failure
    mode this module exists to prevent.
    """
    import vllm.envs as envs

    os.environ[_V2_MODEL_RUNNER_ENV] = "0"
    assert envs.VLLM_USE_V2_MODEL_RUNNER is False

    os.environ[_V2_MODEL_RUNNER_ENV] = "1"
    assert envs.VLLM_USE_V2_MODEL_RUNNER is True


def test_the_pin_defeats_the_upstream_default_for_any_config():
    """``use_v2_model_runner`` must consult the env before anything else.

    Calling the property on a bare object proves the env check short-circuits
    ahead of every config-derived term, so the pin holds regardless of
    architecture, ``is_moe``, or ``HAS_TRITON``. If upstream reorders those
    checks, this raises ``AttributeError`` instead of quietly passing.
    """
    from vllm.config import VllmConfig

    os.environ[_V2_MODEL_RUNNER_ENV] = "0"

    assert VllmConfig.use_v2_model_runner.fget(object()) is False
