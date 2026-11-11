# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The TT backend runs only vLLM's V2 model-runner contract.

The TT runner reads only the V2 scheduler output, where a request resumed after
preemption arrives in ``scheduled_new_reqs``. Left unset, upstream selects V1 on
a host without Triton, so these tests guard both halves of the pin: that the
plugin writes it, and that upstream still honors the knob the plugin writes.
"""

import os
from types import SimpleNamespace

import pytest
import vllm  # noqa: F401

from vllm_tt_plugin.platform import _V2_MODEL_RUNNER_ENV, _pin_v2_model_runner


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

    _pin_v2_model_runner()

    assert os.environ[_V2_MODEL_RUNNER_ENV] == "1"


def test_pin_is_idempotent():
    os.environ[_V2_MODEL_RUNNER_ENV] = "1"

    _pin_v2_model_runner()

    assert os.environ[_V2_MODEL_RUNNER_ENV] == "1"


@pytest.mark.parametrize("requested", ["0", "00", "false"])
def test_pin_refuses_a_value_vllm_does_not_read_as_v2(requested):
    os.environ[_V2_MODEL_RUNNER_ENV] = requested

    with pytest.raises(ValueError, match="does not select vLLM's V2 model runner"):
        _pin_v2_model_runner()


@pytest.mark.parametrize("requested", ["1", "01", "2"])
def test_pin_accepts_every_value_vllm_reads_as_v2(requested):
    import vllm.envs as envs

    os.environ[_V2_MODEL_RUNNER_ENV] = requested
    assert envs.VLLM_USE_V2_MODEL_RUNNER is True

    _pin_v2_model_runner()

    assert os.environ[_V2_MODEL_RUNNER_ENV] == "1"


def test_upstream_still_honors_the_v2_env_knob():
    """Fail loudly if upstream renames the knob or changes how it parses it.

    Either change would silently un-pin the backend, which is the whole failure
    mode this module exists to prevent.
    """
    import vllm.envs as envs

    os.environ[_V2_MODEL_RUNNER_ENV] = "1"
    assert envs.VLLM_USE_V2_MODEL_RUNNER is True

    os.environ[_V2_MODEL_RUNNER_ENV] = "0"
    assert envs.VLLM_USE_V2_MODEL_RUNNER is False


def test_the_pin_defeats_the_upstream_default_for_any_config():
    """``use_v2_model_runner`` must consult the env before the Triton and
    architecture defaults, which pick V1 on a host without Triton.

    Calling the property on a minimal config object proves the env check
    short-circuits ahead of model and hardware config checks.
    """
    from vllm.config import VllmConfig

    os.environ[_V2_MODEL_RUNNER_ENV] = "1"

    cfg = SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=None))
    assert VllmConfig.use_v2_model_runner.fget(cfg) is True


def _config(*, method=None, mamba_cache_mode="none"):
    """The fields vLLM 0.31's ``_get_v2_model_runner_unsupported_features``
    reads, set to what a TT config has, with its real method bound."""
    from vllm.config import CompilationMode, VllmConfig

    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE, pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            distributed_executor_backend=None,
            use_ubatching=False,
            enable_elastic_ep=False,
        ),
        speculative_config=(
            None
            if method is None
            else SimpleNamespace(method=method, parallel_drafting=False)
        ),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    cfg._get_v2_model_runner_unsupported_features = (
        lambda: VllmConfig._get_v2_model_runner_unsupported_features(cfg)
    )
    return cfg


@pytest.mark.parametrize("method", ["ngram", "custom_class"])
def test_upstream_still_names_the_tt_drafters_as_the_patch_expects(method):
    """Fail loudly if upstream stops refusing a TT drafter or renames the refusal:
    the patch would then filter nothing, or the TODO on it could be acted on."""
    from vllm.config import VllmConfig

    from vllm_tt_plugin.platform import _UPSTREAM_V2_SPEC_METHOD_REFUSALS

    refusals = VllmConfig._get_v2_model_runner_unsupported_features(
        _config(method=method)
    )

    assert refusals == [f"speculative method '{method}'"]
    assert set(refusals) <= _UPSTREAM_V2_SPEC_METHOD_REFUSALS


@pytest.mark.parametrize("method", [None, "ngram", "custom_class"])
def test_a_tt_v2_config_validates_without_triton(method, monkeypatch):
    from vllm.config import VllmConfig
    from vllm.config import vllm as vllm_config_module

    from vllm_tt_plugin.platform import _install_tt_v2_model_runner_validation_patch

    # What a TT host reports: Triton absent, or present with no active driver.
    monkeypatch.setattr(vllm_config_module, "HAS_TRITON", False)
    with pytest.raises(ValueError, match="requires Triton"):
        VllmConfig._validate_v2_model_runner(_config(method=method))

    _install_tt_v2_model_runner_validation_patch()

    VllmConfig._validate_v2_model_runner(_config(method=method))


def test_every_other_v2_refusal_still_raises():
    from vllm.config import VllmConfig

    from vllm_tt_plugin.platform import _install_tt_v2_model_runner_validation_patch

    _install_tt_v2_model_runner_validation_patch()

    with pytest.raises(
        ValueError,
        match="^Model Runner V2 does not yet support: mamba cache mode 'all'$",
    ):
        VllmConfig._validate_v2_model_runner(
            _config(method="ngram", mamba_cache_mode="all")
        )


@pytest.mark.parametrize(
    ("path", "value", "option"),
    [
        ("model_config.enable_trace_replay", True, "--enable-trace-replay"),
        ("model_config.return_sampling_mask", True, "--return-sampling-mask"),
        ("watermark_config", object(), "--watermark-config"),
        (
            "attention_config.hisparse_config",
            object(),
            "attention_config.hisparse_config",
        ),
    ],
)
def test_options_tt_does_not_implement_under_v2_are_refused(path, value, option):
    """vLLM refuses trace replay and the sampling mask under V1 only, and
    selects V2 for watermarking and HiSparse whatever the pin says.
    TTModelRunner implements none of these, so none may pass."""
    from vllm_tt_plugin.platform import _refuse_unimplemented_v2_features

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_trace_replay=False,
            return_sampling_mask=False,
        ),
        watermark_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    _refuse_unimplemented_v2_features(config)

    *parents, name = path.split(".")
    target = config
    for parent in parents:
        target = getattr(target, parent)
    setattr(target, name, value)
    with pytest.raises(ValueError, match=f"^{option} not supported on TT"):
        _refuse_unimplemented_v2_features(config)


@pytest.mark.parametrize(
    "request_field",
    [
        {"prompt_logprobs": 1},
        {"prompt_logprob_token_ids": [5, 6]},
        {"trace_decode_token_ids": [5]},
    ],
)
def test_a_request_for_a_scoring_or_replay_feature_is_refused(request_field):
    """vLLM accepts ``prompt_logprob_token_ids`` under V2, and
    ``trace_decode_token_ids`` under either runner; TTModelRunner returns
    neither, so the request is refused instead of answered without them."""
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    from vllm_tt_plugin.platform import TTPlatform

    (name,) = request_field
    with pytest.raises(VLLMValidationError, match=f"Not yet supporting {name}"):
        TTPlatform.validate_request(
            {"prompt_token_ids": [1, 2, 3]},
            SamplingParams(max_tokens=1, **request_field),
        )
