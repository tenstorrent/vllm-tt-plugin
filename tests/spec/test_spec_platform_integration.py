# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The platform hook admits speculation and publishes the resolved plan.

``tests/spec/test_spec_admission.py`` drives the admission function directly.
These tests drive ``TTPlatform.check_and_update_config``, so that deleting the
call site, storing the result under the wrong key, or resolving the plan before
the lane fold rewrites the concurrency all fail here.
"""

from types import SimpleNamespace

import pytest

# vLLM's own bootstrap resolves the platform plugin, which imports plugin
# modules. Letting a plugin import trigger that bootstrap deadlocks the cycle
# on a half-built module, so let vLLM finish importing itself first.
import vllm  # noqa: F401

from tests.spec.fake_spec_model import make_fake_spec_model
from vllm_tt_plugin.config import (
    get_tt_output_tokens_per_step,
    get_tt_spec_plan,
    require_tt_spec_plan,
)


def _speculative(vllm_config, *, method="mtp", requested_k=7):
    vllm_config.speculative_config = SimpleNamespace(
        method=method, num_speculative_tokens=requested_k
    )
    vllm_config.diffusion_config = None
    vllm_config.parallel_config.distributed_executor_backend = None
    return vllm_config


def _run_hook(monkeypatch, vllm_config, model_class):
    from vllm_tt_plugin.platform import TTPlatform

    with monkeypatch.context() as m:
        m.setattr(
            "vllm_tt_plugin.platform.register_tt_models",
            lambda *args, **kwargs: None,
        )
        m.setattr(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            lambda: ["TTDummyModel"],
        )
        m.setattr(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            lambda _model_config: (model_class, None),
        )
        TTPlatform.check_and_update_config(vllm_config)


def test_an_unspeculative_launch_publishes_no_plan(monkeypatch, vllm_config):
    vllm_config.diffusion_config = None
    vllm_config.parallel_config.distributed_executor_backend = None
    model = make_fake_spec_model()
    _run_hook(monkeypatch, vllm_config, model)
    assert get_tt_spec_plan(vllm_config) is None
    # Published, so a later reader can tell "no speculation" from "admission
    # never ran".
    assert require_tt_spec_plan(vllm_config) is None


def test_an_admissible_launch_is_refused_while_nothing_can_execute_it(
    monkeypatch, vllm_config
):
    # The execution path does not exist, so admission resolves the plan and
    # then refuses rather than starting a server that serves no speculation.
    model = make_fake_spec_model(max_supported_num_seqs=4)
    with pytest.raises(ValueError) as excinfo:
        _run_hook(monkeypatch, _speculative(vllm_config), model)
    message = str(excinfo.value)
    assert "take_draft_token_ids" in message
    assert "verify-then-propose" in message


def test_a_declaration_error_is_reported_before_the_execution_refusal(
    monkeypatch, vllm_config
):
    # Ordering matters: a model author must see their own mistake, not the
    # blanket "nothing can execute this" message.
    model = make_fake_spec_model(
        max_supported_num_seqs=4,
        model_capabilities={"supports_spec_decode": True, "spec_requirements": []},
    )
    with pytest.raises(ValueError) as excinfo:
        _run_hook(monkeypatch, _speculative(vllm_config), model)
    assert "spec_requirements" in str(excinfo.value)
    assert "take_draft_token_ids" not in str(excinfo.value)


def test_a_model_refusal_reaches_the_operator_through_the_hook(
    monkeypatch, vllm_config
):
    model = make_fake_spec_model(max_supported_num_seqs=4)
    with pytest.raises(ValueError) as excinfo:
        _run_hook(monkeypatch, _speculative(vllm_config, requested_k=2), model)
    assert "[3, 7, 11]" in str(excinfo.value)


def test_speculation_does_not_turn_on_the_block_output_rail(monkeypatch, vllm_config):
    # The whole reason the plan is not stored as output_tokens_per_step: that
    # key selects a rail which neutralizes sampling controls.
    model = make_fake_spec_model(max_supported_num_seqs=4)
    with pytest.raises(ValueError):
        _run_hook(monkeypatch, _speculative(vllm_config), model)
    assert get_tt_output_tokens_per_step(vllm_config) == 1


def test_a_block_output_model_cannot_also_speculate(monkeypatch, vllm_config):
    model = make_fake_spec_model(
        model_capabilities={
            "supports_spec_decode": True,
            "output_tokens_per_step": 64,
            "supports_prefix_caching": False,
        }
    )
    with pytest.raises(ValueError) as excinfo:
        _run_hook(monkeypatch, _speculative(vllm_config), model)
    message = str(excinfo.value)
    assert "output_tokens_per_step" in message
    assert "drop the speculative flags" in message


def test_the_plan_is_resolved_against_the_concurrency_the_lane_fold_leaves(
    monkeypatch, vllm_config
):
    # _convert_dp_to_lanes multiplies scheduler_config.max_num_seqs. Admission
    # must see the value that survives, or the two runs of the hook disagree.
    seen = []

    class _RecordingModel(make_fake_spec_model(max_supported_num_seqs=1024)):
        @classmethod
        def spec_plan(cls, config, max_num_seqs, requested_k):
            seen.append(max_num_seqs)
            return super().spec_plan(config, max_num_seqs, requested_k)

    vllm_config.scheduler_config.max_num_seqs = 4
    with pytest.raises(ValueError):
        _run_hook(monkeypatch, _speculative(vllm_config), _RecordingModel)
    assert seen == [vllm_config.scheduler_config.max_num_seqs]


def test_a_reduced_draft_length_is_published_to_the_field_vllm_budgets_on(
    monkeypatch, vllm_config
):
    # vLLM's own Scheduler derives its lookahead slots from
    # num_speculative_tokens, so a reduction the model made has to reach it or
    # the scheduler reserves for a draft length the model will not verify.
    model = make_fake_spec_model(max_supported_num_seqs=4)
    with pytest.raises(ValueError):
        _run_hook(monkeypatch, _speculative(vllm_config, requested_k=9), model)
    assert vllm_config.speculative_config.num_speculative_tokens == 7
    assert get_tt_spec_plan(vllm_config).effective_k == 7


def test_an_unreduced_draft_length_is_left_alone(monkeypatch, vllm_config):
    model = make_fake_spec_model(max_supported_num_seqs=4)
    with pytest.raises(ValueError):
        _run_hook(monkeypatch, _speculative(vllm_config, requested_k=7), model)
    assert vllm_config.speculative_config.num_speculative_tokens == 7
