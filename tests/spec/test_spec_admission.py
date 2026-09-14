# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative configurations are admitted or refused at config time.

Every refusal here raises rather than disabling speculation, because a server
that quietly serves without it reports speedups it did not achieve. These tests
drive the admission function against ``FakeSpecModel`` and against models with
deliberately incomplete declarations.
"""

from types import SimpleNamespace

import pytest

# vLLM's own bootstrap resolves the platform plugin, which imports plugin
# modules. Letting a plugin import trigger that bootstrap deadlocks the cycle
# on a half-built module, so let vLLM finish importing itself first.
import vllm  # noqa: F401

from tests.spec.fake_spec_model import FakeSpecModel, make_fake_spec_model
from vllm_tt_plugin.config import get_tt_spec_plan, store_tt_spec_plan
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    DRAFTER_STATE_PAGED,
    HIDDEN_HANDOFF_ON_DEVICE,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
    SpecPlan,
    SpecReject,
    admit_speculative_config,
    method_requirements,
)

# An MTP method name vLLM already carries for the model this plan targets.
MTP_METHOD = "qwen3_5_mtp"


def _config(*, method=MTP_METHOD, requested_k=7, max_num_seqs=1, speculative=True):
    return SimpleNamespace(
        speculative_config=(
            SimpleNamespace(method=method, num_speculative_tokens=requested_k)
            if speculative
            else None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        additional_config={},
    )


def _admit(config, model_class=FakeSpecModel, declared_width=1):
    return admit_speculative_config(
        config,
        model_class,
        model_class.model_capabilities,
        declared_output_tokens_per_step=declared_width,
    )


# --- the method-to-requirement table --------------------------------------


def test_host_drafters_ask_nothing_of_the_model():
    assert method_requirements("ngram") == ()
    assert method_requirements("suffix") == ()


@pytest.mark.parametrize("method", [MTP_METHOD, "eagle", "eagle3", "dflash", "medusa"])
def test_device_drafters_need_propose_and_a_hidden_feed(method):
    assert method_requirements(method) == (
        SPEC_REQUIREMENT_DEVICE_PROPOSE,
        SPEC_REQUIREMENT_HIDDEN_FEED,
    )


def test_a_draft_model_needs_a_scheduler_owned_cache():
    assert method_requirements("draft_model") == (
        SPEC_REQUIREMENT_DEVICE_PROPOSE,
        "paged_drafter_cache",
    )


def test_an_unmapped_method_is_refused_by_name():
    # Refused rather than assumed serviceable: ngram_gpu drafts on a GPU.
    with pytest.raises(ValueError) as excinfo:
        method_requirements("ngram_gpu")
    assert "ngram_gpu" in str(excinfo.value)


# --- admission ------------------------------------------------------------


def test_no_speculative_config_admits_nothing():
    assert _admit(_config(speculative=False)) is None


def test_an_admitted_plan_comes_back():
    plan = _admit(_config(requested_k=7))
    assert isinstance(plan, SpecPlan)
    assert plan.effective_k == 7
    assert plan.accept_modes[0] == ACCEPT_MODE_ARGMAX_IDS


def test_a_requested_draft_length_the_model_reduces_is_accepted():
    # 9 is above the model's 7 and below its 11, so it resolves to 7.
    assert _admit(_config(requested_k=9)).effective_k == 7


def test_a_refusal_names_the_supported_draft_lengths():
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(requested_k=2))
    message = str(excinfo.value)
    assert "[3, 7, 11]" in message
    assert "2" in message


def test_a_refusal_names_the_concurrency_it_cannot_serve():
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(max_num_seqs=8))
    assert "8" in str(excinfo.value)


def test_speculation_is_never_disabled_silently():
    # Every refusal path raises. A caller that asked for speculation and got a
    # server without it would report a speedup it never had.
    for config in (
        _config(requested_k=2),
        _config(max_num_seqs=8),
        _config(method="ngram_gpu"),
    ):
        with pytest.raises(ValueError):
            _admit(config)


# --- capability declarations the model must carry -------------------------


def test_a_model_without_supports_spec_decode_is_refused():
    variant = make_fake_spec_model(model_capabilities={})
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=variant)
    assert "supports_spec_decode" in str(excinfo.value)


def test_a_model_missing_a_required_capability_is_refused_naming_it():
    variant = make_fake_spec_model(
        model_capabilities={
            "supports_spec_decode": True,
            "spec_requirements": [SPEC_REQUIREMENT_DEVICE_PROPOSE],
            "spec_hidden_handoff": [HIDDEN_HANDOFF_ON_DEVICE],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=variant)
    assert SPEC_REQUIREMENT_HIDDEN_FEED in str(excinfo.value)


def test_a_hidden_feed_method_needs_a_declared_handoff():
    variant = make_fake_spec_model(
        model_capabilities={
            "supports_spec_decode": True,
            "spec_requirements": [
                SPEC_REQUIREMENT_DEVICE_PROPOSE,
                SPEC_REQUIREMENT_HIDDEN_FEED,
            ],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=variant)
    assert "spec_hidden_handoff" in str(excinfo.value)


def test_a_host_drafter_needs_no_requirements_but_still_needs_a_plan():
    variant = make_fake_spec_model(
        model_capabilities={"supports_spec_decode": True},
    )
    plan = _admit(_config(method="ngram"), model_class=variant)
    assert plan.effective_k == 7


def test_a_model_declaring_support_without_a_plan_is_refused():
    class _NoPlanModel:
        model_capabilities = {"supports_spec_decode": True}

    with pytest.raises(ValueError) as excinfo:
        _admit(_config(method="ngram"), model_class=_NoPlanModel)
    assert "spec_plan" in str(excinfo.value)


# --- the two rails cannot be combined -------------------------------------


def test_a_block_output_model_cannot_also_speculate():
    # Both own output_tokens_per_step, and the rail's machinery neutralizes
    # sampling controls and disables logprobs, which speculation honours.
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), declared_width=64)
    assert "64" in str(excinfo.value)


def test_a_single_token_model_is_not_treated_as_a_block_model():
    assert _admit(_config(), declared_width=1).effective_k == 7


# --- surfaces refused until something implements them ---------------------


def test_a_fused_sample_mode_is_refused():
    variant = make_fake_spec_model(
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_FUSED_SAMPLE)
    )
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=variant)
    assert ACCEPT_MODE_FUSED_SAMPLE in str(excinfo.value)


def test_a_paged_drafter_cache_is_refused():
    variant = make_fake_spec_model(drafter_state=DRAFTER_STATE_PAGED)
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=variant)
    assert DRAFTER_STATE_PAGED in str(excinfo.value)


# --- a plan the model returned wrongly ------------------------------------


def test_a_plan_above_the_requested_draft_length_is_refused():
    class _GreedyPlanModel(FakeSpecModel):
        @classmethod
        def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
            del vllm_config, max_num_seqs
            return SpecPlan(
                effective_k=requested_k + 1,
                lanes_per_request=requested_k + 2,
                extra_bytes_per_seq=0,
                extra_bytes_per_token=0,
                accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
                drafter_state="internal",
            )

    with pytest.raises(ValueError) as excinfo:
        _admit(_config(requested_k=7), model_class=_GreedyPlanModel)
    assert "8" in str(excinfo.value)


def test_a_plan_of_the_wrong_type_is_refused():
    class _WrongTypeModel(FakeSpecModel):
        @classmethod
        def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
            del vllm_config, max_num_seqs, requested_k
            return "fine"

    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=_WrongTypeModel)
    assert "SpecPlan" in str(excinfo.value)


def test_a_reject_with_no_supported_lengths_still_explains_itself():
    class _NeverModel(FakeSpecModel):
        @classmethod
        def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
            del vllm_config, max_num_seqs, requested_k
            return SpecReject(reason="this mesh has no spare rows")

    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=_NeverModel)
    assert "no spare rows" in str(excinfo.value)


# --- the resolved plan is what downstream reads ---------------------------


def test_the_admitted_plan_round_trips_through_the_config():
    config = _config()
    plan = _admit(config)
    store_tt_spec_plan(config, plan)
    assert get_tt_spec_plan(config) is plan


def test_an_unspeculative_config_stores_no_plan():
    config = _config(speculative=False)
    store_tt_spec_plan(config, _admit(config))
    assert get_tt_spec_plan(config) is None
