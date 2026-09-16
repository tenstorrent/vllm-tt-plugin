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
from vllm_tt_plugin.spec_admission import (
    MODEL_OWNED_DRAFT_METHOD,
    MODEL_OWNED_DRAFT_SENTINEL,
    method_requirements,
    resolve_speculative_plan,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    DRAFTER_STATE_PAGED,
    HIDDEN_HANDOFF_ON_DEVICE,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
    SpecPlan,
    SpecReject,
)

_SENTINEL = object()

# The only MTP name admission can receive: SpeculativeConfig.__post_init__
# rewrites every other MTPModelTypes member to "mtp" during construction.
MTP_METHOD = "mtp"

# The method the runner can propose for, which is what most of these tests want
# when their subject is something other than the method itself.
RUNNABLE_METHOD = "ngram"


def _config(
    *,
    method=RUNNABLE_METHOD,
    requested_k=7,
    max_num_seqs=1,
    speculative=True,
    model=None,
):
    return SimpleNamespace(
        speculative_config=(
            SimpleNamespace(
                method=method, num_speculative_tokens=requested_k, model=model
            )
            if speculative
            else None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        additional_config={},
    )


def _admit(config, model_class=FakeSpecModel, capabilities=_SENTINEL):
    return resolve_speculative_plan(
        config,
        model_class,
        model_class.model_capabilities if capabilities is _SENTINEL else capabilities,
        int(config.scheduler_config.max_num_seqs),
    )


# --- the method-to-requirement table --------------------------------------


def test_host_drafters_ask_nothing_of_the_model():
    assert method_requirements("ngram") == ()
    assert method_requirements("suffix") == ()


@pytest.mark.parametrize(
    "method",
    [
        MTP_METHOD,
        "eagle",
        "eagle3",
        "dflash",
        "medusa",
        "mlp_speculator",
        # Table-only: SpeculativeConfig rewrites this to "mtp" before
        # admission ever sees it, so it can never arrive in production.
        "qwen3_5_mtp",
    ],
)
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
    # The wrapper's own contribution, not the reason string the model wrote.
    assert "num_speculative_tokens=2" in message
    assert "--spec-tokens" in message


def test_a_refusal_names_the_concurrency_it_cannot_serve():
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(max_num_seqs=8))
    message = str(excinfo.value)
    assert "max_num_seqs=8" in message
    assert "--max-num-seqs" in message


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


def test_a_method_with_no_proposer_is_refused():
    """A method the runner cannot draft for must not start a server.

    Admission knowing a method is not the same as the runner being able to
    propose for it. A suffix or device method admitted here would take the
    speculative flags, draft nothing, commit one token per step, and report a
    speedup it never achieved.
    """
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(method="suffix"))
    message = str(excinfo.value)
    assert "no proposer drives it" in message
    assert "ngram" in message
    assert MODEL_OWNED_DRAFT_METHOD in message


def test_a_device_method_is_refused_even_by_a_fully_declaring_model():
    # FakeSpecModel declares device_propose and hidden_feed, so nothing in its
    # declarations refuses an MTP launch. Nothing calls propose_draft_tokens,
    # so the runner must.
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(method=MTP_METHOD))
    assert "no proposer drives it" in str(excinfo.value)


def test_the_model_owned_drafter_is_admitted():
    """The method that means the model's own drafter proposes on device.

    ``FakeSpecModel`` declares ``device_propose``, ``hidden_feed`` and a hidden
    handoff, which is everything the method requires of a model, so the launch
    is admitted and the runner calls ``propose_draft_tokens`` each step.
    """
    plan = _admit(
        _config(method=MODEL_OWNED_DRAFT_METHOD, model=MODEL_OWNED_DRAFT_SENTINEL)
    )

    assert plan is not None
    assert plan.effective_k == 7


def test_the_model_owned_drafter_needs_its_documented_model_path():
    """vLLM requires a dotted proposer path; on TT nothing imports it.

    Any other value names a proposer that will never be loaded and would read
    as the thing doing the drafting, so exactly one value is accepted.
    """
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(method=MODEL_OWNED_DRAFT_METHOD, model="some.other.Proposer"))

    message = str(excinfo.value)
    assert MODEL_OWNED_DRAFT_SENTINEL in message
    assert "propose_draft_tokens" in message


def test_the_model_owned_drafter_needs_a_declared_hidden_handoff():
    """It is a device drafter, so the hidden-state handoff has to be declared."""
    variant = make_fake_spec_model()
    variant.model_capabilities = {
        **FakeSpecModel.model_capabilities,
        "spec_hidden_handoff": [],
    }

    with pytest.raises(ValueError) as excinfo:
        _admit(
            _config(method=MODEL_OWNED_DRAFT_METHOD, model=MODEL_OWNED_DRAFT_SENTINEL),
            model_class=variant,
        )

    assert "spec_hidden_handoff" in str(excinfo.value)


def test_a_model_that_cannot_draft_is_refused_the_model_owned_method():
    """A model declaring no ``device_propose`` cannot be asked to draft."""
    variant = make_fake_spec_model()
    variant.model_capabilities = {
        **FakeSpecModel.model_capabilities,
        "spec_requirements": [],
    }

    with pytest.raises(ValueError) as excinfo:
        _admit(
            _config(method=MODEL_OWNED_DRAFT_METHOD, model=MODEL_OWNED_DRAFT_SENTINEL),
            model_class=variant,
        )

    assert SPEC_REQUIREMENT_DEVICE_PROPOSE in str(excinfo.value)


def test_a_logits_only_plan_is_refused():
    """A mode the runner never asks for cannot be the only one offered.

    The runner requests ``argmax_ids`` on every step and refuses any other
    answer, so a plan offering only ``logits`` would pass admission and fail on
    its first decode.
    """
    variant = make_fake_spec_model(accept_modes=("logits",))
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=variant)
    message = str(excinfo.value)
    assert "logits" in message
    assert "argmax_ids" in message


def test_a_plan_offering_more_than_the_runner_drives_is_still_admitted():
    # Declaring a real capability must never make a model less admissible.
    variant = make_fake_spec_model(accept_modes=("logits", ACCEPT_MODE_ARGMAX_IDS))
    assert _admit(_config(), model_class=variant).effective_k == 7


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
    # Named explicitly, for the same reason: the runnable default requires
    # nothing of the model, so it can miss nothing.
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(method=MTP_METHOD), model_class=variant)
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
    # A hidden-feed method, named explicitly: the default is the one method the
    # runner can propose for, and that one feeds nothing to a device drafter.
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(method=MTP_METHOD), model_class=variant)
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


# --- model_capabilities that is absent entirely ---------------------------


def test_an_absent_capability_dictionary_is_refused_not_crashed():
    # TTPlatform resolves it with getattr(model_class, "model_capabilities",
    # None), so None is a live value.
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), capabilities=None)
    assert "supports_spec_decode" in str(excinfo.value)


# --- surfaces refused until something implements them ---------------------


def test_offering_fused_sample_alongside_a_runnable_mode_is_still_admitted():
    # Declaring a real capability must not make a model less admissible, or a
    # model author's only remedy is to hide what the device can do.
    variant = make_fake_spec_model(
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_FUSED_SAMPLE)
    )
    plan = _admit(_config(), model_class=variant)
    assert ACCEPT_MODE_FUSED_SAMPLE in plan.accept_modes


def test_a_plan_offering_only_unrunnable_modes_is_refused():
    variant = make_fake_spec_model(accept_modes=(ACCEPT_MODE_FUSED_SAMPLE,))
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=variant)
    message = str(excinfo.value)
    assert ACCEPT_MODE_FUSED_SAMPLE in message
    assert ACCEPT_MODE_ARGMAX_IDS in message


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
    message = str(excinfo.value)
    assert "no spare rows" in message
    # The empty-supported_k branch, which tells an operator that no draft
    # length works at this concurrency rather than printing a bare list.
    assert "no draft length is supported" in message


# --- the resolved plan is what downstream reads ---------------------------


def test_the_admitted_plan_round_trips_through_the_config():
    config = _config()
    plan = _admit(config)
    store_tt_spec_plan(config, plan)
    # Equal, not identical: the plan is stored as a dictionary so
    # additional_config stays JSON-encodable, and rebuilt on the way out.
    assert get_tt_spec_plan(config) == plan


def test_an_unspeculative_config_stores_no_plan():
    config = _config(speculative=False)
    store_tt_spec_plan(config, _admit(config))
    assert get_tt_spec_plan(config) is None


# --- the plugin's own method table cannot drift from vLLM's -----------------


def test_every_mapped_method_name_is_one_vllm_knows():
    # The coupling runs one way: vLLM's literals feed the table, but the
    # plugin's own entries are plain strings. An upstream rename would
    # otherwise leave the plugin mapping a dead name while refusing the live
    # one, with every test green.
    from typing import get_args

    from vllm.config.speculative import SpeculativeMethod

    from vllm_tt_plugin.spec_admission import _build_method_requirements

    assert set(_build_method_requirements()) <= set(get_args(SpeculativeMethod))


def test_the_refusal_quotes_method_names_the_backend_does_serve():
    with pytest.raises(ValueError) as excinfo:
        method_requirements("ngram_gpu")
    message = str(excinfo.value)
    assert "ngram_gpu" in message
    assert "eagle" in message
    assert "--spec-method" in message


# --- a spec_plan attribute that is not callable ---------------------------


def test_a_non_callable_spec_plan_is_refused_naming_what_was_found():
    # The natural mistake is a class attribute of that name, which an
    # `is None` test would pass straight through to a bare TypeError.
    class _NotAMethodModel(FakeSpecModel):
        spec_plan = 5

    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=_NotAMethodModel)
    message = str(excinfo.value)
    assert "spec_plan" in message
    assert "5" in message


# --- nothing can execute an admitted plan yet ------------------------------


# --- the paged-drafter gate cannot be walked around ------------------------


def test_a_paged_cache_method_is_refused_even_when_the_plan_says_internal():
    # The gate must fire on the method's requirement, not only on the returned
    # plan, or a model whose two declarations contradict each other is admitted
    # through the gap between them.
    variant = make_fake_spec_model(
        model_capabilities={
            "supports_spec_decode": True,
            "spec_requirements": [
                SPEC_REQUIREMENT_DEVICE_PROPOSE,
                "paged_drafter_cache",
            ],
        },
    )
    assert variant.spec_plan(None, 1, 7).drafter_state == "internal"
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(method="draft_model"), model_class=variant)
    assert "scheduler-owned drafter cache" in str(excinfo.value)


def test_a_plan_returning_a_paged_drafter_state_is_also_refused():
    variant = make_fake_spec_model(drafter_state=DRAFTER_STATE_PAGED)
    with pytest.raises(ValueError) as excinfo:
        _admit(_config(), model_class=variant)
    assert DRAFTER_STATE_PAGED in str(excinfo.value)
