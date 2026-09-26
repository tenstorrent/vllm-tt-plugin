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

from tests.spec.fake_spec_model import FakeSpecModel, make_fake_spec_model
from vllm_tt_plugin.config import (
    get_tt_output_tokens_per_step,
    get_tt_spec_plan,
    require_tt_spec_plan,
)


def _speculative(vllm_config, *, method="ngram", requested_k=7):
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


def test_an_admissible_launch_is_admitted(monkeypatch, vllm_config):
    # The execution path exists now: the worker publishes draft token ids and
    # the runner drives the verify-then-propose loop, so a resolved plan is
    # served rather than refused.
    model = make_fake_spec_model(max_supported_num_seqs=4)
    _run_hook(monkeypatch, _speculative(vllm_config), model)
    plan = get_tt_spec_plan(vllm_config)
    assert plan is not None
    assert plan.effective_k == 7


def test_a_declaration_error_is_reported_before_the_unproposable_refusal(
    monkeypatch, vllm_config
):
    # Ordering matters: a model author must see their own mistake, not the
    # blanket refusal that no proposer drives their method.
    model = make_fake_spec_model(
        max_supported_num_seqs=4,
        model_capabilities={"supports_spec_decode": True, "spec_requirements": []},
    )
    with pytest.raises(ValueError) as excinfo:
        _run_hook(monkeypatch, _speculative(vllm_config, method="mtp"), model)
    assert "spec_requirements" in str(excinfo.value)
    assert "no proposer drives it" not in str(excinfo.value)


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
    _run_hook(monkeypatch, _speculative(vllm_config), _RecordingModel)
    assert seen == [vllm_config.scheduler_config.max_num_seqs]


def test_a_reduced_draft_length_is_published_to_the_field_vllm_budgets_on(
    monkeypatch, vllm_config
):
    # vLLM's own Scheduler derives its lookahead slots from
    # num_speculative_tokens, so a reduction the model made has to reach it or
    # the scheduler reserves for a draft length the model will not verify.
    model = make_fake_spec_model(max_supported_num_seqs=4)
    _run_hook(monkeypatch, _speculative(vllm_config, requested_k=9), model)
    assert vllm_config.speculative_config.num_speculative_tokens == 7
    assert get_tt_spec_plan(vllm_config).effective_k == 7


def test_an_unreduced_draft_length_is_left_alone(monkeypatch, vllm_config):
    model = make_fake_spec_model(max_supported_num_seqs=4)
    _run_hook(monkeypatch, _speculative(vllm_config, requested_k=7), model)
    assert vllm_config.speculative_config.num_speculative_tokens == 7


def test_async_scheduling_cannot_speculate_for_an_undeclared_model(
    monkeypatch, vllm_config
):
    # The runner defers a speculative step now, so the refusal is no longer
    # about a missing accept walk: it is about the model. The deferred path
    # hands read_decode_output a [B, 1+K] verify whose committed length the
    # host decides after the forward, and holds the verify's hidden handle
    # across the readback until the next step's propose call. The decode
    # reload contract that supports_async_decode answers to covers neither,
    # because it was written for a decode committing one token per forward.
    # A model that does not declare async-decode support has async scheduling
    # cleared for it earlier, so the pair only arises for one that does.
    vllm_config.scheduler_config.async_scheduling = True
    model = make_fake_spec_model(
        max_supported_num_seqs=4,
        model_capabilities={
            **FakeSpecModel.model_capabilities,
            "supports_async_decode": True,
        },
    )
    with pytest.raises(ValueError) as excinfo:
        _run_hook(monkeypatch, _speculative(vllm_config), model)
    message = str(excinfo.value)
    assert "supports_async_spec_decode" in message
    assert "--no-async-scheduling" in message
    # Async scheduling stays on in the config: the launch fails instead of
    # quietly serving something the operator did not ask for.
    assert vllm_config.scheduler_config.async_scheduling is True


def test_async_scheduling_speculates_for_a_model_that_declares_it(
    monkeypatch, vllm_config
):
    # The admitted combination: the model declares that its readback and its
    # hidden handle serve a deferred verify, so the path is served and the plan
    # is resolved as it is on a synchronous launch.
    vllm_config.scheduler_config.async_scheduling = True
    model = make_fake_spec_model(
        max_supported_num_seqs=4,
        model_capabilities={
            **FakeSpecModel.model_capabilities,
            "supports_async_decode": True,
            "supports_async_spec_decode": True,
        },
    )
    _run_hook(monkeypatch, _speculative(vllm_config), model)
    assert vllm_config.scheduler_config.async_scheduling is True
    plan = get_tt_spec_plan(vllm_config)
    assert plan is not None
    assert plan.effective_k == 7


def test_declaring_async_spec_decode_does_not_admit_async_decode_itself(
    monkeypatch, vllm_config
):
    # The two declarations answer different questions. supports_async_decode
    # says the model's ordinary decode can be submitted and read back as two
    # calls; supports_async_spec_decode says the same readback serves a verify
    # block and holds its hidden handle. A model declaring only the second one
    # still has async scheduling cleared, because the step it would defer
    # first is an ordinary one.
    vllm_config.scheduler_config.async_scheduling = True
    model = make_fake_spec_model(
        max_supported_num_seqs=4,
        model_capabilities={
            **FakeSpecModel.model_capabilities,
            "supports_async_spec_decode": True,
        },
    )
    _run_hook(monkeypatch, _speculative(vllm_config), model)
    assert vllm_config.scheduler_config.async_scheduling is False


def test_lane_mode_cannot_speculate(monkeypatch, vllm_config):
    # Lane mode builds its device input from TTLaneInputBatch, which has no
    # candidate-block builder, so the refusal is what stops a lane launch from
    # accepting the flags and sending plain single-token decodes.
    vllm_config.additional_config = {"_tt_resolved_lane_count": 2}
    vllm_config.parallel_config.data_parallel_size = 1
    model = make_fake_spec_model(max_supported_num_seqs=1024)
    with pytest.raises(ValueError) as excinfo:
        _run_hook(monkeypatch, _speculative(vllm_config), model)
    message = str(excinfo.value)
    assert "lane mode" in message
    assert "TTLaneInputBatch" in message


# region Per-request semantics


def _greedy_params():
    from vllm.sampling_params import SamplingParams

    return SamplingParams(temperature=0.0)


def _validate(params):
    from vllm_tt_plugin.platform import TTPlatform

    TTPlatform.validate_request({"prompt_token_ids": [1, 2, 3]}, params)


def _speculating_platform(monkeypatch, vllm_config):
    """Run the hook so a spec plan is live, and return the platform."""
    from vllm_tt_plugin.platform import TTPlatform

    model = make_fake_spec_model(max_supported_num_seqs=4)
    _run_hook(monkeypatch, _speculative(vllm_config), model)
    assert get_tt_spec_plan(vllm_config) is not None
    return TTPlatform


def test_a_greedy_request_is_served_while_speculating(monkeypatch, vllm_config):
    _speculating_platform(monkeypatch, vllm_config)
    _validate(_greedy_params())


@pytest.mark.parametrize(
    "field, value",
    [
        ("logprobs", 1),
        ("min_tokens", 4),
        ("bad_words", ["no"]),
    ],
)
def test_a_request_the_greedy_walk_cannot_serve_is_refused(
    monkeypatch, vllm_config, field, value
):
    """Answering greedily anyway would change what was asked for, silently.

    The accept walk compares token ids and never sees logits, so it cannot
    arbitrate a token filter or a penalty, and it cannot produce logprobs. Each
    of those would come back plausible and wrong.

    ``min_p``, ``top_p`` and ``top_k`` are absent from this list because vLLM
    neutralises them itself on a greedy request, so they can only arrive
    alongside a temperature, which the next test covers.
    """
    from vllm.sampling_params import SamplingParams

    _speculating_platform(monkeypatch, vllm_config)
    params = SamplingParams(temperature=0.0, **{field: value})
    with pytest.raises(ValueError) as excinfo:
        _validate(params)
    message = str(excinfo.value)
    assert "cannot serve" in message
    assert field in message


@pytest.mark.parametrize(
    "field, value",
    [
        ("temperature", 0.7),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("repetition_penalty", 1.1),
    ],
)
def test_a_sampled_request_is_admitted_and_decoded_unspeculated(
    monkeypatch, vllm_config, field, value
):
    """Sampling the walk cannot certify is served WITHOUT speculation.

    Refusing these made a speculating launch unusable for any sampled client:
    every prompt of r1_gpqa_diamond and mmlu_pro came back HTTP 400, because
    both send temperature=1.0. The ordinary decode path applies the full
    sampling the runner implements, so the request gets what it asked for at
    baseline speed; only the drafts are withheld
    (TTModelRunner._request_is_speculable).
    """
    from vllm.sampling_params import SamplingParams

    _speculating_platform(monkeypatch, vllm_config)
    kwargs = {"temperature": 0.0} if field != "temperature" else {}
    kwargs[field] = value
    _validate(SamplingParams(**kwargs))  # must not raise


def test_a_request_is_unrestricted_when_nothing_speculates(monkeypatch, vllm_config):
    """The gate is speculation's, not the backend's."""
    from vllm.sampling_params import SamplingParams

    vllm_config.diffusion_config = None
    vllm_config.parallel_config.distributed_executor_backend = None
    _run_hook(monkeypatch, vllm_config, make_fake_spec_model())
    assert get_tt_spec_plan(vllm_config) is None
    _validate(SamplingParams(temperature=0.9, presence_penalty=0.3))


# endregion Per-request semantics


class _FakeBatch:
    def __init__(self, random=(), presence=(), frequency=(), repetition=()):
        self.random_reqs = set(random)
        self.presence_penalties_reqs = set(presence)
        self.frequency_penalties_reqs = set(frequency)
        self.repetition_penalties_reqs = set(repetition)


def _speculable(**batch_kwargs):
    """Call the runner's gate without building a runner."""
    from vllm_tt_plugin.model_runner import TTModelRunner

    runner = object.__new__(TTModelRunner)
    runner.input_batch = _FakeBatch(**batch_kwargs)
    return TTModelRunner._request_is_speculable(runner, "r0")


def test_a_greedy_request_is_speculable():
    assert _speculable() is True


@pytest.mark.parametrize(
    "batch_kwargs",
    [
        {"random": ["r0"]},
        {"presence": ["r0"]},
        {"frequency": ["r0"]},
        {"repetition": ["r0"]},
    ],
    ids=["temperature", "presence", "frequency", "repetition"],
)
def test_sampling_the_walk_cannot_certify_is_not_speculable(batch_kwargs):
    """accept_greedy_drafts compares ids, so it can only certify argmax.

    Proposing for these and accepting on id equality would return greedy text
    for a request that asked to sample -- the silent failure the refusal was
    protecting against. Withholding the drafts keeps that protection without
    refusing the request.
    """
    assert _speculable(**batch_kwargs) is False


def test_another_request_being_sampled_does_not_block_this_one():
    """The gate is per request: a mixed batch still speculates where it can."""
    assert _speculable(random=["other"]) is True


def _publish(tokens, **batch_kwargs):
    """Drive the single publish choke point every proposer goes through."""
    from vllm_tt_plugin.model_runner import TTModelRunner

    runner = object.__new__(TTModelRunner)
    runner.input_batch = _FakeBatch(**batch_kwargs)
    runner._proposed_draft_token_ids = {"r0": [9, 9]}  # a stale earlier offer
    TTModelRunner._publish_draft(runner, "r0", tokens)
    return runner._proposed_draft_token_ids


def test_a_greedy_request_publishes_its_drafts():
    assert _publish([1, 2, 3]) == {"r0": [1, 2, 3]}


def test_a_sampled_request_publishes_nothing_and_erases_the_stale_offer():
    """Both proposers route through here, so neither can bypass the gate.

    The erase matters as much as the withholding: this map is what the
    scheduler verifies next, so a stale entry would be verified against a
    token the request has already moved past.
    """
    assert _publish([1, 2, 3], random=["r0"]) == {}


def test_an_empty_offer_erases_the_stale_offer():
    assert _publish([]) == {}
