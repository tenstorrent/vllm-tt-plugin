# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Sampled requests speculate through the runner's real step path.

``test_spec_sampled_accept.py`` holds the host walk to its references. This
file holds the runner to the walk: that a verify asks for ``logits`` exactly
when one of its rows needs the target distribution, that what the walk samples
under is each request's own sampling, captured when the step was built, and
that the result reaches the request's history as an ordinary sampled step
would have.

Every step goes through ``_prepare_model_inputs``, ``submit_decode`` and
``_finish_front_packed_sync``, the synchronous path a server takes, against
``SampledTarget``, whose distribution does not depend on what was drafted.
The reference arm for a sampled request is a runner that does not speculate at
all, so its tokens come from vLLM's ordinary host sampler.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import MinPLogitsProcessor, build_logitsprocs

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.spec_accept import spec_sampling_inputs
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    PLACEHOLDER_TOKEN_ID,
    VerifyOutput,
)

from .sampled_sync_harness import (
    DRAFT_LEN,
    GREEDY,
    MAX_NUM_REQS,
    _add_request,
    _argmax_drafts,
    _assert_in_support,
    _base_chain,
    _build,
    _drafting_runner,
    _DraftingTarget,
    _output,
    _run_plain,
    _runner,
    _step,
    _tail,
)
from .sampled_target import (
    SAMPLED_VOCAB_SIZE,
    SUPPORT_PROBABILITIES,
    SampledTarget,
    base_token,
    sampled_target,
)

# region Which mode a verify asks for


def test_a_greedy_batch_verifies_in_argmax_ids_mode():
    """Ids are enough when every row is greedy, and they read back far less."""
    model = SampledTarget()
    runner = _runner(model)
    _add_request(runner, "g", GREEDY)
    _step(runner, "g")
    model_input = _build(runner, "g", drafts={"g": _argmax_drafts(runner, "g")})

    assert model_input.spec_mode == ACCEPT_MODE_ARGMAX_IDS


def test_one_sampled_row_makes_the_whole_verify_logits():
    """A row needing its distribution decides the mode for every row.

    The sampled row carries no draft, and on a launch without ``logits`` it
    would commit the argmax with its neighbor's verify.
    """
    model = SampledTarget()
    runner = _runner(model)
    _add_request(runner, "g", GREEDY, first_token=1)
    _add_request(runner, "s", SamplingParams(temperature=1.0, seed=3), first_token=9)
    _step(runner, "g", "s")

    tail = _tail(runner, "g")
    drafts = _argmax_drafts(runner, "g")
    committed = _step(runner, "g", "s", drafts={"g": drafts})

    assert model.verify_modes[-1] == ACCEPT_MODE_LOGITS
    assert committed["g"] == _base_chain(*tail, DRAFT_LEN + 1)
    assert len(committed["s"]) == 1


def test_a_logits_only_model_serves_a_greedy_request_exactly():
    """A model that never returns ids still serves greedy speculation."""
    model = sampled_target(accept_modes=(ACCEPT_MODE_LOGITS,))()
    runner = _runner(model)
    _add_request(runner, "g", GREEDY)
    while len(_output(runner, "g")) < 20:
        _step(runner, "g", drafts={"g": _argmax_drafts(runner, "g")})

    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}
    assert _output(runner, "g")[:20] == _run_plain(GREEDY, 20)


def test_an_argmax_only_model_decodes_a_step_a_sampled_row_shares():
    """Without ``logits`` no verify can sample the row, so none is sent.

    The step runs as the ordinary decode and the greedy neighbor's drafts are
    dropped. A model without ``supports_narrow_decode`` has no ordinary
    decode, so there admission refuses the sampled request instead.
    """
    model = sampled_target(
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS,), supports_narrow_decode=True
    )()
    runner = _runner(model)
    _add_request(runner, "g", GREEDY, first_token=1)
    _add_request(runner, "s", SamplingParams(temperature=1.0, seed=3), first_token=9)
    _step(runner, "g", "s")
    greedy_tail, sampled_tail = _tail(runner, "g"), _tail(runner, "s")

    committed = _step(runner, "g", "s", drafts={"g": _argmax_drafts(runner, "g")})

    assert model.verify_modes == []
    assert committed["g"] == [base_token(*greedy_tail)]
    _assert_in_support(sampled_tail, committed["s"])


def test_a_sampled_request_is_offered_drafts_only_when_logits_is_served():
    """The proposal gate follows the walks the launch can drive."""
    sampled = SamplingParams(temperature=0.8, presence_penalty=0.2, seed=1)
    for modes, speculable in (
        ((ACCEPT_MODE_ARGMAX_IDS,), False),
        ((ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS), True),
        ((ACCEPT_MODE_LOGITS,), True),
    ):
        runner = _runner(sampled_target(accept_modes=modes)())
        _add_request(runner, "s", sampled)
        assert runner._request_is_speculable("s") is speculable, modes
        _add_request(runner, "g", GREEDY, first_token=20)
        assert runner._request_is_speculable("g") is True


# endregion Which mode a verify asks for

# region What the walk samples under


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=1.0, seed=11, ignore_eos=True),
        SamplingParams(temperature=0.7, top_k=3, seed=12, ignore_eos=True),
        SamplingParams(temperature=1.3, top_p=0.8, seed=13, ignore_eos=True),
        SamplingParams(
            temperature=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            repetition_penalty=1.4,
            seed=14,
            ignore_eos=True,
        ),
    ],
    ids=["temperature", "top-k", "top-p", "penalties"],
)
def test_a_seeded_row_without_drafts_reads_exactly_as_ordinary_sampling(params):
    """Token for token, a draftless seeded row equals the unspeculated arm.

    Its neighbor drafts on every step, so every step is a verify, and the row
    commits its bonus from each one. The bonus is drawn first, from the
    request's own generator, under the request's own controls, so the row
    reads exactly what an ordinary sampled step would have drawn.
    """
    model = SampledTarget()
    runner = _runner(model)
    _add_request(runner, "g", GREEDY, first_token=21)
    _add_request(runner, "s", params, first_token=1)
    while len(_output(runner, "s")) < 24:
        _step(runner, "g", "s", drafts={"g": _argmax_drafts(runner, "g")})

    assert _output(runner, "s")[:24] == _run_plain(params, 24)
    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}


def _speculate_penalized(model, params, drafts_for):
    """Thirty tokens of one penalized request, drafted by ``drafts_for``."""
    runner = _runner(model)
    _add_request(runner, "p", params, first_token=1)
    while len(_output(runner, "p")) < 30:
        _step(runner, "p", drafts={"p": drafts_for(runner)})
    return _output(runner, "p")[:30]


@pytest.mark.parametrize("drafts", ["right", "wrong"])
def test_a_penalized_greedy_request_speculates_to_the_ordinary_output(drafts):
    """Penalties apply per candidate column, so drafting changes nothing.

    Temperature 0 with penalties is deterministic, which makes the
    unspeculated arm an exact reference. Right drafts are the reference's own
    continuation, so they are accepted and the penalties have to read them as
    history; wrong drafts are bent to a token the target rules out, so they are
    rejected and the correction has to be the penalized argmax.
    """
    params = SamplingParams(
        temperature=0.0,
        presence_penalty=0.7,
        frequency_penalty=0.3,
        repetition_penalty=1.5,
        ignore_eos=True,
    )
    reference = _run_plain(params, 40)
    assert reference[:30] != _run_plain(GREEDY, 30), (
        "the penalties do not change this request's greedy output, so this "
        "test would pass with them ignored"
    )

    def right(runner):
        done = len(_output(runner, "p"))
        return reference[done : done + DRAFT_LEN]

    def wrong(runner):
        bent = right(runner)
        bent[1] = (bent[1] + 1) % SAMPLED_VOCAB_SIZE
        return bent

    model = SampledTarget()
    emitted = _speculate_penalized(model, params, right if drafts == "right" else wrong)

    assert emitted == reference[:30]
    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}


def test_seeded_sampled_speculation_is_reproducible():
    """Same seed, same drafts, same output, with acceptance and rejection both."""

    def run():
        model = SampledTarget()
        runner = _runner(model)
        _add_request(
            runner, "s", SamplingParams(temperature=1.0, seed=77, ignore_eos=True)
        )
        widths = []
        while len(_output(runner, "s")) < 40:
            committed = _step(runner, "s", drafts={"s": _argmax_drafts(runner, "s")})
            widths.append(len(committed["s"]))
        return _output(runner, "s")[:40], widths

    first, widths = run()
    second, _ = run()

    assert first == second
    assert max(widths) > 1, "no draft was ever accepted"
    assert min(widths) < DRAFT_LEN + 1, "no draft was ever rejected"


def test_sampled_speculation_commits_the_target_distribution():
    """The rank of each committed token within its context's support.

    The target's ranks are independent of the context and of the position,
    distributed as the tempered, top-k-filtered support probabilities, so a
    pooled chi-square over every committed token of every request is a test
    of the whole loop, including acceptance, the correction and the bonus.
    """
    torch.manual_seed(5)
    temperature, top_k = 0.8, 3
    model = SampledTarget()
    runner = _runner(model)
    params = SamplingParams(temperature=temperature, top_k=top_k, ignore_eos=True)
    req_ids = [f"s{index}" for index in range(MAX_NUM_REQS)]
    tails = {}
    for index, req_id in enumerate(req_ids):
        _add_request(runner, req_id, params, first_token=index * 13)
        tails[req_id] = _tail(runner, req_id)
    while min(len(_output(runner, r)) for r in req_ids) < 300:
        _step(runner, *req_ids, drafts={r: _argmax_drafts(runner, r) for r in req_ids})

    counts = [0] * len(SUPPORT_PROBABILITIES)
    for req_id in req_ids:
        for rank in _assert_in_support(tails[req_id], _output(runner, req_id)[:300]):
            counts[rank] += 1
    weights = [p ** (1.0 / temperature) for p in SUPPORT_PROBABILITIES[:top_k]]
    expected = [w / sum(weights) for w in weights] + [0.0] * (
        len(SUPPORT_PROBABILITIES) - top_k
    )
    total = sum(counts)
    assert counts[top_k:] == [0] * (len(SUPPORT_PROBABILITIES) - top_k)
    statistic = sum(
        (counts[rank] - expected[rank] * total) ** 2 / (expected[rank] * total)
        for rank in range(top_k)
    )
    # chi-square with 2 degrees of freedom: P(X > 18.4) = 1e-4.
    assert statistic < 18.4, (counts, expected)


# endregion What the walk samples under

# region What a logits verify must return


class _AnswersWith(SampledTarget):
    """Returns a corrupted ``logits`` answer, built from the correct one."""

    corrupt = staticmethod(lambda logits: logits)

    def answer(self, spec_mode, logits):
        return VerifyOutput(spec_mode=spec_mode, logits=type(self).corrupt(logits))


@pytest.mark.parametrize(
    ("corrupt", "error", "match"),
    [
        (lambda logits: logits[..., :-1], ValueError, "whole vocabulary"),
        (
            lambda logits: logits.reshape(-1, 1, SAMPLED_VOCAB_SIZE // 2),
            ValueError,
            r"\[B, 1\+K, V\]",
        ),
        (lambda logits: logits[:, :1], ValueError, "every candidate column"),
        (lambda logits: logits[:1], ValueError, "every submitted row"),
        (lambda logits: logits.argmax(-1), TypeError, "floating point"),
    ],
    ids=["truncated-vocabulary", "row-folded", "narrow-block", "missing-rows", "ids"],
)
def test_a_malformed_logits_answer_is_refused_by_name(corrupt, error, match):
    model = type("Corrupt", (_AnswersWith,), {"corrupt": staticmethod(corrupt)})()
    runner = _runner(model)
    _add_request(runner, "s", SamplingParams(temperature=1.0, seed=1))

    with pytest.raises(error, match=match):
        _step(runner, "s", drafts={"s": _argmax_drafts(runner, "s")})


def test_padding_rows_never_reach_the_output():
    """The walk samples only live rows, and only live rows are published.

    A padding row commits its column 0 argmax and a count of 1, which is
    what the greedy walk gives it too, so the drafter can be called with every
    row; nothing of it reaches a request.
    """
    model = SampledTarget()
    runner = _runner(model)
    _add_request(runner, "s", SamplingParams(temperature=1.0, seed=4))
    _step(runner, "s")
    tail = _tail(runner, "s")
    model_input = _build(runner, "s", drafts={"s": _argmax_drafts(runner, "s")})
    rows, width = model_input.input_tokens.shape
    assert rows == MAX_NUM_REQS
    verify = model.decode_forward(
        model_input.input_tokens,
        model_input.input_positions,
        spec_mode=model_input.spec_mode,
        num_valid_drafts=model_input.num_valid_drafts,
        accepted_counts=model_input.accepted_counts,
    )

    committed, counts, _ = runner.walk_spec_acceptance(model_input, verify.logits)

    assert counts[1:].tolist() == [1] * (rows - 1)
    assert committed[1:, 0].tolist() == verify.logits[1:, 0].argmax(-1).tolist()
    assert bool((committed[1:, 1:] == PLACEHOLDER_TOKEN_ID).all())
    count = int(counts[0])
    assert 1 <= count <= width
    assert PLACEHOLDER_TOKEN_ID not in committed[0, :count].tolist()
    _assert_in_support(tail, committed[0, :count].tolist())
    assert list(spec_sampling_inputs(model_input, SAMPLED_VOCAB_SIZE).generators) == [0]

    output = runner.build_spec_runner_output(
        model_input.row_req_ids,
        runner.spec_committed_prefixes(model_input, committed, counts),
    )
    assert output.req_ids == ["s"]
    assert len(output.sampled_token_ids) == 1


def test_the_walk_samples_under_the_step_s_own_controls():
    """What the walk reads is the step's input, never the persistent batch.

    The asynchronous path walks on another thread after the persistent batch
    has moved on, so reading the batch there would sample a later step's
    controls. ``_prepare_model_inputs`` copies the rows into the step, and the
    walk reads only that copy, and only its live rows.
    """
    model = SampledTarget()
    runner = _runner(model)
    _add_request(
        runner,
        "s",
        SamplingParams(
            temperature=0.6, top_k=5, top_p=0.9, presence_penalty=0.4, seed=2
        ),
        first_token=3,
    )
    _add_request(runner, "g", GREEDY, first_token=30)
    _step(runner, "s", "g")
    model_input = _build(runner, "s", "g", drafts={"s": _argmax_drafts(runner, "s")})
    row = runner.input_batch.req_id_to_index["s"]
    history = _output(runner, "s")

    runner.input_batch.sampling.temperature[row] = 0.0
    runner.input_batch.sampling.presence_penalty[row] = 0.0
    runner.input_batch.sampling.top_k[row] = 1
    runner.input_batch.token_ids_cpu[row, :] = 0
    captured = spec_sampling_inputs(model_input, SAMPLED_VOCAB_SIZE)

    assert not model_input.perform_device_sampling
    assert captured.vocab_size == SAMPLED_VOCAB_SIZE
    assert captured.temperature.tolist() == pytest.approx([0.6, 0.0])
    assert captured.top_k.tolist() == [5, SAMPLED_VOCAB_SIZE]
    assert captured.top_p.tolist()[0] == pytest.approx(0.9)
    assert list(captured.generators) == [row]
    assert captured.generators[row] is runner.requests["s"].generator
    assert captured.penalties is not None
    assert captured.penalties.presence.tolist() == pytest.approx([0.4, 0.0])
    assert captured.penalties.output_token_ids[row] == history
    assert captured.penalties.prompt_token_ids.shape[0] == 2


def test_reading_the_controls_does_not_write_into_the_step():
    """The walk's copy is its own, so the step's tensors stay as built."""
    runner = _runner(SampledTarget())
    _add_request(runner, "s", SamplingParams(temperature=1.0, top_k=5, seed=2))
    _step(runner, "s")
    model_input = _build(runner, "s", drafts={"s": _argmax_drafts(runner, "s")})
    before = model_input.tt_sampling_params.top_k.clone()

    spec_sampling_inputs(model_input, SAMPLED_VOCAB_SIZE)

    assert torch.equal(model_input.tt_sampling_params.top_k, before)


# endregion What a logits verify must return

# region Transitions


def test_narrow_steps_and_logits_verifies_alternate_on_one_request():
    """A draftless step decodes ordinarily; a drafted one verifies in logits mode.

    The step after a multi-token commit verifies even without drafts, because
    the model needs the count, and its row then commits one sampled token.
    """
    model = sampled_target(supports_narrow_decode=True)()
    runner = _runner(model)
    _add_request(runner, "s", SamplingParams(temperature=1.0, seed=9, ignore_eos=True))
    tail = _tail(runner, "s")
    kinds = []
    for step in range(40):
        plain_before, verifies_before = model.plain_calls, len(model.verify_modes)
        drafts = {"s": _argmax_drafts(runner, "s")} if step % 3 == 0 else None
        _step(runner, "s", drafts=drafts)
        kinds.append(
            "plain"
            if model.plain_calls > plain_before
            else model.verify_modes[verifies_before]
        )

    assert "plain" in kinds and ACCEPT_MODE_LOGITS in kinds
    assert ACCEPT_MODE_ARGMAX_IDS not in kinds
    _assert_in_support(tail, _output(runner, "s"))


def test_a_model_drafter_drafts_for_a_sampled_request():
    """Proposals reach a sampled request, verified with the step's hidden handle."""
    model = _DraftingTarget()
    runner = _drafting_runner(model)
    params = SamplingParams(temperature=1.0, seed=31, ignore_eos=True)
    _add_request(runner, "s", params)
    tail = _tail(runner, "s")
    drafts = None
    widths = []
    while len(_output(runner, "s")) < 40:
        committed = _step(runner, "s", drafts=drafts)
        widths.append(len(committed["s"]))
        handed = runner.take_draft_token_ids()
        drafts = dict(zip(handed.req_ids, handed.draft_token_ids)) if handed else None

    assert model.proposals > 0
    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}
    assert max(widths) > 1, "no proposal was ever accepted"
    _assert_in_support(tail, _output(runner, "s"))


# endregion Transitions

# region What vLLM lets through


def test_a_speculating_launch_admits_a_small_min_p_and_applies_it_nowhere():
    """A min_p up to 1e-5 reaches neither sampler on a speculating launch.

    vLLM admits it with speculation and builds no min_p processor for the
    launch, so the ordinary sampler ignores it, and the walk has no min_p. A
    vLLM that builds one fails here: the walk needs min_p then, or the two
    paths sample different distributions.
    """
    speculative = object()
    SamplingParams(min_p=1e-5)._validate_spec_decode(speculative)
    with pytest.raises(VLLMValidationError):
        SamplingParams(min_p=2e-5)._validate_spec_decode(speculative)

    processors = build_logitsprocs(
        vllm_config=SimpleNamespace(speculative_config=speculative),
        device=torch.device("cpu"),
        is_pin_memory=False,
        is_pooling_model=False,
    )
    assert not any(isinstance(p, MinPLogitsProcessor) for p in processors.all)


# endregion What vLLM lets through
