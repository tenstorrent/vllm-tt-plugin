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

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin import model_runner as model_runner_module
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    PLACEHOLDER_TOKEN_ID,
    VerifyOutput,
)

from .sampled_target import (
    SAMPLED_VOCAB_SIZE,
    SUPPORT_PROBABILITIES,
    SampledTarget,
    base_token,
    sampled_target,
    support,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 512
MAX_NUM_REQS = 4
PROMPT_LEN = 4
DRAFT_LEN = 3

# region Test helpers


def _runner(model: SampledTarget, num_speculative_tokens: int = DRAFT_LEN):
    """A synchronous runner over the real InputBatch, speculating or not."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=SAMPLED_VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    plan = type(model).spec_plan(None, MAX_NUM_REQS, max(num_speculative_tokens, 1))
    runner = SimpleNamespace(
        model=model,
        input_batch=batch,
        vocab_size=SAMPLED_VOCAB_SIZE,
        requests={},
        kv_caches=object(),
        trace_mode="decode_only",
        request_specific_rope=False,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=num_speculative_tokens,
        _spec_accept_modes=plan.accept_modes if num_speculative_tokens else (),
        async_decode_scheduling=False,
        # The drafts come from each test through the scheduler output, so a
        # case can make them right or wrong exactly where it means to.
        _spec_method=None,
        _spec_supports_narrow_decode=plan.supports_narrow_decode,
        _spec_drafts_from_model=False,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _num_unspeculable_verify_rows=0,
        _ngram_proposer=None,
        vllm_config=SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
        ),
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=MAX_MODEL_LEN
        ),
        check_perform_device_sampling=lambda **_: False,
        # The real sampler: the reference arm's tokens have to come from the
        # ordinary host-sampling tail.
        host_sampler=Sampler(),
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    for name, member in vars(TTModelRunner).items():
        if hasattr(runner, name):
            continue
        if isinstance(member, staticmethod):
            setattr(runner, name, member.__func__)
        elif isinstance(member, classmethod):
            setattr(runner, name, member.__func__.__get__(TTModelRunner))
        elif inspect.isfunction(member):
            setattr(runner, name, member.__get__(runner))
    return runner


def _add_request(runner, req_id: str, params: SamplingParams, first_token: int = 1):
    """A fully prefilled request, with its own generator if it is seeded."""
    generator = None
    if params.sampling_type == SamplingType.RANDOM_SEED:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(params.seed)
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[
            (first_token + index) % SAMPLED_VOCAB_SIZE for index in range(PROMPT_LEN)
        ],
        mm_features=None,
        sampling_params=params,
        generator=generator,
        block_ids=([0],),
        num_computed_tokens=PROMPT_LEN,
        output_token_ids=[],
    )
    runner.requests[req_id] = request
    runner.input_batch.add_request(request)


def _build(runner, *req_ids: str, drafts: dict[str, list[int]] | None = None):
    for req_id in req_ids:
        row = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_spec_decode_tokens = dict(drafts or {})
    scheduler_output.num_scheduled_tokens = dict.fromkeys(req_ids, 1)
    scheduler_output.total_num_scheduled_tokens = len(req_ids)
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in req_ids],
        num_computed_tokens=[
            int(runner.input_batch.num_tokens[runner.input_batch.req_id_to_index[r]])
            for r in req_ids
        ],
        num_output_tokens=[0 for _ in req_ids],
    )
    return TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)


def _step(runner, *req_ids: str, drafts: dict[str, list[int]] | None = None):
    """One whole synchronous decode step; returns each request's committed ids."""
    model_input = _build(runner, *req_ids, drafts=drafts)
    submission = TTAsyncDecodeController(runner).submit_decode(
        model_input, read_from_device=True
    )
    fwd = _SyncForward(
        tt_out=submission.tt_out,
        tt_log_probs=None,
        sampling_params=model_input.tt_sampling_params,
        model_input=model_input,
        batch_size_per_dp=[len(req_ids)],
        perform_device_sampling=model_input.perform_device_sampling,
        is_decode=True,
        spec_hidden=submission.spec_hidden,
    )
    output = runner._finish_front_packed_sync(None, fwd=fwd)
    return {
        req_id: list(output.sampled_token_ids[output.req_id_to_index[req_id]])
        for req_id in req_ids
    }


def _tail(runner, req_id: str) -> tuple[int, int]:
    row = runner.input_batch.req_id_to_index[req_id]
    length = int(runner.input_batch.num_tokens[row])
    return int(runner.input_batch.token_ids_cpu[row, length - 1]), length - 1


def _output(runner, req_id: str) -> list[int]:
    return list(runner.requests[req_id].output_token_ids)


def _base_chain(token: int, position: int, count: int) -> list[int]:
    """The target's argmax continuation, which is what a greedy request emits."""
    chain = []
    for _ in range(count):
        token = base_token(token, position)
        position += 1
        chain.append(token)
    return chain


def _argmax_drafts(runner, req_id: str) -> list[int]:
    return _base_chain(*_tail(runner, req_id), DRAFT_LEN)


def _run_plain(params: SamplingParams, count: int, first_token: int = 1) -> list[int]:
    """The reference arm: a runner that does not speculate, one request."""
    runner = _runner(SampledTarget(), num_speculative_tokens=0)
    _add_request(runner, "r", params, first_token)
    while len(_output(runner, "r")) < count:
        _step(runner, "r")
    return _output(runner, "r")[:count]


def _assert_in_support(prompt_tail: tuple[int, int], tokens: list[int]) -> list[int]:
    """Every token is one the target allows after its predecessor; returns ranks."""
    token, position = prompt_tail
    ranks = []
    for index, emitted in enumerate(tokens):
        allowed = support(token, position)
        assert emitted in allowed, (
            f"token {index} ({emitted}) has probability zero after {token} at "
            f"position {position}; the target allows {allowed}"
        )
        ranks.append(allowed.index(emitted))
        token, position = emitted, position + 1
    return ranks


GREEDY = SamplingParams(temperature=0.0, ignore_eos=True)

# endregion Test helpers

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
    assert runner._num_unspeculable_verify_rows == 0


def test_a_logits_only_model_serves_a_greedy_request_exactly():
    """A model that never returns ids still serves greedy speculation."""
    model = sampled_target(accept_modes=(ACCEPT_MODE_LOGITS,))()
    runner = _runner(model)
    _add_request(runner, "g", GREEDY)
    while len(_output(runner, "g")) < 20:
        _step(runner, "g", drafts={"g": _argmax_drafts(runner, "g")})

    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}
    assert _output(runner, "g")[:20] == _run_plain(GREEDY, 20)


def test_an_argmax_only_model_keeps_its_greedy_verify_and_counts_the_row():
    """Without ``logits`` the gap stays what it was, and stays reported.

    The sampled row commits the target argmax with its neighbor's verify, and
    ``_note_unspeculable_verify_rows`` counts it, because no walk on this
    launch can sample it.
    """
    model = sampled_target(accept_modes=(ACCEPT_MODE_ARGMAX_IDS,))()
    runner = _runner(model)
    _add_request(runner, "g", GREEDY, first_token=1)
    _add_request(runner, "s", SamplingParams(temperature=1.0, seed=3), first_token=9)
    _step(runner, "g", "s")
    tail = _tail(runner, "s")
    counted = runner._num_unspeculable_verify_rows

    committed = _step(runner, "g", "s", drafts={"g": _argmax_drafts(runner, "g")})

    assert model.verify_modes[-1] == ACCEPT_MODE_ARGMAX_IDS
    assert committed["s"] == [base_token(*tail)]
    assert runner._num_unspeculable_verify_rows == counted + 1


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

    committed, counts = runner.walk_spec_acceptance(model_input, verify.logits)

    assert counts[1:].tolist() == [1] * (rows - 1)
    assert committed[1:, 0].tolist() == verify.logits[1:, 0].argmax(-1).tolist()
    assert bool((committed[1:, 1:] == PLACEHOLDER_TOKEN_ID).all())
    count = int(counts[0])
    assert 1 <= count <= width
    assert PLACEHOLDER_TOKEN_ID not in committed[0, :count].tolist()
    _assert_in_support(tail, committed[0, :count].tolist())
    assert list(
        model_runner_module._spec_sampling_inputs(
            model_input, SAMPLED_VOCAB_SIZE
        ).generators
    ) == [0]

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
    captured = model_runner_module._spec_sampling_inputs(
        model_input, SAMPLED_VOCAB_SIZE
    )

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
    assert captured.min_p is None


def test_reading_the_controls_does_not_write_into_the_step():
    """The walk's copy is its own, so the step's tensors stay as built."""
    runner = _runner(SampledTarget())
    _add_request(runner, "s", SamplingParams(temperature=1.0, top_k=5, seed=2))
    _step(runner, "s")
    model_input = _build(runner, "s", drafts={"s": _argmax_drafts(runner, "s")})
    before = model_input.tt_sampling_params.top_k.clone()

    model_runner_module._spec_sampling_inputs(model_input, SAMPLED_VOCAB_SIZE)

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


class _DraftingTarget(SampledTarget):
    """``SampledTarget`` with its own drafter: the rule's argmax chain.

    Each verify hands out a fresh hidden handle that the next proposal must
    receive back.
    """

    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": ["device_propose", "hidden_feed"],
        "spec_hidden_handoff": ["roundtrip"],
    }

    def __init__(self) -> None:
        super().__init__()
        self.hidden = None
        self.proposals = 0

    def answer(self, spec_mode, logits):
        out = super().answer(spec_mode, logits)
        self.hidden = object()
        return VerifyOutput(
            spec_mode=out.spec_mode,
            argmax_ids=out.argmax_ids,
            logits=out.logits,
            hidden=self.hidden,
        )

    def propose_draft_tokens(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ):
        from vllm_tt_plugin.spec_decode import DraftOutput

        assert hidden is self.hidden, "the proposal got another step's hidden handle"
        self.proposals += 1
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        token = committed_tokens.to(torch.int64).gather(1, index)
        position = committed_positions.to(torch.int64).gather(1, index)
        columns = []
        for _ in range(num_drafts):
            token = (token * 31 + position * 7 + 11) % SAMPLED_VOCAB_SIZE
            position = position + 1
            columns.append(token)
        return DraftOutput(draft_token_ids=torch.cat(columns, dim=1).to(torch.int32))


def _drafting_runner(model):
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    runner._spec_method = "custom_class"
    return runner


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
