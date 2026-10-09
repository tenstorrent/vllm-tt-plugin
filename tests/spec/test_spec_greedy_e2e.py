# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The greedy speculative loop, driven end to end on the host.

One speculative step is: build the ``[B, 1+K]`` candidate block from the drafts
the scheduler delivered, verify it in one forward, walk acceptance over the
returned ids, commit the accepted prefix, and propose the next drafts. These
tests drive that whole sequence through ``TTModelRunner``'s own methods against
``FakeSpecModel``, with no device and no engine.

What they can reach: the builder, the accept walk, the commit, the accepted
count carried into the next step, and the drafts handed back for the scheduler
to collect. What they cannot reach, because it needs a mesh device: the engine
core's ``take_draft_token_ids`` handshake, the real scheduler's lookahead
budget, and ``initialize_kv_cache``.

The committed output is checked against the tokens the same model produces with
no speculation at all, because that equality is the only thing speculation is
allowed to preserve.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import (
    TTAsyncDecodeController,
    TTFinalizedDecode,
    _verify_output_tensor,
)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    PLACEHOLDER_TOKEN_ID,
    VerifyOutput,
)

from .fake_spec_model import FAKE_VOCAB_SIZE, FakeSpecModel, make_fake_spec_model

BLOCK_SIZE = 16
MAX_MODEL_LEN = 128
MAX_NUM_REQS = 2
PROMPT_LEN = 4
DRAFT_LEN = 3

# region Test helpers


def _vllm_config(num_speculative_tokens: int) -> SimpleNamespace:
    """Just the fields ``NgramProposer`` reads from a config."""
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            prompt_lookup_min=2,
            prompt_lookup_max=3,
            num_speculative_tokens=num_speculative_tokens,
            method="ngram",
        ),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


_SHARED_PROPOSER: list = []


def _proposer(num_speculative_tokens: int):
    """One proposer for the whole module: building it compiles numba kernels."""
    if not _SHARED_PROPOSER:
        from vllm.v1.spec_decode.ngram_proposer import NgramProposer

        _SHARED_PROPOSER.append(NgramProposer(_vllm_config(num_speculative_tokens)))
    return _SHARED_PROPOSER[0]


def _runner(
    model: FakeSpecModel,
    num_speculative_tokens: int = DRAFT_LEN,
    method: str | None = "ngram",
    drafts_from_model: bool = False,
) -> SimpleNamespace:
    """A runner fake carrying only what the speculative step path reads."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=FAKE_VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    runner = SimpleNamespace(
        model=model,
        input_batch=batch,
        requests={},
        kv_caches=object(),
        trace_mode="decode_only",
        request_specific_rope=False,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=num_speculative_tokens,
        # Synchronous harness: the drafts reach the runner through the
        # scheduler output, which is what ``_drafts_to_verify`` reads when
        # asynchronous scheduling is off.
        async_decode_scheduling=False,
        _spec_method=method,
        _spec_supports_narrow_decode=False,
        _spec_drafts_from_model=drafts_from_model,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _ngram_proposer=(
            _proposer(num_speculative_tokens) if method == "ngram" else None
        ),
        vllm_config=_vllm_config(num_speculative_tokens),
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=MAX_MODEL_LEN
        ),
        check_perform_device_sampling=lambda **_: False,
        # A draftless step commits through the ordinary host-sampling tail, so
        # the sampler has to be the real one.
        host_sampler=Sampler(),
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
        _build_host_generators=TTModelRunner._build_host_generators,
        _spec_row_state=TTModelRunner._spec_row_state,
        _spec_candidate_block=TTModelRunner._spec_candidate_block,
        _committed_positions=TTModelRunner._committed_positions,
    )
    # Every remaining method comes from the real class. A draftless step runs
    # the ordinary sampling tail now, and that tail reaches a chain of helpers
    # this test has no business enumerating. The attributes set above win, so
    # the stubs stay stubs, and each descriptor keeps its own binding: copying
    # a static method onto an instance as a bound one shifts every argument.
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


def _add_request(runner: SimpleNamespace, req_id: str, first_token: int = 1) -> None:
    """Add a decoding request. ``first_token`` shifts its whole prompt.

    Two requests in one step need different token histories, or a defect that
    reads one row's state for another cannot change any assertion.
    """
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([0],),
        # Fully prefilled, so the next step is a decode rather than prompt work.
        num_computed_tokens=PROMPT_LEN,
        output_token_ids=[],
    )
    runner.requests[req_id] = request
    runner.input_batch.add_request(request)


def _scheduler_output_for(
    runner: SimpleNamespace, *req_ids: str, drafts: dict[str, list[int]] | None = None
) -> SchedulerOutput:
    """What the scheduler would hand the runner for these decoding rows."""
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_spec_decode_tokens = dict(drafts or {})
    scheduler_output.num_scheduled_tokens = {req: 1 for req in req_ids}
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
    return scheduler_output


def _step(
    runner: SimpleNamespace, *req_ids: str, drafts: dict[str, list[int]] | None = None
):
    """Run one whole decode step: build, verify, accept, commit, propose."""
    # Stands in for ``_update_states``, which the real step runs first: the
    # scheduler's computed-token count advances to cover whatever the previous
    # step committed. Without it the builder reads a request as still
    # prefilling and sends prompt work instead of a candidate block.
    for req in req_ids:
        row = runner.input_batch.req_id_to_index[req]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]

    scheduler_output = _scheduler_output_for(runner, *req_ids, drafts=drafts)
    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)
    submission = TTAsyncDecodeController(runner).submit_decode(
        model_input, read_from_device=True
    )
    fwd = _SyncForward(
        tt_out=submission.tt_out,
        tt_log_probs=None,
        sampling_params=model_input.tt_sampling_params,
        model_input=model_input,
        batch_size_per_dp=[len(req_ids)],
        perform_device_sampling=False,
        is_decode=True,
        # The real path carries the verify's handle from the submission to the
        # accept walk; a test that built it by hand would not exercise that.
        spec_hidden=submission.spec_hidden,
    )
    # Through the routing entry point, not straight into the speculative
    # finish: which of the two tails a step takes is itself part of the loop.
    return runner._finish_front_packed_sync(None, fwd=fwd)


# endregion Test helpers

# region The loop


def test_a_speculative_step_commits_its_accepted_prefix():
    """Every draft the model agrees with commits, plus the bonus."""
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "r")
    # The stand-in drafts ``last + 1 + j``, which is what it also verifies, so
    # supplying that as the scheduler's drafts accepts all three.
    last = int(runner.input_batch.token_ids_cpu[0, PROMPT_LEN - 1])
    drafts = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    output = _step(runner, "r", drafts={"r": drafts})

    assert output.sampled_token_ids[0][:DRAFT_LEN] == drafts
    assert len(output.sampled_token_ids[0]) == DRAFT_LEN + 1
    assert runner._req_accepted_counts["r"] == DRAFT_LEN + 1


def test_a_rejected_draft_stops_the_row_and_still_commits_one_token():
    """A row whose first draft the model disagrees with commits the correction.

    The stand-in agrees with whatever is drafted up to its accept depth, so a
    depth of 0 is how a rejection is arranged; drafting an unlikely token would
    not produce one.
    """
    model = make_fake_spec_model(accept_depth=0)()
    runner = _runner(model)
    _add_request(runner, "r")

    output = _step(runner, "r", drafts={"r": [11, 12, 13]})

    assert len(output.sampled_token_ids[0]) == 1
    assert runner._req_accepted_counts["r"] == 1


def test_an_accept_depth_commits_exactly_that_many_drafts():
    """The stand-in's accept depth decides how long the prefix is."""
    model = make_fake_spec_model(accept_depth=2)()
    runner = _runner(model)
    _add_request(runner, "r")
    last = int(runner.input_batch.token_ids_cpu[0, PROMPT_LEN - 1])
    drafts = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    output = _step(runner, "r", drafts={"r": drafts})

    # Two drafts stand, and the third's position commits the correction.
    assert len(output.sampled_token_ids[0]) == 3
    assert output.sampled_token_ids[0][:2] == drafts[:2]
    assert output.sampled_token_ids[0][2] != drafts[2]


def test_a_draftless_step_commits_one_token():
    """A step with nothing drafted is an ordinary decode inside a wide block."""
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "r")

    output = _step(runner, "r")

    assert len(output.sampled_token_ids[0]) == 1
    assert runner._req_accepted_counts["r"] == 1


def test_one_row_s_rejection_does_not_shorten_another():
    """Two requests in one step commit independently."""
    model = FakeSpecModel()
    runner = _runner(model)
    for req_id in ("a", "b"):
        _add_request(runner, req_id)
    last = int(runner.input_batch.token_ids_cpu[0, PROMPT_LEN - 1])
    good = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    # "a" carries a full set of drafts and "b" carries none, which is the
    # mixed case a batch-wide accept count would flatten.
    output = _step(runner, "a", "b", drafts={"a": good, "b": []})

    by_req = dict(zip(output.req_ids, output.sampled_token_ids))
    assert len(by_req["a"]) == DRAFT_LEN + 1
    assert len(by_req["b"]) == 1


def test_the_committed_tokens_reach_the_persistent_batch_and_the_request():
    """A committed prefix lands in both places the runner tracks tokens."""
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "r")
    last = int(runner.input_batch.token_ids_cpu[0, PROMPT_LEN - 1])
    drafts = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    output = _step(runner, "r", drafts={"r": drafts})
    committed = output.sampled_token_ids[0]

    assert int(runner.input_batch.num_tokens[0]) == PROMPT_LEN + len(committed)
    assert runner.requests["r"].output_token_ids == committed
    written = runner.input_batch.token_ids_cpu[
        0, PROMPT_LEN : PROMPT_LEN + len(committed)
    ]
    assert written.tolist() == committed


def test_the_accepted_count_reaches_the_next_verify():
    """The count a step commits is what the next step hands the model.

    A model that defers its candidate-state select reads it to know which of
    the ``1+K`` states the previous step actually left standing, so the count
    has to travel from one step's commit into the next step's call.
    """
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "r")
    last = int(runner.input_batch.token_ids_cpu[0, PROMPT_LEN - 1])
    drafts = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    _step(runner, "r", drafts={"r": drafts})
    _step(runner, "r")

    assert model.verify_calls[0]["accepted_counts"][0] == 1
    assert int(model.verify_calls[1]["accepted_counts"][0]) == DRAFT_LEN + 1


def test_the_verify_is_asked_for_the_mode_the_runner_can_walk():
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "r")

    _step(runner, "r")

    assert model.verify_calls[0]["spec_mode"] == ACCEPT_MODE_ARGMAX_IDS
    assert model.verify_calls[0]["block_width"] == DRAFT_LEN + 1


def test_a_roundtrip_hidden_drafter_keeps_every_step_a_verify():
    """The gate on the plain path, and the reason for it.

    A step with nothing to verify returns no ``VerifyOutput``, so it produces
    no hidden handle. A drafter that is fed its target hidden state through the
    runner would then be asked to draft from nothing, so a model declaring that
    feed keeps every step a verify instead. Decided when the model is loaded,
    because a launch-time decision is checkable and passing ``None`` to that
    drafter at step time is not.
    """
    runner = TTModelRunner.__new__(TTModelRunner)
    runner._spec_drafts_from_model = True

    class RoundtripHidden:
        model_capabilities = {
            "supports_spec_decode": True,
            "spec_requirements": ["device_propose", "hidden_feed"],
            "spec_hidden_handoff": ["roundtrip"],
        }

    class OnDeviceHidden:
        model_capabilities = {
            "supports_spec_decode": True,
            "spec_requirements": ["device_propose", "hidden_feed"],
            "spec_hidden_handoff": ["on_device"],
        }

    class NoHiddenFeed:
        model_capabilities = {
            "supports_spec_decode": True,
            "spec_requirements": ["device_propose"],
        }

    runner.model = RoundtripHidden()
    assert runner._narrow_steps_serve_the_drafter() is False
    runner.model = OnDeviceHidden()
    assert runner._narrow_steps_serve_the_drafter() is True
    runner.model = NoHiddenFeed()
    assert runner._narrow_steps_serve_the_drafter() is True

    # And a launch that speculates with a host proposer needs nothing from the
    # model here at all: its drafter is never handed hidden state.
    runner._spec_drafts_from_model = False
    runner.model = RoundtripHidden()
    assert runner._narrow_steps_serve_the_drafter() is True


def test_a_draftless_step_commits_through_the_ordinary_decode_tail():
    """With nothing to verify, the step is an ordinary decode.

    ``supports_narrow_decode`` says the model also serves its own decode call
    inside a speculating launch. A step where no row carries a draft and no row
    has a multi-token commit to resolve has nothing for a verify to do, so it
    is sent as that call and commits through the sampling tail. No verify runs,
    and the accepted count stays at its post-prefill default of 1, which for
    this map means absent.
    """
    model = FakeSpecModel()
    runner = _runner(model)
    runner._spec_supports_narrow_decode = True
    _add_request(runner, "r")

    output = _step(runner, "r")

    assert len(output.sampled_token_ids[0]) == 1
    assert model.verify_calls == [], "a step with nothing to verify verified"
    assert len(model.plain_calls) == 1
    assert model.plain_calls[0]["width"] == 1
    assert "r" not in runner._req_accepted_counts


def test_a_draftless_step_still_verifies_while_a_commit_is_unresolved():
    """The step after a multi-token commit carries the count that resolves it.

    ``accepted_counts`` is how a model finds which candidate state slot its
    previous step's commit landed on, so a row whose last step committed more
    than one token has to be told even on a step that drafts nothing. One step
    resolves it: that step commits a single token, and the step after it is an
    ordinary decode.
    """
    model = FakeSpecModel()
    runner = _runner(model)
    runner._spec_supports_narrow_decode = True
    _add_request(runner, "r")
    runner._req_accepted_counts["r"] = 3

    _step(runner, "r")

    assert len(model.verify_calls) == 1, "the unresolved count was not carried"
    assert model.verify_calls[0]["accepted_counts"][0] == 3
    assert model.plain_calls == []

    # Resolved: the next draftless step is an ordinary decode.
    assert runner._req_accepted_counts["r"] == 1
    _step(runner, "r")
    assert len(model.verify_calls) == 1
    assert len(model.plain_calls) == 1


# endregion The loop

# region Equality with an unspeculated run

# Losslessness against ordinary decoding lives in ``test_spec_lossless.py``,
# which needs a target whose choice does not depend on what was drafted, a
# genuinely unspeculated reference arm, and drafts that are deliberately wrong.
# None of those can be built from ``FakeSpecModel``, whose verify returns each
# draft unchanged wherever it agrees.


# endregion Equality with an unspeculated run

# region Drafts handed back to the scheduler


def test_a_launch_without_speculation_reports_no_drafts():
    """A non-speculating runner has nothing to hand over."""
    runner = _runner(FakeSpecModel(), num_speculative_tokens=0, method=None)

    assert runner.take_draft_token_ids() is None


def test_the_reported_count_is_what_actually_committed_at_the_length_cap():
    """A row clipped by ``max_model_len`` records what it committed.

    The accept walk's count is what the model agreed to; the commit may fit
    fewer of those tokens. Recording the walk's count instead would tell the
    next verify to continue from a candidate whose token was never emitted.
    """
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "r")
    # Two token slots left, against a block that would commit four.
    runner.input_batch.num_tokens[0] = MAX_MODEL_LEN - 2
    runner.input_batch.num_computed_tokens_cpu[0] = MAX_MODEL_LEN - 2
    last = int(runner.input_batch.token_ids_cpu[0, MAX_MODEL_LEN - 3])
    drafts = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    output = _step(runner, "r", drafts={"r": drafts})

    assert len(output.sampled_token_ids[0]) == 2
    assert runner._req_accepted_counts["r"] == 2
    assert int(runner.input_batch.num_tokens[0]) == MAX_MODEL_LEN


class _StubProposer:
    """A proposer that always drafts, so the handoff can be tested alone.

    The real n-gram proposer only drafts where a request's own text repeats,
    which makes it a poor way to test whether a proposal reaches the engine.
    """

    def __init__(self, drafts: list[int]):
        self.drafts = drafts
        self.calls: list[list[list[int]]] = []

    def propose(self, num_speculative_tokens, sampled, num_tokens, token_ids_cpu):
        self.calls.append(sampled)
        return [list(self.drafts) for _ in sampled]


def test_a_proposal_reaches_the_engine_exactly_once():
    """``take_draft_token_ids`` hands each proposal over and forgets it.

    The engine collects after the step that produced it, and replaying it
    would have the scheduler speculate on a continuation already spent.
    """
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "r")
    runner._ngram_proposer = _StubProposer([41, 42, 43])

    _step(runner, "r")
    first = runner.take_draft_token_ids()
    second = runner.take_draft_token_ids()

    assert first is not None
    assert first.req_ids == ["r"]
    assert first.draft_token_ids == [[41, 42, 43]]
    assert second is None


def test_the_proposer_sees_the_tokens_the_step_committed():
    """Proposal runs after the commit, on this step's own tokens."""
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "r")
    proposer = _StubProposer([41])
    runner._ngram_proposer = proposer
    last = int(runner.input_batch.token_ids_cpu[0, PROMPT_LEN - 1])
    drafts = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    output = _step(runner, "r", drafts={"r": drafts})

    assert proposer.calls == [[output.sampled_token_ids[0]]]


# endregion Drafts handed back to the scheduler

# region What a step may answer a verify with


class _DeclaresSpecDecodeWithoutAVerify:
    """A model class that declares the contract and implements no verify.

    ``supports_spec_decode`` is a claim about ``decode_forward``, and a model
    can be launched with the claim made and the verify unwritten: the
    speculative keywords then land in ``**kwargs`` and are ignored, and the
    step answers with the return an ordinary decode makes. Device sampling
    makes that a ``[B, 1]`` tensor of token ids, which is what this returns.
    """

    def decode_forward(self, tokens, **kwargs):
        del kwargs
        return torch.zeros((int(tokens.shape[0]), 1), dtype=torch.int32)


class _AnswersEveryStepWithAVerify:
    """A model class that returns a ``VerifyOutput`` from a plain decode."""

    def decode_forward(self, tokens, **kwargs):
        del kwargs
        return VerifyOutput(
            spec_mode=ACCEPT_MODE_ARGMAX_IDS,
            argmax_ids=torch.zeros((int(tokens.shape[0]), 1), dtype=torch.int32),
        )


def test_a_step_that_answers_a_verify_with_a_plain_decode_is_refused():
    """The unimplemented verify has to stop the run, not shorten it.

    Nothing below the submission boundary can catch this. A ``[B, 1]`` id
    tensor has the two dimensions the accept walk expects and an integer
    dtype, and the walk reads it as a verify that claimed one token on every
    row: each step then commits a single token, the drafts are all rejected,
    and the server produces correct text at a fraction of the speed with no
    error anywhere. The refusal names the model class and the declaration that
    is not backed.
    """
    runner = _runner(_DeclaresSpecDecodeWithoutAVerify())
    _add_request(runner, "r")

    with pytest.raises(TypeError, match="must not declare supports_spec_decode"):
        _step(runner, "r", drafts={"r": [11, 12, 13]})


def test_a_verify_output_from_an_unspeculated_step_is_refused():
    """A verify's return has no accepted count to be read against here.

    The runner sends ``spec_mode`` on every step it speculates on and on no
    other, so a ``VerifyOutput`` arriving without one is the model answering a
    question that was not asked.
    """
    runner = _runner(
        _AnswersEveryStepWithAVerify(), num_speculative_tokens=0, method=None
    )
    _add_request(runner, "r")

    with pytest.raises(TypeError, match="asked for no verify"):
        _step(runner, "r")


def test_a_verify_answering_in_another_mode_is_refused():
    """A mode the runner cannot walk fails by name, not by misreading ids.

    ``logits`` is a legal accept mode and a model may offer it, but nothing
    drives it yet. A model that ignores the requested mode and answers in its
    own must not have its ``[B, 1+K, V]`` logits read as token ids: that would
    commit vocabulary indices of float rows as output. A conformant stand-in
    refuses the request instead, so the guard is exercised on its own.
    """
    logits = torch.zeros(1, 4, FAKE_VOCAB_SIZE)
    answer = VerifyOutput(spec_mode=ACCEPT_MODE_LOGITS, logits=logits)

    with pytest.raises(NotImplementedError, match="no accept walk for that mode"):
        _verify_output_tensor(answer, "FakeSpecModel", ACCEPT_MODE_ARGMAX_IDS)


def test_a_tuple_of_host_tensors_does_not_pass_as_a_verify():
    """A plain decode's other return shape is refused by type too.

    Host sampling returns logits, and a model may return them beside other
    tensors. Neither is a ``VerifyOutput``, and the guard is on the type
    rather than on any shape, so neither reaches the accept walk.
    """
    answer = (torch.zeros(1, 1, FAKE_VOCAB_SIZE), None)

    with pytest.raises(TypeError, match="returned tuple"):
        _verify_output_tensor(answer, "SomeTTModel", ACCEPT_MODE_ARGMAX_IDS)


# endregion What a step may answer a verify with

# region The model's own drafter


def _model_drafter_runner(model, num_speculative_tokens: int = DRAFT_LEN):
    """A runner whose drafts come from the model, not from a host proposer."""
    return _runner(
        model,
        num_speculative_tokens=num_speculative_tokens,
        method="custom_class",
        drafts_from_model=True,
    )


def test_the_model_is_asked_for_the_next_drafts():
    """One propose per step, over the rows the verify ran on.

    The drafter's state is indexed by row and a device graph has one shape, so
    it is handed the verify's rows rather than only the live ones, and the
    committed block plus each row's count rather than a single token.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    _step(runner, "r")

    assert len(model.propose_calls) == 1
    call = model.propose_calls[0]
    assert call["num_drafts"] == DRAFT_LEN
    # MAX_NUM_REQS rows: the one live request and the padding the verify saw.
    assert call["rows"] == MAX_NUM_REQS
    assert call["committed_tokens"].shape == (MAX_NUM_REQS, DRAFT_LEN + 1)
    assert call["committed_positions"].shape == (MAX_NUM_REQS, DRAFT_LEN + 1)


def test_the_drafts_the_model_proposed_reach_the_engine():
    """``take_draft_token_ids`` hands over what the model proposed."""
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    output = _step(runner, "r")
    drafts = runner.take_draft_token_ids()

    assert drafts is not None
    assert drafts.req_ids == ["r"]
    # The stand-in drafts ``last + 1 + j`` from the row's last committed token,
    # which is the same arithmetic its verify agrees with.
    last = output.sampled_token_ids[0][-1]
    assert drafts.draft_token_ids[0] == [
        (last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)
    ]


def test_the_verify_hidden_handle_reaches_the_drafter_unchanged():
    """The handoff no host fake can fake: identity, not value.

    A real handle is a device tensor whose layout and tensor-parallel
    fracturing the runner must not interpret, so the only thing that can be
    checked is that the object the verify returned is the object the drafter
    receives.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    _step(runner, "r")

    assert model.verify_hidden is not None
    assert model.propose_calls[0]["hidden"] is model.verify_hidden


def test_the_drafter_continues_from_each_row_s_own_committed_token():
    """Two rows with different histories and different accepted counts.

    The committed block is one fixed width and each row's count says how much
    of it is real, so a drafter reading a fixed column, or one row's entry for
    another, continues from the wrong token or from padding. Both rows have to
    differ in both respects for that to be visible, which is why they get
    different prompts and different numbers of drafts.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "a", first_token=1)
    _add_request(runner, "b", first_token=200)

    # This stand-in agrees with whatever is drafted at full depth, so a row's
    # accepted count follows how many drafts it carried: row "a" carries a full
    # set and commits 1+K, row "b" carries one and commits two.
    last_a = int(runner.input_batch.token_ids_cpu[0, PROMPT_LEN - 1])
    last_b = int(runner.input_batch.token_ids_cpu[1, PROMPT_LEN - 1])
    output = _step(
        runner,
        "a",
        "b",
        drafts={
            "a": [(last_a + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)],
            "b": [(last_b + 1) % FAKE_VOCAB_SIZE],
        },
    )
    drafts = runner.take_draft_token_ids()

    assert drafts is not None
    committed = dict(zip(output.req_ids, output.sampled_token_ids))
    assert len(committed["a"]) == DRAFT_LEN + 1
    assert len(committed["b"]) == 2
    # The counts the drafter was handed, which are what select each row's tail.
    assert model.propose_calls[0]["accepted_counts"].tolist()[:2] == [
        DRAFT_LEN + 1,
        2,
    ]

    by_req = dict(zip(drafts.req_ids, drafts.draft_token_ids))
    for req_id in ("a", "b"):
        assert by_req[req_id] == [
            (committed[req_id][-1] + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)
        ]
    # The point of the two rows: they draft from different places.
    assert by_req["a"] != by_req["b"]


def test_a_full_draft_set_commits_every_step_with_the_model_drafting():
    """The property an n-gram drafter cannot give: no draftless step.

    The stand-in proposes what its own verify accepts, so with every draft
    accepted each step commits ``1+K`` tokens. An n-gram drafter stalls
    whenever the text stops repeating, which is what made the device run's
    accept-all measurement alternate between wide and narrow steps.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    widths = []
    pending: dict[str, list[int]] = {}
    for _ in range(4):
        output = _step(runner, "r", drafts=dict(pending))
        widths.append(len(output.sampled_token_ids[0]))
        handed = runner.take_draft_token_ids()
        pending = {"r": list(handed.draft_token_ids[0])} if handed else {}

    # The first step has no drafts in flight yet, and every step after it does.
    assert widths == [1] + [DRAFT_LEN + 1] * 3


def test_a_drafter_returning_the_wrong_shape_is_refused():
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    def wrong_shape(num_drafts, committed, positions, counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        return DraftOutput(draft_token_ids=torch.zeros(1, 1, dtype=torch.int32))

    model.propose_draft_tokens = wrong_shape

    with pytest.raises(ValueError, match="one row per verified row"):
        _step(runner, "r")


def test_a_drafter_returning_a_bare_tensor_is_refused():
    """The same fail-fast rule the verify return follows."""
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    model.propose_draft_tokens = lambda *a, **k: torch.zeros(
        MAX_NUM_REQS, DRAFT_LEN, dtype=torch.int32
    )

    with pytest.raises(TypeError, match="returns a DraftOutput"):
        _step(runner, "r")


def test_a_draft_outside_the_vocabulary_is_refused():
    """A drafted id is committed if the model agrees with it next step."""
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    def out_of_range(num_drafts, committed, positions, counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        return DraftOutput(
            draft_token_ids=torch.full(
                (MAX_NUM_REQS, num_drafts), FAKE_VOCAB_SIZE, dtype=torch.int32
            )
        )

    model.propose_draft_tokens = out_of_range

    with pytest.raises(ValueError, match="outside"):
        _step(runner, "r")


def test_a_drafter_offering_nothing_for_a_row_proposes_nothing():
    """``num_valid`` is how a drafter declines, and it is per row.

    A device graph has one shape, so a drafter with nothing to offer still
    returns ids for every row. ``num_valid`` at 0 is what says those ids are
    not a proposal; encoding the refusal as a dummy token id would be
    indistinguishable from a real draft and the runner would verify it.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")
    _add_request(runner, "s")

    def declines_the_first_row(num_drafts, committed, positions, counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        rows = int(committed.shape[0])
        offered = torch.zeros(rows, dtype=torch.int32)
        # The second row offers one draft; every other row offers none.
        offered[1] = 1
        return DraftOutput(
            draft_token_ids=torch.full((rows, num_drafts), 7, dtype=torch.int32),
            num_valid=offered,
        )

    model.propose_draft_tokens = declines_the_first_row
    _step(runner, "r", "s")

    assert "r" not in runner._proposed_draft_token_ids
    assert runner._proposed_draft_token_ids["s"] == [7]


def test_an_unoffered_row_may_carry_any_ids_including_the_placeholder():
    """The range check covers what can reach the scheduler, and no more.

    A row the drafter is not offering is never read, so its ids are its own
    business. Checking them would force a drafter with nothing to say to
    fabricate in-vocabulary tokens.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    def pads_with_the_placeholder(
        num_drafts, committed, positions, counts, hidden=None
    ):
        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, DraftOutput

        rows = int(committed.shape[0])
        return DraftOutput(
            draft_token_ids=torch.full(
                (rows, num_drafts), PLACEHOLDER_TOKEN_ID, dtype=torch.int32
            ),
            num_valid=torch.zeros(rows, dtype=torch.int32),
        )

    model.propose_draft_tokens = pads_with_the_placeholder
    _step(runner, "r")

    assert runner._proposed_draft_token_ids == {}


def test_an_offered_row_is_still_range_checked():
    """What a row does offer has to be a token the verify could choose."""
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    def offers_one_bad_id(num_drafts, committed, positions, counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        rows = int(committed.shape[0])
        return DraftOutput(
            draft_token_ids=torch.full(
                (rows, num_drafts), FAKE_VOCAB_SIZE, dtype=torch.int32
            ),
            num_valid=torch.ones(rows, dtype=torch.int32),
        )

    model.propose_draft_tokens = offers_one_bad_id

    with pytest.raises(ValueError, match="outside"):
        _step(runner, "r")


@pytest.mark.parametrize(
    "bad, match",
    [
        ("dtype", "dtype"),
        ("shape", "shape"),
        ("range", r"outside \[0, 3\]"),
    ],
)
def test_a_malformed_num_valid_is_refused_by_name(bad, match):
    """Each way the count can be wrong is named, not inferred downstream."""
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    def malformed(num_drafts, committed, positions, counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        rows = int(committed.shape[0])
        offered = {
            "dtype": torch.zeros(rows, dtype=torch.float32),
            "shape": torch.zeros(rows + 1, dtype=torch.int32),
            "range": torch.full((rows,), num_drafts + 1, dtype=torch.int32),
        }[bad]
        return DraftOutput(
            draft_token_ids=torch.zeros(rows, num_drafts, dtype=torch.int32),
            num_valid=offered,
        )

    model.propose_draft_tokens = malformed

    with pytest.raises(ValueError, match=match):
        _step(runner, "r")


def test_the_committed_positions_follow_the_input_block():
    """The drafter is told where each committed token sits.

    The verify's input column 0 holds the row's last committed token at its own
    position, and the return's column ``j`` is the choice that follows it, so
    the committed block starts one past that and runs consecutively.
    """
    positions = TTModelRunner._committed_positions(
        torch.tensor([[7, 8, 9, 10]], dtype=torch.int32), 4
    )

    assert positions.tolist() == [[8, 9, 10, 11]]


def test_the_committed_positions_of_a_narrow_step():
    """A narrow step's positions arrive 1-D, and still produce a block."""
    positions = TTModelRunner._committed_positions(
        torch.tensor([7, 11], dtype=torch.int32), 1
    )

    assert positions.tolist() == [[8], [12]]


def test_a_padding_row_s_committed_positions_stay_negative():
    """A padding row must not look like a request starting from nothing.

    The drafter is handed the verify's rows, padding included, and no live-row
    mask, so the position is the only thing marking a row as owned by no
    request. A padding row's input position is -1, and adding the block's
    offsets to it would produce 0, 1, 2 and so on: a position-aware drafter
    would then allocate or advance state for a request that does not exist.
    """
    positions = TTModelRunner._committed_positions(
        torch.tensor([[7, 8, 9], [-1, -1, -1]], dtype=torch.int32), 3
    )

    assert positions.tolist() == [[8, 9, 10], [-1, -1, -1]]


def test_a_draftless_step_still_proposes_at_the_uniform_width():
    """A model serving its own decode and its own drafter sees one shape.

    ``supports_narrow_decode`` lets a step with nothing to verify run as the
    ordinary decode it is, so the committed block that step produces is one
    column wide. The drafter is a separate call with a fixed shape of its own,
    and a model implementing the documented ``[B, 1+K]`` contract refuses
    anything narrower, so the runner pads before proposing.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    runner._spec_supports_narrow_decode = True
    _add_request(runner, "r")

    # No drafts in flight, so this step is the ordinary one.
    output = _step(runner, "r")

    assert len(output.sampled_token_ids[0]) == 1
    call = model.propose_calls[0]
    assert call["committed_tokens"].shape == (MAX_NUM_REQS, DRAFT_LEN + 1)
    assert call["committed_positions"].shape == (MAX_NUM_REQS, DRAFT_LEN + 1)
    # The one real column is the committed token; the rest is padding.
    assert int(call["committed_tokens"][0, 0]) == output.sampled_token_ids[0][0]
    assert (
        call["committed_tokens"][0, 1:].tolist() == [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN
    )
    # And the drafter still drafts from that one token.
    drafts = runner.take_draft_token_ids()
    assert drafts is not None
    assert drafts.draft_token_ids[0] == [
        (output.sampled_token_ids[0][0] + 1 + j) % FAKE_VOCAB_SIZE
        for j in range(DRAFT_LEN)
    ]


def test_a_fractional_draft_tensor_is_refused():
    """An id read out with ``int()`` would be truncated, not rejected.

    A float tensor passes a range check, so without a dtype check the
    scheduler would store a different token from the one the drafter returned,
    verify that one next step, and commit it if the model agreed.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    _add_request(runner, "r")

    def fractional(num_drafts, committed, positions, counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        return DraftOutput(
            draft_token_ids=torch.full(
                (MAX_NUM_REQS, num_drafts), 1.9, dtype=torch.float32
            )
        )

    model.propose_draft_tokens = fractional

    with pytest.raises(ValueError, match="int32 token ids"):
        _step(runner, "r")


def test_the_forward_carries_the_hidden_handle_from_the_submission(monkeypatch):
    """The production copy, not the test harness's own.

    ``_step`` builds its ``_SyncForward`` by hand, so it proves nothing about
    ``_forward_with_model_input``, which is what moves the submission's handle
    onto the forward on a real launch. Dropping that would leave every device
    drafter running against no hidden state, and every other test here would
    stay green.

    ``finalize_decode`` is stubbed because it waits on device events and
    normalizes the sampling tensors this fake runner does not build. What is
    under test is the two lines around it: the handle read off the submission
    and written onto the forward.
    """
    model = FakeSpecModel()
    runner = _model_drafter_runner(model)
    controller = TTAsyncDecodeController(runner)
    runner.async_decode = controller
    monkeypatch.setattr(
        TTAsyncDecodeController,
        "finalize_decode",
        lambda self, submission: TTFinalizedDecode(
            tt_out=submission.tt_out, tt_log_probs=None
        ),
    )
    _add_request(runner, "r")
    row = runner.input_batch.req_id_to_index["r"]
    runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[row]
    model_input = TTModelRunner._prepare_model_inputs(
        runner, _scheduler_output_for(runner, "r"), None
    )

    fwd = TTModelRunner._forward_with_model_input(runner, model_input)

    assert model.verify_hidden is not None
    assert fwd.spec_hidden is model.verify_hidden


# endregion The model's own drafter
