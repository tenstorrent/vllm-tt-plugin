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

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController, _verify_output_tensor
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
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
        _num_speculative_tokens=num_speculative_tokens,
        _spec_method=method,
        _spec_supports_narrow_decode=False,
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
    )
    for name in (
        "_finish_spec_decode",
        "_finish_front_packed_sync",
        "_apply_committed_spec_tokens_to_state",
        "_apply_grammar_to_input",
        "_propose_ngram_drafts",
        "_reorder_grammar_bitmask",
        "take_draft_token_ids",
    ):
        setattr(runner, name, getattr(TTModelRunner, name).__get__(runner))
    return runner


def _add_request(runner: SimpleNamespace, req_id: str) -> None:
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(1, PROMPT_LEN + 1)),
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


# endregion The loop

# region Equality with an unspeculated run


def test_speculation_emits_exactly_what_no_speculation_would():
    """The only thing speculation may preserve is the token sequence.

    Both runs drive the same stand-in over the same prompt. One receives the
    drafts the stand-in's own proposer would produce and so accepts them; the
    other receives none and commits one token per step. The sequences must be
    identical, which is the whole correctness claim of the accept walk.
    """
    steps = 4

    def run(with_drafts: bool) -> list[int]:
        model = FakeSpecModel()
        runner = _runner(model)
        _add_request(runner, "r")
        emitted: list[int] = []
        for _ in range(steps):
            drafts = None
            if with_drafts:
                row = runner.input_batch.req_id_to_index["r"]
                last = int(
                    runner.input_batch.token_ids_cpu[
                        row, int(runner.input_batch.num_tokens[row]) - 1
                    ]
                )
                drafts = {
                    "r": [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]
                }
            output = _step(runner, "r", drafts=drafts)
            emitted.extend(output.sampled_token_ids[0])
        return emitted

    speculated = run(with_drafts=True)
    plain = run(with_drafts=False)

    # The speculated run commits more per step, so compare the common prefix:
    # what both produced has to agree token for token.
    shared = min(len(speculated), len(plain))
    assert shared > 0
    assert speculated[:shared] == plain[:shared]
    assert len(speculated) > len(plain), (
        "speculation committed no more tokens than plain decode, so the test "
        "proves nothing about acceptance"
    )


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
        _verify_output_tensor(answer)


# endregion Drafts handed back to the scheduler
