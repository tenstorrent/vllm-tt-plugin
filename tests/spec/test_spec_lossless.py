# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculation emits exactly what ordinary decoding would, token for token.

This is the whole correctness claim of the accept walk, and it needs three
things that a test against ``FakeSpecModel`` cannot give.

An **independent target**. ``DeterministicTarget`` chooses its next token by a
rule that never looks at the drafts, so "the model agreed with the draft" is a
fact about the draft rather than a property of the stand-in. See
``deterministic_target.py``.

**Wrong drafts**. Every case below offers drafts that are wrong somewhere: at
the first position, in the middle, at the last, or everywhere. A rejection is
where the accept walk can commit the wrong token, drop a token, or carry the
wrong count into the next step, and a test whose drafts are all correct never
reaches that code.

**The whole output**. Each arm runs until it has emitted the same number of
tokens rather than the same number of steps, and the full sequences are
compared. Comparing a shared prefix hides a divergence that starts at the first
token the speculated arm committed beyond the plain arm's length.

The reference arm is a genuinely unspeculated runner: ``_num_speculative_tokens``
of 0, so ``_prepare_model_inputs`` builds a ``[B, 1]`` decode and the ordinary
host-sampling tail commits one token per step. The accept walk is not in it at
all.
"""

import inspect
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward

from .deterministic_target import (
    TARGET_VOCAB_SIZE,
    DeterministicTarget,
    continuation,
    next_token,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 256
MAX_NUM_REQS = 2
PROMPT_LEN = 4
DRAFT_LEN = 3
# Enough tokens that every arm passes several accept-walk outcomes, and not a
# multiple of 1+K, so the speculated arm has to stop mid-block.
COMPARE_TOKENS = 25


def _runner(model: DeterministicTarget, num_speculative_tokens: int) -> SimpleNamespace:
    """A runner fake over the real InputBatch, speculating or not."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=TARGET_VOCAB_SIZE,
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
        # No drafting method: the drafts come from each test, through the
        # scheduler output, so that a case can make them wrong exactly where it
        # means to. ``_propose_ngram_drafts`` returns immediately on this, which
        # leaves the accept walk as the only thing under test.
        _spec_method=None,
        _spec_supports_narrow_decode=False,
        _spec_drafts_from_model=False,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        # The drafts come from each test rather than from a proposer, so that a
        # case can make them wrong exactly where it means to.
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
        # The real sampler, because the plain arm's committed token has to come
        # from the ordinary host-sampling tail rather than from a stand-in.
        host_sampler=Sampler(),
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
        _spec_row_state=TTModelRunner._spec_row_state,
        _spec_candidate_block=TTModelRunner._spec_candidate_block,
        _committed_positions=TTModelRunner._committed_positions,
        _build_host_generators=TTModelRunner._build_host_generators,
    )
    # Every remaining method comes from the real class, because the plain arm
    # runs the ordinary sampling tail and that tail reaches a chain of helpers
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


def _add_request(runner: SimpleNamespace, req_id: str = "r") -> None:
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(11, 11 + PROMPT_LEN)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=PROMPT_LEN,
        output_token_ids=[],
    )
    runner.requests[req_id] = request
    runner.input_batch.add_request(request)


def _step(runner: SimpleNamespace, drafts: list[int] | None = None):
    """One decode step for request "r", speculative or plain."""
    row = runner.input_batch.req_id_to_index["r"]
    runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[row]

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_spec_decode_tokens = (
        {"r": list(drafts)} if drafts else {}
    )
    scheduler_output.num_scheduled_tokens = {"r": 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[int(runner.input_batch.num_tokens[row])],
        num_output_tokens=[0],
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
        batch_size_per_dp=[1],
        perform_device_sampling=False,
        is_decode=True,
        spec_hidden=submission.spec_hidden,
    )
    return runner._finish_front_packed_sync(None, fwd=fwd)


def _tail(runner: SimpleNamespace) -> tuple[int, int]:
    """The request's last committed token and that token's position."""
    row = runner.input_batch.req_id_to_index["r"]
    length = int(runner.input_batch.num_tokens[row])
    return int(runner.input_batch.token_ids_cpu[row, length - 1]), length - 1


def _run_plain(at_least: int) -> list[int]:
    """Ordinary decoding, one token per step, no accept walk anywhere."""
    runner = _runner(DeterministicTarget(), num_speculative_tokens=0)
    _add_request(runner)
    emitted: list[int] = []
    while len(emitted) < at_least:
        output = _step(runner)
        emitted.extend(output.sampled_token_ids[0])
    assert runner.model.plain_calls == len(emitted), (
        "the plain arm must commit exactly one token per decode step, or it is "
        "not the reference it claims to be"
    )
    return emitted


# How each case bends the drafts away from what the target will choose. Each
# takes the tokens the target would emit next and returns what to offer.
_DRAFT_CASES = {
    "all correct": lambda truth: list(truth),
    "wrong at the first": lambda truth: [truth[0] + 1] + truth[1:],
    "wrong in the middle": lambda truth: [truth[0], truth[1] + 1, truth[2]],
    "wrong at the last": lambda truth: truth[:-1] + [truth[-1] + 1],
    "all wrong": lambda truth: [token + 1 for token in truth],
    "shorter than K": lambda truth: list(truth[:1]),
    "one wrong and short": lambda truth: [truth[0] + 1],
}


@pytest.mark.parametrize("case", sorted(_DRAFT_CASES))
def test_speculation_emits_exactly_what_ordinary_decoding_would(case):
    """Whatever is drafted, the committed sequence is the plain one."""
    bend = _DRAFT_CASES[case]
    runner = _runner(DeterministicTarget(), num_speculative_tokens=DRAFT_LEN)
    _add_request(runner)

    emitted: list[int] = []
    steps = 0
    while len(emitted) < COMPARE_TOKENS:
        token, position = _tail(runner)
        # What the target will choose at each candidate position, which is what
        # a perfect drafter would offer; each case bends it.
        truth = continuation(token, position, DRAFT_LEN)
        output = _step(runner, drafts=bend(truth))
        emitted.extend(output.sampled_token_ids[0])
        steps += 1

    plain = _run_plain(COMPARE_TOKENS)

    assert emitted[:COMPARE_TOKENS] == plain[:COMPARE_TOKENS]
    # And against a third, independent expectation, so a fault shared by both
    # runners cannot pass: both arms started from the same prompt tail.
    assert plain[:COMPARE_TOKENS] == continuation(
        11 + PROMPT_LEN - 1, PROMPT_LEN - 1, COMPARE_TOKENS
    )
    if case == "all correct":
        # Sanity on the test itself: a correct draft set must be accepted, or
        # none of the cases above prove anything about acceptance.
        assert steps < COMPARE_TOKENS


def test_a_rejected_draft_commits_the_target_s_own_choice():
    """The token at a rejecting position comes from the model, not the draft.

    This is the one place a wrong draft could reach the output. The row's first
    draft is wrong, so the walk must stop there and commit what the target
    chose at that position, which is neither the draft nor nothing.
    """
    runner = _runner(DeterministicTarget(), num_speculative_tokens=DRAFT_LEN)
    _add_request(runner)
    token, position = _tail(runner)
    truth = continuation(token, position, DRAFT_LEN)

    output = _step(runner, drafts=[truth[0] + 1, truth[1], truth[2]])

    assert output.sampled_token_ids[0] == [truth[0]]
    assert runner._req_accepted_counts["r"] == 1


def test_an_accepted_prefix_stops_at_the_first_wrong_draft():
    """A row accepts up to the wrong draft and no further.

    The drafts after a rejection are not candidates any more, even when they
    would have been right, because the sequence they continue never happened.
    """
    runner = _runner(DeterministicTarget(), num_speculative_tokens=DRAFT_LEN)
    _add_request(runner)
    token, position = _tail(runner)
    truth = continuation(token, position, DRAFT_LEN)

    # The first draft stands, the second does not, and the third is what the
    # target would have chosen had the second been right.
    output = _step(runner, drafts=[truth[0], truth[1] + 1, truth[2]])

    assert output.sampled_token_ids[0] == [truth[0], truth[1]]
    assert runner._req_accepted_counts["r"] == 2


def test_a_fully_accepted_row_commits_the_bonus_the_target_chose():
    """The extra token past the last draft is the target's own next choice."""
    runner = _runner(DeterministicTarget(), num_speculative_tokens=DRAFT_LEN)
    _add_request(runner)
    token, position = _tail(runner)
    truth = continuation(token, position, DRAFT_LEN + 1)

    output = _step(runner, drafts=truth[:DRAFT_LEN])

    assert output.sampled_token_ids[0] == truth
    assert runner._req_accepted_counts["r"] == DRAFT_LEN + 1


def test_the_verify_is_asked_about_the_positions_the_block_occupies():
    """A position error would change the target's answer, and the output.

    The target's choice depends on the position of the token it follows, so the
    candidate block's positions are load-bearing here rather than decorative:
    this is what makes the comparison above cover the block builder's position
    arithmetic and not only its tokens.
    """
    runner = _runner(DeterministicTarget(), num_speculative_tokens=DRAFT_LEN)
    _add_request(runner)
    token, position = _tail(runner)

    _step(runner, drafts=continuation(token, position, DRAFT_LEN))

    call = runner.model.verify_calls[0]
    assert call["positions"][0].tolist() == [
        position,
        position + 1,
        position + 2,
        position + 3,
    ]
    assert int(call["tokens"][0, 0]) == token


def test_two_rows_with_different_drafts_each_commit_their_own_sequence():
    """A mixed batch: one row accepts everything, the other rejects at once.

    Each row's committed tokens must be the continuation of its own history,
    which a walk that mixed rows or a commit that crossed them would break
    while both rows still looked plausible on their own.
    """
    runner = _runner(DeterministicTarget(), num_speculative_tokens=DRAFT_LEN)
    for req_id in ("a", "b"):
        request = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=(
                list(range(11, 11 + PROMPT_LEN))
                if req_id == "a"
                else list(range(211, 211 + PROMPT_LEN))
            ),
            mm_features=None,
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            block_ids=([0],),
            num_computed_tokens=PROMPT_LEN,
            output_token_ids=[],
        )
        runner.requests[req_id] = request
        runner.input_batch.add_request(request)

    rows = {req: runner.input_batch.req_id_to_index[req] for req in ("a", "b")}
    tails = {
        req: (
            int(
                runner.input_batch.token_ids_cpu[
                    row, int(runner.input_batch.num_tokens[row]) - 1
                ]
            ),
            int(runner.input_batch.num_tokens[row]) - 1,
        )
        for req, row in rows.items()
    }
    truth = {
        req: continuation(token, position, DRAFT_LEN + 1)
        for req, (token, position) in tails.items()
    }
    for req, row in rows.items():
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_spec_decode_tokens = {
        "a": truth["a"][:DRAFT_LEN],
        "b": [truth["b"][0] + 1] + truth["b"][1:DRAFT_LEN],
    }
    scheduler_output.num_scheduled_tokens = {"a": 1, "b": 1}
    scheduler_output.total_num_scheduled_tokens = 2
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["a", "b"],
        resumed_req_ids=set(),
        new_token_ids=[[], []],
        all_token_ids={},
        new_block_ids=[None, None],
        num_computed_tokens=[
            int(runner.input_batch.num_tokens[rows["a"]]),
            int(runner.input_batch.num_tokens[rows["b"]]),
        ],
        num_output_tokens=[0, 0],
    )
    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)
    submission = TTAsyncDecodeController(runner).submit_decode(
        model_input, read_from_device=True
    )
    output = runner._finish_front_packed_sync(
        None,
        fwd=_SyncForward(
            tt_out=submission.tt_out,
            tt_log_probs=None,
            sampling_params=model_input.tt_sampling_params,
            model_input=model_input,
            batch_size_per_dp=[2],
            perform_device_sampling=False,
            is_decode=True,
            spec_hidden=submission.spec_hidden,
        ),
    )

    committed = dict(zip(output.req_ids, output.sampled_token_ids))
    assert committed["a"] == truth["a"]
    assert committed["b"] == [truth["b"][0]]


def test_the_target_rejects_a_draft_that_merely_looks_plausible():
    """Guards the test fixture: the rule does not agree by accident.

    Every case above rests on a bent draft being rejected. If the rule were
    monotonic, or the bend landed on a value it happened to choose anyway, the
    cases would quietly pass while testing nothing.
    """
    for token in range(0, TARGET_VOCAB_SIZE, 37):
        for position in (0, 1, 7, 64):
            choice = next_token(token, position)
            assert choice != (choice + 1) % TARGET_VOCAB_SIZE
            # And the rule is not the successor function, which is what makes a
            # bent draft a genuinely different token.
            assert choice != (token + 1) % TARGET_VOCAB_SIZE
