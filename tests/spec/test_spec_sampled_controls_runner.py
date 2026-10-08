# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Grammars, token filters and logprobs through the runner's real step path.

``test_spec_sampled_controls.py`` holds the host walk to its references. This
file holds the runner to the walk: that a verify asks for ``logits`` whenever a
row carries one of these controls, that what the walk applies is captured from
the step itself, that the scheduler's grammar bitmask reaches the right column
of the right row, and that logprobs reach vLLM in the layout it slices by.

Every synchronous step goes through ``_prepare_model_inputs``,
``submit_decode`` and ``_finish_front_packed_sync`` against ``SampledTarget``,
whose distribution does not depend on what was drafted. The reference arm is a
runner that does not speculate, so its tokens come from vLLM's ordinary host
sampler with the same grammar bitmask and the same logits processors.

The asynchronous tests drive ``execute_model`` and ``sample_tokens`` with the
held readback of ``test_spec_async_sampled.py``, and between the two they do
what ``EngineCore.step_with_batch_queue`` does for a step with a structured
request: take the runner's drafts, validate them against each grammar, and
build the bitmask from what survived.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    PLACEHOLDER_TOKEN_ID,
)
from vllm_tt_plugin.structured_output import spec_grammar_bitmask_for_tt_batch

from . import test_spec_async as async_harness
from . import test_spec_async_sampled as async_sampled
from . import test_spec_sampled_runner as sync_harness
from .sampled_target import (
    SAMPLED_VOCAB_SIZE,
    SUPPORT_PROBABILITIES,
    SampledTarget,
    support,
)
from .test_spec_sampled_controls import _pack

wait_for_released_reads = async_harness.wait_for_released_reads

DRAFT_LEN = sync_harness.DRAFT_LEN
GREEDY = sync_harness.GREEDY

# region Test helpers


def _grammar_allows(history: list[int], token: int) -> bool:
    """A grammar whose state is the output length: one residue mod 3 is out.

    ``SampledTarget``'s support ``base, base+3, base+7, base+11`` covers every
    residue mod 3, twice for ``base``'s, so every context keeps two allowed
    support tokens at least, and the row for column ``j`` differs from column
    ``j+1``'s, which is what a misaligned bitmask would get wrong.
    """
    return token % 3 != len(history) % 3


def _allowlist(history: list[int]) -> torch.Tensor:
    return torch.tensor(
        [_grammar_allows(history, token) for token in range(SAMPLED_VOCAB_SIZE)]
    )


def _validated(history: list[int], drafts: list[int]) -> list[int]:
    """``StructuredOutputGrammar.validate_tokens``: the longest valid prefix."""
    valid: list[int] = []
    for token in drafts:
        if not _grammar_allows(history + valid, token):
            break
        valid.append(token)
    return valid


def _scheduler_rows(history: list[int], scheduled: list[int]) -> torch.Tensor:
    """The rows ``StructuredOutputManager.grammar_bitmask`` writes for a request.

    One row per scheduled token and one for the bonus. The row at a ``-1`` is
    still filled from the current state, and every row after it is all ones
    except the bonus, which is filled from the state the valid tokens reached.
    """
    rows = []
    state = list(history)
    constrained = True
    for token in scheduled:
        rows.append(
            _allowlist(state) if constrained else torch.ones(SAMPLED_VOCAB_SIZE)
        )
        if token == PLACEHOLDER_TOKEN_ID:
            constrained = False
        elif constrained:
            state.append(token)
    rows.append(_allowlist(state))
    return _pack(torch.stack(rows).bool())


def _grammar_output(runner, scheduled: dict[str, list[int]]) -> GrammarOutput | None:
    """The grammar output for this step's structured requests, in batch order."""
    structured = [
        req_id
        for req_id in runner.input_batch.req_ids[: runner.input_batch.num_reqs]
        if runner.requests[req_id].sampling_params.structured_outputs is not None
    ]
    if not structured:
        return None
    rows = [
        _scheduler_rows(
            list(runner.requests[req_id].output_token_ids), scheduled.get(req_id, [])
        )
        for req_id in structured
    ]
    return GrammarOutput(structured, torch.cat(rows).numpy())


def _with_min_tokens(runner) -> None:
    """vLLM's min_tokens processor on the batch, which the reference arm needs."""
    runner.input_batch = InputBatch(
        max_num_reqs=sync_harness.MAX_NUM_REQS,
        max_model_len=sync_harness.MAX_MODEL_LEN,
        max_num_batched_tokens=sync_harness.MAX_MODEL_LEN,
        vocab_size=SAMPLED_VOCAB_SIZE,
        block_sizes=[sync_harness.BLOCK_SIZE],
        kernel_block_sizes=[sync_harness.BLOCK_SIZE],
        logitsprocs=LogitsProcessors(
            [MinTokensLogitsProcessor(None, torch.device("cpu"), False)]
        ),
    )


def _runner(model=None, num_speculative_tokens: int = DRAFT_LEN):
    runner = sync_harness._runner(model or SampledTarget(), num_speculative_tokens)
    _with_min_tokens(runner)
    # Set in TTModelRunner.__init__, which the harness does not run; the
    # ordinary host path unpacks the grammar bitmask with it.
    runner.structured_output_arange = torch.arange(0, 32)
    return runner


def _step(runner, *req_ids: str, drafts: dict[str, list[int]] | None = None):
    """One synchronous step with its grammar; returns ids and logprobs by request.

    ``drafts`` are what the scheduler hands the step, so a structured
    request's are validated first, as ``Scheduler.update_draft_token_ids``
    validates them before they are scheduled.
    """
    drafts = dict(drafts or {})
    for req_id in list(drafts):
        if runner.requests[req_id].sampling_params.structured_outputs is not None:
            drafts[req_id] = _validated(
                list(runner.requests[req_id].output_token_ids), drafts[req_id]
            )
    runner.input_batch.refresh_logitsprocs()
    model_input = sync_harness._build(runner, *req_ids, drafts=drafts)
    grammar = _grammar_output(runner, drafts)
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
    output = runner._finish_front_packed_sync(grammar, fwd=fwd)
    committed = {
        req_id: list(output.sampled_token_ids[output.req_id_to_index[req_id]])
        for req_id in req_ids
    }
    return committed, _logprobs_by_request(output, committed)


def _logprobs_by_request(output, committed):
    """Each request's logprob rows, sliced the way the scheduler slices them."""
    if output.logprobs is None:
        return {}
    by_request = {}
    for req_id, tokens in committed.items():
        sliced = output.logprobs.slice_request(
            output.req_id_to_index[req_id], len(tokens)
        )
        by_request[req_id] = [
            (
                ids.tolist(),
                [round(float(v), 5) for v in values],
                int(rank),
            )
            for ids, values, rank in zip(
                sliced.logprob_token_ids, sliced.logprobs, sliced.sampled_token_ranks
            )
        ]
    return by_request


def _run(runner, req_id, count, *, neighbor=None, drafts_for=None):
    """Step until ``req_id`` has ``count`` tokens; returns its ids and logprobs."""
    logprobs: list = []
    req_ids = (req_id,) if neighbor is None else (neighbor, req_id)
    while len(sync_harness._output(runner, req_id)) < count:
        drafts = {}
        if neighbor is not None:
            drafts[neighbor] = sync_harness._argmax_drafts(runner, neighbor)
        if drafts_for is not None:
            drafts[req_id] = drafts_for(runner)
        _, step_logprobs = _step(runner, *req_ids, drafts=drafts)
        logprobs.extend(step_logprobs.get(req_id, []))
    return sync_harness._output(runner, req_id)[:count], logprobs[:count]


def _run_plain(params: SamplingParams, count: int, first_token: int = 1):
    runner = _runner(num_speculative_tokens=0)
    sync_harness._add_request(runner, "r", params, first_token)
    return _run(runner, "r", count)


def _structured() -> StructuredOutputsParams:
    # The plugin never reads the grammar itself: the scheduler fills the
    # bitmask, which these tests fill from ``_grammar_allows``.
    return StructuredOutputsParams(regex="[a-z]+")


def _even_tokens() -> list[int]:
    return list(range(0, SAMPLED_VOCAB_SIZE, 2))


def _greedy_chain(first_token: int, count: int) -> list[int]:
    token = first_token + sync_harness.PROMPT_LEN - 1
    return sync_harness._base_chain(token, sync_harness.PROMPT_LEN - 1, count)


# The controls under test, each on top of whatever sampling the case sets. A
# multi-token bad word is taken from the greedy chain itself, so the ban bites.
CHAIN = _greedy_chain(1, 8)
CONTROLS = {
    "allowed": {"allowed_token_ids": _even_tokens()},
    "bad-words": {"bad_words_token_ids": [[CHAIN[2], CHAIN[3]], [CHAIN[5]]]},
    "min-tokens": {"min_tokens": 6, "stop_token_ids": [CHAIN[1], CHAIN[4]]},
    "logprobs": {"logprobs": 2},
    "structured": {"structured_outputs": _structured()},
}


# Every control at once except the allowlist, which with the grammar can leave
# a context nothing to sample.
ALL = ("bad-words", "min-tokens", "logprobs", "structured")


def _params(case: str, **sampling) -> SamplingParams:
    controls = dict(CONTROLS[case]) if case != "all" else {}
    if case == "all":
        for name in ALL:
            controls.update(CONTROLS[name])
    bad_words = controls.pop("bad_words_token_ids", None)
    params = SamplingParams(ignore_eos=True, max_tokens=64, **controls, **sampling)
    if bad_words is not None:
        # What ``update_from_tokenizer`` leaves; the test has no tokenizer.
        params._bad_words_token_ids = bad_words
    return params


ALL_CASES = [*CONTROLS, "all"]

# endregion Test helpers

# region Which mode a verify asks for


@pytest.mark.parametrize("case", ALL_CASES)
def test_a_greedy_row_with_a_control_makes_its_verify_ask_for_logits(case):
    """Ids cannot carry a filter, a grammar or a logprob, so the mode moves."""
    model = SampledTarget()
    runner = _runner(model)
    sync_harness._add_request(runner, "g", GREEDY, first_token=21)
    sync_harness._add_request(runner, "c", _params(case, temperature=0.0))
    _step(runner, "g", "c")

    _step(runner, "g", "c", drafts={"g": sync_harness._argmax_drafts(runner, "g")})
    assert model.verify_modes[-1] == ACCEPT_MODE_LOGITS
    assert runner._num_unspeculable_verify_rows == 0

    plain = _runner(SampledTarget())
    sync_harness._add_request(plain, "g", GREEDY, first_token=21)
    _step(plain, "g")
    _step(plain, "g", drafts={"g": sync_harness._argmax_drafts(plain, "g")})
    assert plain.model.verify_modes[-1] == ACCEPT_MODE_ARGMAX_IDS


def test_min_tokens_stops_needing_logits_once_it_is_reached():
    """A greedy row whose output reached min_tokens is certifiable again."""
    runner = _runner()
    sync_harness._add_request(
        runner, "m", SamplingParams(temperature=0.0, min_tokens=2, stop_token_ids=[5])
    )
    assert runner._request_is_argmax_certifiable("m") is False
    _step(runner, "m")
    _step(runner, "m")
    assert len(sync_harness._output(runner, "m")) >= 2
    assert runner._request_is_argmax_certifiable("m") is True


# endregion Which mode a verify asks for

# region Token for token against ordinary decoding


@pytest.mark.parametrize("case", ALL_CASES)
def test_a_seeded_draftless_row_reads_exactly_as_ordinary_decoding(case):
    """Its neighbor drafts every step, so every step is a logits verify.

    The row commits its bonus from each one, which is drawn first, from its
    own generator, under its own controls and grammar row 0. Logprobs, where
    asked for, are compared row for row as the scheduler slices them.
    """
    params = _params(case, temperature=1.0, seed=31)
    runner = _runner()
    sync_harness._add_request(runner, "g", GREEDY, first_token=21)
    sync_harness._add_request(runner, "s", params, first_token=1)
    emitted, logprobs = _run(runner, "s", 24, neighbor="g")

    reference, reference_logprobs = _run_plain(params, 24)
    assert emitted == reference
    assert logprobs == reference_logprobs
    assert set(runner.model.verify_modes) == {ACCEPT_MODE_LOGITS}
    if params.logprobs is not None:
        assert len(logprobs) == 24
    _assert_follows_its_controls(emitted, params, first_token=1)


@pytest.mark.parametrize("drafts", ["right", "wrong"])
@pytest.mark.parametrize("case", ALL_CASES)
def test_a_greedy_row_with_a_control_speculates_to_the_ordinary_output(case, drafts):
    """Deterministic, so the unspeculated arm is an exact reference.

    Right drafts are the reference's own continuation and are accepted, so
    every filter has to read them as history and each column's grammar row has
    to be that column's; wrong drafts are bent, so they are rejected and the
    correction has to be the filtered argmax.
    """
    params = _params(case, temperature=0.0)
    reference, reference_logprobs = _run_plain(params, 40)
    if case != "logprobs":
        assert reference[:30] != _run_plain(GREEDY, 30)[0], (
            f"{case} does not change this request's greedy output, so this test "
            "would pass with it ignored"
        )

    def right(runner):
        done = len(sync_harness._output(runner, "c"))
        return reference[done : done + DRAFT_LEN]

    def wrong(runner):
        bent = right(runner)
        if len(bent) > 1:
            bent[1] = (bent[1] + 1) % SAMPLED_VOCAB_SIZE
        return bent

    model = SampledTarget()
    runner = _runner(model)
    sync_harness._add_request(runner, "c", params)
    emitted, logprobs = _run(
        runner, "c", 30, drafts_for=right if drafts == "right" else wrong
    )

    assert emitted == reference[:30]
    assert logprobs == reference_logprobs[:30]
    # A greedy row whose output reached min_tokens is certifiable by ids again.
    expected_modes = {ACCEPT_MODE_LOGITS}
    if case == "min-tokens":
        expected_modes.add(ACCEPT_MODE_ARGMAX_IDS)
    assert set(model.verify_modes) == expected_modes
    _assert_follows_its_controls(emitted, params, first_token=1)


def _assert_follows_its_controls(emitted, params, first_token):
    """Independently of either arm: what every control allows, at every token.

    Two arms that sampled from a context with nothing allowed would agree on
    the same garbage, so the output is also checked against the support and
    each control directly.
    """
    sync_harness._assert_in_support(
        (first_token + sync_harness.PROMPT_LEN - 1, sync_harness.PROMPT_LEN - 1),
        emitted,
    )
    history: list[int] = []
    for token in emitted:
        if params.structured_outputs is not None:
            assert _grammar_allows(history, token), (history, token)
        if params.allowed_token_ids:
            assert token in params.allowed_token_ids
        for word in params.bad_words_token_ids or ():
            assert not (
                token == word[-1]
                and history[len(history) - len(word) + 1 :] == word[:-1]
            ), (history, token, word)
        if len(history) < params.min_tokens:
            assert token not in params.all_stop_token_ids, (history, token)
        history.append(token)


def test_drafted_commits_report_the_target_s_own_logprobs():
    """Raw logprobs: the support probabilities, whatever the sampling did.

    ``SampledTarget`` puts 0.4, 0.3, 0.2 and 0.1 on its support, so every
    committed token, accepted draft or correction or bonus alike, must report
    the log of its own support probability, rank its support position, and
    list the support in order as its top two. The penalties and temperature
    here change what is committed but not what is reported for it.
    """
    params = SamplingParams(
        temperature=0.9,
        presence_penalty=0.6,
        logprobs=2,
        seed=12,
        ignore_eos=True,
    )
    runner = _runner()
    sync_harness._add_request(runner, "s", params)
    tail = sync_harness._tail(runner, "s")
    widths = []
    rows = []
    while len(sync_harness._output(runner, "s")) < 40:
        committed, logprobs = _step(
            runner, "s", drafts={"s": sync_harness._argmax_drafts(runner, "s")}
        )
        widths.append(len(committed["s"]))
        rows.extend(logprobs["s"])
    assert max(widths) > 1, "no draft was ever accepted"

    token, position = tail
    for emitted, (ids, values, rank) in zip(sync_harness._output(runner, "s"), rows):
        allowed = support(token, position)
        member = allowed.index(emitted)
        assert ids[0] == emitted
        assert values[0] == pytest.approx(
            math.log(SUPPORT_PROBABILITIES[member]), abs=1e-4
        )
        assert rank == member + 1
        assert ids[1:] == allowed[:2]
        token, position = emitted, position + 1


def test_a_multi_token_bad_word_is_applied_on_an_ordinary_decode():
    """The ordinary host sampler needs the output history to match a word.

    vLLM's bad-words op bans a word's last token only after the rest of it,
    and reads that from the output history it is handed. A decode that hands
    it none bans one-token words only, so a greedy request whose chain holds
    the word ``[a, b]`` would emit it.
    """
    params = _params("bad-words", temperature=0.0)
    emitted, _ = _run_plain(params, 8)
    pairs = list(zip(emitted, emitted[1:]))
    assert (CHAIN[2], CHAIN[3]) not in pairs
    assert CHAIN[5] not in emitted
    assert emitted[:3] == CHAIN[:3]


def test_the_published_logprobs_follow_a_length_capped_prefix():
    """A row the length cap shortened publishes fewer logprob rows.

    vLLM slices a request's logprobs by how many tokens it was published, from
    the request's start offset, so the dropped tail must not shift the next
    request's rows.
    """
    ids = torch.arange(6 * 3, dtype=torch.int32).reshape(6, 3)
    values = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
    ranks = torch.arange(6, dtype=torch.int32)
    walked = LogprobsTensors(ids, values, ranks, cu_num_generated_tokens=[0, 3, 4, 6])

    published = TTModelRunner.spec_committed_logprobs(
        walked, [[10, 11], [12], [13, 14]]
    )

    assert published.cu_num_generated_tokens == [0, 2, 3, 5]
    assert published.sampled_token_ranks.tolist() == [0, 1, 3, 4, 5]
    assert published.logprob_token_ids[:, 0].tolist() == [0, 3, 9, 12, 15]

    whole = TTModelRunner.spec_committed_logprobs(walked, [[1, 2, 3], [4], [5, 6]])
    assert whole.cu_num_generated_tokens == [0, 3, 4, 6]
    assert TTModelRunner.spec_committed_logprobs(None, [[1]]) is None


# endregion Token for token against ordinary decoding

# region The scheduler's bitmask layout


def test_the_bitmask_rows_are_located_per_request_and_per_column():
    """Each structured request owns ``1 + scheduled drafts`` consecutive rows."""
    words = 2
    bitmask = torch.arange(1, 8, dtype=torch.int32).unsqueeze(1).repeat(1, words)
    rows_per_request = {"a": 3, "plain": 1, "b": 1, "c": 3}
    # "plain" is scheduled but not structured, and "c" is structured but not in
    # this batch; its rows still count towards the offsets after it.
    ids = ["a", "b", "c"]

    per_column = spec_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=ids,
        rows_per_request=rows_per_request,
        row_req_ids=["b", "plain", "a"],
        batch_length=4,
        width=4,
    )
    assert per_column[:, :, 0].tolist() == [
        [4, -1, -1, -1],
        [-1, -1, -1, -1],
        [1, 2, 3, -1],
        [-1, -1, -1, -1],
    ]

    first_rows = spec_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=ids,
        rows_per_request=rows_per_request,
        row_req_ids=["b", "plain", "a"],
        batch_length=4,
        width=None,
    )
    assert first_rows[:, 0].tolist() == [4, -1, 1, -1]


@pytest.mark.parametrize(
    "rows_per_request, match",
    [({"a": 2, "b": 1, "c": 3}, "has 7 rows"), ({"a": 3, "b": 1}, "recorded None")],
    ids=["short", "missing"],
)
def test_a_bitmask_the_step_cannot_account_for_is_refused(rows_per_request, match):
    """A wrong row count shifts the mask of every later request, silently."""
    with pytest.raises(RuntimeError, match=match):
        spec_grammar_bitmask_for_tt_batch(
            bitmask=torch.zeros(7, 1, dtype=torch.int32),
            structured_output_request_ids=["a", "b", "c"],
            rows_per_request=rows_per_request,
            row_req_ids=["a"],
            batch_length=1,
            width=None,
        )


def test_an_ordinary_decode_on_a_speculating_launch_reads_each_request_s_row_0():
    """An asynchronous launch reserves drafts on every step, plain ones too.

    So the scheduler writes ``1 + K`` rows per structured request even when
    the runner verifies nothing, and a plain decode that read one row per
    request would hand the second request the first one's bonus row.
    """
    runner = _runner()
    for req_id, first in (("a", 1), ("b", 9)):
        sync_harness._add_request(
            runner,
            req_id,
            SamplingParams(temperature=0.0, structured_outputs=_structured()),
            first_token=first,
        )
    model_input = sync_harness._build(
        runner,
        "a",
        "b",
        drafts={
            "a": [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN,
            "b": [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN,
        },
    )
    assert model_input.grammar_rows_per_request == {
        "a": DRAFT_LEN + 1,
        "b": DRAFT_LEN + 1,
    }
    plain_input = replace(model_input, spec_mode=None)
    marks = torch.arange(2 * (DRAFT_LEN + 1), dtype=torch.int32).unsqueeze(1)
    bitmask = runner._reorder_grammar_bitmask(
        GrammarOutput(["a", "b"], marks.numpy()), plain_input, lane_total=None
    )
    assert bitmask.shape[0] == plain_input.input_tokens.shape[0]
    assert bitmask[:2, 0].tolist() == [0, DRAFT_LEN + 1]


def test_a_grammar_output_missing_a_structured_row_is_refused():
    """A structured row the grammar output omits would walk unconstrained.

    The step captured both structured requests when it was built, so a
    synchronous sample that receives rows for one of them only is refused by
    name. An asynchronous sample tolerates it, because a request finished by
    the result before it may be absent and its token is discarded.
    """
    runner = _runner()
    for req_id, first in (("a", 1), ("b", 9)):
        sync_harness._add_request(
            runner,
            req_id,
            SamplingParams(temperature=0.0, structured_outputs=_structured()),
            first_token=first,
        )
    model_input = sync_harness._build(
        runner, "a", "b", drafts={"a": [5] * DRAFT_LEN, "b": [6] * DRAFT_LEN}
    )
    assert model_input.structured_output_req_ids == {"a", "b"}
    only_a = GrammarOutput(
        ["a"], torch.zeros(DRAFT_LEN + 1, 1, dtype=torch.int32).numpy()
    )

    with pytest.raises(RuntimeError, match=r"missing TT batch request IDs: \['b'\]"):
        runner._reorder_grammar_bitmask(only_a, model_input, lane_total=None)
    tolerated = runner._reorder_grammar_bitmask(
        only_a, model_input, lane_total=None, require_complete=False
    )
    assert bool((tolerated[1] == -1).all())


# endregion The scheduler's bitmask layout

# region Asynchronous: the grammar handoff


def _async_runner():
    model, runner = async_sampled._sampled_runner()
    runner.structured_output_arange = torch.arange(0, 32)
    return model, runner


def _submit_async(runner, *req_ids, proposals, handoff: bool):
    """Execute, then do what the engine does before sampling a structured step.

    The scheduler reserved ``[-1] * K`` for every request; the runner verifies
    its own proposals. With ``handoff`` the engine took the drafts back through
    ``take_draft_token_ids``, validated them, and filled the bitmask from what
    survived; without it the step was not deferred and the bitmask was filled
    from the placeholders.
    """
    for req_id, drafts in proposals.items():
        runner._proposed_draft_token_ids[req_id] = list(drafts)
    reservation = {req_id: [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN for req_id in req_ids}
    scheduler_output = async_harness._scheduler_output(
        runner, decoding=req_ids, drafts=reservation
    )
    for req_id in req_ids:
        row = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    assert runner.execute_model(scheduler_output) is None
    scheduled = {req_id: list(tokens) for req_id, tokens in reservation.items()}
    handed = runner.take_draft_token_ids() if handoff else None
    if handed is not None:
        for req_id, drafts in zip(handed.req_ids, handed.draft_token_ids):
            history = list(runner.requests[req_id].output_token_ids)
            valid = _validated(history, drafts[:DRAFT_LEN])
            scheduled[req_id] = valid + [PLACEHOLDER_TOKEN_ID] * (
                DRAFT_LEN - len(valid)
            )
    return runner.sample_tokens(_grammar_output(runner, scheduled)), handed


def _finish(model, runner, wrapper, *req_ids):
    model.release()
    output = wrapper.get_output()
    async_harness._drain(runner, *req_ids)
    return output


def _grammar_chain(runner, req_id, count):
    """The greedy continuation under the grammar, from the row's tail."""
    token, position = async_harness._tail(runner, req_id)
    history = list(runner.requests[req_id].output_token_ids)
    chain = []
    for _ in range(count):
        allowed = [t for t in support(token, position) if _grammar_allows(history, t)]
        token = allowed[0]
        position += 1
        history.append(token)
        chain.append(token)
    return chain


def test_take_draft_token_ids_hands_back_only_a_verify_s_structured_drafts():
    """The deferred grammar path reads the drafts the device already verifies.

    A greedy neighbor's proposal stays with the runner; only the structured
    row's drafts go to the scheduler, once.
    """
    model, runner = _async_runner()
    async_sampled._admit(
        runner,
        ("j", 3, SamplingParams(temperature=0.0, structured_outputs=_structured())),
        ("g", 21, SamplingParams(temperature=0.0)),
    )
    runner._proposed_draft_token_ids["later"] = [1, 2]
    wrapper, handed = _submit_async(
        runner,
        "j",
        "g",
        proposals={"j": [2, 3, 4], "g": async_sampled._drafts(runner, "g")},
        handoff=True,
    )
    assert handed.req_ids == ["j"]
    assert handed.draft_token_ids == [[2, 3, 4]]
    assert runner._proposed_draft_token_ids == {"later": [1, 2]}
    assert runner.take_draft_token_ids() is None
    _finish(model, runner, wrapper, "j", "g")


@pytest.mark.parametrize("handoff", [True, False], ids=["deferred", "immediate"])
def test_an_asynchronous_structured_row_follows_its_grammar(handoff):
    """Greedy under the grammar, so the output is exact either way.

    Handed over, the bitmask holds a row per drafted column and right drafts
    are accepted. Not handed over, the bitmask was filled from placeholders,
    so the row walks as if it had no drafts and commits one token a step.
    """
    model, runner = _async_runner()
    params = SamplingParams(temperature=0.0, structured_outputs=_structured())
    async_sampled._admit(runner, ("j", 3, params))
    expected = _grammar_chain(runner, "j", 40)
    widths = []
    while len(runner.requests["j"].output_token_ids) < 30:
        done = len(runner.requests["j"].output_token_ids)
        wrapper, _ = _submit_async(
            runner,
            "j",
            proposals={"j": expected[done : done + DRAFT_LEN]},
            handoff=handoff,
        )
        output = _finish(model, runner, wrapper, "j")
        widths.append(len(output.sampled_token_ids[output.req_id_to_index["j"]]))

    assert runner.requests["j"].output_token_ids[:30] == expected[:30]
    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}
    if handoff:
        assert max(widths) == DRAFT_LEN + 1
    else:
        assert set(widths) == {1}


def test_an_asynchronous_structured_row_rejects_the_drafts_its_grammar_forbids():
    """Drafts the scheduler found invalid were verified anyway; none commits.

    The drafter proposes the target's unconstrained argmax, which the grammar
    often forbids, so the walk sees forbidden drafts in columns the bitmask
    constrains, and must reject them there.
    """
    model, runner = _async_runner()
    params = SamplingParams(temperature=1.0, seed=5, structured_outputs=_structured())
    async_sampled._admit(runner, ("j", 3, params))
    while len(runner.requests["j"].output_token_ids) < 60:
        wrapper, _ = _submit_async(
            runner,
            "j",
            proposals={"j": async_sampled._drafts(runner, "j")},
            handoff=True,
        )
        _finish(model, runner, wrapper, "j")
    history: list[int] = []
    for token in runner.requests["j"].output_token_ids:
        assert _grammar_allows(history, token), (history, token)
        history.append(token)
    async_sampled.sync_harness._assert_in_support(
        (3 + async_sampled.PROMPT_LEN - 1, async_sampled.PROMPT_LEN - 1),
        runner.requests["j"].output_token_ids,
    )


def test_a_seeded_structured_row_without_the_handoff_reads_as_ordinary_decoding():
    """No handoff means no drafts, which is exactly an ordinary sampled step."""
    params = SamplingParams(
        temperature=1.0, seed=17, ignore_eos=True, structured_outputs=_structured()
    )
    model, runner = _async_runner()
    async_sampled._admit(runner, ("j", 1, params))
    while len(runner.requests["j"].output_token_ids) < 20:
        wrapper, _ = _submit_async(
            runner,
            "j",
            proposals={"j": async_sampled._drafts(runner, "j")},
            handoff=False,
        )
        _finish(model, runner, wrapper, "j")

    reference, _ = _run_plain(params, 20)
    assert runner.requests["j"].output_token_ids[:20] == reference


def test_asynchronous_logprobs_equal_the_synchronous_path_s():
    """Same seed, same drafts, same tokens and the same logprob rows."""
    params = SamplingParams(temperature=1.1, logprobs=3, seed=23, ignore_eos=True)

    model, runner = _async_runner()
    async_sampled._admit(runner, ("s", 7, params))
    deferred_logprobs: list = []
    while len(runner.requests["s"].output_token_ids) < 30:
        wrapper = async_harness._submit_step(
            runner, "s", drafts={"s": async_sampled._drafts(runner, "s")}
        )
        output = _finish(model, runner, wrapper, "s")
        committed = {"s": output.sampled_token_ids[output.req_id_to_index["s"]]}
        deferred_logprobs.extend(_logprobs_by_request(output, committed)["s"])

    sync_runner = _runner()
    sync_harness._add_request(sync_runner, "s", params, first_token=7)
    immediate, immediate_logprobs = _run(
        sync_runner, "s", 30, drafts_for=lambda r: sync_harness._argmax_drafts(r, "s")
    )
    assert runner.requests["s"].output_token_ids[:30] == immediate
    assert deferred_logprobs[:30] == immediate_logprobs


# endregion Asynchronous: the grammar handoff
