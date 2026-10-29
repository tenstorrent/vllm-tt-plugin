# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Plumbing of a chained masked step: the drafts of the step before it.

A masked step chained on the step before it is built before that step is read,
so the drafts it verifies are on the device only. The model returns them in
``FusedReadout.drafts``; the reader thread keeps them, with the committed
prefixes, in a record by tip id (``ChainResolved``); and
``TTModelRunner.take_draft_token_ids`` hands them to the scheduler from that
record (``_chain_handoff``). These tests pin that plumbing on the chain-capable
fake, with the real xgrammar scheduler rows for the end to end case:

- the record exists before the output is returned, and the handoff equals the
  drafts the model reports, trimmed to each row's hint;
- an unresolved step is waited for through ``ensure_finalized``, exactly once;
- a request with a hint and no drafts to give is an error, never a silent pass;
- the commit view (applied output plus the step's prefix) equals the state after
  the commit is applied;
- the scheduler rows made from the handed-over drafts give the commit of the
  ``logits`` walk, for a plain grammar row and for a row at its reasoning end.

The second half (section T4) pins the chained masked step itself, on the
mask-capable chain fake (``MaskTarget``, whose begin takes the chain arguments
and whose readout carries the drafts): the chain blockers (a model that serves
masks chains grammar, allowed ids, bad words and ``min_tokens`` rows, and with the
capability off the blockers are today's), the two-call step and the tip it leaves,
the pending guards, the rows that leave in the window, the sample-time wait that
only an N-dependent static part needs, and the thinking row's all-ones mask.
"""

from __future__ import annotations

import dataclasses
import threading
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import ChainResolved, TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_decode import FusedReadout
from vllm_tt_plugin.spec_masks import (
    SpecMaskPlan,
    StaticMasks,
    all_ones,
    bad_word_last_token,
    deferred_mask_plan,
    device_mask_stats,
    is_unconstrained,
    mask_words,
    reset_device_mask_stats,
    resolve_deferred_plan,
    step_mask,
    worst_case_row_empty,
)

from .fake_spec_model import make_fake_spec_model
from .test_spec_async import BLOCK_SIZE, MAX_MODEL_LEN, MAX_NUM_REQS
from .test_spec_chain_async import Engine, K, _decode_so, _make_runner
from .test_spec_chain_async import _env as _chain_env  # noqa: F401  (autouse)
from .test_spec_device_mask_plugin import (
    EVENS,
    REAL_VOCAB,
    SV,
    THINK_END,
    WIDTH,
    MaskEngine,
    MaskTarget,
    _grammar_request,
    _logits,
    _mask_params,
    _plugin_rows,
    _scheduler_step,
    _tokens,
    _walk,
    unpack_allowed,
)
from .test_spec_device_mask_plugin import xg as _xg  # noqa: F401  (a fixture)
from .test_spec_sampled_controls import _pack
from .test_spec_sampled_controls_runner import (
    _grammar_allows,
    _scheduler_rows,
    _structured,
)


@pytest.fixture(scope="module")
def xg(request):
    """vLLM's xgrammar manager of the mask tests (skips where it is unavailable)."""
    return request.getfixturevalue("_xg")


@pytest.fixture(autouse=True)
def _stats():
    reset_device_mask_stats()


# region The model: a chain fake whose readout carries the next step's drafts


class DraftsModel(make_fake_spec_model(serve_chain=True)):
    """The chain fake with ``FusedReadout.drafts``: per row, the drafts the next
    chained step verifies (``_chain_drafts`` of the row's last committed token
    and next position, the rule the fake's chained step itself applies).

    ``with_drafts=False`` is a model that does not report them."""

    with_drafts = True
    supported_k = (K,)
    max_supported_num_seqs = MAX_NUM_REQS

    def _chain_forward(self, *args, **kwargs):
        out = super()._chain_forward(*args, **kwargs)
        handle = out.handle
        if self.with_drafts:
            readout = handle._readout
            rows = int(readout.accepted_counts.shape[0])
            drafts = torch.zeros((rows, self._num_drafts), dtype=torch.int32)
            for row, (last, next_p0) in self._dev_state.items():
                drafts[row] = torch.tensor(
                    self._chain_drafts(last, next_p0), dtype=torch.int32
                )
            handle._readout = dataclasses.replace(readout, drafts=drafts)
        return out


def _engine(monkeypatch, *, with_drafts=True, hold=False, steps=0, **knobs):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = DraftsModel()
    model.with_drafts = with_drafts
    model.hold_reads = hold
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("b", 40)
    for _ in range(steps):
        engine.iterate()
    return model, runner, engine


def _tip(wrapper):
    return wrapper._controller._deferred_handle(wrapper._submission).tip_id


def _handle(model, tip):
    return next(h for h in model.handles if h.tip_id == tip)


# endregion

# region a. The handoff from a resolved step


def test_a_the_record_is_stored_before_the_output_is_returned(monkeypatch):
    model, runner, engine = _engine(monkeypatch, steps=1)
    wrapper, _ = engine.inflight[0]
    tip = _tip(wrapper)
    controller = runner.async_decode
    assert controller.resolved_view(tip) is None  # nothing read yet
    out = wrapper.get_output()
    record = controller.resolved_view(tip)
    assert isinstance(record, ChainResolved) and record.tip == tip
    assert record.row_req_ids == ("a", "b")
    # the prefixes are what the output published, row by row
    for req_id, prefix in zip(record.row_req_ids, record.prefixes):
        assert prefix == list(out.sampled_token_ids[out.req_id_to_index[req_id]])
    assert torch.equal(record.drafts, _handle(model, tip)._readout.drafts)
    # resolved_view is a pure read: no wait was counted
    assert device_mask_stats()["waits"] == 0


def test_a_the_handoff_is_the_drafts_of_the_tip_trimmed_to_each_hint(monkeypatch):
    model, runner, engine = _engine(monkeypatch, steps=4)
    controller = runner.async_decode
    tip = max(controller._resolved)
    record = controller.resolved_view(tip)
    readout = _handle(model, tip)._readout
    runner._chain_handoff = (tip, {"a": 2, "b": K, "gone": 0})
    handed = runner.take_draft_token_ids()
    assert handed.req_ids == ["a", "b"]
    assert handed.draft_token_ids == [
        readout.drafts[0, :2].tolist(),
        readout.drafts[1, :K].tolist(),
    ]
    # independent of the readout: the rule the fake's next chained step applies
    for row, req_id in enumerate(record.row_req_ids):
        last = record.prefixes[row][-1]
        next_p0 = int(readout.p0[row]) + int(readout.accepted_counts[row])
        assert (
            handed.draft_token_ids[row]
            == model._chain_drafts(last, next_p0)[: len(handed.draft_token_ids[row])]
        )
    assert runner._grammar_handed_over_req_ids == {"a", "b"}
    assert runner._chain_handoff is None
    # a second call hands nothing: the handoff is taken once
    assert runner.take_draft_token_ids() is None
    assert runner._grammar_handed_over_req_ids == set()
    engine.finish()


def test_a_the_handoff_keeps_the_build_time_drafts_of_the_other_rows(monkeypatch):
    model, runner, engine = _engine(monkeypatch, steps=3)
    tip = max(runner.async_decode._resolved)
    runner._grammar_draft_handoff = {"other": [7, 8]}
    runner._chain_handoff = (tip, {"a": 1})
    handed = runner.take_draft_token_ids()
    assert dict(zip(handed.req_ids, handed.draft_token_ids))["other"] == [7, 8]
    assert set(runner._grammar_handed_over_req_ids) == {"other", "a"}
    # with no chain handoff the build-time path is exactly as it was
    runner._grammar_draft_handoff = {"other": [7, 8]}
    handed = runner.take_draft_token_ids()
    assert handed.req_ids == ["other"] and handed.draft_token_ids == [[7, 8]]
    engine.finish()


def test_a_only_the_newest_records_are_kept(monkeypatch):
    model, runner, engine = _engine(monkeypatch, steps=12)
    engine.finish()
    controller = runner.async_decode
    assert len(controller._resolved) == 4
    assert sorted(controller._resolved) == sorted(h.tip_id for h in model.handles)[-4:]
    assert len(controller._steps_by_tip) == 4


# endregion

# region b. Waiting for an unresolved step


def test_b_the_handoff_waits_for_an_unresolved_step_exactly_once(monkeypatch):
    model, runner, engine = _engine(monkeypatch, hold=True, steps=1)
    wrapper, _ = engine.inflight[0]
    tip = _tip(wrapper)
    handle = _handle(model, tip)
    controller = runner.async_decode
    assert controller.resolved_view(tip) is None and handle.waits == 0
    runner._chain_handoff = (tip, {"a": 2})
    handed = runner.take_draft_token_ids()
    assert handed.draft_token_ids == [handle._readout.drafts[0, :2].tolist()]
    assert handle.waits == 1 and device_mask_stats()["waits"] == 1
    # the record is there now: a second handoff and the core's own read do not
    # read the step again
    runner._chain_handoff = (tip, {"b": 1})
    runner.take_draft_token_ids()
    out = wrapper.get_output()
    assert handle.waits == 1 and device_mask_stats()["waits"] == 1
    assert out.req_ids == ["a", "b"]
    engine.inflight.clear()


def test_b_a_wait_on_the_reader_thread_and_one_on_the_worker_read_once(monkeypatch):
    model, runner, engine = _engine(monkeypatch, hold=True, steps=1)
    wrapper, _ = engine.inflight[0]
    tip = _tip(wrapper)
    handle = _handle(model, tip)
    handle.release()
    box = []
    reader = threading.Thread(target=lambda: box.append(wrapper.get_output()))
    reader.start()
    record = runner.async_decode.wait_resolved(tip)
    reader.join(timeout=30)
    assert box and handle.waits == 1
    assert record is runner.async_decode.resolved_view(tip)
    engine.inflight.clear()


def test_b_a_step_that_is_not_remembered_cannot_be_waited_for(monkeypatch):
    model, runner, engine = _engine(monkeypatch, steps=1)
    with pytest.raises(RuntimeError, match="not one of the newest"):
        runner.async_decode.wait_resolved(999)
    engine.finish()


# endregion

# region c. A row with a hint and no drafts to give


def test_c_a_request_that_is_not_a_row_of_the_tip_is_refused(monkeypatch):
    model, runner, engine = _engine(monkeypatch, steps=3)
    tip = max(runner.async_decode._resolved)
    runner._chain_handoff = (tip, {"a": 2, "late-joiner": 1})
    with pytest.raises(RuntimeError, match="late-joiner"):
        runner.take_draft_token_ids()
    engine.finish()


def test_c_a_hint_of_zero_needs_no_drafts_and_no_record_row(monkeypatch):
    model, runner, engine = _engine(monkeypatch, with_drafts=False, steps=3)
    tip = max(runner.async_decode._resolved)
    runner._chain_handoff = (tip, {"a": 0, "late-joiner": 0})
    assert runner.take_draft_token_ids() is None
    assert runner._grammar_handed_over_req_ids == set()
    engine.finish()


# endregion

# region d. A model without drafts (the handoff side; the blocker is the chain build's)


def test_d_a_model_without_drafts_gives_none_and_a_hint_is_a_loud_error(monkeypatch):
    model, runner, engine = _engine(monkeypatch, with_drafts=False, steps=3)
    tip = max(runner.async_decode._resolved)
    assert runner.async_decode.resolved_view(tip).drafts is None
    runner._chain_handoff = (tip, {"a": 2})
    with pytest.raises(RuntimeError, match="FusedReadout.drafts"):
        runner.take_draft_token_ids()
    engine.finish()


def test_d_the_readout_drafts_are_checked_like_the_candidates():
    ids = torch.zeros((2, K + 1), dtype=torch.int32)
    counts = torch.ones(2, dtype=torch.int32)
    p0 = torch.zeros(2, dtype=torch.int32)
    ok = FusedReadout(ids, counts, p0, drafts=torch.zeros((2, K), dtype=torch.int32))
    assert ok.drafts.shape == (2, K) and FusedReadout(ids, counts, p0).drafts is None
    with pytest.raises(TypeError, match="int32"):
        FusedReadout(ids, counts, p0, drafts=torch.zeros((2, K), dtype=torch.int64))
    with pytest.raises(TypeError, match="2-D"):
        FusedReadout(ids, counts, p0, drafts=torch.zeros(K, dtype=torch.int32))
    with pytest.raises(TypeError, match="int32"):
        FusedReadout(ids, counts, p0, drafts=[[1, 2, 3]])
    with pytest.raises(ValueError, match="rows"):
        FusedReadout(ids, counts, p0, drafts=torch.zeros((3, K), dtype=torch.int32))


# endregion

# region e. The commit view


def test_e_the_commit_view_equals_the_state_after_the_commit(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = DraftsModel()
    runner = _make_runner(model)
    engine = Engine(runner)
    min_tokens = SamplingParams(
        temperature=0.0, ignore_eos=True, min_tokens=9, stop_token_ids=[3, 5]
    )
    engine.add("a", 1, min_tokens)
    engine.add("b", 40)
    controller = runner.async_decode
    checked = 0
    for step in range(14):
        views = {}
        if controller._resolved:
            tip = max(controller._resolved)
            views = {
                req_id: controller.commit_view(req_id, tip)
                for req_id in ("a", "b")
                if req_id in runner.requests
            }
        before = {r: list(s.output_token_ids) for r, s in runner.requests.items()}
        engine.iterate()  # its build applies the commit of the step `tip`
        for req_id, view in views.items():
            state = runner.requests[req_id]
            assert view == list(state.output_token_ids), (step, req_id)
            assert len(view) >= len(before[req_id])
            # what min_tokens and a two-token bad word read off it
            params = state.sampling_params
            assert params.min_tokens - len(view) == params.min_tokens - len(
                state.output_token_ids
            )
            word = view[-2:]
            assert bad_word_last_token(
                [word[0], 5], list(state.output_token_ids)
            ) == bad_word_last_token([word[0], 5], view)
            checked += 1
    assert checked >= 10
    # the view is a pure read: a second call gives the same, and state is unchanged
    tip = max(controller._resolved)
    state_output = list(runner.requests["a"].output_token_ids)
    assert controller.commit_view("a", tip) == controller.commit_view("a", tip)
    assert list(runner.requests["a"].output_token_ids) == state_output
    engine.finish()


def test_e_the_commit_view_of_an_unresolved_step_is_refused(monkeypatch):
    model, runner, engine = _engine(monkeypatch, hold=True, steps=1)
    tip = _tip(engine.inflight[0][0])
    with pytest.raises(RuntimeError, match="wait_resolved"):
        runner.async_decode.commit_view("a", tip)
    engine.inflight[0][0].get_output()
    assert (
        runner.async_decode.commit_view("a", tip)[-1]
        == (runner.async_decode.resolved_view(tip).prefixes[0][-1])
    )
    engine.inflight.clear()


# endregion

# region counters


def test_the_submission_line_adds_the_chain_counters_only_when_there_are_some():
    report = TTAsyncDecodeController._device_masks_report
    assert report() == ""
    from vllm_tt_plugin.spec_masks import (
        count_mask_chained,
        count_mask_late,
        count_mask_step,
        count_mask_wait,
    )

    count_mask_step(2)
    plain = report()
    assert "chained" not in plain and "waits" not in plain
    count_mask_chained()
    count_mask_late()
    count_mask_wait()
    count_mask_wait()
    assert report() == plain + ", chained 1, late 1, waits 2"


# endregion

# region f. Real xgrammar: scheduler rows made from the handed-over drafts


def _handoff_runner(drafts: list[int], hint: int):
    """A runner stub whose ``take_draft_token_ids`` is the real one, over a real
    controller holding the resolved record of a step whose drafts are ``drafts``."""
    controller = TTAsyncDecodeController(SimpleNamespace())
    padded = drafts + [-1] * (K - len(drafts))
    controller._resolved[7] = ChainResolved(
        tip=7,
        row_req_ids=("pad", "r"),
        prefixes=[[1], [2]],
        drafts=torch.tensor([[0] * K, padded], dtype=torch.int32),
    )
    runner = SimpleNamespace(
        async_decode_scheduling=True,
        async_decode=controller,
        _grammar_draft_handoff={},
        _grammar_handed_over_req_ids=set(),
        _chain_handoff=(7, {"r": hint}),
    )
    handed = TTModelRunner.take_draft_token_ids(runner)
    return dict(zip(handed.req_ids, handed.draft_token_ids))["r"]


def _chain_plan():
    """What a chained build makes: every structured row is a grammar row."""
    static = StaticMasks(all_ones(1, WIDTH, words=mask_words(REAL_VOCAB)), ())
    return SpecMaskPlan(static, (0,))


@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(
    "case", ["valid", "first_invalid", "middle_invalid", "trimmed"]
)
def test_f_a_handed_over_grammar_row_commits_what_the_logits_walk_commits(
    xg, case, seed, temperature
):
    ids = _tokens(xg, '{"a": 12}')
    wrong = _tokens(xg, "zebra")[0]
    drafts = {
        "valid": ids[:3],
        "first_invalid": [wrong, ids[1], ids[2]],
        "middle_invalid": [ids[0], wrong, ids[2]],
        "trimmed": ids[:3],
    }[case]
    hint = 2 if case == "trimmed" else K
    handed = _handoff_runner(drafts, hint)
    assert handed == drafts[:hint]
    request = _grammar_request(xg)
    bitmask, scheduled = _scheduler_step(xg, request, handed)
    grammar = _plugin_rows(bitmask, scheduled)
    mask, constrained = step_mask(
        _chain_plan(), grammar, ["r"], rows=1, width=WIDTH, vocab_size=REAL_VOCAB
    )
    assert constrained == 1
    cut = len(xg.validate_tokens(request, list(handed)))
    assert bool(is_unconstrained(mask[0, cut + 1 :]).all())
    reference, masked = _walk(
        _logits(seed, handed), handed, grammar, mask, temperature, seed
    )
    assert torch.equal(masked.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(masked.accepted_counts, reference.accepted_counts)
    committed = reference.committed_token_ids[
        0, : int(reference.accepted_counts[0])
    ].tolist()
    assert xg.validate_tokens(request, committed) == committed


@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("end_at", [0, 1, 2, None])
def test_f_a_thinking_row_is_masked_by_the_scheduler_rows_after_its_end(
    xg, end_at, temperature
):
    """In a chained step a row in its reasoning phase is a grammar row too: the
    scheduler rows are all ones up to and including ``</think>`` and the grammar
    after it, because the drafts it was given are the ones the device verifies."""
    start = _tokens(xg, '{"a": 1')[:3]
    thought = _tokens(xg, "think")[0]
    drafts = [thought, thought, thought]
    if end_at is not None:
        drafts[end_at] = THINK_END
        for column in range(end_at + 1, K):
            drafts[column] = start[column - end_at - 1]
    handed = _handoff_runner(drafts, K)
    assert handed == drafts
    request = _grammar_request(xg, reasoning_ended=None)
    bitmask, scheduled = _scheduler_step(xg, request, handed)
    grammar = _plugin_rows(bitmask, scheduled)
    constrained_rows = (~is_unconstrained(grammar[0])).nonzero().reshape(-1).tolist()
    assert constrained_rows == (
        [] if end_at is None else list(range(end_at + 1, WIDTH))
    )
    mask, constrained = step_mask(
        _chain_plan(), grammar, ["r"], rows=1, width=WIDTH, vocab_size=REAL_VOCAB
    )
    if end_at is None:
        # nothing to constrain: the finish replays no mask
        assert mask is None and constrained == 0
        return
    assert constrained == 1
    reference, masked = _walk(_logits(7, handed), handed, grammar, mask, temperature, 7)
    assert torch.equal(masked.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(masked.accepted_counts, reference.accepted_counts)
    committed = reference.committed_token_ids[
        0, : int(reference.accepted_counts[0])
    ].tolist()
    assert xg.validate_tokens(request, committed) == committed


# endregion


# region T4. The harness: a masked, chained run on the mask-capable chain fake


def _batch(*procs):
    """An input batch with vLLM's logits processors, as a launch builds it."""
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS)
    )
    return InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=SV,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
        logitsprocs=LogitsProcessors(
            [proc(config, torch.device("cpu"), False) for proc in procs]
        ),
    )


def _masked_runner(
    monkeypatch,
    *,
    mask=True,
    with_drafts=True,
    hold=False,
    engine_cls=MaskEngine,
    procs=(MinTokensLogitsProcessor,),
):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    torch.manual_seed(5)
    model = MaskTarget()
    model.with_drafts = with_drafts
    model.hold_reads = hold
    runner = _make_runner(model)
    runner.input_batch = _batch(*procs)
    runner._spec_device_mask = mask
    runner.model_config.logprobs_mode = "raw_logprobs"
    return model, runner, engine_cls(runner)


def _run(monkeypatch, params=None, *, steps=16, stop_after=None, **kw):
    """Two plain greedy rows and, when ``params`` is given, a row ``x`` with them."""
    model, runner, engine = _masked_runner(monkeypatch, **kw)
    engine.add("g1", 1)
    engine.add("g2", 9)
    if params is not None:
        engine.add("x", 17, params, stop_after=stop_after)
    for _ in range(steps):
        engine.iterate()
    engine.finish()
    return model, runner, engine


def _step_of_calls(model):
    """The fake's calls as ``(step number, name)``: a begin or a one-call step
    starts a step, its finish belongs to it."""
    out, step = [], -1
    for call in model.calls:
        if call[0] != "finish":
            step += 1
        out.append((step, call[0]))
    return out


def _chained_steps(model):
    return [i for i, log in enumerate(model.chain_log) if bool(log["chain_rows"].any())]


def _streams(engine, rows=("g1", "g2")):
    n = min(len(engine.outputs[r]) for r in rows)
    return {r: engine.outputs[r][:n] for r in rows}


# endregion

# region T4 a. The chain blockers


def _blocker_runner(monkeypatch, params, *, mask=True, drafts_seen=True):
    model, runner, engine = _masked_runner(
        monkeypatch,
        mask=mask,
        procs=(MinTokensLogitsProcessor, MinPLogitsProcessor, LogitBiasLogitsProcessor),
    )
    engine.add("a", 1)
    engine.add("x", 40, params)
    if drafts_seen:
        runner.async_decode._latest_readout_drafts = True
    return runner


def _blocker(runner, drafts=None):
    rows = ["a", "x"]
    kw = {} if drafts is None else {"drafts": drafts}
    return runner._chain_rows_blocker(rows, _decode_so(runner, rows, **kw), 1)[0]


def _p(**kw):
    return SamplingParams(temperature=0.0, ignore_eos=True, **kw)


def _bad(words, **kw):
    params = _p(**kw)
    params._bad_words_token_ids = words  # what update_from_tokenizer leaves
    return params


# (params, the blocker with the capability, the blocker without it)
BLOCKERS = {
    "plain": (_p(), None, None),
    "allowed": (_p(allowed_token_ids=EVENS), None, "allowed_token_ids"),
    "one-token bad word": (_bad([[4]]), None, "bad_words"),
    "multi-token bad word": (_bad([[4, 5]]), None, "bad_words"),
    "min_tokens": (_mask_params("min_tokens"), None, "logitsprocs"),
    "grammar": (_mask_params("structured"), None, "grammar"),
    "grammar+allowed": (
        _p(structured_outputs=_structured(), allowed_token_ids=EVENS),
        "grammar_with_controls",
        "allowed_token_ids",
    ),
    "grammar+min_tokens": (
        _p(structured_outputs=_structured(), min_tokens=5, stop_token_ids=[3]),
        "grammar_with_controls",
        "logitsprocs",
    ),
    "logprobs+allowed": (
        _p(logprobs=2, allowed_token_ids=EVENS),
        "logprobs",
        "allowed_token_ids",
    ),
    "min_p": (
        SamplingParams(temperature=0.8, top_k=5, min_p=0.1, ignore_eos=True),
        "logitsprocs",
        "logitsprocs",
    ),
    "logit_bias": (_p(logit_bias={3: 1.0}), "logitsprocs", "logitsprocs"),
    # the worst case of the static row (see test_a_the_worst_case_row ...)
    "allowed = the stop ids, min_tokens pending": (
        _p(allowed_token_ids=[3, 5], min_tokens=7, stop_token_ids=[3, 5, 9]),
        "mask_empty_risk",
        "allowed_token_ids",
    ),
    "allowed = the last token of a longer bad word": (
        _bad([[4, 5]], allowed_token_ids=[5]),
        "mask_empty_risk",
        "allowed_token_ids",
    ),
    "allowed with one id to spare": (
        _bad([[4, 5]], allowed_token_ids=[5, 6]),
        None,
        "allowed_token_ids",
    ),
}


@pytest.mark.parametrize("case", list(BLOCKERS))
def test_a_blocker_matrix_with_the_capability(monkeypatch, case):
    params, on, _ = BLOCKERS[case]
    assert _blocker(_blocker_runner(monkeypatch, params)) == on


@pytest.mark.parametrize("case", list(BLOCKERS))
def test_a_with_the_capability_off_the_blockers_are_todays(monkeypatch, case):
    """Todays order: allowed ids, bad words, logprobs, the processors (min_tokens
    among them), then the grammar. The drafts rule does not exist."""
    params, _, off = BLOCKERS[case]
    runner = _blocker_runner(monkeypatch, params, mask=False, drafts_seen=False)
    assert _blocker(runner) == off


def test_a_real_drafts_and_a_certificate_row_still_refuse_the_chain(monkeypatch):
    for mask in (True, False):
        runner = _blocker_runner(monkeypatch, _p(), mask=mask)
        assert _blocker(runner, drafts={"a": [7] * K, "x": [7] * K}) == "real_drafts"
    runner = _blocker_runner(
        monkeypatch, SamplingParams(temperature=0.8, top_p=0.9, ignore_eos=True)
    )
    runner._spec_device_top_p = True
    assert _blocker(runner) == "device_top_p"


def test_a_a_penalty_row_keeps_todays_rules(monkeypatch):
    runner = _blocker_runner(monkeypatch, _p(presence_penalty=0.5))
    assert _blocker(runner) == "penalties"
    runner = _blocker_runner(monkeypatch, _p(presence_penalty=0.5), mask=False)
    assert _blocker(runner) == "penalties"


@pytest.mark.parametrize(
    "case",
    [
        "allowed",
        "one-token bad word",
        "multi-token bad word",
        "min_tokens",
        "grammar",
    ],
)
def test_a_each_control_alone_chains_in_a_run(monkeypatch, case):
    """The matrix answer is what a run does: the row's steps chain."""
    params = BLOCKERS[case][0]
    model, runner, engine = _run(monkeypatch, params, steps=14)
    assert len(_chained_steps(model)) >= 8, case
    assert all(c[0] != "one_call" for c in model.calls[-4:]) or case == "min_tokens"


def test_a_the_worst_case_row_predicate():
    def empty(allowed=None, bad=(), min_tokens=0, stops=(), output=0, vocab=64):
        params = SimpleNamespace(
            allowed_token_ids=allowed,
            bad_words_token_ids=list(bad),
            min_tokens=min_tokens,
            all_stop_token_ids=set(stops),
        )
        return worst_case_row_empty(params, output, vocab)

    # allowed ids that are only stop ids, while min_tokens is pending at build
    assert empty([3, 5], min_tokens=7, stops=[3, 5, 9])
    assert not empty([3, 5, 6], min_tokens=7, stops=[3, 5, 9])
    # min_tokens reached at build: the stop ids are free
    assert not empty([3, 5], min_tokens=7, stops=[3, 5, 9], output=7)
    # allowed ids and a longer bad word whose last token is the only allowed id
    assert empty([5], bad=[[4, 5]])
    assert not empty([5, 6], bad=[[4, 5]])
    # the FIRST token of a longer word is not banned by it
    assert not empty([4], bad=[[4, 5]])
    # a one-token word is banned outright
    assert empty([5], bad=[[5]])
    # ids outside the vocabulary are no help
    assert empty([70, 80], vocab=64)
    # no allowed ids: empty only if the bad words cover the vocabulary
    assert not empty(bad=[[1], [2, 3]])
    assert empty(bad=[[i] for i in range(64)])
    assert empty(None, min_tokens=3, stops=range(64))


# endregion

# region T4 no_device_drafts


def test_a_no_device_drafts_until_a_readout_carried_them(monkeypatch):
    runner = _blocker_runner(monkeypatch, _p(), drafts_seen=False)
    controller = runner.async_decode
    assert controller._latest_readout_drafts is None
    assert _blocker(runner) == "no_device_drafts"
    controller._latest_readout_drafts = True
    assert _blocker(runner) is None
    # true again when the latest fused readout has none
    controller._latest_readout_drafts = False
    assert _blocker(runner) == "no_device_drafts"
    # the capability off: no such rule
    runner = _blocker_runner(monkeypatch, _p(), mask=False, drafts_seen=False)
    assert _blocker(runner) is None


def test_a_a_model_without_drafts_never_chains_a_masked_run_and_streams_are_equal(
    monkeypatch,
):
    model, runner, engine = _run(
        monkeypatch, _mask_params("allowed"), with_drafts=False
    )
    assert _chained_steps(model) == []
    assert runner.async_decode._chain_breaks["no_device_drafts"] > 0
    assert runner.async_decode._latest_readout_drafts is False
    off_model, off, off_engine = _run(monkeypatch, _mask_params("allowed"), mask=False)
    assert engine.outputs["x"][:12] == off_engine.outputs["x"][:12]
    assert _streams(engine) == _streams(off_engine)


def test_a_the_flag_follows_the_newest_readout(monkeypatch):
    model, runner, engine = _run(monkeypatch, None, steps=4)
    assert runner.async_decode._latest_readout_drafts is True
    assert runner.async_decode.latest_readout_has_drafts()


# endregion

# region T4 b. Two calls, and the tip is the finished step


def test_b_a_chained_masked_step_is_two_calls_and_chains_onto_the_finished_one(
    monkeypatch,
):
    model, runner, engine = _run(monkeypatch, _mask_params("allowed"), steps=16)
    controller = runner.async_decode
    # every step the allowed row lives in is a begin then a finish, nothing else
    assert [c[0] for c in model.calls] == ["begin", "finish"] * 16
    begins = model.begin_inputs
    chained = [k for k, b in enumerate(begins) if b["chain_rows"] is not None]
    assert 0 not in chained and len(chained) >= 12
    # step k chains onto the handle of step k-1, the one its finish returned, and
    # the begin sent the chain arguments of a chained step (placeholders only)
    for k in chained:
        assert begins[k]["chain_tip"] == model.handles[k - 1].tip_id
        assert bool(begins[k]["chain_rows"][:3].all())  # the live rows
        assert bool((begins[k]["tokens"][:3] == -1).all())
    assert controller._chained_submissions == len(chained)
    # one counter write per chained masked step
    assert device_mask_stats()["chained"] == len(chained)
    # every mask the model got keeps x on its even ids
    assert model.masks and all(mask is not None for mask in model.masks)
    for step, mask in enumerate(model.masks):
        row = unpack_allowed(mask[engine.step_rows[step].index("x")], SV)
        assert bool((row[:, 1::2] == 0).all()) and bool(row[:, ::2].all())
    # the same streams as the logits run
    off_model, off, off_engine = _run(
        monkeypatch, _mask_params("allowed"), steps=16, mask=False
    )
    assert _streams(engine) == _streams(off_engine)
    assert engine.outputs["x"] == off_engine.outputs["x"]


def test_b_a_masked_step_is_a_chain_tip_only_after_its_finish(monkeypatch):
    model, runner, engine = _masked_runner(monkeypatch)
    engine.add("x", 17, _mask_params("allowed"))
    controller = runner.async_decode
    engine.iterate()  # step 0 begun and finished
    assert controller._last_fused_tip is not None and not controller.mask_pending()
    so = engine.schedule()
    assert runner.execute_model(so) is None  # step 1: begun, pending
    assert controller.mask_pending()
    assert model.calls[-1][0] == "begin"
    runner.sample_tokens(None)
    assert not controller.mask_pending() and model.calls[-1][0] == "finish"
    engine.finish()


# endregion

# region T4 c. The pending guards


def test_c_execute_model_while_a_masked_step_is_pending_raises(monkeypatch):
    model, runner, engine = _masked_runner(monkeypatch)
    engine.add("x", 17, _mask_params("allowed"))
    engine.iterate()
    engine.iterate()
    so = engine.schedule()
    assert runner.execute_model(so) is None
    assert runner.async_decode.mask_pending()
    calls = list(model.calls)
    with pytest.raises(RuntimeError, match="before its mask finish"):
        runner.execute_model(engine.schedule())
    assert model.calls == calls  # nothing was begun
    # the finish is still possible: the guard changed nothing
    runner.sample_tokens(None)
    assert not runner.async_decode.mask_pending()


def test_c_chain_evaluate_has_the_reason_mask_pending(monkeypatch):
    model, runner, engine = _masked_runner(monkeypatch)
    engine.add("x", 17, _mask_params("allowed"))
    engine.iterate()
    engine.iterate()
    controller = runner.async_decode
    so = engine.schedule()
    assert runner.execute_model(so) is None
    assert controller.chain_evaluate(so) == ("mask_pending", None)
    assert controller.chain_blocker(so) == "mask_pending"
    runner.sample_tokens(None)
    assert controller.chain_blocker(so) != "mask_pending"


def test_c_a_structured_row_with_a_hint_that_was_not_handed_over_raises(monkeypatch):
    class Skipping(MaskEngine):
        """A core that builds the bitmask but never calls take_draft_token_ids."""

        skip_from = 6

        def _grammar(self):
            if self.step <= self.skip_from:
                return super()._grammar()
            rows = [
                _scheduler_rows(list(self.outputs.get(r, ())), [-1] * K)
                for r in self._structured_rows()
            ]
            return GrammarOutput(self._structured_rows(), torch.cat(rows).numpy())

    model, runner, engine = _masked_runner(monkeypatch, engine_cls=Skipping)
    engine.add("x", 17, _mask_params("structured"))
    with pytest.raises(RuntimeError, match="were not handed over"):
        for _ in range(12):
            engine.iterate()
    # the guard is the sample-time check, so the step was begun and not finished
    assert model.calls[-1][0] == "begin"


def test_c_the_chain_handoff_is_set_for_structured_rows_and_cleared_otherwise(
    monkeypatch,
):
    model, runner, engine = _masked_runner(monkeypatch)
    engine.add("x", 17, _mask_params("structured"))
    for _ in range(4):
        engine.iterate()
    so = engine.schedule()
    assert runner.execute_model(so) is None
    tip, hints = runner._chain_handoff
    assert hints == {"x": K} and tip == model.begin_inputs[-1]["chain_tip"]
    runner.sample_tokens(engine._grammar())
    assert runner._chain_handoff is None  # taken by the core once
    engine.finish()
    # a build that is not chained over a grammar row clears a stale handoff
    model, runner, engine = _masked_runner(monkeypatch)
    engine.add("g1", 1)
    for _ in range(3):
        engine.iterate()
    runner._chain_handoff = (99, {"g1": 1})
    engine.iterate()
    assert runner._chain_handoff is None
    engine.finish()


# endregion

# region T4 d. Rows that leave in the window


class _LeavingEngine(MaskEngine):
    """The scheduler no longer holds a request that finished or aborted in the
    window: ``get_grammar_bitmask`` skips it, so it is absent from the output."""

    def _structured_rows(self):
        return [r for r in super()._structured_rows() if r not in self.done]

    def _grammar(self):
        # the core calls take_draft_token_ids only while a structured request
        # is still scheduled
        if not self._structured_rows():
            self.handed.append({})
            return None
        return super()._grammar()


def test_d_a_row_that_finishes_in_the_window_is_unconstrained_never_an_error(
    monkeypatch,
):
    model, runner, engine = _masked_runner(monkeypatch, engine_cls=_LeavingEngine)
    engine.add("g1", 1)
    engine.add("x", 17, _mask_params("structured"), stop_after=7)
    engine.add("y", 33, _mask_params("structured"))
    for _ in range(14):
        engine.iterate()
    engine.finish()
    assert "x" in engine.finish_step
    # y is still constrained after x left, and its stream is grammar valid
    history: list[int] = []
    for token in engine.outputs["y"]:
        assert _grammar_allows(history, token)
        history.append(token)
    assert len(engine.outputs["y"]) > 20
    assert [c[0] for c in model.calls] == ["begin", "finish"] * 14


def test_d_when_all_grammar_rows_left_the_step_finishes_with_none(monkeypatch):
    model, runner, engine = _masked_runner(monkeypatch, engine_cls=_LeavingEngine)
    engine.add("g1", 1)
    engine.add("x", 17, _mask_params("structured"), stop_after=7)
    for _ in range(14):
        engine.iterate()
    engine.finish()
    k = engine.finish_step["x"]
    # the step built right after x's finishing output carries x, unconstrained:
    # a finish with no mask at all, and no error
    assert model.masks[k + 1] is None
    assert _chained_steps(model)[-1] == 13


def test_d_an_abort_race_leaves_the_row_unconstrained(monkeypatch):
    model, runner, engine = _masked_runner(monkeypatch)
    engine.add("g1", 1)
    engine.add("x", 17, _mask_params("structured"))
    for _ in range(5):
        engine.iterate()
    taken = []

    def absent():  # the abort removed x from the grammar output
        taken.append(runner.take_draft_token_ids())
        return None

    engine._grammar = absent
    for _ in range(3):
        engine.iterate()
    engine.finish()
    assert taken and taken[-1].req_ids == ["x"]
    assert model.masks[-1] is None and model.masks[-2] is None


# endregion

# region T4 e. A row that leaves, a preempted row


class _PreemptEngine(MaskEngine):
    """Preempts ``victim`` at step ``at``: the scheduler drops it from the batch and
    reports it in ``preempted_req_ids``."""

    victim = "x"
    at = 7

    def schedule(self):
        if self.step == self.at:
            self.live.remove(self.victim)
            so = super().schedule()
            so.preempted_req_ids = {self.victim}
            return so
        return super().schedule()


def test_e_a_preempted_row_drops_its_output_and_the_other_rows_go_on(monkeypatch):
    model, runner, engine = _masked_runner(monkeypatch, engine_cls=_PreemptEngine)
    engine.add("g1", 1)
    engine.add("g2", 9)
    engine.add("x", 17, _mask_params("allowed"))
    for _ in range(14):
        engine.iterate()
    engine.finish()
    # the steps after the preemption have no x row, and no mask row for it either
    assert all("x" not in rows for k, rows in engine.step_rows.items() if k >= 7)
    assert all(m is None for m in model.masks[8:])
    # the chain broke once, at the layout change, and ran again
    assert runner.async_decode._chain_breaks.get("layout", 0) >= 1
    assert len(_chained_steps(model)) >= 10
    # the others equal a run in which nobody was preempted, on the logits path
    off_model, off, off_engine = _run(
        monkeypatch, _mask_params("allowed"), steps=14, mask=False
    )
    n = min(len(engine.outputs["g1"]), len(off_engine.outputs["g1"]))
    assert engine.outputs["g1"][:n] == off_engine.outputs["g1"][:n]


def test_e_a_row_that_finishes_keeps_the_chain_and_the_mask_of_the_others(monkeypatch):
    model2, runner2, engine2 = _masked_runner(monkeypatch)
    engine2.add("g1", 1)
    engine2.add("g2", 9, stop_after=8)
    engine2.add("x", 17, _mask_params("allowed"))
    for _ in range(16):
        engine2.iterate()
    engine2.finish()
    assert "g2" in engine2.finish_step
    assert [c[0] for c in model2.calls] == ["begin", "finish"] * 16
    assert all(
        bool(unpack_allowed(m[1], SV)[:, ::2].any())
        for m in model2.masks
        if m is not None
    )
    assert len(_chained_steps(model2)) >= 12
    assert all(v in EVENS for v in engine2.outputs["x"])


# endregion

# region T4 f. The wait at sample time


def _x_mask_rows(model, engine):
    """``(step number, row x of the mask)`` of every finish that carried a mask."""
    by_step = {}
    finish_index = 0
    for step, name in _step_of_calls(model):
        if name == "finish":
            by_step[step] = model.masks[finish_index]
            finish_index += 1
    out = {}
    for step, mask in by_step.items():
        if mask is not None:
            out[step] = unpack_allowed(mask[engine.step_rows[step].index("x")], SV)
    return out


def test_f_min_tokens_waits_for_the_step_before_and_composes_from_its_commit(
    monkeypatch,
):
    params = _mask_params("min_tokens")  # min_tokens 7, stops 3 5 9
    model, runner, engine = _run(monkeypatch, params, steps=10, hold=True)
    stops = [3, 5, 9]
    masks = _x_mask_rows(model, engine)
    assert masks
    # one wait per masked chained step whose predecessor was unresolved
    assert device_mask_stats()["waits"] >= len(masks) - 1
    for step, rows in masks.items():
        # the commit view: what x committed through the step before this one
        committed = sum(len(engine.published[k].get("x", ())) for k in range(step))
        remaining = 7 - committed
        for column in range(rows.shape[0]):
            banned = column < remaining
            assert all(bool(rows[column, t]) != banned for t in stops), (
                step,
                column,
                committed,
            )
    # once min_tokens is reached the row needs no mask: the steps run one call
    assert max(masks) < 9


def test_f_allowed_ids_and_one_token_bad_words_do_not_wait(monkeypatch):
    for case in ("allowed", "one-token bad word"):
        reset_device_mask_stats()
        model, runner, engine = _run(
            monkeypatch, BLOCKERS[case][0], steps=12, hold=True
        )
        assert device_mask_stats()["waits"] == 0, case
        assert device_mask_stats()["chained"] >= 8, case
        # no step was waited for by the sample-time compose: every handle the
        # reader waited on was waited on by the drain at the end
    reset_device_mask_stats()
    model, runner, engine = _run(
        monkeypatch, BLOCKERS["multi-token bad word"][0], steps=12, hold=True
    )
    assert device_mask_stats()["waits"] > 0  # its history needs the commit


def test_f_a_multi_token_bad_word_reads_the_commit_view_and_the_device_drafts(
    monkeypatch,
):
    """The word [a, b] bans b when the history ends with a: the history of column
    j is the committed output plus the first j drafts of the step before."""
    model, runner, engine = _masked_runner(monkeypatch, hold=True)
    engine.add("g1", 1)
    engine.add("x", 17, _bad([[4, 5]]))
    for _ in range(3):
        engine.iterate()
    tip = max(runner.async_decode._steps_by_tip)
    handle = model.handles[-1]
    assert tip == handle.tip_id
    # make the word match: pretend the drafts of the tip are [4, x, x] and the
    # last committed token of x is 4 (so column 0 bans 5, and column 1 bans 5)
    record_drafts = torch.zeros_like(handle._readout.drafts)
    record_drafts[1] = torch.tensor([4] * K)
    handle._readout = dataclasses.replace(handle._readout, drafts=record_drafts)
    so = engine.schedule()
    assert runner.execute_model(so) is None
    plan = runner._pending_samples[0].keywords["model_input"].spec_mask
    assert plan.deferred and plan.static_rows == (1,) and plan.needs_wait
    runner.sample_tokens(None)
    mask = unpack_allowed(model.masks[-1][1], SV)
    committed = runner.async_decode.commit_view("x", tip)
    for column in range(K + 1):
        history = committed + [4] * column
        banned = bad_word_last_token([4, 5], history) is not None
        assert bool(mask[column, 5]) != banned, column
    engine.finish()


# endregion

# region T4 g. The thinking row, the boundary


class _ThinkEngine(MaskEngine):
    """A structured row inside its reasoning phase: the scheduler holds all ones
    until the end token is committed or drafted, then the grammar applies."""

    end_id = 11

    def _grammar(self):
        runner = self.runner
        handed = runner.take_draft_token_ids()
        by_request = dict(zip(handed.req_ids, handed.draft_token_ids)) if handed else {}
        self.handed.append({r: list(d) for r, d in by_request.items()})
        structured = self._structured_rows()
        if not structured:
            return None
        rows = []
        for req_id in structured:
            history = list(self.outputs.get(req_id, ()))
            drafts = by_request.get(req_id, [])
            rows.append(self._think_rows(history, drafts))
        return GrammarOutput(structured, torch.cat(rows).numpy())

    def _think_rows(self, history, drafts):
        """One row per position 0..K: all ones up to and including the first end
        token (the vLLM rule), the grammar of what followed after it."""
        scheduled = list(drafts) + [-1] * (K - len(drafts))
        out = []
        ended = self.end_id in history
        after = history[history.index(self.end_id) + 1 :] if ended else []
        ok = True
        for position in range(K + 1):
            if ended and ok:
                out.append(
                    torch.tensor(
                        [_grammar_allows(after, t) for t in range(SV)],
                    )
                )
            else:
                out.append(torch.ones(SV))
            if position == K:
                break
            token = scheduled[position]
            if token == -1:
                ok = False
                continue
            if ended:
                if not _grammar_allows(after, token):
                    ok = False
                    continue
                after = after + [token]
            elif token == self.end_id:
                ended, after = True, []
        return _pack(torch.stack(out).bool())


def test_g_a_thinking_row_chains_with_all_ones_rows_and_the_mask_is_none(monkeypatch):
    seen = []
    model, runner, engine = _masked_runner(monkeypatch, engine_cls=_ThinkEngine)
    engine.end_id = 10_000  # never reached: the row stays in its reasoning phase
    real = runner._spec_step_mask

    def spy(model_input, *a, **k):
        seen.append(model_input.spec_mask)
        return real(model_input, *a, **k)

    runner._spec_step_mask = spy
    engine.add("g1", 1)
    engine.add("x", 17, _mask_params("structured"))
    for _ in range(12):
        engine.iterate()
    engine.finish()
    chained = [plan for plan in seen if plan.deferred]
    assert len(chained) >= 8
    # every structured row is a grammar row; no reasoning rows in a chained build
    assert all(
        plan.grammar_rows == (1,) and plan.reasoning_rows == () for plan in chained
    )
    # all-ones scheduler rows: the finish gets no mask and the step is a normal one
    assert all(m is None for m in model.masks)
    assert len(_chained_steps(model)) >= 8
    # the commit is the plain greedy commit: a run with no grammar row at all
    off_model, off, off_engine = _run(monkeypatch, _p(), steps=12, mask=False)
    n = min(len(engine.outputs["x"]), len(off_engine.outputs["x"]))
    assert n >= 12


@pytest.mark.parametrize("end_at", [0, 1, 3, 6])
def test_g_a_reasoning_end_inside_a_chained_step_is_masked_by_the_scheduler_rows(
    monkeypatch, end_at
):
    """The end id turns up in the device drafts: the build could not know, the
    scheduler's rows (from the handed-over drafts) do. The masked chained run
    commits what the ``logits`` run commits, and what follows the end id is
    grammar valid."""
    # the end id: a token the greedy stream of x commits, found on a control run
    control_model, control, control_engine = _run(
        monkeypatch, _mask_params("structured"), steps=14, mask=False
    )
    plain_model, plain, plain_engine = _run(monkeypatch, _p(), steps=14, mask=False)
    stream = plain_engine.outputs["x"]
    end_id = stream[end_at]

    def run(mask):
        model, runner, engine = _masked_runner(
            monkeypatch, mask=mask, engine_cls=_ThinkEngine
        )
        engine.end_id = end_id
        engine.add("g1", 1)
        engine.add("g2", 9)
        engine.add("x", 17, _mask_params("structured"))
        for _ in range(14):
            engine.iterate()
        engine.finish()
        return model, runner, engine

    on_model, on, on_engine = run(True)
    off_model, off, off_engine = run(False)
    n = min(len(on_engine.outputs["x"]), len(off_engine.outputs["x"]))
    assert n >= 14
    assert on_engine.outputs["x"][:n] == off_engine.outputs["x"][:n]
    out = on_engine.outputs["x"]
    after = out[out.index(end_id) + 1 :] if end_id in out else []
    history: list[int] = []
    for token in after:
        assert _grammar_allows(history, token)
        history.append(token)
    assert len(_chained_steps(on_model)) >= 8


# endregion

# region T4 resolve_deferred_plan and the sample-time empty row


def _state(params, output=()):
    return SimpleNamespace(sampling_params=params, output_token_ids=list(output))


def _unit_params(**kw):
    base = {
        "allowed_token_ids": None,
        "bad_words_token_ids": [],
        "min_tokens": 0,
        "all_stop_token_ids": set(),
        "structured_outputs": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_deferred_plan_has_every_structured_row_as_a_grammar_row():
    requests = {
        "a": _state(_unit_params()),
        "s": _state(_unit_params(structured_outputs=object())),
        "m": _state(_unit_params(min_tokens=4, all_stop_token_ids={3})),
        "w": _state(_unit_params(bad_words_token_ids=[[1, 2]])),
        "k": _state(_unit_params(allowed_token_ids=[1, 2])),
    }
    rows = list(requests)
    plan = deferred_mask_plan(requests, rows, dict.fromkeys(rows, 2), tip=9)
    assert plan.deferred and plan.static is None and plan.tip == 9
    assert plan.grammar_rows == (1,) and plan.reasoning_rows == ()
    assert plan.static_rows == (2, 3, 4) and plan.needs_wait
    assert plan.hints == (2, 2, 2, 2, 2)
    # allowed ids and one-token bad words alone need no wait
    only = {"k": requests["k"], "o": _state(_unit_params(bad_words_token_ids=[[7]]))}
    plan = deferred_mask_plan(only, list(only), {"k": 1, "o": 1}, tip=1)
    assert plan.static_rows == (0, 1) and not plan.needs_wait
    # no control and no grammar row: no plan, the step stays one call
    assert deferred_mask_plan({"a": requests["a"]}, ["a"], {"a": 1}, tip=1) is None


def test_an_empty_composed_row_at_sample_time_raises_naming_the_request():
    """The build refuses the chain when the worst case can be empty
    (``mask_empty_risk``); if a row is empty anyway the step cannot go to logits:
    ``step_mask`` raises, naming the request."""
    params = _unit_params(allowed_token_ids=[5], bad_words_token_ids=[[4, 5]])
    requests = {"r": _state(params, output=[1])}
    plan = deferred_mask_plan(requests, ["r"], {"r": 1}, tip=3)
    assert plan.needs_wait
    # the step before committed a 4: the commit view ends with the word's prefix
    resolved = resolve_deferred_plan(
        plan,
        requests,
        ["r"],
        __import__("vllm_tt_plugin.spec_masks", fromlist=["x"]).SpecMaskCache(),
        view=lambda req_id: [1, 4],
        drafts=(torch.tensor([[9, 9]], dtype=torch.int32), {"r": 0}.__getitem__),
        width=3,
        vocab_size=64,
    )
    with pytest.raises(RuntimeError, match=r"allows no token.*'r'"):
        step_mask(resolved, None, ["r"], rows=1, width=3, vocab_size=64)
    # the same row with the history that does not end with the prefix is fine
    fine = resolve_deferred_plan(
        plan,
        requests,
        ["r"],
        __import__("vllm_tt_plugin.spec_masks", fromlist=["x"]).SpecMaskCache(),
        view=lambda req_id: [1, 2],
        drafts=(torch.tensor([[9, 9]], dtype=torch.int32), {"r": 0}.__getitem__),
        width=3,
        vocab_size=64,
    )
    mask, constrained = step_mask(fine, None, ["r"], rows=1, width=3, vocab_size=64)
    assert constrained == 1


def test_a_plan_that_needs_the_step_before_refuses_to_resolve_without_it():
    requests = {"r": _state(_unit_params(min_tokens=4, all_stop_token_ids={3}))}
    plan = deferred_mask_plan(requests, ["r"], {"r": 1}, tip=3)
    with pytest.raises(RuntimeError, match="commit view"):
        resolve_deferred_plan(
            plan, requests, ["r"], None, view=None, drafts=None, width=3, vocab_size=64
        )


# endregion
