# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative decoding through the asynchronous execution path.

Asynchronous scheduling splits a step in two. ``execute_model`` submits and
returns nothing; the engine collects the output later, from a thread that is
not the engine thread; and the runner applies it to request state at the top of
the following step. A speculative step cannot be finished where an ordinary one
is, because its output is a candidate block whose committed length the accept
walk decides, and the walk's result has to reach the request's token history
and the next proposal, both of which only the engine thread may touch.

So the speculative path splits the same way: ``walk_spec_acceptance`` runs where
the readback completes and touches no host state, and ``commit_spec_acceptance``
runs on the engine thread when the next step drains this one.

These tests drive that through the real entry points, ``execute_model`` and
``sample_tokens``, against a target whose choice does not depend on what was
drafted. Completion is held open deliberately: ``DeferredVerifyTarget``
implements the ``read_decode_output`` hook, hands back an output buffer whose
contents arrive only when the test releases them, and the patched
``ttnn.event_synchronize`` blocks until then. That is what makes an assertion
about ordering an assertion rather than a race: every "before the result lands"
below is a real point in the step, not a sleep.
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin import async_decode as async_decode_module
from vllm_tt_plugin.async_decode import (
    AsyncTTSpecDecodeOutput,
    CompletedSpecDecodeStep,
    TTAsyncDecodeController,
)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner

from .deterministic_target import (
    TARGET_VOCAB_SIZE,
    DeterministicTarget,
    continuation,
)

# The output-equality suite's own draft cases and unspeculated reference arm.
# Imported rather than restated so that both execution paths answer to one
# standard, and so that a case added there is run here too.
from .test_spec_lossless import _DRAFT_CASES, COMPARE_TOKENS, _run_plain

BLOCK_SIZE = 16
MAX_MODEL_LEN = 256
MAX_NUM_REQS = 4
PROMPT_LEN = 4
DRAFT_LEN = 3


class DeferredVerifyTarget(DeterministicTarget):
    """A target whose verify output becomes readable only when released.

    The plugin's asynchronous read path is what this reaches: ``submit_decode``
    calls ``read_decode_output`` when a model implements it, and
    ``finalize_decode`` waits on the events it returned. Returning a buffer
    that is filled at release time, rather than the answer itself, is what
    makes the deferral observable: a test can assert that nothing has been
    committed while the step is outstanding.

    The buffer is populated before the event is set, in that order, because the
    completion reads it as soon as the wait returns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[tuple[torch.Tensor, torch.Tensor, threading.Event]] = []
        self.reads = 0
        self.propose_calls: list[object] = []
        self.verify_hidden: object | None = None

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        """The rule's own answer, with a fresh hidden handle on each verify.

        The handle stands for the target state a device drafter reads. It is a
        distinct object per verify so that a runner holding the wrong step's
        handle is a failure rather than a coincidence.
        """
        out = super().decode_forward(tokens, start_pos, spec_mode=spec_mode, **kwargs)
        if spec_mode is None:
            return out
        self.verify_hidden = SimpleNamespace(verify=len(self.verify_calls))
        return replace(out, hidden=self.verify_hidden)

    def read_decode_output(self, tt_out, async_read: bool = False):
        self.reads += 1
        buffer = torch.full_like(tt_out, -7)
        event = threading.Event()
        self.pending.append((buffer, tt_out.clone(), event))
        return buffer, [event]

    def propose_draft_tokens(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ):
        """Draft the rule's own continuation, and check the handle.

        The handle is what this asserts: on the asynchronous path the verify
        that produced it completed on another thread and its result sat in a
        queue before this call, so a runner that dropped or replaced it would
        show up here rather than in a wrong token.
        """
        from vllm_tt_plugin.spec_decode import DraftOutput

        if hidden is not self.verify_hidden:
            raise AssertionError(
                "propose_draft_tokens received a hidden handle that is not the "
                "one this model's verify returned"
            )
        self.propose_calls.append(hidden)
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        token = committed_tokens.to(torch.int64).gather(1, index)
        position = committed_positions.to(torch.int64).gather(1, index)
        columns = []
        for _ in range(num_drafts):
            token = self._choice(token, position).to(torch.int64)
            position = position + 1
            columns.append(token)
        return DraftOutput(draft_token_ids=torch.cat(columns, dim=1).to(torch.int32))

    def release(self) -> None:
        """Make the oldest outstanding verify readable."""
        buffer, answer, event = self.pending.pop(0)
        buffer.copy_(answer)
        event.set()

    @property
    def outstanding(self) -> int:
        return len(self.pending)


@pytest.fixture(autouse=True)
def wait_for_released_reads(monkeypatch):
    """``ttnn.event_synchronize`` waits for the test's release.

    The synthetic events stay inside this process: the real runtime never sees
    one, which is why this patches the plugin's own reference rather than
    handing a Python event to a device.
    """

    def event_synchronize(event):
        if isinstance(event, threading.Event):
            assert event.wait(timeout=30), "a held verify was never released"
            return
        raise AssertionError(f"unexpected event {event!r}")

    monkeypatch.setattr(
        async_decode_module.ttnn, "event_synchronize", event_synchronize
    )


def _runner(model: DeferredVerifyTarget) -> SimpleNamespace:
    """An async-scheduling runner fake whose methods are the real ones."""
    from collections import deque

    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=TARGET_VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    released: list[int] = []
    runner = SimpleNamespace(
        model=model,
        input_batch=batch,
        requests={},
        encoder_cache={},
        kv_caches=object(),
        trace_mode="decode_only",
        request_specific_rope=False,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _num_speculative_tokens=DRAFT_LEN,
        _spec_method=None,
        _spec_supports_narrow_decode=False,
        _spec_drafts_from_model=False,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _ngram_proposer=None,
        _req_state_slot={},
        released_slots=released,
        # The asynchronous half of the runner's state.
        async_decode_scheduling=True,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        _steady_decode_lock=threading.Lock(),
        _pending_async_steps=deque(),
        _pending_async_overlap_ok=deque(),
        _completed_decode_steps=deque(),
        _pending_samples=deque(),
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
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        # ``finalize_decode`` requires the per-row log-probability flags the
        # real runner builds here, and the asynchronous path goes through it.
        _sampling_params_for_padded_decode=(
            lambda params, req_indices, n: SimpleNamespace(
                enable_log_probs=torch.zeros(n, dtype=torch.bool)
            )
        ),
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
        _spec_row_state=TTModelRunner._spec_row_state,
        _spec_candidate_block=TTModelRunner._spec_candidate_block,
        _committed_positions=TTModelRunner._committed_positions,
    )
    runner.model.release_request = released.append
    for name, member in vars(TTModelRunner).items():
        if hasattr(runner, name):
            continue
        if isinstance(member, staticmethod):
            setattr(runner, name, member.__func__)
        elif isinstance(member, classmethod):
            setattr(runner, name, member.__func__.__get__(TTModelRunner))
        elif inspect.isfunction(member):
            setattr(runner, name, member.__get__(runner))
    runner.async_decode = TTAsyncDecodeController(runner)
    return runner


def _new_request(req_id: str, first_token: int) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=PROMPT_LEN,
        lora_request=None,
    )


def _computed_tokens(runner, req_id: int | str) -> int:
    """What a scheduler reports for a decoding request: all of it.

    Taken from the row rather than from ``CachedRequestState``, because the
    commit advances the row's length and the scheduler is the thing that would
    have advanced the request's own counter in production.
    """
    row = runner.input_batch.req_id_to_index.get(req_id)
    if row is None:
        return int(runner.requests[req_id].num_computed_tokens)
    return int(runner.input_batch.num_tokens[row])


def _scheduler_output(
    runner,
    *,
    new=(),
    decoding=(),
    finished=(),
    preempted=(),
    drafts=None,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [_new_request(req_id, first) for req_id, first in new]
    scheduled = [req_id for req_id, _ in new] + list(decoding)
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=list(decoding),
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in decoding],
        all_token_ids={},
        new_block_ids=[None for _ in decoding],
        num_computed_tokens=[_computed_tokens(runner, req_id) for req_id in decoding],
        num_output_tokens=[
            len(runner.requests[req_id].output_token_ids) for req_id in decoding
        ],
    )
    output.num_scheduled_tokens = {req_id: 1 for req_id in scheduled}
    output.total_num_scheduled_tokens = len(scheduled)
    output.finished_req_ids = set(finished)
    output.preempted_req_ids = set(preempted)
    output.scheduled_spec_decode_tokens = dict(drafts or {})
    return output


def _admit(runner, *specs) -> None:
    runner._update_states(_scheduler_output(runner, new=specs))


def _tail(runner, req_id) -> tuple[int, int]:
    row = runner.input_batch.req_id_to_index[req_id]
    length = int(runner.input_batch.num_tokens[row])
    return int(runner.input_batch.token_ids_cpu[row, length - 1]), length - 1


def _submit_step(runner, *req_ids, drafts=None):
    """Run one asynchronous step up to the deferred output, not past it."""
    scheduler_output = _scheduler_output(runner, decoding=req_ids, drafts=drafts)
    for req_id in req_ids:
        row = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    assert runner.execute_model(scheduler_output) is None, (
        "an asynchronous step defers its output, so execute_model returns None"
    )
    wrapper = runner.sample_tokens(None)
    assert isinstance(wrapper, AsyncTTSpecDecodeOutput), (
        f"a speculative async step must defer through the speculative path, "
        f"got {type(wrapper).__name__}"
    )
    return wrapper


def _drain(runner, *req_ids, finished=(), preempted=()):
    """The next step's start, where the engine thread applies what completed.

    The two calls ``build_model_input`` makes before it builds anything:
    lifecycle first, so a request that finished or was preempted is out of the
    batch before its late result is considered, then the apply. The build
    itself is left to the next ``_submit_step``, which sets the computed-token
    counts a real scheduler output would carry.
    """
    scheduler_output = _scheduler_output(
        runner, decoding=req_ids, finished=finished, preempted=preempted
    )
    runner._update_states(scheduler_output)
    runner.async_decode.apply_ready_completed_decode_steps(
        suppress_output_req_ids=runner.async_decode.suppressed_output_req_ids(
            scheduler_output
        ),
        forced_reset_discard_counts=None,
    )


def _accept_everything(runner, req_id):
    token, position = _tail(runner, req_id)
    return continuation(token, position, DRAFT_LEN)


# region Deferred completion


def test_a_speculative_step_defers_its_output_and_commits_nothing_yet():
    """Nothing reaches request state while the verify is outstanding.

    The whole point of the split: the accept walk cannot run until the readback
    completes, and the commit cannot run until the engine thread drains it. A
    commit on the readback thread would race the next step's input build.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    before = int(runner.input_batch.num_tokens[0])

    wrapper = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})

    assert model.outstanding == 1, "the verify was read back immediately"
    assert int(runner.input_batch.num_tokens[0]) == before
    assert runner._req_accepted_counts == {}
    assert not wrapper.is_resolved()

    model.release()
    output = wrapper.get_output()

    # Resolved, published, and still not applied.
    assert wrapper.is_resolved()
    assert len(output.sampled_token_ids[0]) == DRAFT_LEN + 1
    assert int(runner.input_batch.num_tokens[0]) == before
    assert runner._req_accepted_counts == {}

    _drain(runner, "a")

    assert int(runner.input_batch.num_tokens[0]) == before + DRAFT_LEN + 1
    assert runner._req_accepted_counts["a"] == DRAFT_LEN + 1


def test_the_published_output_matches_what_the_commit_writes():
    """One step, two consumers, one answer.

    The engine gets the published prefix and the request's own history gets the
    committed one. They are produced at different times on different threads,
    so a test that checked only one of them would not notice them diverging.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    expected = continuation(*_tail(runner, "a"), DRAFT_LEN + 1)

    wrapper = _submit_step(runner, "a", drafts={"a": expected[:DRAFT_LEN]})
    model.release()
    output = wrapper.get_output()
    _drain(runner, "a")

    assert output.sampled_token_ids[0] == expected
    assert runner.requests["a"].output_token_ids == expected


@pytest.mark.parametrize("bend", ["all correct", "wrong at the first", "all wrong"])
def test_acceptance_through_the_async_path_matches_the_target(bend):
    """Zero, partial and full acceptance, deferred.

    The accept walk is the same code on both paths, but what it is handed is
    not: on this one the block and the verify's answer have crossed a thread
    boundary and a queue first.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    token, position = _tail(runner, "a")
    truth = continuation(token, position, DRAFT_LEN + 1)
    offered = {
        "all correct": truth[:DRAFT_LEN],
        "wrong at the first": [truth[0] + 1] + truth[1:DRAFT_LEN],
        "all wrong": [value + 1 for value in truth[:DRAFT_LEN]],
    }[bend]

    wrapper = _submit_step(runner, "a", drafts={"a": offered})
    model.release()
    output = wrapper.get_output()
    _drain(runner, "a")

    committed = output.sampled_token_ids[0]
    if bend == "all correct":
        assert committed == truth
    else:
        # A rejection at the first draft commits the target's own choice there
        # and nothing more.
        assert committed == [truth[0]]
    assert runner.requests["a"].output_token_ids == committed
    assert runner._req_accepted_counts["a"] == len(committed)


def test_two_steps_in_a_row_each_commit_their_own_prefix():
    """The second step's block is built from the first step's commit.

    A speculative step is not overlap-safe for exactly this reason, and this is
    the test that the ordering holds: the second block's first column has to be
    the token the first step committed last.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    expected = continuation(*_tail(runner, "a"), 2 * (DRAFT_LEN + 1))

    first = _submit_step(runner, "a", drafts={"a": expected[:DRAFT_LEN]})
    model.release()
    first_output = first.get_output()
    _drain(runner, "a")

    second = _submit_step(
        runner, "a", drafts={"a": expected[DRAFT_LEN + 1 : 2 * DRAFT_LEN + 1]}
    )
    model.release()
    second_output = second.get_output()
    _drain(runner, "a")

    assert first_output.sampled_token_ids[0] + second_output.sampled_token_ids[0] == (
        expected
    )
    assert runner.requests["a"].output_token_ids == expected


def test_a_speculative_step_is_not_overlap_safe():
    """The serialization the first milestone relies on, asserted.

    The next candidate block is built from this step's committed tokens, so the
    build has to wait for this commit. The runner expresses that by refusing to
    treat the step as overlap-safe, which makes the next build drain first.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))

    _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})

    assert list(runner._pending_async_overlap_ok) == [False]
    assert runner.async_decode.must_drain_pending_async_steps(
        steady_decode_candidate=True
    ), "a pending speculative step must force the next build to drain"


def test_resolving_the_same_step_twice_reads_the_device_once():
    """Two threads race for every step; the readback must happen once.

    vLLM's executor resolves the deferred output on its own thread while the
    runner's drain can reach the same step from the engine thread. A second
    readback of one submission corrupts it, and a second accept walk would
    commit the same tokens twice.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    wrapper = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()

    outputs: list[object] = []
    errors: list[BaseException] = []

    def resolve():
        try:
            outputs.append(wrapper.get_output())
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=resolve) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert model.reads == 1, "the verify was read back more than once"
    assert len({id(output) for output in outputs}) == 1, (
        "each caller has to receive the same published output"
    )
    assert len(runner._completed_decode_steps) == 1, (
        "one step must enqueue one commit, or the tokens commit twice"
    )


# endregion Deferred completion

# region Lifecycle across the deferral


def test_a_request_that_finished_before_its_result_lands_is_not_revived():
    """A late result must not write tokens for a request that is gone.

    The engine can report a request finished on the output that finished it
    while a later speculative step for the same request is still outstanding.
    Committing that one would write to a row the request no longer owns, and
    proposing from it would draft a continuation of a finished response.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11), ("b", 211))
    wrapper = _submit_step(
        runner,
        "a",
        "b",
        drafts={req: _accept_everything(runner, req) for req in ("a", "b")},
    )
    model.release()
    wrapper.get_output()

    # "a" finishes before the next step drains the completed one.
    _drain(runner, "b", finished=["a"])

    assert "a" not in runner.requests
    assert "a" not in runner._req_accepted_counts
    assert "a" not in runner._proposed_draft_token_ids
    # And "b", which is still live, committed its own prefix.
    assert len(runner.requests["b"].output_token_ids) == DRAFT_LEN + 1


def test_a_preempted_request_does_not_take_its_late_result():
    """Preemption frees the blocks the outstanding verify was computed against.

    A resume re-prefills, so the accepted count and any pending proposal from
    before the preemption name a candidate state the model has overwritten.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11), ("b", 211))
    runner._req_state_slot["a"] = 0
    wrapper = _submit_step(
        runner,
        "a",
        "b",
        drafts={req: _accept_everything(runner, req) for req in ("a", "b")},
    )
    model.release()
    wrapper.get_output()

    _drain(runner, "b", preempted=["a"])

    assert "a" in runner.requests, "a preempted request keeps its cached state"
    assert "a" not in runner.input_batch.req_id_to_index
    assert "a" not in runner._req_state_slot
    assert 0 in runner.released_slots


def test_a_row_reused_after_the_deferral_starts_from_its_own_prompt():
    """A new request taking a freed row inherits nothing from the late result."""
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    wrapper = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    wrapper.get_output()
    _drain(runner, finished=["a"])

    _admit(runner, ("z", 151))
    assert runner.input_batch.req_id_to_index["z"] == 0
    assert "z" not in runner._req_accepted_counts

    expected = continuation(*_tail(runner, "z"), DRAFT_LEN + 1)
    second = _submit_step(runner, "z", drafts={"z": expected[:DRAFT_LEN]})
    model.release()
    output = second.get_output()
    _drain(runner, "z")

    assert output.sampled_token_ids[0] == expected
    assert runner.requests["z"].output_token_ids == expected


def test_the_hidden_handle_survives_the_deferral():
    """The drafter is handed the handle its own verify returned.

    The handle crosses a thread boundary and a queue on this path, and the
    proposal that consumes it runs long after the verify that produced it.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    _admit(runner, ("a", 11))

    wrapper = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    wrapper.get_output()
    handle_at_completion = runner._completed_decode_steps[0].spec_hidden
    _drain(runner, "a")

    assert handle_at_completion is not None
    assert handle_at_completion is model.verify_hidden


def test_a_completed_speculative_step_enqueues_a_speculative_commit():
    """The queue carries the speculative record, not an ordinary one.

    Both kinds share one queue so the engine applies them in submission order.
    A speculative record applied as an ordinary one would put a candidate block
    through the single-width commit.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    wrapper = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    wrapper.get_output()

    queued = list(runner._completed_decode_steps)
    assert len(queued) == 1
    assert isinstance(queued[0], CompletedSpecDecodeStep)
    assert queued[0].runner_output is not None


# endregion Lifecycle across the deferral


# region Output equality through the async path


@pytest.mark.parametrize("case", sorted(_DRAFT_CASES))
def test_the_async_path_emits_exactly_what_ordinary_decoding_would(case):
    """The deferred path's whole output, against unspeculated decoding.

    This is ``test_spec_lossless.test_speculation_emits_exactly_what_ordinary_
    decoding_would`` driven through ``execute_model``, ``sample_tokens`` and a
    held completion instead of the synchronous tail. The draft cases and the
    reference arm are that suite's own, imported rather than restated, so the
    two paths are held to one standard: whatever is drafted, the committed
    sequence is the sequence ordinary decoding would have produced.

    The instrumentation is in the loop and in the counters after it.
    ``DeferredVerifyTarget.reads`` counts calls to ``read_decode_output``,
    which only the deferred path makes, so a step that had quietly finished
    inside ``execute_model`` would leave it short. ``plain_calls`` staying at
    zero says no step fell back to an ordinary decode. And the request's own
    history is checked while each step is outstanding, which is what makes
    "deferred" a claim about this run rather than about the code.
    """
    bend = _DRAFT_CASES[case]
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))

    emitted: list[int] = []
    steps = 0
    while len(emitted) < COMPARE_TOKENS:
        token, position = _tail(runner, "a")
        truth = continuation(token, position, DRAFT_LEN)
        wrapper = _submit_step(runner, "a", drafts={"a": bend(truth)})
        assert runner.requests["a"].output_token_ids == emitted, (
            "the submission committed a token: nothing was deferred"
        )
        model.release()
        output = wrapper.get_output()
        assert runner.requests["a"].output_token_ids == emitted, (
            "completion mutated request state; the commit belongs to the "
            "engine thread, at the next step's drain"
        )
        _drain(runner, "a")
        emitted.extend(output.sampled_token_ids[0])
        steps += 1

    plain = _run_plain(COMPARE_TOKENS)

    assert emitted[:COMPARE_TOKENS] == plain[:COMPARE_TOKENS]
    # Against a third, independent expectation as well, so a fault shared by
    # both runners cannot pass: both arms started from the same prompt tail.
    assert plain[:COMPARE_TOKENS] == continuation(
        11 + PROMPT_LEN - 1, PROMPT_LEN - 1, COMPARE_TOKENS
    )
    assert model.reads == steps
    assert model.plain_calls == 0
    if case == "all correct":
        # Sanity on the test itself: a correct draft set must be accepted, or
        # none of the cases above prove anything about acceptance.
        assert steps < COMPARE_TOKENS


# endregion Output equality through the async path
