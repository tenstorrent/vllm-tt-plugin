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
import io
import logging
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
from vllm_tt_plugin.scheduler import (
    get_tt_forced_reset_discard_counts,
    set_tt_forced_reset_discard_counts,
)
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

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
        self.proposal_positions: list[torch.Tensor] = []
        self.verify_hidden: object | None = None
        self.wait_failure: BaseException | None = None
        self.adaptive = False
        self.device_sampled_calls = 0
        self._propose_failure: BaseException | None = None

    def fail_the_readback(self, error: BaseException) -> None:
        """Make the wait for this step's completion raise.

        Where the deferral actually is: ``read_decode_output`` runs when the
        step is submitted, and what happens later is the wait on the events it
        returned. A failure raised there is a readback failure; one raised in
        the hook would be a submission failure.
        """
        self.wait_failure = error

    def fail_next_propose(self, error: BaseException) -> None:
        """Make the next proposal raise, the way a device error would."""
        self._propose_failure = error

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        """The rule's own answer, with a fresh hidden handle on each verify.

        The handle stands for the target state a device drafter reads. It is a
        distinct object per verify so that a runner holding the wrong step's
        handle is a failure rather than a coincidence.
        """
        if spec_mode is None and kwargs.get("sampling_params") is not None:
            # Device sampling, which is what the plugin requires before it
            # overlaps a decode at all: the model returns the chosen ids, not
            # logits for a host sampler to argmax.
            self.device_sampled_calls += 1
            self.plain_calls += 1
            rows = int(tokens.shape[0])
            positions = start_pos.reshape(rows, -1)[:, :1]
            choice = self._choice(tokens.reshape(rows, -1)[:, :1], positions)
            return choice.reshape(rows).to(torch.int32)
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

    def offer_drafts_only_when_solo(self) -> None:
        """Draft for a lone request and for nobody else.

        Ashai's workload in miniature: speculation is worth it for a solo
        request and not for a batch, so the model offers ``K`` drafts when one
        request is live and none when more are. Live requests are counted from
        the committed positions, because the rows are padded to the wire width
        and a padding row is not a request.

        This mode also stops requiring a hidden handle, and that is not a
        detail. A step with nothing to verify returns no ``VerifyOutput`` and
        so produces no handle, so a drafter that is fed its target hidden state
        through the runner cannot be asked to draft after one. A model wanting
        drafts on both kinds of step therefore drafts from the committed block
        alone, which is all this one's arithmetic needs.
        """
        self.adaptive = True

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

        if self._propose_failure is not None:
            error, self._propose_failure = self._propose_failure, None
            raise error
        if self.adaptive and hidden is None:
            # A step that verified nothing, in the mode that requires no
            # hidden feed. Nothing to check, and nothing missing.
            pass
        elif hidden is not self.verify_hidden:
            raise AssertionError(
                "propose_draft_tokens received a hidden handle that is not the "
                "one this model's verify returned"
            )
        self.propose_calls.append(hidden)
        self.proposal_positions.append(committed_positions.clone())
        rows = int(committed_tokens.shape[0])
        offered = None
        if self.adaptive:
            live = int((committed_positions[:, 0] >= 0).sum())
            offered = torch.full(
                (rows,), num_drafts if live == 1 else 0, dtype=torch.int32
            )
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        token = committed_tokens.to(torch.int64).gather(1, index)
        position = committed_positions.to(torch.int64).gather(1, index)
        columns = []
        for _ in range(num_drafts):
            token = self._choice(token, position).to(torch.int64)
            position = position + 1
            columns.append(token)
        return DraftOutput(
            draft_token_ids=torch.cat(columns, dim=1).to(torch.int32),
            num_valid=offered,
        )

    def release(self) -> None:
        """Make the oldest outstanding verify readable."""
        buffer, answer, event = self.pending.pop(0)
        buffer.copy_(answer)
        if self.wait_failure is not None:
            event.tt_test_failure = self.wait_failure
            self.wait_failure = None
        event.set()

    @property
    def outstanding(self) -> int:
        return len(self.pending)


class ResidentDeferredTarget(DeferredVerifyTarget):
    """A deferred target implementing resident decode input updates."""

    decode_input_update_contract = 1
    model_capabilities = {
        "supports_async_decode": True,
        "supports_spec_decode": True,
        "spec_requirements": ["device_propose"],
    }

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        if spec_mode is not None:
            return super().decode_forward(tokens, start_pos, spec_mode, **kwargs)
        if kwargs["reload_inputs"]:
            self.resident_tokens = tokens.clone()
            self.resident_positions = start_pos.clone()
        self.executed_positions = getattr(self, "executed_positions", [])
        self.executed_positions.append(self.resident_positions.clone())
        answer = super().decode_forward(
            self.resident_tokens, self.resident_positions, **kwargs
        )
        self.resident_tokens = answer.view(-1, 1)
        self.resident_positions = torch.where(
            self.resident_positions >= 0,
            self.resident_positions + 1,
            self.resident_positions,
        )
        return answer


@pytest.fixture(autouse=True)
def wait_for_released_reads(monkeypatch):
    """``ttnn.event_synchronize`` waits for the test's release.

    The synthetic events stay inside this process: the real runtime never sees
    one, which is why this patches the plugin's own reference rather than
    handing a Python event to a device.
    """

    waits: list[threading.Event] = []

    def event_synchronize(event):
        if isinstance(event, threading.Event):
            assert event.wait(timeout=30), "a held verify was never released"
            waits.append(event)
            failure = getattr(event, "tt_test_failure", None)
            if failure is not None:
                raise failure
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
        _is_adaptive_block_output=False,
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
            is_multimodal_model=False,
            max_model_len=MAX_MODEL_LEN,
            # Read by the steady-decode predicate: a launch carrying logits
            # processors samples on the host whatever else is true of it.
            logits_processors=None,
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
    resumed=(),
    drafts=None,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [_new_request(req_id, first) for req_id, first in new]
    scheduled = [req_id for req_id, _ in new] + list(decoding)
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=list(decoding),
        resumed_req_ids=set(resumed),
        new_token_ids=[[] for _ in decoding],
        all_token_ids={},
        # A resumed request carries fresh blocks, which
        # ``apply_cached_req_state_update`` requires of a resume: the scheduler
        # freed the request's blocks when it preempted it.
        new_block_ids=[
            ([1],) if req_id in set(resumed) else None for req_id in decoding
        ],
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


def _drain(
    runner,
    *req_ids,
    new=(),
    finished=(),
    preempted=(),
    resumed=(),
    forced_reset=None,
):
    """The next step's start, where the engine thread applies what completed.

    The two calls ``build_model_input`` makes before it builds anything:
    lifecycle first, so a request that finished or was preempted is out of the
    batch before its late result is considered, then the apply. The build
    itself is left to the next ``_submit_step``, which sets the computed-token
    counts a real scheduler output would carry.
    """
    scheduler_output = _scheduler_output(
        runner,
        new=new,
        decoding=req_ids,
        finished=finished,
        preempted=preempted,
        resumed=resumed,
    )
    if forced_reset:
        set_tt_forced_reset_discard_counts(scheduler_output, forced_reset)
    runner._update_states(scheduler_output)
    runner.async_decode.apply_ready_completed_decode_steps(
        suppress_output_req_ids=runner.async_decode.suppressed_output_req_ids(
            scheduler_output
        ),
        forced_reset_discard_counts=get_tt_forced_reset_discard_counts(
            scheduler_output
        ),
    )


def _baseline_runner(model: DeferredVerifyTarget) -> SimpleNamespace:
    """A runner that can overlap: device sampling, and narrow steps allowed.

    Two settings away from ``_runner``, and both are what the plugin already
    requires of any overlapped decode. It samples on device, because
    ``can_use_steady_decode_fast_path`` refuses a host-sampled step whatever
    else is true of it. And the model declares that it serves its own decode
    call inside a speculating launch, without which every step stays a verify.
    """
    runner = _runner(model)
    runner._spec_supports_narrow_decode = True
    runner._spec_drafts_from_model = True
    runner._spec_method = "custom_class"
    runner.check_perform_device_sampling = lambda **_: True
    # The real one, because device sampling reads its fields as a dataclass.
    del runner._sampling_params_for_padded_decode
    runner._sampling_params_for_padded_decode = (
        TTModelRunner._sampling_params_for_padded_decode.__get__(runner)
    )
    return runner


def _settle_layout(runner) -> None:
    """What a decode that consumed the layout leaves behind.

    The persistent batch reports a changed layout after a request is admitted,
    and an overlapped step requires a stable one. The real runner clears this
    in ``note_decode_layout_consumed``, which this harness stubs out, so a test
    that wants the steady state says so here.
    """
    runner._decode_layout_changed_since_last_decode = False


def _submit_plain_step(runner, *req_ids):
    """Run one asynchronous step that has nothing to verify."""
    scheduler_output = _scheduler_output(runner, decoding=req_ids)
    for req_id in req_ids:
        row = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    assert runner.execute_model(scheduler_output) is None
    wrapper = runner.sample_tokens(None)
    assert not isinstance(wrapper, AsyncTTSpecDecodeOutput), (
        "a step with nothing to verify went down the speculative path"
    )
    return wrapper


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


# region Adverse ordering


def test_a_condensed_row_takes_its_own_late_result():
    """A row that moved while its result was outstanding still gets it.

    The step carries the row order it was built with. Between the submission
    and the commit, ``_update_states`` removes the finished request and
    ``InputBatch.condense`` slides the survivor down into the freed row, so the
    row a request occupied when the verify ran is not the row it occupies when
    the commit writes. The commit resolves each row through
    ``req_id_to_index``, and this is what says so: a commit that trusted the
    step's own row index would write the survivor's tokens into a row it no
    longer owns.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11), ("b", 41))
    assert runner.input_batch.req_id_to_index == {"a": 0, "b": 1}
    expected_b = continuation(*_tail(runner, "b"), DRAFT_LEN + 1)

    wrapper = _submit_step(
        runner,
        "a",
        "b",
        drafts={
            "a": _accept_everything(runner, "a"),
            "b": _accept_everything(runner, "b"),
        },
    )
    model.release()
    wrapper.get_output()
    # "a" finishes while the result is outstanding, so the drain removes it and
    # condense moves "b" into row 0 before the commit runs.
    _drain(runner, "b", finished=["a"])

    assert runner.input_batch.req_id_to_index == {"b": 0}, "condense did not move it"
    row = runner.input_batch.req_id_to_index["b"]
    length = int(runner.input_batch.num_tokens[row])
    committed = runner.input_batch.token_ids_cpu[
        row, length - len(expected_b) : length
    ].tolist()
    assert committed == expected_b
    assert runner.requests["b"].output_token_ids == expected_b


def test_a_late_result_does_not_revive_a_preempted_candidate_state():
    """Preemption takes the state the late result would name, then a replay.

    Two halves of one late result, and they are treated differently on
    purpose. The accepted tokens are kept: the engine has them, an ordinary
    preemption is not a cancellation, and the resume restores the request from
    this history. The candidate state they named is not: ``_update_states``
    released the request's device state slot and its row when it preempted the
    request, so an accepted count recorded here would tell the next verify to
    select a slot the model no longer holds, and a draft proposed here would
    continue it.

    Only the asynchronous path can reach this. On the synchronous path the
    commit runs inside the step, before any preemption reaches the runner.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    _admit(runner, ("a", 11))
    runner._req_state_slot["a"] = 0
    expected = continuation(*_tail(runner, "a"), DRAFT_LEN + 1)

    wrapper = _submit_step(runner, "a", drafts={"a": expected[:DRAFT_LEN]})
    model.release()
    wrapper.get_output()
    _drain(runner, preempted=["a"])

    assert "a" in runner.requests, "a preempted request keeps its cached state"
    assert runner.requests["a"].output_token_ids == expected, (
        "an ordinary preemption keeps the accepted tokens: the engine has them"
    )
    assert "a" not in runner.input_batch.req_id_to_index
    assert "a" not in runner._req_state_slot
    assert "a" not in runner._req_accepted_counts, (
        "the late commit recorded a candidate state the preemption released"
    )
    assert "a" not in runner._proposed_draft_token_ids, (
        "the late commit drafted a continuation of state that is gone"
    )

    # The resume: fresh blocks, and the row comes back. It speculates again,
    # from the history the preemption kept.
    runner._update_states(_scheduler_output(runner, decoding=["a"], resumed=["a"]))
    assert "a" in runner.input_batch.req_id_to_index

    next_expected = continuation(*_tail(runner, "a"), DRAFT_LEN + 1)
    wrapper = _submit_step(runner, "a", drafts={"a": next_expected[:DRAFT_LEN]})
    model.release()
    output = wrapper.get_output()
    _drain(runner, "a")

    assert output.sampled_token_ids[0] == next_expected
    assert runner.requests["a"].output_token_ids == expected + next_expected


def test_one_step_commits_zero_partial_and_full_acceptance_apart():
    """Three rows, three acceptance outcomes, three histories, one step.

    Every other acceptance case here runs one row at a time, which cannot
    catch a walk that mixes rows: the same accepted count applied to every row
    passes a single-row test. Each row starts from a different prompt and is
    offered drafts that are wrong at a different position, so a row taking
    another row's count or another row's tokens fails.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("none", 11), ("part", 41), ("full", 71))

    truth = {
        req_id: continuation(*_tail(runner, req_id), DRAFT_LEN + 1)
        for req_id in ("none", "part", "full")
    }
    offered = {
        # Wrong at the first draft, so the row commits the target's own choice
        # there and stops.
        "none": [truth["none"][0] + 1] + truth["none"][1:DRAFT_LEN],
        # Right, then wrong: two tokens commit.
        "part": [truth["part"][0], truth["part"][1] + 1, truth["part"][2]],
        "full": truth["full"][:DRAFT_LEN],
    }
    expected = {
        "none": truth["none"][:1],
        "part": truth["part"][:2],
        "full": truth["full"],
    }

    wrapper = _submit_step(runner, "none", "part", "full", drafts=offered)
    model.release()
    output = wrapper.get_output()
    _drain(runner, "none", "part", "full")

    for req_id, tokens in expected.items():
        row = output.req_id_to_index[req_id]
        assert output.sampled_token_ids[row] == tokens, req_id
        assert runner.requests[req_id].output_token_ids == tokens, req_id
        assert runner._req_accepted_counts[req_id] == len(tokens), req_id


def test_each_step_proposes_from_its_own_hidden_handle():
    """The handle the drafter receives belongs to the step being committed.

    The runner holds a handle per outstanding step, not one field per runner.
    Two steps in a row therefore have to reach the drafter with two different
    handles, each the one its own verify returned. ``DeferredVerifyTarget``
    raises if it is handed a handle it did not produce, so a runner that kept
    the first step's handle would fail on the second step rather than commit a
    wrong token.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    _admit(runner, ("a", 11))

    handles = []
    for _ in range(2):
        wrapper = _submit_step(
            runner, "a", drafts={"a": _accept_everything(runner, "a")}
        )
        model.release()
        wrapper.get_output()
        handles.append(runner._completed_decode_steps[0].spec_hidden)
        _drain(runner, "a")

    assert len(model.propose_calls) == 2
    assert handles[0] is not handles[1], "both steps carried one handle"
    assert model.propose_calls == handles, (
        "a step proposed from a handle that was not its own verify's"
    )


def test_a_failed_readback_is_terminal_and_commits_nothing():
    """A readback that raises leaves no commit and no pending step.

    The failure is cached as terminal rather than retried, because the
    submission it belonged to was consumed: a second readback of the same
    device submission corrupts it. So both callers see the same exception, the
    device is read once, nothing reaches request state, and the step does not
    stay pending and block every following build.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    wrapper = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})

    failure = RuntimeError("the device read failed")
    model.fail_the_readback(failure)
    model.release()

    with pytest.raises(RuntimeError) as first:
        wrapper.get_output()
    with pytest.raises(RuntimeError) as second:
        wrapper.get_output()
    assert first.value is failure
    assert second.value is failure, "the failure was not cached as terminal"
    assert model.reads == 1, "the same submission was read back twice"
    assert not runner._completed_decode_steps, "a failed step enqueued a commit"

    _drain(runner, "a")
    assert runner.requests["a"].output_token_ids == []
    assert not runner._pending_async_steps, (
        "a failed step stayed pending, so every following build would drain it"
    )


def test_a_failed_proposal_does_not_commit_twice():
    """A proposal that raises surfaces, and its step is not applied again.

    ``commit_spec_acceptance`` writes the accepted prefix and then asks the
    model to draft. A drafter that raises leaves the prefix written, which is
    correct: those tokens were accepted. What must not happen is the same step
    being applied a second time on a later drain, which would append the
    prefix twice and corrupt the request's text.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    _admit(runner, ("a", 11))
    expected = continuation(*_tail(runner, "a"), DRAFT_LEN + 1)

    wrapper = _submit_step(runner, "a", drafts={"a": expected[:DRAFT_LEN]})
    model.release()
    wrapper.get_output()

    failure = RuntimeError("the drafter failed")
    model.fail_next_propose(failure)
    with pytest.raises(RuntimeError) as raised:
        _drain(runner, "a")
    assert raised.value is failure

    assert runner.requests["a"].output_token_ids == expected
    # The queue was drained before the apply, so the next step cannot replay it.
    assert not runner._completed_decode_steps
    _drain(runner, "a")
    assert runner.requests["a"].output_token_ids == expected, (
        "the step was applied twice, so the prefix is in the history twice"
    )


def test_a_cancelled_request_s_row_is_taken_before_its_result_is_applied():
    """The row belongs to someone else by the time the commit runs.

    A cancellation reaches the runner the way a completion does, in
    ``finished_req_ids``, and the scheduler is free to admit a new request in
    the same step: ``_update_states`` removes the cancelled request, and the
    new one takes the row it freed, all before
    ``apply_ready_completed_decode_steps`` applies the step that was
    outstanding. So the row named by the completed step now holds a different
    request's prompt, and the cancelled request's tokens must reach neither it
    nor the request that is gone.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    runner._req_state_slot["a"] = 0
    wrapper = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    wrapper.get_output()

    # One step: "a" is cancelled and "z" takes its row, then the result lands.
    _drain(runner, new=[("z", 151)], finished=["a"])

    assert "a" not in runner.requests
    assert runner.input_batch.req_id_to_index == {"z": 0}
    assert runner.requests["z"].output_token_ids == [], (
        "the cancelled request's tokens were written into the row it left"
    )
    row = runner.input_batch.req_id_to_index["z"]
    assert int(runner.input_batch.num_tokens[row]) == PROMPT_LEN, (
        "the new request's row grew by tokens it never generated"
    )
    assert "z" not in runner._req_accepted_counts
    assert "z" not in runner._proposed_draft_token_ids

    # And "z" then speculates from its own prompt tail.
    expected = continuation(*_tail(runner, "z"), DRAFT_LEN + 1)
    second = _submit_step(runner, "z", drafts={"z": expected[:DRAFT_LEN]})
    model.release()
    output = second.get_output()
    _drain(runner, "z")
    assert output.sampled_token_ids[0] == expected
    assert runner.requests["z"].output_token_ids == expected


def test_a_forced_reset_publishes_its_frame_without_applying_it():
    """The one case where a live request's committed prefix is not written.

    A forced prefix-cache reset calls
    ``reset_prefix_cache(reset_running_requests=True)``, which preempts live
    requests, frees their blocks, and records in
    ``_tt_forced_reset_discard_counts`` how many in-flight results it made
    stale. vLLM resumes each request from its saved history and discards that
    many published outputs, so the runner must publish the frame and not apply
    it: applying it would replay tokens the scheduler has already thrown away,
    and recording its accepted count would name candidate state the reset
    destroyed.

    This request keeps its row, which is what separates this case from a
    cancellation or an ordinary preemption, and is why the skip has to be
    explicit rather than a consequence of the row being gone.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    _admit(runner, ("a", 11))

    wrapper = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    output = wrapper.get_output()
    assert output.sampled_token_ids[0], "the frame has to be published"

    row = runner.input_batch.req_id_to_index["a"]
    length_before = int(runner.input_batch.num_tokens[row])
    _drain(runner, "a", forced_reset={"a": 1})

    assert runner.input_batch.req_id_to_index["a"] == row, (
        "a forced reset keeps the request's row, unlike a cancellation"
    )
    assert runner.requests["a"].output_token_ids == [], (
        "the discarded frame was applied, so these tokens are in the history "
        "twice: once here and once in what vLLM replays"
    )
    assert int(runner.input_batch.num_tokens[row]) == length_before
    assert "a" not in runner._req_accepted_counts
    assert "a" not in runner._proposed_draft_token_ids


def test_a_condensed_row_is_drafted_against_its_own_remaining_context():
    """The proposal trims by the row the request holds now, not the step's.

    Two row spaces meet in the proposal. The committed block and the accepted
    counts are indexed by the step's rows, which is the order the model
    answered in, while how much context the request has left is a property of
    the row it occupies now. A completion between the submission and the commit
    makes ``condense`` slide a request into another row, and the vacated row
    keeps a stale copy of that request's pre-commit length.

    Here "c" sits at step row 2, moves to row 0 when "a" finishes, and is close
    enough to ``max_model_len`` that the two readings disagree: its own row
    leaves room for two more tokens after this commit, the stale copy at row 2
    leaves room for six. A proposal trimmed by the stale value drafts a third
    token past the context, which the next verify spends a candidate column on
    and the commit then drops.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    _admit(runner, ("a", 11), ("b", 41), ("c", 71))
    runner.input_batch.num_tokens[runner.input_batch.req_id_to_index["c"]] = (
        MAX_MODEL_LEN - DRAFT_LEN - 3
    )

    wrapper = _submit_step(
        runner,
        "a",
        "b",
        "c",
        drafts={req: _accept_everything(runner, req) for req in ("a", "b", "c")},
    )
    model.release()
    wrapper.get_output()
    _drain(runner, "b", "c", finished=["a"])

    assert runner.input_batch.req_id_to_index == {"c": 0, "b": 1}, (
        "condense did not move c out of the row its step was built with"
    )
    row = runner.input_batch.req_id_to_index["c"]
    room = MAX_MODEL_LEN - int(runner.input_batch.num_tokens[row])
    assert room == 2, "the test no longer places c where the two readings differ"
    assert len(runner._proposed_draft_token_ids["c"]) == room
    # The row c left still holds its pre-commit length, which is what a
    # proposal indexed by the step's rows would have read.
    assert int(runner.input_batch.num_tokens[2]) == MAX_MODEL_LEN - DRAFT_LEN - 3
    # b did not move and has context to spare, so it keeps the full draft set.
    assert len(runner._proposed_draft_token_ids["b"]) == DRAFT_LEN


# endregion Adverse ordering


# region Solo and batched transitions


def test_a_draftless_step_is_overlap_safe_and_a_verify_is_not():
    """The two-sided barrier, asserted from both sides.

    This is the milestone's performance objective reduced to the one decision
    it rests on. A step with nothing to verify is registered as overlap-safe,
    so the next build does not wait for it. A verify is registered as not
    overlap-safe, so the next build does.

    The second half is worth pinning because nothing else provides it:
    ``check_perform_device_sampling`` does not look at speculation, so on a
    launch that samples on device a verify satisfies every condition
    ``can_use_steady_decode_fast_path`` checks and comes back eligible. The
    explicit registration in ``submit_async_decode`` is the whole barrier.
    """
    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    _settle_layout(runner)

    plain = _submit_plain_step(runner, "a")
    assert list(runner._pending_async_overlap_ok) == [True]
    assert not runner.async_decode.must_drain_pending_async_steps(
        steady_decode_candidate=True
    ), "a step with nothing to verify made the next build wait"
    model.release()
    plain.get_output()
    _drain(runner, "a")
    assert model.device_sampled_calls == 1

    # And a verify, on the same runner, with drafts in flight.
    _settle_layout(runner)
    drafted = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    assert list(runner._pending_async_overlap_ok) == [False]
    assert runner.async_decode.must_drain_pending_async_steps(
        steady_decode_candidate=True
    )
    # The predicate the ordinary path would have used says otherwise, which is
    # why the registration cannot be derived from it.
    assert runner.async_decode.can_use_steady_decode_fast_path(drafted._model_input)
    model.release()
    drafted.get_output()
    _drain(runner, "a")


def test_a_second_baseline_step_submits_before_the_first_completes():
    """Overlap, as an ordering fact rather than a launch option.

    The first step's completion is held. The second step is then submitted and
    reaches the model, which is what overlap means: the runner did not wait for
    the outstanding readback before building and submitting the next forward.
    The model's own call log is the evidence, and the hold is explicit, so this
    is an ordering the test forces rather than one it hopes for.
    """
    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    _settle_layout(runner)

    first = _submit_plain_step(runner, "a")
    assert model.outstanding == 1, "the first step's readback is not held"
    assert model.device_sampled_calls == 1

    second = _submit_plain_step(runner, "a")
    assert model.device_sampled_calls == 2, (
        "the second forward waited for the first step's readback"
    )
    assert model.outstanding == 2

    # Both land, oldest first, which is the order the device stream permits.
    model.release()
    first.get_output()
    model.release()
    second.get_output()
    _drain(runner, "a")
    assert len(runner.requests["a"].output_token_ids) == 2


def test_solo_speculation_yields_to_a_peer_and_resumes_when_it_leaves():
    """The sequence the third milestone asks for, end to end.

    A lone request speculates. A peer arrives, so the model's drafter offers
    nothing and the batched steps run as ordinary overlapping decodes. The peer
    leaves, the drafter offers again, and the solo request speculates from
    where it got to. What the assertions hold onto is which call each step
    made, because "a response came back" says nothing about whether the batch
    spent those steps in a verify it did not need.
    """
    model = DeferredVerifyTarget()
    model.offer_drafts_only_when_solo()
    runner = _baseline_runner(model)
    _admit(runner, ("solo", 11))
    _settle_layout(runner)

    # Solo: the drafter offered K at the last commit, so this step verifies.
    plain = _submit_plain_step(runner, "solo")
    model.release()
    plain.get_output()
    _drain(runner, "solo")
    drafts = runner.take_draft_token_ids()
    assert drafts is not None and drafts.draft_token_ids[0], (
        "a solo request was offered no drafts"
    )

    _settle_layout(runner)
    verified = _submit_step(
        runner, "solo", drafts={"solo": list(drafts.draft_token_ids[0])}
    )
    model.release()
    verified.get_output()
    _drain(runner, "solo")
    solo_tokens = len(runner.requests["solo"].output_token_ids)
    assert solo_tokens > 2, "the verify committed no prefix"

    # The peer arrives in a step that also schedules the solo request: a
    # running request the scheduler output omits leaves the persistent batch,
    # so admitting the peer on its own would take the solo request out of it.
    _drain(runner, "solo", new=[("peer", 71)])
    _settle_layout(runner)

    # One verify first, and it is not speculation: the solo request's last
    # commit was several tokens wide, and ``accepted_counts`` is how the model
    # finds which candidate state slot that commit landed on. The step carries
    # the count, drafts nothing, and resolves it. That is the fixed cost of
    # leaving speculation, one step per transition.
    resolving = _submit_step(runner, "solo", "peer")
    assert model.verify_calls[-1]["num_valid_drafts"].tolist() == [0] * MAX_NUM_REQS
    assert int(model.verify_calls[-1]["accepted_counts"][0]) > 1
    model.release()
    resolving.get_output()
    _drain(runner, "solo", "peer")
    _settle_layout(runner)

    verifies_before = len(model.verify_calls)
    for _ in range(3):
        step = _submit_plain_step(runner, "solo", "peer")
        model.release()
        step.get_output()
        _drain(runner, "solo", "peer")
        _settle_layout(runner)
    assert len(model.verify_calls) == verifies_before, (
        "a batched baseline step ran a verify"
    )
    assert runner.take_draft_token_ids() is None, (
        "the drafter offered drafts for a batch it declined"
    )
    # One token from the resolving verify plus one from each ordinary step.
    assert len(runner.requests["peer"].output_token_ids) == 4

    # The peer leaves. The drafter offers again, and the next step verifies.
    _drain(runner, "solo", finished=["peer"])
    _settle_layout(runner)
    step = _submit_plain_step(runner, "solo")
    model.release()
    step.get_output()
    _drain(runner, "solo")
    drafts = runner.take_draft_token_ids()
    assert drafts is not None and drafts.draft_token_ids[0], (
        "speculation never resumed after the batch went solo again"
    )

    _settle_layout(runner)
    verified = _submit_step(
        runner, "solo", drafts={"solo": list(drafts.draft_token_ids[0])}
    )
    model.release()
    verified.get_output()
    _drain(runner, "solo")
    assert len(model.verify_calls) == verifies_before + 1
    # And every token the solo request emitted is still its own continuation.
    assert runner.requests["solo"].output_token_ids == continuation(
        11 + PROMPT_LEN - 1,
        PROMPT_LEN - 1,
        len(runner.requests["solo"].output_token_ids),
    )


def test_a_peer_cancelled_mid_transition_leaves_the_solo_request_intact():
    """A cancellation at the transition, with a result outstanding.

    The peer is cancelled while a batched step's result is still in flight, so
    the runner applies that step to one request and not the other, and the
    batch is solo again in the same scheduler step. The solo request has to
    come out of it with its own continuation and with speculation able to
    resume.
    """
    model = DeferredVerifyTarget()
    model.offer_drafts_only_when_solo()
    runner = _baseline_runner(model)
    _admit(runner, ("solo", 11), ("peer", 71))
    _settle_layout(runner)

    step = _submit_plain_step(runner, "solo", "peer")
    model.release()
    step.get_output()
    # The peer is cancelled before its result is applied.
    _drain(runner, "solo", finished=["peer"])

    assert "peer" not in runner.requests
    assert "peer" not in runner._proposed_draft_token_ids
    assert runner.requests["solo"].output_token_ids == continuation(
        11 + PROMPT_LEN - 1, PROMPT_LEN - 1, 1
    )
    # The drafter decided during the step that ran, when two requests were
    # live, so it offered nothing and there is nothing to report yet. One
    # ordinary step later the batch is solo and it offers again: a transition
    # costs one step before speculation resumes, which is inherent to deciding
    # at the commit of the step that produced the token.
    assert runner.take_draft_token_ids() is None

    _settle_layout(runner)
    step = _submit_plain_step(runner, "solo")
    model.release()
    step.get_output()
    _drain(runner, "solo")

    drafts = runner.take_draft_token_ids()
    assert drafts is not None
    assert drafts.req_ids == ["solo"]


# endregion Solo and batched transitions


# region Drafts on an asynchronously scheduled launch


def _reservation(runner, *req_ids):
    """The lookahead ``AsyncScheduler`` leaves on a scheduled request.

    ``[-1] * num_spec_tokens_to_schedule``, which is what the real scheduler
    delivers under asynchronous scheduling: a reservation of that many
    positions, not a proposal. ``TTScheduler`` leaves it standing there for
    exactly this reason.
    """
    return {req_id: [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN for req_id in req_ids}


def test_the_runner_verifies_its_own_proposal_when_the_scheduler_sends_none():
    """Where the drafts come from on an asynchronous launch.

    ``EngineCore.post_step`` does not call ``take_draft_token_ids`` when
    asynchronous scheduling is on, so nothing hands the scheduler a proposal
    and nothing comes back. The runner holds the proposal its own drafter
    published at the last commit, and that is what this step verifies. Without
    this, a launch with speculation configured and asynchronous scheduling
    enabled would decode every step plainly and never speculate, silently.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    expected = continuation(*_tail(runner, "a"), DRAFT_LEN + 1)
    runner._proposed_draft_token_ids["a"] = expected[:DRAFT_LEN]

    wrapper = _submit_step(runner, "a", drafts=_reservation(runner, "a"))
    model.release()
    output = wrapper.get_output()
    _drain(runner, "a")

    assert model.verify_calls, "the step did not verify anything"
    assert model.verify_calls[-1]["num_valid_drafts"].tolist()[0] == DRAFT_LEN
    assert output.sampled_token_ids[0] == expected
    # Handed over once: the entry is consumed by the step that verified it, so
    # a later step cannot replay a spent proposal.
    assert "a" not in runner._proposed_draft_token_ids


def test_a_placeholder_is_never_verified_as_a_draft():
    """The reservation is read as a count, never as token ids.

    A ``-1`` reaching the accept walk matches whatever the model returns for
    that column, because the model is handed the same placeholder, and commits
    as an output token. With no proposal held, the step verifies nothing.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    assert not runner._proposed_draft_token_ids

    wrapper = _submit_step(runner, "a", drafts=_reservation(runner, "a"))
    model.release()
    output = wrapper.get_output()
    _drain(runner, "a")

    assert model.verify_calls[-1]["num_valid_drafts"].tolist()[0] == 0
    assert output.sampled_token_ids[0] == continuation(
        11 + PROMPT_LEN - 1, PROMPT_LEN - 1, 1
    )
    assert PLACEHOLDER_TOKEN_ID not in runner.requests["a"].output_token_ids


def test_a_proposal_longer_than_the_reservation_is_trimmed_to_it():
    """The scheduler's reservation is the budget, and it wins.

    The request was scheduled ``1 + reserved`` positions, so verifying a
    longer block would write tokens into a row whose KV the scheduler budgeted
    for fewer.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))
    expected = continuation(*_tail(runner, "a"), DRAFT_LEN + 1)
    runner._proposed_draft_token_ids["a"] = expected[:DRAFT_LEN]

    reserved = 1
    wrapper = _submit_step(runner, "a", drafts={"a": [PLACEHOLDER_TOKEN_ID] * reserved})
    model.release()
    output = wrapper.get_output()
    _drain(runner, "a")

    assert model.verify_calls[-1]["num_valid_drafts"].tolist()[0] == reserved
    assert output.sampled_token_ids[0] == expected[: reserved + 1]


def test_a_mixed_reservation_and_proposal_is_refused():
    """Either the scheduler reserved positions or it delivered drafts.

    A list holding both says the two conventions have been crossed, and the
    runner cannot tell which half of it to verify. Raised by name rather than
    guessed at, because guessing wrong commits a placeholder as a token.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    _admit(runner, ("a", 11))

    with pytest.raises(RuntimeError, match="mix of placeholder and real"):
        _submit_step(runner, "a", drafts={"a": [PLACEHOLDER_TOKEN_ID, 11, 12]})


def test_a_synchronous_launch_still_takes_the_scheduler_s_drafts():
    """The other half of the switch, so neither path drifts into the other.

    Synchronously the scheduler owns the drafts: it budgets them, runs them
    through the grammar, and delivers what survived. A runner that preferred
    its own copy there would bypass both.
    """
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner.async_decode_scheduling = False
    _admit(runner, ("a", 11))
    expected = continuation(*_tail(runner, "a"), DRAFT_LEN + 1)
    # Held by the runner and not delivered: this must not be verified.
    runner._proposed_draft_token_ids["a"] = [expected[0] + 7] * DRAFT_LEN

    scheduler_output = _scheduler_output(
        runner, decoding=["a"], drafts={"a": expected[:DRAFT_LEN]}
    )
    row = runner.input_batch.req_id_to_index["a"]
    runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[row]
    runner._update_states(scheduler_output)
    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input.draft_token_ids[0, :DRAFT_LEN].tolist() == expected[:DRAFT_LEN]
    assert runner._proposed_draft_token_ids["a"] == [expected[0] + 7] * DRAFT_LEN


# endregion Drafts on an asynchronously scheduled launch


# region Serialization across a mixed sequence


def test_a_verify_is_not_submitted_over_an_outstanding_ordinary_step():
    """The drain the pending flags cannot ask for.

    An ordinary step is overlap-safe, so nothing about it forces a drain. A
    verify built while it is outstanding would start its candidate block from
    each row's last committed token, and that token is the one the outstanding
    step has not handed back yet: the block would begin one token behind and
    the accept walk would commit for the wrong prefix. So the decision looks at
    what the next step is going to be, not only at what is already pending.
    """
    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    _settle_layout(runner)

    outstanding = _submit_plain_step(runner, "a")
    assert list(runner._pending_async_overlap_ok) == [True]

    # A proposal in hand means the next step verifies.
    runner._proposed_draft_token_ids["a"] = _accept_everything(runner, "a")
    scheduler_output = _scheduler_output(
        runner, decoding=["a"], drafts={"a": [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN}
    )

    assert runner.async_decode.must_drain_pending_async_steps(
        steady_decode_candidate=True, scheduler_output=scheduler_output
    ), "a verify would have been built over an outstanding step"

    model.release()
    outstanding.get_output()
    _drain(runner, "a")


def test_an_ordinary_step_still_overlaps_an_outstanding_ordinary_step():
    """The overlap this must not cost: baseline steps still run together."""
    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    _settle_layout(runner)

    outstanding = _submit_plain_step(runner, "a")
    scheduler_output = _scheduler_output(runner, decoding=["a"])

    assert not runner.async_decode.must_drain_pending_async_steps(
        steady_decode_candidate=True, scheduler_output=scheduler_output
    )

    model.release()
    outstanding.get_output()
    _drain(runner, "a")


def test_a_proposal_created_during_apply_serializes_the_verify_transition():
    """A newly published proposal drains every earlier plain submission."""
    model = ResidentDeferredTarget()
    model.offer_drafts_only_when_solo()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    warmup = _submit_plain_step(runner, "a")
    model.release()
    warmup.get_output()
    _drain(runner, "a")
    _settle_layout(runner)
    expected_plain = continuation(*_tail(runner, "a"), 2)

    first = _submit_plain_step(runner, "a")
    second = _submit_plain_step(runner, "a")
    model.release()
    first.get_output()
    model.release()

    scheduler_output = _scheduler_output(
        runner, decoding=["a"], drafts={"a": [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN}
    )
    scheduler_output.scheduled_cached_reqs.num_computed_tokens[0] += 2
    assert runner.execute_model(scheduler_output) is None
    verified = runner.sample_tokens(None)
    assert isinstance(verified, AsyncTTSpecDecodeOutput)

    assert second.is_resolved(), "the earlier plain step was not drained"
    assert runner.async_decode._overlapped_unsafe_submissions == 0
    assert int(model.verify_calls[-1]["tokens"][0, 0]) == expected_plain[-1]

    model.release()
    verified.get_output()
    _drain(runner, "a")


def test_overlapped_plain_completions_report_authoritative_positions():
    """Each proposal receives the position of its own committed plain token."""
    model = ResidentDeferredTarget()
    model.offer_drafts_only_when_solo()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    warmup = _submit_plain_step(runner, "a")
    model.release()
    warmup.get_output()
    _drain(runner, "a")
    _settle_layout(runner)
    _, initial_position = _tail(runner, "a")

    first = _submit_plain_step(runner, "a")
    second = _submit_plain_step(runner, "a")
    model.release()
    first.get_output()
    model.release()
    second.get_output()
    _drain(runner, "a")

    positions = [int(value[0, 0]) for value in model.proposal_positions[-2:]]
    assert positions == [initial_position + 1, initial_position + 2]


def test_temporary_unscheduling_preserves_the_completed_accepted_count():
    """A request that retains its model slot also retains candidate selection."""
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    _admit(runner, ("a", 11))
    runner._req_state_slot["a"] = 0

    accepted = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    accepted.get_output()
    _drain(runner, new=[("peer", 71)])

    assert "a" not in runner.input_batch.req_id_to_index
    assert runner._req_state_slot["a"] == 0
    assert runner._req_accepted_counts["a"] == DRAFT_LEN + 1
    assert runner.released_slots == []

    runner.requests["a"].num_computed_tokens = runner.requests["a"].num_tokens
    resumed_input = runner.build_model_input(
        _scheduler_output(runner, decoding=["a"]), None
    )
    assert resumed_input is not None and resumed_input.prompt_lens is None
    assert int(resumed_input.accepted_counts[0]) == DRAFT_LEN + 1


def test_an_unresolved_accepted_count_also_forces_the_drain():
    """A step carrying a count is a verify too, drafts or not.

    The step after a multi-token commit sends ``accepted_counts`` so the model
    can find the candidate state slot that commit landed on, and it is built
    from the row's last committed token like any other verify.
    """
    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    _settle_layout(runner)

    outstanding = _submit_plain_step(runner, "a")
    runner._req_accepted_counts["a"] = 4
    scheduler_output = _scheduler_output(runner, decoding=["a"])

    assert runner.async_decode.must_drain_pending_async_steps(
        steady_decode_candidate=True, scheduler_output=scheduler_output
    )

    model.release()
    outstanding.get_output()
    _drain(runner, "a")


def test_overlapping_submissions_are_counted_and_unsafe_ones_are_not():
    """The instrumentation a server has no other way to report.

    Nothing else says whether a step overlapped: a launch option does not, a
    completed request does not, and per-step wall clock cannot separate
    overlap from a faster model. The device suite reads these counters, so what
    they count has to be exactly this: a submission that found a step already
    outstanding, and separately one of those that was not overlap-safe, which
    must never happen.
    """
    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    _settle_layout(runner)
    controller = runner.async_decode
    assert controller._overlapped_submissions == 0

    first = _submit_plain_step(runner, "a")
    assert controller._overlapped_submissions == 0, (
        "the first submission of a chain overlapped nothing"
    )
    second = _submit_plain_step(runner, "a")

    assert controller._overlapped_submissions == 1
    assert controller._overlapped_unsafe_submissions == 0

    model.release()
    first.get_output()
    model.release()
    second.get_output()
    _drain(runner, "a")

    # And a verify, which drains first, so it overlaps nothing and the unsafe
    # counter stays where it belongs.
    _settle_layout(runner)
    verified = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    verified.get_output()
    _drain(runner, "a")

    assert controller._overlapped_unsafe_submissions == 0


def test_an_unsafe_overlap_is_logged_at_its_exact_counter_value(tmp_path):
    """The device parser must observe an unsafe non-power-of-two overlap."""
    from tests.tt.spec.test_async_transitions import _overlap_counts

    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    controller = runner.async_decode

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    async_decode_module.logger.addHandler(handler)
    try:
        for overlap_ok in (True, True, True, False):
            controller.register_pending_async_step(object(), overlap_ok=overlap_ok)
    finally:
        async_decode_module.logger.removeHandler(handler)

    server_log = tmp_path / "server.log"
    server_log.write_text(stream.getvalue())
    assert (
        controller._overlapped_submissions,
        controller._overlapped_unsafe_submissions,
    ) == (
        3,
        1,
    )
    assert _overlap_counts(server_log) == (3, 1)


# endregion Serialization across a mixed sequence


def test_a_reservation_alone_does_not_force_a_drain():
    """The placeholder list is on every scheduled request, and means nothing.

    Under asynchronous scheduling ``AsyncScheduler`` gives each scheduled
    request ``[-1] * num_spec_tokens_to_schedule``, so a drain decision that
    read that list as a proposal would drain on every step and the launch
    would never overlap anything: exactly the behaviour this whole path exists
    to remove. The signal is the proposal the runner holds, not the
    reservation.
    """
    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    _settle_layout(runner)

    outstanding = _submit_plain_step(runner, "a")
    assert not runner._proposed_draft_token_ids
    reserved = _scheduler_output(
        runner, decoding=["a"], drafts=_reservation(runner, "a")
    )

    assert not runner.async_decode.must_drain_pending_async_steps(
        steady_decode_candidate=True, scheduler_output=reserved
    ), "the lookahead reservation was read as a proposal"

    # And with a proposal in hand, the same scheduler output does drain.
    runner._proposed_draft_token_ids["a"] = _accept_everything(runner, "a")
    assert runner.async_decode.must_drain_pending_async_steps(
        steady_decode_candidate=True, scheduler_output=reserved
    )

    model.release()
    outstanding.get_output()
    _drain(runner, "a")


def test_the_submission_counters_separate_verifies_from_ordinary_decodes():
    """What the overhead benchmark reads, and why it cannot read the metrics.

    vLLM's speculative counters describe the scheduler's lookahead
    reservation, which under asynchronous scheduling exists on every scheduled
    request whether or not a draft was ever verified. A cost-per-step
    comparison needs the two things only the runner knows: how many verifies it
    sent and how many ordinary decodes. Counted in ``submit_decode``, which is
    the one funnel both execution modes go through, so a synchronous run and an
    asynchronous one are counted by the same code.
    """
    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    _admit(runner, ("a", 11))
    _settle_layout(runner)
    controller = runner.async_decode
    assert controller._ordinary_decode_submissions == 0
    assert controller._verify_submissions == 0

    plain = _submit_plain_step(runner, "a")
    model.release()
    plain.get_output()
    _drain(runner, "a")

    assert controller._ordinary_decode_submissions == 1
    assert controller._verify_submissions == 0

    _settle_layout(runner)
    verified = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    verified.get_output()
    _drain(runner, "a")

    assert controller._ordinary_decode_submissions == 1
    assert controller._verify_submissions == 1


def test_the_submission_report_is_the_line_the_benchmark_parses():
    """The counters reach a measurement as text, so the text is the contract.

    ``bench_spec_overhead`` has no access to the worker process that holds
    ``TTAsyncDecodeController``; it diffs the last reported line before its
    measured interval against the last one after. A reworded line does not
    fail anything by itself: the benchmark's pattern stops matching, every
    diff becomes zero, and the run still prints a result. This pins the
    wording that pattern is compiled from, the cadence that decides which
    submissions report, and the third count, which is what says whether a
    configuration's extra cost is work it added or overlap it lost.
    """
    from tests.tt.spec.bench_spec_overhead import SUBMISSIONS

    model = DeferredVerifyTarget()
    runner = _baseline_runner(model)
    controller = runner.async_decode
    # One short of ``_SUBMISSION_LOG_INTERVAL`` in total, so the next
    # submission is the one that reports and the cadence itself is under test.
    controller._ordinary_decode_submissions = 17
    controller._verify_submissions = async_decode_module._SUBMISSION_LOG_INTERVAL - 18
    controller._overlapped_submissions = 9
    ordinary = SimpleNamespace(spec_mode=None)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    async_decode_module.logger.addHandler(handler)
    try:
        controller.count_decode_submission(ordinary)
        controller.count_decode_submission(ordinary)
    finally:
        async_decode_module.logger.removeHandler(handler)

    lines = stream.getvalue().splitlines()
    reported = [found.groups() for line in lines if (found := SUBMISSIONS.search(line))]
    assert reported == [
        ("18", str(async_decode_module._SUBMISSION_LOG_INTERVAL - 18), "9")
    ], (
        "the submission report no longer matches the pattern the overhead "
        f"benchmark parses, or no longer reports on the interval; the runner "
        f"logged {lines!r}"
    )
