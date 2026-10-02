# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A ``logits`` verify through the asynchronous path, with its readback held.

Asynchronously, the accept walk runs on whichever thread resolves the
deferred output, after the step's own submission and before the engine thread
commits it. For a sampled walk that adds three obligations the greedy walk did
not have. It must sample only once the logits have actually arrived. It must
sample under the controls captured when the step was built, because the
persistent batch may have moved on by then. And it must advance each request's
generator exactly once, however many callers race to resolve the step.

A dummy model served over HTTP cannot test any of that: it answers on the host
before the runner ever waits. So the readback here is held open. The model
returns its logits through ``read_decode_output`` as a buffer that is filled
only when the test releases it, and the patched ``ttnn.event_synchronize``
blocks until then, so "before the logits arrive" is a real point in the step.
"""

from __future__ import annotations

import threading

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.sample.sampler import Sampler

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS

from . import test_spec_async as async_harness
from . import test_spec_sampled_runner as sync_harness
from .sampled_target import SAMPLED_VOCAB_SIZE, SampledTarget, base_token

# The held-readback fixture, applied to every test here as it is there.
wait_for_released_reads = async_harness.wait_for_released_reads

DRAFT_LEN = async_harness.DRAFT_LEN
PROMPT_LEN = async_harness.PROMPT_LEN


class DeferredSampledTarget(SampledTarget):
    """``SampledTarget`` whose verify logits are readable only once released."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[tuple[torch.Tensor, torch.Tensor, threading.Event]] = []
        self.reads = 0
        self.wait_failure: BaseException | None = None

    def read_decode_output(self, tt_out, async_read: bool = False):
        self.reads += 1
        # NaN until released, so a walk that read the buffer early could not
        # produce a valid sample from it.
        buffer = torch.full_like(tt_out, float("nan"))
        event = threading.Event()
        self.pending.append((buffer, tt_out.clone(), event))
        return buffer, [event]

    def release(self) -> None:
        buffer, answer, event = self.pending.pop(0)
        buffer.copy_(answer)
        if self.wait_failure is not None:
            event.tt_test_failure = self.wait_failure
            self.wait_failure = None
        event.set()


def _runner(model: DeferredSampledTarget):
    runner = async_harness._runner(model)
    runner.vocab_size = SAMPLED_VOCAB_SIZE
    # An ordinary decode on a narrow-capable model samples on the host.
    runner.host_sampler = Sampler()
    runner._spec_accept_modes = type(model).accept_modes
    runner._num_unspeculable_verify_rows = 0
    # The real builder: the walk samples under these tensors.
    del runner._sampling_params_for_padded_decode
    runner._sampling_params_for_padded_decode = (
        TTModelRunner._sampling_params_for_padded_decode.__get__(runner)
    )
    return runner


def _rebuild_batch(runner) -> None:
    """The harness batch was sized for the greedy target's vocabulary."""
    from vllm_tt_plugin.input_batch import InputBatch

    runner.input_batch = InputBatch(
        max_num_reqs=async_harness.MAX_NUM_REQS,
        max_model_len=async_harness.MAX_MODEL_LEN,
        max_num_batched_tokens=async_harness.MAX_MODEL_LEN,
        vocab_size=SAMPLED_VOCAB_SIZE,
        block_sizes=[async_harness.BLOCK_SIZE],
        kernel_block_sizes=[async_harness.BLOCK_SIZE],
    )


def _sampled_runner():
    model = DeferredSampledTarget()
    runner = _runner(model)
    _rebuild_batch(runner)
    return model, runner


def _admit(runner, *specs: tuple[str, int, SamplingParams]) -> None:
    """Admit requests through ``_update_states``, which seeds their generators.

    All in one scheduler output: a running request that a step does not
    schedule leaves the persistent batch.
    """
    output = async_harness._scheduler_output(runner)
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=[
                (first_token + index) % SAMPLED_VOCAB_SIZE
                for index in range(PROMPT_LEN)
            ],
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=PROMPT_LEN,
            lora_request=None,
        )
        for req_id, first_token, params in specs
    ]
    output.num_scheduled_tokens = {req_id: 1 for req_id, _, _ in specs}
    output.total_num_scheduled_tokens = len(specs)
    runner._update_states(output)


def _drafts(runner, req_id: str) -> list[int]:
    token, position = async_harness._tail(runner, req_id)
    drafts = []
    for _ in range(DRAFT_LEN):
        token = base_token(token, position)
        position += 1
        drafts.append(token)
    return drafts


def _generator(runner, req_id: str) -> torch.Generator:
    return runner.requests[req_id].generator


SEEDED = SamplingParams(temperature=1.0, top_k=3, seed=41, ignore_eos=True)

# region Sampling waits for the logits


def test_a_logits_verify_samples_only_after_its_readback_completes():
    """The generator does not move until the held logits are released.

    The buffer the step holds is NaN until release, so a walk that ran early
    would either raise or commit garbage; a generator that advanced early is
    the quieter form of the same bug, and is what this checks first.
    """
    model, runner = _sampled_runner()
    _admit(runner, ("s", 5, SEEDED))
    untouched = _generator(runner, "s").get_state().clone()

    wrapper = async_harness._submit_step(
        runner, "s", drafts={"s": _drafts(runner, "s")}
    )

    assert model.verify_modes == [ACCEPT_MODE_LOGITS]
    assert len(model.pending) == 1, "the verify was read back immediately"
    assert not wrapper.is_resolved()
    assert torch.equal(_generator(runner, "s").get_state(), untouched)

    model.release()
    output = wrapper.get_output()

    assert not torch.equal(_generator(runner, "s").get_state(), untouched)
    committed = output.sampled_token_ids[output.req_id_to_index["s"]]
    assert 1 <= len(committed) <= DRAFT_LEN + 1
    assert runner.requests["s"].output_token_ids == []

    async_harness._drain(runner, "s")
    assert runner.requests["s"].output_token_ids == committed


def test_resolving_a_logits_step_twice_samples_once():
    """Racing resolutions read once, walk once, and advance the generator once."""
    model, runner = _sampled_runner()
    _admit(runner, ("s", 5, SEEDED))
    wrapper = async_harness._submit_step(
        runner, "s", drafts={"s": _drafts(runner, "s")}
    )
    model.release()

    outputs: list[object] = []
    threads = [
        threading.Thread(target=lambda: outputs.append(wrapper.get_output()))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "a resolver hung"
    assert len(outputs) == len(threads)
    after_race = _generator(runner, "s").get_state().clone()
    wrapper.ensure_finalized()

    assert model.reads == 1
    assert len({id(output) for output in outputs}) == 1
    assert len(runner._completed_decode_steps) == 1
    assert torch.equal(_generator(runner, "s").get_state(), after_race)

    # And it advanced exactly as far as one synchronous step of the same
    # request does.
    sync_runner = sync_harness._runner(SampledTarget())
    sync_harness._add_request(sync_runner, "s", SEEDED, first_token=5)
    sync_harness._step(sync_runner, "s", drafts={"s": _drafts_sync(sync_runner)})
    assert torch.equal(after_race, sync_runner.requests["s"].generator.get_state())


def _drafts_sync(runner) -> list[int]:
    return sync_harness._argmax_drafts(runner, "s")


def test_a_failed_logits_readback_samples_nothing():
    """A readback that raises never reaches the walk, so no draw is taken."""
    model, runner = _sampled_runner()
    _admit(runner, ("s", 5, SEEDED))
    untouched = _generator(runner, "s").get_state().clone()
    wrapper = async_harness._submit_step(
        runner, "s", drafts={"s": _drafts(runner, "s")}
    )

    failure = RuntimeError("the device read failed")
    model.wait_failure = failure
    model.release()
    with pytest.raises(RuntimeError) as raised:
        wrapper.get_output()

    assert raised.value is failure
    assert torch.equal(_generator(runner, "s").get_state(), untouched)
    assert not runner._completed_decode_steps


# endregion Sampling waits for the logits

# region The walk reads the step's own controls


def test_the_walk_samples_under_the_controls_captured_at_submission():
    """Changing the persistent batch after submission changes nothing.

    The reference is the same seeded step resolved before anything changed.
    The mutated run turns the row greedy and unpenalized in the batch, and
    removes it from the batch's random set, between submission and release.
    """

    def run(mutate: bool) -> list[int]:
        model, runner = _sampled_runner()
        _admit(
            runner,
            ("s", 5, SamplingParams(temperature=0.9, presence_penalty=0.8, seed=8)),
        )
        for _ in range(3):
            wrapper = async_harness._submit_step(
                runner, "s", drafts={"s": _drafts(runner, "s")}
            )
            if mutate:
                row = runner.input_batch.req_id_to_index["s"]
                runner.input_batch.sampling.temperature[row] = 0.0
                runner.input_batch.sampling.presence_penalty[row] = 0.0
                runner.input_batch.random_reqs.discard("s")
            model.release()
            wrapper.get_output()
            async_harness._drain(runner, "s")
            if mutate:
                row = runner.input_batch.req_id_to_index["s"]
                runner.input_batch.sampling.temperature[row] = 0.9
                runner.input_batch.sampling.presence_penalty[row] = 0.8
                runner.input_batch.random_reqs.add("s")
        return list(runner.requests["s"].output_token_ids)

    assert run(mutate=True) == run(mutate=False)


def test_the_async_path_commits_what_the_synchronous_path_commits():
    """Same seed, same drafts, same tokens, deferred or not.

    The asynchronous path splits the step at the readback and commits a step
    later. Neither split may change what is drawn or in which order.
    """
    params = SamplingParams(temperature=1.1, top_p=0.85, seed=19, ignore_eos=True)

    model, runner = _sampled_runner()
    _admit(runner, ("s", 7, params))
    while len(runner.requests["s"].output_token_ids) < 30:
        wrapper = async_harness._submit_step(
            runner, "s", drafts={"s": _drafts(runner, "s")}
        )
        model.release()
        wrapper.get_output()
        async_harness._drain(runner, "s")
    deferred = list(runner.requests["s"].output_token_ids)

    sync_runner = sync_harness._runner(SampledTarget())
    sync_harness._add_request(sync_runner, "s", params, first_token=7)
    while len(sync_runner.requests["s"].output_token_ids) < 30:
        sync_harness._step(sync_runner, "s", drafts={"s": _drafts_sync(sync_runner)})
    immediate = list(sync_runner.requests["s"].output_token_ids)

    assert deferred[:30] == immediate[:30]
    sync_harness._assert_in_support((7 + PROMPT_LEN - 1, PROMPT_LEN - 1), deferred[:30])


# endregion The walk reads the step's own controls

# region Lifecycle with a sampled step outstanding


def test_a_cancelled_request_s_late_logits_step_is_not_applied():
    """A request finished while its readback is pending commits nothing more.

    Its neighbor's step resolves normally and commits its own prefix.
    """
    model, runner = _sampled_runner()
    _admit(
        runner,
        ("a", 5, SEEDED),
        ("b", 30, SamplingParams(temperature=0.0, ignore_eos=True)),
    )
    wrapper = async_harness._submit_step(
        runner, "a", "b", drafts={"a": _drafts(runner, "a"), "b": _drafts(runner, "b")}
    )

    # "a" is cancelled while the readback is still held.
    async_harness._drain(runner, "b", finished=["a"])
    model.release()
    wrapper.get_output()
    async_harness._drain(runner, "b")

    assert "a" not in runner.requests
    assert "a" not in runner._req_accepted_counts
    assert len(runner.requests["b"].output_token_ids) == DRAFT_LEN + 1


def test_a_greedy_neighbor_keeps_its_exact_output_in_a_sampled_verify():
    """A mixed batch walks each row by its own temperature, deferred too."""
    model, runner = _sampled_runner()
    _admit(
        runner,
        ("s", 5, SEEDED),
        ("g", 30, SamplingParams(temperature=0.0, ignore_eos=True)),
    )
    token, position = async_harness._tail(runner, "g")
    while len(runner.requests["g"].output_token_ids) < 20:
        wrapper = async_harness._submit_step(
            runner,
            "s",
            "g",
            drafts={"s": _drafts(runner, "s"), "g": _drafts(runner, "g")},
        )
        model.release()
        wrapper.get_output()
        async_harness._drain(runner, "s", "g")

    greedy = runner.requests["g"].output_token_ids[:20]
    expected = []
    for _ in range(20):
        token = base_token(token, position)
        position += 1
        expected.append(token)
    assert greedy == expected
    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}
    assert ACCEPT_MODE_ARGMAX_IDS not in model.verify_modes
    sync_harness._assert_in_support(
        (5 + PROMPT_LEN - 1, PROMPT_LEN - 1), runner.requests["s"].output_token_ids
    )


# endregion Lifecycle with a sampled step outstanding


def test_a_condensed_sampled_row_takes_its_own_late_result():
    """A sampled row that moved while its readback was held keeps its draw.

    The walk ran on the row order the step was built with, using the
    generator captured for that row; ``condense`` then moves the survivor
    before the commit. Its tokens must be the ones a synchronous step of that
    request alone commits from the same seed.
    """
    model, runner = _sampled_runner()
    _admit(
        runner,
        ("a", 30, SamplingParams(temperature=0.0, ignore_eos=True)),
        ("b", 5, SEEDED),
    )
    assert runner.input_batch.req_id_to_index == {"a": 0, "b": 1}
    wrapper = async_harness._submit_step(
        runner, "a", "b", drafts={"a": _drafts(runner, "a"), "b": _drafts(runner, "b")}
    )
    model.release()
    published = wrapper.get_output()
    async_harness._drain(runner, "b", finished=["a"])

    assert runner.input_batch.req_id_to_index == {"b": 0}, "condense did not move it"
    committed = runner.requests["b"].output_token_ids
    assert committed == published.sampled_token_ids[published.req_id_to_index["b"]]

    sync_runner = sync_harness._runner(SampledTarget())
    sync_harness._add_request(sync_runner, "s", SEEDED, first_token=5)
    sync_harness._step(sync_runner, "s", drafts={"s": _drafts_sync(sync_runner)})
    assert committed == sync_runner.requests["s"].output_token_ids


def test_a_preempted_sampled_request_does_not_take_its_late_result():
    """Preemption frees the state the held logits verify was computed against."""
    model, runner = _sampled_runner()
    _admit(
        runner,
        ("a", 5, SEEDED),
        ("b", 30, SamplingParams(temperature=0.0, ignore_eos=True)),
    )
    runner._req_state_slot["a"] = 0
    wrapper = async_harness._submit_step(
        runner, "a", "b", drafts={"a": _drafts(runner, "a"), "b": _drafts(runner, "b")}
    )
    model.release()
    wrapper.get_output()

    async_harness._drain(runner, "b", preempted=["a"])

    assert "a" in runner.requests, "a preempted request keeps its cached state"
    assert "a" not in runner.input_batch.req_id_to_index
    assert "a" not in runner._req_state_slot
    assert "a" not in runner._req_accepted_counts
    assert len(runner.requests["b"].output_token_ids) == DRAFT_LEN + 1


# region More of the lifecycle, deferred


def _submit_any(runner, *req_ids, drafts=None):
    """One asynchronous step, whichever wrapper it resolves through."""
    scheduler_output = async_harness._scheduler_output(
        runner, decoding=req_ids, drafts=drafts
    )
    for req_id in req_ids:
        row = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    assert runner.execute_model(scheduler_output) is None
    return runner.sample_tokens(None)


def _finish(model, runner, wrapper, *req_ids, **drain):
    model.release()
    wrapper.get_output()
    async_harness._drain(runner, *req_ids, **drain)


def test_a_draftless_seeded_row_reads_as_ordinary_sampling_when_deferred():
    """The deferred twin of the synchronous exactness check.

    The greedy neighbour drafts every step, so every step is a ``logits``
    verify, and the seeded row commits only its bonus, drawn on the readback
    thread. It must read token for token what an unspeculated runner draws.
    """
    params = SamplingParams(temperature=0.9, top_k=3, seed=23, ignore_eos=True)
    model, runner = _sampled_runner()
    _admit(
        runner,
        ("g", 40, SamplingParams(temperature=0.0, ignore_eos=True)),
        ("s", 7, params),
    )
    while len(runner.requests["s"].output_token_ids) < 24:
        wrapper = async_harness._submit_step(
            runner, "g", "s", drafts={"g": _drafts(runner, "g")}
        )
        _finish(model, runner, wrapper, "g", "s")

    assert runner.requests["s"].output_token_ids[:24] == sync_harness._run_plain(
        params, 24, first_token=7
    )


def test_a_penalized_seeded_request_reads_the_same_deferred_or_not():
    """The captured penalty history, compared across the two decode tails."""
    params = SamplingParams(
        temperature=0.8,
        presence_penalty=0.6,
        frequency_penalty=0.3,
        repetition_penalty=1.3,
        seed=29,
        ignore_eos=True,
    )
    model, runner = _sampled_runner()
    _admit(runner, ("s", 7, params))
    while len(runner.requests["s"].output_token_ids) < 30:
        wrapper = async_harness._submit_step(
            runner, "s", drafts={"s": _drafts(runner, "s")}
        )
        _finish(model, runner, wrapper, "s")

    sync_runner = sync_harness._runner(SampledTarget())
    sync_harness._add_request(sync_runner, "s", params, first_token=7)
    while len(sync_runner.requests["s"].output_token_ids) < 30:
        sync_harness._step(sync_runner, "s", drafts={"s": _drafts_sync(sync_runner)})

    assert (
        runner.requests["s"].output_token_ids[:30]
        == sync_runner.requests["s"].output_token_ids[:30]
    )


def test_a_request_joining_while_a_logits_readback_is_held_changes_no_draw():
    """A join reshapes the batch between submission and commit, and nothing more.

    The held step's walk reads the rows it was built with; the sampled
    request's draws match its run alone, and the joiner starts from its own
    prompt.
    """
    model, runner = _sampled_runner()
    _admit(runner, ("s", 5, SEEDED))
    wrapper = async_harness._submit_step(
        runner, "s", drafts={"s": _drafts(runner, "s")}
    )
    # "j" is admitted while the verify is still held.
    async_harness._drain(runner, "s", new=(("j", 50),))
    model.release()
    wrapper.get_output()
    async_harness._drain(runner, "s", "j")
    token, position = async_harness._tail(runner, "j")
    while len(runner.requests["s"].output_token_ids) < 20:
        wrapper = async_harness._submit_step(
            runner,
            "s",
            "j",
            drafts={"s": _drafts(runner, "s"), "j": _drafts(runner, "j")},
        )
        _finish(model, runner, wrapper, "s", "j")

    sync_runner = sync_harness._runner(SampledTarget())
    sync_harness._add_request(sync_runner, "s", SEEDED, first_token=5)
    while len(sync_runner.requests["s"].output_token_ids) < 20:
        sync_harness._step(sync_runner, "s", drafts={"s": _drafts_sync(sync_runner)})
    assert (
        runner.requests["s"].output_token_ids[:20]
        == sync_runner.requests["s"].output_token_ids[:20]
    )
    joined = runner.requests["j"].output_token_ids
    assert joined == sync_harness._base_chain(token, position, len(joined))


def test_narrow_decodes_and_held_logits_verifies_alternate_as_they_do_synchronously():
    """Host-sampled ordinary decodes between sampled verifies, deferred.

    On a model serving narrow decodes, a draftless step that follows a
    single-token commit is an ordinary decode and samples through vLLM's
    sampler on the readback thread; a drafted step is a ``logits`` verify.
    The seeded request's tokens must match the synchronous run of the same
    schedule.
    """
    params = SamplingParams(temperature=1.0, seed=37, ignore_eos=True)
    model, runner = _sampled_runner()
    runner._spec_supports_narrow_decode = True
    _admit(runner, ("s", 9, params))
    step = 0
    while len(runner.requests["s"].output_token_ids) < 30:
        drafts = {"s": _drafts(runner, "s")} if step % 3 == 0 else None
        wrapper = _submit_any(runner, "s", drafts=drafts)
        _finish(model, runner, wrapper, "s")
        step += 1
    assert model.plain_calls > 0 and model.verify_modes, "no alternation happened"

    sync_runner = sync_harness._runner(
        sync_harness.sampled_target(supports_narrow_decode=True)()
    )
    sync_harness._add_request(sync_runner, "s", params, first_token=9)
    step = 0
    while len(sync_runner.requests["s"].output_token_ids) < 30:
        drafts = {"s": _drafts_sync(sync_runner)} if step % 3 == 0 else None
        sync_harness._step(sync_runner, "s", drafts=drafts)
        step += 1
    assert (
        runner.requests["s"].output_token_ids[:30]
        == sync_runner.requests["s"].output_token_ids[:30]
    )


class _DeferredDraftingTarget(DeferredSampledTarget, sync_harness._DraftingTarget):
    """A held readback and a model drafter that checks its hidden handle."""


def test_a_model_drafter_proposes_from_a_held_logits_verify_s_hidden_handle():
    """Proposals on the asynchronous path come from the runner, not the scheduler.

    The scheduler reserves lookahead with placeholders; the runner verifies the
    proposal its drafter published at the last commit, from the hidden handle
    of the verify that was held, and the tokens match the synchronous path.
    """
    params = SamplingParams(temperature=1.0, seed=41, ignore_eos=True)
    model = _DeferredDraftingTarget()
    runner = _runner(model)
    _rebuild_batch(runner)
    runner._spec_drafts_from_model = True
    runner._spec_method = "custom_class"
    _admit(runner, ("s", 11, params))
    reservation = {"s": [-1] * DRAFT_LEN}
    while len(runner.requests["s"].output_token_ids) < 30:
        wrapper = async_harness._submit_step(runner, "s", drafts=reservation)
        _finish(model, runner, wrapper, "s")
    assert model.proposals > 0

    sync_model = sync_harness._DraftingTarget()
    sync_runner = sync_harness._drafting_runner(sync_model)
    sync_harness._add_request(sync_runner, "s", params, first_token=11)
    drafts = None
    while len(sync_runner.requests["s"].output_token_ids) < 30:
        sync_harness._step(sync_runner, "s", drafts=drafts)
        handed = sync_runner.take_draft_token_ids()
        drafts = dict(zip(handed.req_ids, handed.draft_token_ids)) if handed else None
    assert (
        runner.requests["s"].output_token_ids[:30]
        == sync_runner.requests["s"].output_token_ids[:30]
    )


# endregion More of the lifecycle, deferred
