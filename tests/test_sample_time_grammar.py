# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for the grammar-at-sample-time seam.

The non-DP/lane-DP forward runs without the grammar bitmask and defers sampling
to ``sample_tokens``; the bitmask is reordered into the live batch/lane layout
and attached at that point. These tests pin the seam that makes that safe: FIFO
pairing of each deferred forward with its grammar, the lane-vs-non-DP reorder
dispatch, and grammar being applied before sampling in every finisher. No device
execution; the index logic runs on plain tensors with fake collaborators.
Requires the ttnn-enabled environment because importing the plugin modules pulls
in ttnn.
"""

import threading
from collections import deque
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

from vllm_tt_plugin.async_decode import (
    SubmittedStepContext,
    TTAsyncDecodeController,
    TTDecodeSubmission,
)
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_input import TTModelInput, TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward

VOCAB_WORDS = 2  # int32 words per grammar bitmask row


def _model_input(**overrides) -> TTModelInput:
    """A minimal real ``TTModelInput`` so ``dataclasses.replace`` works."""
    base = dict(
        input_tokens=torch.zeros((1, 1), dtype=torch.int32),
        input_positions=torch.zeros((1,), dtype=torch.int32),
        prompt_lens=None,
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        block_tables_per_group=[torch.zeros((1, 1), dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=1,
        tt_sampling_params=object(),
        multi_modal_kwargs={},
        perform_device_sampling=False,
        grammar_bitmask=[None],
        logitsprocs_list=[None],
        bad_words_token_ids_list=[{}],
        allowed_token_ids_mask_list=[None],
        generators_list=[{}],
        max_num_logprobs=[None],
        row_req_ids=["req-0"],
    )
    base.update(overrides)
    if "grammar_row_req_ids" not in overrides:
        row_req_ids = base["row_req_ids"] or []
        batch_length = base["input_tokens"].shape[0]
        base["grammar_row_req_ids"] = tuple(row_req_ids) + (None,) * (
            batch_length - len(row_req_ids)
        )
    return TTModelInput(**base)


def _grammar(num_structured: int) -> GrammarOutput:
    bitmask = np.arange(num_structured * VOCAB_WORDS, dtype=np.int32).reshape(
        num_structured, VOCAB_WORDS
    )
    req_ids = [f"req-{i}" for i in range(num_structured)]
    return GrammarOutput(structured_output_request_ids=req_ids, grammar_bitmask=bitmask)


def _device_sampling_runner(*, supports_device_grammar: bool):
    return SimpleNamespace(
        sample_on_device_mode="all",
        seeded_sampling_policy="auto",
        supports_device_grammar=supports_device_grammar,
        num_devices=8,
        tt_data_parallel_size=1,
        supports_topk_logprobs=False,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        input_batch=SimpleNamespace(
            no_allowed_token_ids=True,
            no_penalties=True,
            sampling=SimpleNamespace(
                bad_words_token_ids={},
                has_active_logitsprocs=lambda: False,
            ),
            max_num_logprobs=None,
        ),
        model_config=SimpleNamespace(logits_processors=[]),
        model=SimpleNamespace(model_capabilities={}),
    )


# --------------------------------------------------------------------------
# sample_tokens FIFO pairing
# --------------------------------------------------------------------------


def test_sample_tokens_pops_fifo_and_passes_grammar_through():
    """Each deferred forward must sample with the grammar of its own step."""
    from collections import deque

    seen: list[tuple[str, object]] = []

    def finisher(tag, grammar_output):
        seen.append((tag, grammar_output))
        return tag

    runner = SimpleNamespace(_pending_samples=deque())
    runner._pending_samples.append(partial(finisher, "first"))
    runner._pending_samples.append(partial(finisher, "second"))

    out0 = TTModelRunner.sample_tokens(runner, "grammar-A")
    out1 = TTModelRunner.sample_tokens(runner, "grammar-B")

    assert out0 == "first"
    assert out1 == "second"
    assert seen == [("first", "grammar-A"), ("second", "grammar-B")]
    assert len(runner._pending_samples) == 0


def test_execute_model_defers_structured_device_sampling_until_sample_tokens():
    events: list[str] = []
    model_input = _model_input(
        perform_device_sampling=True,
        defer_device_sampling=True,
    )
    submission = SimpleNamespace(stage_outputs=None)
    fwd = replace(
        _sync_forward(model_input),
        perform_device_sampling=True,
        decode_submission=submission,
    )

    def build_model_input(_scheduler_output, grammar_output):
        assert grammar_output is None
        events.append("build-without-grammar")
        return model_input

    def forward(received_input):
        assert received_input.grammar_bitmask == [None]
        events.append("forward")
        return fwd

    def sample_sync(received_fwd):
        bitmask = received_fwd.model_input.grammar_bitmask[0]
        assert bitmask is not None
        events.append("device-sample")
        return [torch.tensor([[42]], dtype=torch.int32)], [None]

    async_decode = SimpleNamespace(
        can_attempt_steady_decode_from_scheduler=lambda _output: False,
        must_drain_pending_async_steps=lambda *_args: False,
        submit_async_decode=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a synchronous decode must not use async overlap")
        ),
    )
    runner = SimpleNamespace(
        async_decode=async_decode,
        async_decode_scheduling=False,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _pending_samples=deque(),
        build_model_input=build_model_input,
        _forward_with_model_input=forward,
        _sample_sync_forward=sample_sync,
        apply_and_build_runner_output=lambda sampled, _logprobs, **_kwargs: sampled,
        _num_speculative_tokens=0,
    )
    runner._reorder_grammar_bitmask = partial(
        TTModelRunner._reorder_grammar_bitmask,
        runner,
    )
    runner._apply_grammar_to_input = partial(
        TTModelRunner._apply_grammar_to_input,
        runner,
    )
    runner._finish_front_packed_sync = partial(
        TTModelRunner._finish_front_packed_sync,
        runner,
    )

    assert TTModelRunner.execute_model(runner, SimpleNamespace()) is None
    assert events == ["build-without-grammar", "forward"]

    output = TTModelRunner.sample_tokens(runner, _grammar(1))

    assert events == ["build-without-grammar", "forward", "device-sample"]
    assert output.tolist() == [[42]]


def test_device_grammar_sampling_is_capability_gated_for_decode():
    unsupported = _device_sampling_runner(supports_device_grammar=False)
    supported = _device_sampling_runner(supports_device_grammar=True)

    assert not TTModelRunner.check_perform_device_sampling(
        unsupported,
        is_decode=True,
        has_structured_outputs=True,
    )
    assert TTModelRunner.check_perform_device_sampling(
        supported,
        is_decode=True,
        has_structured_outputs=True,
    )


def test_structured_prefill_remains_host_sampled_with_device_grammar_capability():
    runner = _device_sampling_runner(supports_device_grammar=True)

    assert not TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=False,
        has_structured_outputs=True,
    )


def test_structured_decode_samples_on_device_with_async_scheduling():
    runner = _device_sampling_runner(supports_device_grammar=True)
    runner.scheduler_config.async_scheduling = True

    assert TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=True,
    )


def test_structured_device_logprobs_remain_host_sampled():
    runner = _device_sampling_runner(supports_device_grammar=True)
    runner.input_batch.max_num_logprobs = 0

    assert not TTModelRunner.check_perform_device_sampling(
        runner,
        is_decode=True,
        has_structured_outputs=True,
    )


def test_model_load_downgrades_device_grammar_for_incompatible_runtime(
    monkeypatch,
):
    class Loader:
        def __init__(self, _load_config):
            pass

        def load_model(self, **_kwargs):
            return SimpleNamespace(
                device_grammar_enabled=False,
                sample_deferred_decode=lambda *_args, **_kwargs: None,
            )

    monkeypatch.setattr(
        "vllm_tt_plugin.model_runner.TTModelLoader",
        Loader,
    )
    runner = SimpleNamespace(
        load_config=object(),
        vllm_config=object(),
        model_config=object(),
        supports_device_grammar=True,
        _spec_supports_narrow_decode=False,
    )

    TTModelRunner.load_model(runner)

    assert runner.supports_device_grammar is False


def test_model_load_activates_device_grammar_runtime(monkeypatch):
    events = []

    class Model:
        device_grammar_enabled = False

        def enable_device_grammar(self):
            events.append("activate")
            self.device_grammar_enabled = True

        def sample_deferred_decode(self, *_args, **_kwargs):
            return None

    class Loader:
        def __init__(self, _load_config):
            pass

        def load_model(self, **_kwargs):
            return Model()

    monkeypatch.setattr(
        "vllm_tt_plugin.model_runner.TTModelLoader",
        Loader,
    )
    runner = SimpleNamespace(
        load_config=object(),
        vllm_config=object(),
        model_config=object(),
        supports_device_grammar=True,
        _spec_supports_narrow_decode=False,
    )

    TTModelRunner.load_model(runner)

    assert events == ["activate"]
    assert runner.supports_device_grammar is True


def test_runner_poison_rejects_all_later_work():
    runner = SimpleNamespace(_device_grammar_poisoned=False)

    TTModelRunner._poison_device_grammar(runner)

    assert runner._device_grammar_poisoned is True
    with pytest.raises(RuntimeError, match="reconstruct the model runner"):
        TTModelRunner._raise_if_device_grammar_poisoned(runner)


def test_front_packed_grammar_remap_failure_poisons_runner():
    poisoned = []
    runner = SimpleNamespace(
        _apply_grammar_to_input=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("missing grammar row")
        ),
        _poison_device_grammar=lambda: poisoned.append(True),
    )
    fwd = _sync_forward(
        _model_input(
            perform_device_sampling=True,
            defer_device_sampling=True,
            structured_output_req_ids=frozenset({"req-0"}),
        )
    )

    with pytest.raises(RuntimeError, match="missing grammar row"):
        TTModelRunner._finish_front_packed_sync(
            runner,
            _grammar(1),
            fwd=fwd,
        )

    assert poisoned == [True]


# --------------------------------------------------------------------------
# _reorder_grammar_bitmask dispatch (lane vs non-DP vs none)
# --------------------------------------------------------------------------


def test_reorder_grammar_bitmask_none_returns_none():
    runner = SimpleNamespace()
    result = TTModelRunner._reorder_grammar_bitmask(
        runner, None, _model_input(), lane_total=None
    )
    assert result is None


def test_reorder_grammar_bitmask_requires_mask_for_structured_forward():
    runner = SimpleNamespace()
    model_input = _model_input(structured_output_req_ids=frozenset({"req-0"}))

    with pytest.raises(RuntimeError, match="grammar output is absent.*req-0"):
        TTModelRunner._reorder_grammar_bitmask(
            runner,
            None,
            model_input,
            lane_total=None,
        )


def test_reorder_grammar_bitmask_lane_path_uses_submitted_row_snapshot():
    runner = SimpleNamespace(
        lane_batch=SimpleNamespace(
            slot_grammar_bitmask=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("sample-time mapping must not read mutable lane state")
            )
        )
    )
    grammar = GrammarOutput(
        structured_output_request_ids=["req-1"],
        grammar_bitmask=np.array([[7, 8]], dtype=np.int32),
    )
    model_input = _model_input(
        input_tokens=torch.zeros((3, 1), dtype=torch.int32),
        row_req_ids=None,
        grammar_row_req_ids=("req-0", None, "req-1"),
        structured_output_req_ids=frozenset({"req-1"}),
    )

    result = TTModelRunner._reorder_grammar_bitmask(
        runner,
        grammar,
        model_input,
        lane_total=3,
    )

    assert result.tolist() == [[-1, -1], [-1, -1], [7, 8]]


def test_reorder_grammar_bitmask_non_dp_path_uses_front_packed_reorder():
    """Non-DP (``lane_total=None``) reorders against the build's own rows."""
    # Two requests, only req-1 structured; batch_length comes from input_tokens.
    runner = SimpleNamespace()
    grammar = GrammarOutput(
        structured_output_request_ids=["req-1"],
        grammar_bitmask=np.array([[5, 6]], dtype=np.int32),
    )
    model_input = _model_input(
        input_tokens=torch.zeros((2, 1), dtype=torch.int32),
        row_req_ids=["req-0", "req-1"],
        structured_output_req_ids=frozenset({"req-1"}),
    )

    result = TTModelRunner._reorder_grammar_bitmask(
        runner, grammar, model_input, lane_total=None
    )

    assert result.shape == (2, VOCAB_WORDS)
    # req-1 (forward row 1) gets its scheduler bitmask row; req-0 is left
    # all-ones (-1 in int32 = every token allowed).
    assert result[1].tolist() == [5, 6]
    assert torch.all(result[0] == -1)


def test_reorder_grammar_bitmask_non_dp_path_ignores_filtered_rows():
    """A prefill build drops mixed-in decode rows; the reorder must follow the
    forward's rows, not the persistent batch's.
    """
    # Persistent batch req-0..req-3, forward kept req-0 and req-2. Under a
    # ``range(num_reqs)`` reorder req-2 lands past the end of a 2-row tensor.
    runner = SimpleNamespace()
    grammar = GrammarOutput(
        structured_output_request_ids=["req-1", "req-2"],
        grammar_bitmask=np.array([[1, 2], [5, 6]], dtype=np.int32),
    )
    model_input = _model_input(
        input_tokens=torch.zeros((2, 8), dtype=torch.int32),
        row_req_ids=["req-0", "req-2"],
        structured_output_req_ids=frozenset({"req-2"}),
    )

    result = TTModelRunner._reorder_grammar_bitmask(
        runner, grammar, model_input, lane_total=None
    )

    assert result.shape == (2, VOCAB_WORDS)
    assert torch.all(result[0] == -1)
    # req-2's own mask, not req-1's, which the forward never ran.
    assert result[1].tolist() == [5, 6]


# --------------------------------------------------------------------------
# _apply_grammar_to_input (attach only when a bitmask exists)
# --------------------------------------------------------------------------


def test_apply_grammar_to_input_returns_input_unchanged_when_no_bitmask():
    runner = SimpleNamespace(_reorder_grammar_bitmask=lambda *a, **k: None)
    model_input = _model_input()

    result = TTModelRunner._apply_grammar_to_input(
        runner, model_input, None, lane_total=None
    )

    assert result is model_input


def test_apply_grammar_to_input_wraps_bitmask_in_single_element_list():
    bitmask = torch.tensor([[1, 2]], dtype=torch.int32)
    runner = SimpleNamespace(_reorder_grammar_bitmask=lambda *a, **k: bitmask)
    model_input = _model_input()

    result = TTModelRunner._apply_grammar_to_input(
        runner, model_input, _grammar(1), lane_total=None
    )

    assert result is not model_input  # replace() returns a new frozen instance
    assert len(result.grammar_bitmask) == 1
    assert torch.equal(result.grammar_bitmask[0], bitmask)


# --------------------------------------------------------------------------
# _finish_async_decode (reorder on engine thread, attach to wrapper)
# --------------------------------------------------------------------------


def test_finish_async_decode_skips_bitmask_when_grammar_absent():
    calls: list = []
    wrapper = SimpleNamespace(set_grammar_bitmask=lambda bm: calls.append(bm))
    runner = SimpleNamespace(_reorder_grammar_bitmask=lambda *a, **k: None)

    result = TTModelRunner._finish_async_decode(
        runner, None, wrapper=wrapper, model_input=_model_input(), lane_total=32
    )

    assert result is wrapper
    assert calls == []  # no bitmask set on the wrapper


def test_finish_async_decode_sets_bitmask_when_grammar_present():
    calls: list = []
    bitmask = torch.tensor([[3, 4]], dtype=torch.int32)
    wrapper = SimpleNamespace(set_grammar_bitmask=lambda bm: calls.append(bm))
    runner = SimpleNamespace(_reorder_grammar_bitmask=lambda *a, **k: bitmask)

    result = TTModelRunner._finish_async_decode(
        runner, _grammar(1), wrapper=wrapper, model_input=_model_input(), lane_total=32
    )

    assert result is wrapper
    assert len(calls) == 1 and torch.equal(calls[0], bitmask)


def test_finish_async_decode_allows_missing_stale_grammar_rows():
    calls: list[torch.Tensor] = []
    wrapper = SimpleNamespace(set_grammar_bitmask=lambda bm: calls.append(bm))
    runner = SimpleNamespace()
    runner._reorder_grammar_bitmask = partial(
        TTModelRunner._reorder_grammar_bitmask,
        runner,
    )
    model_input = _model_input(
        input_tokens=torch.zeros((2, 1), dtype=torch.int32),
        row_req_ids=["stale", "active"],
        grammar_row_req_ids=("stale", "active"),
        structured_output_req_ids=frozenset({"stale", "active"}),
    )
    grammar = GrammarOutput(
        structured_output_request_ids=["active"],
        grammar_bitmask=np.array([[7, 8]], dtype=np.int32),
    )

    result = TTModelRunner._finish_async_decode(
        runner,
        grammar,
        wrapper=wrapper,
        model_input=model_input,
        lane_total=None,
    )

    assert result is wrapper
    assert calls[0].tolist() == [[-1, -1], [7, 8]]


def test_finish_async_decode_allows_all_structured_rows_to_be_stale():
    calls: list = []
    wrapper = SimpleNamespace(set_grammar_bitmask=lambda bm: calls.append(bm))
    runner = SimpleNamespace()
    runner._reorder_grammar_bitmask = partial(
        TTModelRunner._reorder_grammar_bitmask,
        runner,
    )
    model_input = _model_input(
        structured_output_req_ids=frozenset({"stale"}),
    )

    result = TTModelRunner._finish_async_decode(
        runner,
        None,
        wrapper=wrapper,
        model_input=model_input,
        lane_total=None,
    )

    assert result is wrapper
    assert calls == []


# --------------------------------------------------------------------------
# _finish_front_packed_sync / _finish_lane_sync (grammar before sampling)
# --------------------------------------------------------------------------


def _sync_forward(model_input: TTModelInput) -> _SyncForward:
    return _SyncForward(
        tt_out=object(),
        tt_log_probs=None,
        sampling_params=object(),
        model_input=model_input,
        batch_size_per_dp=[1],
        perform_device_sampling=False,
        is_decode=True,
    )


def test_sample_sync_forward_consumes_deferred_grammar_before_extraction():
    bitmask = torch.tensor([[3, 4]], dtype=torch.int32)
    model_input = _model_input(
        perform_device_sampling=True,
        grammar_bitmask=[bitmask],
        defer_device_sampling=True,
    )
    submission = object()
    seen: dict = {}

    def complete_deferred(received_submission, received_input):
        seen["submission"] = received_submission
        seen["bitmask"] = received_input.grammar_bitmask[0]
        return SimpleNamespace(tt_out=torch.tensor([[42]]), tt_log_probs=None)

    def get_output_tokens(**kwargs):
        seen["extraction_input"] = kwargs["model_input"]
        return [kwargs["tt_out"]], [None]

    runner = SimpleNamespace(
        async_decode=SimpleNamespace(
            complete_deferred_device_sampling=complete_deferred
        ),
        _get_output_tokens=get_output_tokens,
    )
    fwd = replace(
        _sync_forward(model_input),
        perform_device_sampling=True,
        decode_submission=submission,
    )

    sampled, _ = TTModelRunner._sample_sync_forward(runner, fwd)

    assert seen["submission"] is submission
    assert seen["bitmask"] is bitmask
    assert seen["extraction_input"].grammar_bitmask == [None]
    assert seen["extraction_input"].defer_device_sampling is False
    assert sampled[0].tolist() == [[42]]


def test_deferred_device_sampling_transports_sample_time_grammar():
    calls: list[tuple] = []
    device_params = object()
    raw_logits = object()
    sampled = torch.tensor([[17]], dtype=torch.int32)

    def sample_deferred_decode(tt_out, *, sampling_params, grammar_bitmask):
        calls.append((tt_out, sampling_params, grammar_bitmask))
        return sampled

    runner = SimpleNamespace(
        model=SimpleNamespace(sample_deferred_decode=sample_deferred_decode)
    )
    controller = TTAsyncDecodeController(runner)
    sampling_params = SimpleNamespace(enable_log_probs=torch.tensor([False]))
    submission = TTDecodeSubmission(
        tt_out=raw_logits,
        read_events=None,
        batch_size_per_dp=[1],
        sampling_params=sampling_params,
        perform_device_sampling=True,
        device_sampling_params=device_params,
        deferred_device_sampling=True,
    )
    bitmask = torch.tensor([[5, 6]], dtype=torch.int32)
    model_input = _model_input(
        perform_device_sampling=True,
        grammar_bitmask=[bitmask],
        defer_device_sampling=True,
    )

    finalized = controller.complete_deferred_device_sampling(
        submission,
        model_input,
    )

    assert calls == [(raw_logits, device_params, bitmask)]
    assert finalized.tt_out is sampled
    assert controller.device_grammar_sample_count == 1


def test_deferred_device_sampling_rejects_missing_sample_time_grammar():
    poisoned = []
    runner = SimpleNamespace(
        model=SimpleNamespace(
            sample_deferred_decode=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("device sampler must not run without grammar")
            )
        ),
        _poison_device_grammar=lambda: poisoned.append(True),
    )
    controller = TTAsyncDecodeController(runner)
    controller._decode_chain_valid = True
    controller._submitted_page_tables = (torch.zeros((1, 1)),)
    submission = TTDecodeSubmission(
        tt_out=object(),
        read_events=None,
        batch_size_per_dp=[1],
        sampling_params=SimpleNamespace(enable_log_probs=torch.tensor([False])),
        perform_device_sampling=True,
        device_sampling_params=object(),
        deferred_device_sampling=True,
    )

    with pytest.raises(RuntimeError, match="without a sample-time grammar"):
        controller.complete_deferred_device_sampling(
            submission,
            _model_input(
                perform_device_sampling=True,
                grammar_bitmask=[None],
                defer_device_sampling=True,
            ),
        )

    assert controller._decode_chain_valid is False
    assert controller._submitted_page_tables is None
    assert poisoned == [True]


def test_deferred_device_sampling_no_output_invalidates_decode_chain():
    poisoned = []
    runner = SimpleNamespace(
        model=SimpleNamespace(sample_deferred_decode=lambda *_args, **_kwargs: None),
        _poison_device_grammar=lambda: poisoned.append(True),
    )
    controller = TTAsyncDecodeController(runner)
    controller._decode_chain_valid = True
    controller._submitted_page_tables = (torch.zeros((1, 1)),)
    submission = TTDecodeSubmission(
        tt_out=object(),
        read_events=None,
        batch_size_per_dp=[1],
        sampling_params=SimpleNamespace(enable_log_probs=torch.tensor([False])),
        perform_device_sampling=True,
        device_sampling_params=object(),
        deferred_device_sampling=True,
    )

    with pytest.raises(RuntimeError, match="produced no output"):
        controller.complete_deferred_device_sampling(
            submission,
            _model_input(
                perform_device_sampling=True,
                grammar_bitmask=[torch.full((1, VOCAB_WORDS), -1, dtype=torch.int32)],
                defer_device_sampling=True,
            ),
        )

    assert controller._decode_chain_valid is False
    assert controller._submitted_page_tables is None
    assert poisoned == [True]


def test_finish_front_packed_sync_applies_grammar_before_sampling():
    order: list[str] = []

    def apply_grammar(model_input, grammar_output, *, lane_total):
        order.append("grammar")
        assert lane_total is None
        return replace(model_input, decode_layout_changed=True)  # mark it was applied

    def sample_sync(fwd):
        order.append("sample")
        # Grammar must already be on the input the sampler runs against.
        assert fwd.model_input.decode_layout_changed is True
        return [torch.tensor([[42]], dtype=torch.int32)], []

    def build_output(sampled, logprobs, **kwargs):
        order.append("build")
        return SimpleNamespace(sampled=sampled, logprobs=logprobs)

    runner = SimpleNamespace(
        _apply_grammar_to_input=apply_grammar,
        _sample_sync_forward=sample_sync,
        apply_and_build_runner_output=build_output,
        # This launch does not speculate, so the decode tail owes no proposal.
        _num_speculative_tokens=0,
    )
    fwd = _sync_forward(_model_input())

    output = TTModelRunner._finish_front_packed_sync(runner, _grammar(1), fwd=fwd)

    assert order == ["grammar", "sample", "build"]
    assert output.sampled.tolist() == [[42]]


def test_v0_sync_decode_defers_state_under_upstream_async_scheduler():
    order: list[str] = []

    def apply_grammar(model_input, grammar_output, *, lane_total):
        order.append("grammar")
        return model_input

    def sample_sync(fwd):
        order.append("sample")
        return [torch.tensor([[42]], dtype=torch.int32)], []

    def defer_output(sampled, logprobs, **kwargs):
        order.append("defer")
        assert kwargs["req_ids"] == ["req-0"]
        return SimpleNamespace(sampled=sampled)

    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        # Contract-v0 standard DP disables device async decode, but the engine
        # still uses AsyncScheduler placeholder accounting.
        async_decode_scheduling=False,
        input_batch=SimpleNamespace(req_ids=["req-0"], num_reqs=1),
        _apply_grammar_to_input=apply_grammar,
        _sample_sync_forward=sample_sync,
        defer_state_apply_and_build_runner_output=defer_output,
        apply_and_build_runner_output=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("async scheduler output must defer runner state")
        ),
    )

    output = TTModelRunner._finish_front_packed_sync(
        runner, _grammar(1), fwd=_sync_forward(_model_input())
    )

    assert order == ["grammar", "sample", "defer"]
    assert output.sampled.tolist() == [[42]]


def test_finish_front_packed_sync_with_no_forward_builds_empty_output():
    captured: dict = {}

    def build_output(sampled, logprobs, **kwargs):
        captured["sampled"] = sampled
        captured["logprobs"] = logprobs
        return "empty"

    runner = SimpleNamespace(apply_and_build_runner_output=build_output)

    output = TTModelRunner._finish_front_packed_sync(runner, None, fwd=None)

    assert output == "empty"
    assert captured["sampled"].numel() == 0
    assert captured["logprobs"] is None


def test_finish_lane_sync_builds_req_ids_in_scheduled_row_order():
    order: list[str] = []

    def apply_grammar(model_input, grammar_output, *, lane_total):
        order.append("grammar")
        assert lane_total == 32
        return model_input

    def extract_output(
        runner, tt_out, tt_log_probs, model_input, scheduled_rows, *, is_decode
    ):
        order.append("extract")
        return torch.tensor([[11], [12]], dtype=torch.int32), None

    captured: dict = {}

    def build_output(sampled, logprobs, *, req_ids):
        order.append("build")
        captured["req_ids"] = req_ids
        return SimpleNamespace(sampled=sampled)

    # Row index -> req_id; scheduled_rows pick a subset out of slot order.
    lane_req_ids = {4: "req-d", 1: "req-a"}
    runner = SimpleNamespace(
        _apply_grammar_to_input=apply_grammar,
        lane_batch=SimpleNamespace(extract_output=extract_output, req_ids=lane_req_ids),
        apply_and_build_runner_output=build_output,
    )

    output = TTModelRunner._finish_lane_sync(
        runner,
        _grammar(1),
        tt_out=object(),
        tt_log_probs=None,
        model_input=_model_input(
            grammar_row_req_ids=(None, "req-a", None, None, "req-d"),
        ),
        scheduled_rows=[4, 1],
        is_decode=True,
        lane_total=32,
    )

    assert order == ["grammar", "extract", "build"]
    # req_ids follow scheduled_rows order, not slot order.
    assert captured["req_ids"] == ["req-d", "req-a"]
    assert output.sampled.tolist() == [[11], [12]]


def test_finish_lane_sync_consumes_device_grammar_before_slot_extraction():
    bitmask = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    submission = object()
    seen: dict = {}

    def complete_deferred(received_submission, received_input):
        seen["submission"] = received_submission
        seen["bitmask"] = received_input.grammar_bitmask[0]
        return SimpleNamespace(
            tt_out=torch.tensor([11, 12], dtype=torch.int32),
            tt_log_probs=None,
        )

    def extract_output(
        _runner,
        tt_out,
        _tt_log_probs,
        model_input,
        _scheduled_rows,
        *,
        is_decode,
    ):
        assert is_decode
        assert model_input.grammar_bitmask == [None]
        assert model_input.defer_device_sampling is False
        return tt_out.reshape(2, 1), None

    runner = SimpleNamespace(
        _apply_grammar_to_input=lambda model_input, _grammar_output, **_kwargs: (
            replace(model_input, grammar_bitmask=[bitmask])
        ),
        async_decode=SimpleNamespace(
            complete_deferred_device_sampling=complete_deferred
        ),
        lane_batch=SimpleNamespace(
            extract_output=extract_output,
            req_ids={0: "req-a", 1: "req-b"},
        ),
        apply_and_build_runner_output=lambda sampled, _logprobs, **_kwargs: sampled,
    )

    output = TTModelRunner._finish_lane_sync(
        runner,
        _grammar(2),
        tt_out=object(),
        tt_log_probs=None,
        model_input=_model_input(
            perform_device_sampling=True,
            defer_device_sampling=True,
            grammar_row_req_ids=("req-a", "req-b"),
        ),
        scheduled_rows=[0, 1],
        is_decode=True,
        lane_total=2,
        decode_submission=submission,
    )

    assert seen["submission"] is submission
    assert seen["bitmask"] is bitmask
    assert output.tolist() == [[11], [12]]


# --------------------------------------------------------------------------
# Device grammar under asynchronous scheduling
# --------------------------------------------------------------------------


def _only(token: int) -> torch.Tensor:
    """A one-row packed mask allowing ``token`` alone."""
    row = torch.zeros((1, VOCAB_WORDS), dtype=torch.int32)
    row[0, token // 32] = 1 << (token % 32)
    return row


def _allowed(bitmask: torch.Tensor, row: int) -> list[int]:
    return [
        token
        for token in range(VOCAB_WORDS * 32)
        if (int(bitmask[row, token // 32]) >> (token % 32)) & 1
    ]


class _ResidentGrammarModel:
    """A device grammar adapter reduced to what the async overlap relies on.

    Like a traced adapter it keeps the fed-back token on device: a step that
    does not reload its inputs decodes from the token the last sample wrote,
    whatever the host still holds. Its forward prefers token ``input + 1``;
    its sampler takes that token when the row's mask allows it, else the
    lowest allowed one. It holds one deferred forward at a time, as
    DEVICE_GRAMMAR_CONTRACT.md requires.
    """

    decode_input_update_contract = 1
    model_capabilities = {"supports_async_decode": True}

    def __init__(self):
        self.resident: torch.Tensor | None = None
        self.pending = None
        self.reload_inputs: list[bool] = []

    def decode_forward(self, *, tokens, reload_inputs, read_from_device, **kwargs):
        if self.pending is not None:
            raise RuntimeError("a deferred decode step is pending")
        assert kwargs["defer_device_sampling"] is True
        assert read_from_device is False
        self.reload_inputs.append(reload_inputs)
        source = tokens if reload_inputs else self.resident
        self.pending = SimpleNamespace(preferred=source.reshape(-1) + 1)
        return self.pending

    def sample_deferred_decode(self, payload, *, sampling_params, grammar_bitmask):
        assert payload is self.pending
        self.pending = None
        tokens = []
        for row, preferred in enumerate(payload.preferred.tolist()):
            allowed = _allowed(grammar_bitmask, row)
            tokens.append(preferred if preferred in allowed else allowed[0])
        sampled = torch.tensor(tokens, dtype=torch.int32)
        self.resident = sampled.reshape(-1, 1)
        return {"device": sampled}

    def read_decode_output(self, out, async_read=False):
        assert async_read
        return {"host": out["device"].clone()}, []

    def process_decode_output_host(self, out, is_tokens=False):
        assert is_tokens
        return out["host"]


def _async_grammar_controller(model) -> TTAsyncDecodeController:
    runner = SimpleNamespace(
        model=model,
        trace_mode="decode_only",
        async_decode_scheduling=True,
        kv_caches=object(),
        request_specific_rope=False,
        supports_stage_outputs=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
        _steady_decode_lock=threading.Lock(),
        _pending_async_steps=deque(),
        _pending_async_overlap_ok=deque(),
        _completed_decode_steps=deque(),
        _get_output_tokens=lambda **kwargs: (
            [kwargs["tt_out"].reshape(-1, 1)],
            [None],
        ),
        _build_runner_output=lambda **kwargs: SimpleNamespace(
            sampled_token_ids=kwargs["sampled_token_ids"]
        ),
    )
    return TTAsyncDecodeController(runner)


def _deferred_step_input(token: int, *, layout_changed: bool) -> TTModelInput:
    return _model_input(
        input_tokens=torch.tensor([[token]], dtype=torch.int32),
        tt_sampling_params=TTSamplingParams(
            temperature=torch.zeros(1),
            top_k=torch.ones(1, dtype=torch.int32),
            top_p=torch.ones(1),
            presence_penalty=torch.zeros(1),
            frequency_penalty=torch.zeros(1),
            repetition_penalty=torch.ones(1),
            seed=torch.full((1,), -1),
            num_logprobs=torch.full((1,), -2),
            enable_log_probs=torch.zeros(1, dtype=torch.bool),
        ),
        perform_device_sampling=True,
        defer_device_sampling=True,
        decode_layout_changed=layout_changed,
        structured_output_req_ids=frozenset({"req-0"}),
    )


def _context() -> SubmittedStepContext:
    return SubmittedStepContext(
        req_ids=["req-0"], req_id_to_index={"req-0": 0}, submit_time_ns=0
    )


def test_async_device_grammar_step_decodes_from_the_masked_token_it_overlaps():
    """The next forward is submitted before this step's tokens are read, and
    decodes from the token the grammar mask chose, not the host's or the
    unmasked forward's."""
    model = _ResidentGrammarModel()
    controller = _async_grammar_controller(model)

    first_input = _deferred_step_input(10, layout_changed=True)
    first = controller.submit_deferred_async_decode(first_input, _context())
    first_step = controller.sample_deferred_async_decode(
        first, replace(first_input, grammar_bitmask=[_only(3)])
    )
    # The host still holds token 10: the step is in flight.
    second_input = _deferred_step_input(10, layout_changed=False)
    second = controller.submit_deferred_async_decode(second_input, _context())
    assert not first_step.is_resolved()
    second_step = controller.sample_deferred_async_decode(
        second,
        replace(
            second_input,
            grammar_bitmask=[torch.full((1, VOCAB_WORDS), -1, dtype=torch.int32)],
        ),
    )

    assert model.reload_inputs == [True, False]
    assert (first.overlap_ok, second.overlap_ok) == (False, True)
    assert list(controller.runner._pending_async_overlap_ok) == [False, True]
    assert first_step.get_output().sampled_token_ids.tolist() == [[3]]
    assert second_step.get_output().sampled_token_ids.tolist() == [[4]]
    assert controller.device_grammar_sample_count == 2


def test_async_device_grammar_sample_failure_is_terminal():
    poisoned = []
    model = _ResidentGrammarModel()
    controller = _async_grammar_controller(model)
    controller.runner._poison_device_grammar = lambda: poisoned.append(True)
    model.read_decode_output = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("read failed")
    )
    step_input = _deferred_step_input(10, layout_changed=True)
    deferred = controller.submit_deferred_async_decode(step_input, _context())

    with pytest.raises(RuntimeError, match="read failed"):
        controller.sample_deferred_async_decode(
            deferred, replace(step_input, grammar_bitmask=[_only(3)])
        )

    assert poisoned == [True]
    assert controller._decode_chain_valid is False
    assert not controller.runner._pending_async_steps


def _async_deferred_runner(events: list, model_input: TTModelInput):
    """A front-packed runner under async scheduling whose controller records
    the deferred submission and sampling it is asked for."""

    def submit_deferred(received_input, context, scheduled_rows=None):
        events.append(("submit", received_input, scheduled_rows))
        return "deferred-step"

    def sample_deferred(deferred, received_input):
        events.append(("sample", deferred, received_input.grammar_bitmask[0]))
        return "async-output"

    def fail(name):
        return lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"a deferred async decode must not call {name}")
        )

    runner = SimpleNamespace(
        async_decode=SimpleNamespace(
            can_attempt_steady_decode_from_scheduler=lambda _output: True,
            must_drain_pending_async_steps=lambda *_args: False,
            wait_for_all_pending_async_steps=fail("wait_for_all_pending_async_steps"),
            submit_async_decode=fail("submit_async_decode"),
            capture_submitted_step_context=lambda: "context",
            submit_deferred_async_decode=submit_deferred,
            sample_deferred_async_decode=sample_deferred,
        ),
        async_decode_scheduling=True,
        _pending_samples=deque(),
        build_model_input=lambda _output, _grammar: model_input,
        _forward_with_model_input=fail("_forward_with_model_input"),
        bitmask_size=VOCAB_WORDS,
        _poison_device_grammar=lambda: events.append(("poison",)),
    )
    runner._reorder_grammar_bitmask = partial(
        TTModelRunner._reorder_grammar_bitmask, runner
    )
    runner._finish_async_deferred_decode = partial(
        TTModelRunner._finish_async_deferred_decode, runner
    )
    return runner


def test_execute_model_submits_async_deferred_decode_without_draining_or_reading():
    events: list = []
    model_input = _model_input(
        perform_device_sampling=True,
        defer_device_sampling=True,
        structured_output_req_ids=frozenset({"req-0"}),
    )
    runner = _async_deferred_runner(events, model_input)

    assert TTModelRunner.execute_model(runner, SimpleNamespace()) is None
    assert events == [("submit", model_input, None)]

    output = TTModelRunner.sample_tokens(runner, _grammar(1))

    assert output == "async-output"
    assert events[1][:2] == ("sample", "deferred-step")
    assert events[1][2].tolist() == [[0, 1]]


def test_async_deferred_decode_samples_unconstrained_when_its_grammar_finished():
    """vLLM builds the mask after the previous step's output, which may have
    finished every structured request this step still carries: the step is
    sampled with an all-allowed mask and its late tokens are discarded."""
    events: list = []
    model_input = _model_input(
        input_tokens=torch.zeros((2, 1), dtype=torch.int32),
        row_req_ids=["req-0", "req-1"],
        perform_device_sampling=True,
        defer_device_sampling=True,
        structured_output_req_ids=frozenset({"req-0", "req-1"}),
    )
    runner = _async_deferred_runner(events, model_input)
    TTModelRunner.execute_model(runner, SimpleNamespace())

    TTModelRunner.sample_tokens(runner, None)

    assert events[1][2].tolist() == [[-1, -1], [-1, -1]]


def test_async_deferred_decode_leaves_a_finished_structured_row_unconstrained():
    events: list = []
    model_input = _model_input(
        input_tokens=torch.zeros((2, 1), dtype=torch.int32),
        row_req_ids=["req-0", "req-1"],
        perform_device_sampling=True,
        defer_device_sampling=True,
        structured_output_req_ids=frozenset({"req-0", "req-1"}),
    )
    runner = _async_deferred_runner(events, model_input)
    TTModelRunner.execute_model(runner, SimpleNamespace())
    grammar = GrammarOutput(
        structured_output_request_ids=["req-1"],
        grammar_bitmask=np.array([[7, 8]], dtype=np.int32),
    )

    TTModelRunner.sample_tokens(runner, grammar)

    assert events[1][2].tolist() == [[-1, -1], [7, 8]]


def test_async_deferred_decode_failure_poisons_runner():
    events: list = []
    model_input = _model_input(
        perform_device_sampling=True,
        defer_device_sampling=True,
        structured_output_req_ids=frozenset({"req-0"}),
    )
    runner = _async_deferred_runner(events, model_input)

    def sample_deferred(*_args):
        raise RuntimeError("device sampler failed")

    runner.async_decode.sample_deferred_async_decode = sample_deferred
    TTModelRunner.execute_model(runner, SimpleNamespace())

    with pytest.raises(RuntimeError, match="device sampler failed"):
        TTModelRunner.sample_tokens(runner, _grammar(1))

    assert events[-1] == ("poison",)


def test_lane_step_submits_async_deferred_decode_with_its_slot_layout():
    events: list = []
    plan = TTStepPlan(
        is_decode=True,
        capacity=3,
        scheduled_req_ids=("req-c", "req-a"),
        scheduled_rows=(2, 0),
        input_rows=(0, 1, 2),
        req_id_to_row={"req-a": 0, "req-c": 2},
        batch_size_per_dp=(3,),
        prefill_empty_slots=None,
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.total_num_scheduled_tokens = 2
    scheduler_output.num_scheduled_tokens = {"req-c": 1, "req-a": 1}
    scheduler_output._tt_step_state = SimpleNamespace(plan=plan)
    model_input = _model_input(
        input_tokens=torch.zeros((3, 1), dtype=torch.int32),
        row_req_ids=["req-a", None, "req-c"],
        grammar_row_req_ids=("req-a", None, "req-c"),
        perform_device_sampling=True,
        defer_device_sampling=True,
        structured_output_req_ids=frozenset({"req-a"}),
    )
    runner = _async_deferred_runner(events, model_input)
    runner.request_specific_rope = False
    runner.requests = {}
    runner.encoder_cache = {}
    runner.lane_batch = SimpleNamespace(
        apply_step_plan=lambda *_args: False,
        build_model_input=lambda *_args: model_input,
    )
    runner.async_decode.can_attempt_steady_lane_decode_from_scheduler = (
        lambda *_args, **_kwargs: True
    )
    runner.async_decode.apply_completed_decode_steps_before_build = lambda _o: None
    runner.async_decode.capture_submitted_step_context = lambda req_ids: req_ids
    contexts: list = []

    def submit_deferred(received_input, context, scheduled_rows=None):
        contexts.append(context)
        events.append(("submit", received_input, scheduled_rows))
        return "deferred-step"

    runner.async_decode.submit_deferred_async_decode = submit_deferred

    assert TTModelRunner._execute_lane_step(runner, scheduler_output) is None
    assert contexts == [["req-c", "req-a"]]
    assert events == [("submit", model_input, [2, 0])]

    grammar = GrammarOutput(
        structured_output_request_ids=["req-a"],
        grammar_bitmask=np.array([[5, 6]], dtype=np.int32),
    )
    assert TTModelRunner.sample_tokens(runner, grammar) == "async-output"
    assert events[1][2].tolist() == [[5, 6], [-1, -1], [-1, -1]]
