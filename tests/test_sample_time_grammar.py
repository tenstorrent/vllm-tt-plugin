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

from collections import deque
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import GrammarOutput

from vllm_tt_plugin.async_decode import TTAsyncDecodeController, TTDecodeSubmission
from vllm_tt_plugin.model_input import TTModelInput
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.platform import TTPlatform

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
        supports_device_grammar=supports_device_grammar,
        num_devices=8,
        tt_data_parallel_size=1,
        supports_topk_logprobs=False,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        input_batch=SimpleNamespace(
            no_allowed_token_ids=True,
            sampling=SimpleNamespace(
                bad_words_token_ids={},
                has_active_logitsprocs=lambda: False,
            ),
            max_num_logprobs=None,
        ),
        model_config=SimpleNamespace(logits_processors=[]),
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
    submission = object()
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
        wait_for_all_pending_async_steps=lambda: events.append("drain"),
        submit_async_decode=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("structured device sampling must not use async overlap")
        ),
    )
    runner = SimpleNamespace(
        async_decode=async_decode,
        async_decode_scheduling=True,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _pending_samples=deque(),
        build_model_input=build_model_input,
        _forward_with_model_input=forward,
        _sample_sync_forward=sample_sync,
        apply_and_build_runner_output=lambda sampled, _logprobs, **_kwargs: sampled,
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
    assert events == ["build-without-grammar", "drain", "forward"]

    output = TTModelRunner.sample_tokens(runner, _grammar(1))

    assert events == [
        "build-without-grammar",
        "drain",
        "forward",
        "device-sample",
    ]
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


def test_structured_decode_remains_host_sampled_with_async_scheduling():
    runner = _device_sampling_runner(supports_device_grammar=True)
    runner.scheduler_config.async_scheduling = True

    assert not TTModelRunner.check_perform_device_sampling(
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
                sample_decode_on_device=lambda *_args, **_kwargs: None,
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

        def sample_decode_on_device(self, *_args, **_kwargs):
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
    )

    TTModelRunner.load_model(runner)

    assert events == ["activate"]
    assert runner.supports_device_grammar is True


def test_runner_poison_rejects_all_later_work():
    model_poisoned = []
    runner = SimpleNamespace(
        model=SimpleNamespace(
            poison_deferred_device_sampling=lambda: model_poisoned.append(True)
        ),
        _device_grammar_poisoned=False,
    )

    TTModelRunner._poison_device_grammar(runner)

    assert runner._device_grammar_poisoned is True
    assert model_poisoned == [True]
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


def test_trace_all_warmup_prepares_decode_before_prefill_capture(monkeypatch):
    events = []

    class Model:
        already_warmed_up_prefill = True

        def warmup_model_prefill(self, *, enable_trace, **_kwargs):
            events.append(("prefill", enable_trace))

        def warmup_model_decode(self, *, enable_trace, **kwargs):
            events.append(
                (
                    "decode",
                    enable_trace,
                    kwargs.get("sampling_trace_variants_prepared", False),
                )
            )

        def prepare_device_grammar_decode_trace_warmup(self, **_kwargs):
            events.append(("prepare-decode",))
            return True

        def capture_prepared_device_grammar_decode_trace(self):
            events.append(("capture-decode",))

    monkeypatch.setattr(TTPlatform, "sample_on_device_mode", "decode_only")
    runner = SimpleNamespace(
        trace_mode="all",
        supports_device_grammar=True,
        model=Model(),
        kv_caches=object(),
        tt_max_batch_size=4,
        max_num_blocks_per_req=8,
    )

    TTModelRunner.warmup_model(runner)

    assert events == [
        ("prefill", False),
        ("decode", False, False),
        ("prepare-decode",),
        ("prefill", True),
        ("capture-decode",),
        ("decode", True, True),
    ]


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

    def sample_decode_on_device(tt_out, *, sampling_params, grammar_bitmask):
        calls.append((tt_out, sampling_params, grammar_bitmask))
        return sampled

    runner = SimpleNamespace(
        model=SimpleNamespace(sample_decode_on_device=sample_decode_on_device)
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
            sample_decode_on_device=lambda *_args, **_kwargs: (_ for _ in ()).throw(
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
        model=SimpleNamespace(sample_decode_on_device=lambda *_args, **_kwargs: None),
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
