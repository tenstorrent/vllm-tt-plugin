# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A pipeline-stage model's decode frames reach the step output they belong to.

The model returns one frame-or-None per decode row from ``pop_stage_outputs()``.
The runner pops right after ``decode_forward``, tags each frame with its request
and ``frame_index``, publishes them on that step's ``ModelRunnerOutput``, and
withdraws the frames of requests whose output must not reach the next stage.
"""

import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_tt_plugin.async_decode import (
    CompletedDecodeStep,
    SubmittedStepContext,
    TTAsyncDecodeController,
)
from vllm_tt_plugin.config import (
    get_tt_supports_stage_outputs,
    resolve_tt_supports_stage_outputs,
)
from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.stage_outputs import (
    TTStageFrame,
    get_tt_stage_outputs,
    set_tt_stage_outputs,
)

EOS = 2150


class _StageModel:
    """Completes, in each decode, the frame whose codec-0 is the row's input token."""

    decode_input_update_contract = 1
    model_capabilities = {"supports_async_decode": True}

    def __init__(self, padding_frames: bool = False):
        self._pending: list = []
        self.padding_frames = padding_frames

    def decode_forward(self, **kwargs):
        rows = kwargs["tokens"].shape[0]
        live = self.live_rows
        self._pending = [
            ("frame", int(kwargs["tokens"][row, 0]))
            if row < live or self.padding_frames
            else None
            for row in range(rows)
        ]
        return torch.zeros((rows, 1))

    def pop_stage_outputs(self):
        frames, self._pending = self._pending, []
        return frames


def _sampling_params(rows):
    return TTSamplingParams(
        temperature=torch.ones(rows),
        top_k=torch.full((rows,), 32),
        top_p=torch.ones(rows),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.full((rows,), -1),
        num_logprobs=torch.full((rows,), -2),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )


def _decode_input(row_req_ids, tokens, positions, padded_rows=0):
    rows = len(row_req_ids) + padded_rows
    tokens = list(tokens) + [0] * padded_rows
    positions = list(positions) + [0] * padded_rows
    page_table = torch.zeros((rows, 1), dtype=torch.int32)
    return SimpleNamespace(
        input_tokens=torch.tensor(tokens, dtype=torch.int32).reshape(rows, 1),
        input_positions=torch.tensor(positions, dtype=torch.int32),
        block_tables=page_table,
        block_tables_per_group=[page_table],
        block_tables_per_layer=None,
        unpadded_batch_size=len(row_req_ids),
        row_req_ids=list(row_req_ids),
        tt_sampling_params=_sampling_params(rows),
        perform_device_sampling=False,
        prompt_lens=None,
        prompt_tokens=None,
        output_tokens=None,
        decode_layout_changed=False,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        slot_remap=None,
    )


def _controller(prompt_lens, model=None):
    model = model or _StageModel()
    runner = SimpleNamespace(
        model=model,
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        supports_stage_outputs=True,
        requests={
            req_id: SimpleNamespace(num_prompt_tokens=n)
            for req_id, n in prompt_lens.items()
        },
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    return TTAsyncDecodeController(runner), model


def _submit(controller, model, model_input):
    model.live_rows = len(model_input.row_req_ids)
    return controller.submit_decode(model_input, read_from_device=True)


def test_each_frame_is_tagged_with_its_request_and_frame_index():
    """Frame k is completed by the step that consumes output token k, and carries
    that token: the codes of frame k arrive with codec-0 of frame k only."""
    controller, model = _controller({"a": 4, "b": 6})

    submission = _submit(
        controller,
        model,
        _decode_input(["a", "b"], tokens=[100, 200], positions=[4, 8]),
    )

    assert submission.stage_outputs == {
        "a": TTStageFrame(frame_index=0, data=("frame", 100)),
        "b": TTStageFrame(frame_index=2, data=("frame", 200)),
    }


def test_frames_follow_the_submission_not_the_live_batch():
    """Two overlapped steps around a batch condense: each keeps its own rows."""
    controller, model = _controller({"a": 4, "b": 4})

    first = _submit(
        controller, model, _decode_input(["a", "b"], tokens=[10, 20], positions=[5, 5])
    )
    # "a" left; "b" moved to row 0 before the first step completed.
    second = _submit(
        controller, model, _decode_input(["b"], tokens=[21], positions=[6])
    )

    assert first.stage_outputs == {
        "a": TTStageFrame(1, ("frame", 10)),
        "b": TTStageFrame(1, ("frame", 20)),
    }
    assert second.stage_outputs == {"b": TTStageFrame(2, ("frame", 21))}


def test_a_row_without_a_frame_publishes_nothing():
    controller, model = _controller({"a": 4})
    model.decode_forward = lambda **kwargs: torch.zeros((1, 1))
    model._pending = [None]

    submission = controller.submit_decode(
        _decode_input(["a"], tokens=[1], positions=[4]), read_from_device=True
    )

    assert submission.stage_outputs == {}


def test_a_frame_for_a_padding_row_is_refused():
    controller, model = _controller({"a": 4}, _StageModel(padding_frames=True))

    with pytest.raises(ValueError, match="padding rows"):
        _submit(
            controller,
            model,
            _decode_input(["a"], tokens=[1], positions=[4], padded_rows=1),
        )


def test_fewer_entries_than_live_rows_is_refused():
    controller, model = _controller({"a": 4, "b": 4})
    model.pop_stage_outputs = lambda: [None]

    with pytest.raises(ValueError, match="1 entries for 2 decode rows"):
        _submit(controller, model, _decode_input(["a", "b"], [1, 2], [4, 4]))


def test_prefill_never_asks_for_stage_outputs():
    captured = {}

    class Model:
        def prefill_forward(self, **kwargs):
            captured.update(kwargs)
            return object()

        def pop_stage_outputs(self):
            raise AssertionError("prefill completes no frame")

    runner = SimpleNamespace(
        kv_caches=object(),
        trace_mode="none",
        request_specific_rope=False,
        supports_stage_outputs=True,
        model=Model(),
        async_decode=SimpleNamespace(note_prefill_submitted=lambda: None),
    )
    model_input = SimpleNamespace(
        input_tokens=torch.zeros((1, 4), dtype=torch.int32),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        prompt_lens=[4],
        input_positions=torch.zeros((1,), dtype=torch.int32),
        block_tables_per_layer=None,
        stage_inputs=None,
        multi_modal_kwargs={},
        perform_device_sampling=False,
        prefill_empty_slots=[0],
    )

    TTModelRunner.submit_prefill(runner, model_input, [1])

    assert captured, "the prefill ran"


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_the_sync_path_publishes_the_frame_of_a_step_that_samples_eos(
    async_scheduling,
):
    """EOS on the first decode: that forward still completes frame 0."""
    frames = {"a": TTStageFrame(0, ("frame", 7))}
    fwd = _SyncForward(
        tt_out=None,
        tt_log_probs=None,
        sampling_params=None,
        model_input=SimpleNamespace(
            spec_mode=None, row_req_ids=["a"], prompt_lens=None
        ),
        batch_size_per_dp=[1],
        perform_device_sampling=False,
        is_decode=True,
        decode_submission=SimpleNamespace(stage_outputs=frames),
    )

    def build(sampled, logprobs, req_ids=None):
        return ModelRunnerOutput(
            req_ids=["a"],
            req_id_to_index={"a": 0},
            sampled_token_ids=[sampled[0].tolist()],
        )

    runner = SimpleNamespace(
        _apply_grammar_to_input=lambda model_input, grammar, lane_total: model_input,
        _sample_sync_forward=lambda fwd: ([torch.tensor([[EOS]])], []),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        apply_and_build_runner_output=build,
        defer_state_apply_and_build_runner_output=build,
        input_batch=SimpleNamespace(req_ids=["a"], num_reqs=1),
        _num_speculative_tokens=0,
    )

    output = TTModelRunner._finish_front_packed_sync_impl(runner, None, fwd=fwd)

    assert output.sampled_token_ids == [[EOS]]
    assert get_tt_stage_outputs(output) == frames


def _completed_step(token, runner_output, frames):
    return CompletedDecodeStep(
        sampled_token_ids=torch.tensor([[token]], dtype=torch.int32),
        logprobs=None,
        context=SubmittedStepContext(
            req_ids=["request"], req_id_to_index={"request": 0}, submit_time_ns=1
        ),
        completion_time_ns=2,
        runner_output=runner_output,
        stage_outputs=frames,
    )


def _published(token, frame_index):
    output = ModelRunnerOutput(
        req_ids=["request"], req_id_to_index={"request": 0}, sampled_token_ids=[[token]]
    )
    set_tt_stage_outputs(output, {"request": TTStageFrame(frame_index, token)})
    return output


def _async_apply_runner(request_state):
    runner = SimpleNamespace(
        _num_speculative_tokens=0,
        _output_tokens_per_step=1,
        _is_adaptive_block_output=False,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        requests={"request": request_state},
        input_batch=SimpleNamespace(
            req_id_to_index={},
            num_tokens=np.zeros(1, dtype=np.int32),
            token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
        ),
        model_config=SimpleNamespace(max_model_len=8),
        _steady_decode_lock=threading.Lock(),
        _completed_decode_steps=deque(),
        _pending_async_steps=deque(),
        _pending_async_overlap_ok=deque(),
    )
    runner._tt_committed_width = lambda toks: TTModelRunner._tt_committed_width(
        runner, toks
    )
    runner._apply_sampled_tokens_to_state = lambda **kwargs: (
        TTModelRunner._apply_sampled_tokens_to_state(runner, **kwargs)
    )
    return runner


def test_the_async_path_publishes_the_submissions_frames():
    frames = {"request": TTStageFrame(3, ("frame", 9))}
    runner = SimpleNamespace(
        _build_runner_output=lambda **kw: ModelRunnerOutput(
            req_ids=kw["req_ids"],
            req_id_to_index=kw["req_id_to_index"],
            sampled_token_ids=[[9]],
        )
    )
    controller = TTAsyncDecodeController(runner)

    output = controller.build_runner_output_from_completed_step(
        _completed_step(9, None, frames)
    )

    assert get_tt_stage_outputs(output) == frames


def test_a_frame_in_flight_when_its_request_is_cancelled_is_withdrawn():
    request_state = SimpleNamespace(output_token_ids=[])
    controller = TTAsyncDecodeController(_async_apply_runner(request_state))
    output = _published(7, 0)

    controller.apply_completed_decode_step(
        _completed_step(7, output, get_tt_stage_outputs(output)),
        suppress_output_req_ids={"request"},
        skip_state_req_ids={"request"},
    )

    assert get_tt_stage_outputs(output) == {}


def test_forced_reset_withdraws_only_the_frames_it_discards():
    request_state = SimpleNamespace(output_token_ids=[])
    runner = _async_apply_runner(request_state)
    controller = TTAsyncDecodeController(runner)
    outputs = [_published(token, index) for index, token in enumerate((5, 6, 7))]
    runner._completed_decode_steps.extend(
        _completed_step(token, output, get_tt_stage_outputs(output))
        for token, output in zip((5, 6, 7), outputs)
    )

    controller.apply_ready_completed_decode_steps(
        forced_reset_discard_counts={"request": 2}
    )

    assert [get_tt_stage_outputs(output) for output in outputs] == [
        {"request": TTStageFrame(0, 5)},
        {},
        {},
    ]
    # The tokens stay published, for the scheduler's stale-token count.
    assert [output.sampled_token_ids for output in outputs] == [[[5]], [[6]], [[7]]]


def test_preempting_a_stage_request_fails_loudly():
    runner = SimpleNamespace(
        supports_stage_outputs=True,
        requests={"a": object()},
        encoder_cache={},
        _release_model_request=lambda req_id: None,
    )
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids={"a"},
    )
    runner.input_batch = SimpleNamespace(remove_request=lambda req_id: None)

    with pytest.raises(RuntimeError, match="Request a was preempted"):
        TTModelRunner._update_states(runner, scheduler_output)


class _Model:
    pass


def _config(output_tokens_per_step=1, speculative_config=None):
    return SimpleNamespace(
        additional_config={"_tt_output_tokens_per_step": output_tokens_per_step},
        speculative_config=speculative_config,
    )


def test_the_resolved_capability_is_stored_on_the_config():
    config = _config()

    resolve_tt_supports_stage_outputs(
        config, {"supports_stage_outputs": True}, _Model, False
    )

    assert get_tt_supports_stage_outputs(config) is True
    resolve_tt_supports_stage_outputs(config, None, _Model, False)
    assert get_tt_supports_stage_outputs(config) is False


@pytest.mark.parametrize(
    "config, is_lane_mode, message",
    [
        (_config(), True, "lane mode"),
        (_config(output_tokens_per_step=4), False, "output_tokens_per_step=4"),
        (_config(speculative_config=object()), False, "speculative decoding"),
    ],
)
def test_stage_outputs_need_one_token_front_packed_steps(config, is_lane_mode, message):
    with pytest.raises(ValueError, match=message):
        resolve_tt_supports_stage_outputs(
            config, {"supports_stage_outputs": True}, _Model, is_lane_mode
        )


def test_a_non_boolean_declaration_is_refused():
    with pytest.raises(ValueError, match="as a boolean"):
        resolve_tt_supports_stage_outputs(
            _config(), {"supports_stage_outputs": "yes"}, _Model, False
        )
