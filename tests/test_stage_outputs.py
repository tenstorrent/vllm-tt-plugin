# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A pipeline-stage model's decode frames reach the step output they belong to.

The model returns one frame-or-None per decode row from ``pop_stage_outputs()``.
The runner pops right after ``decode_forward``, tags each frame with its request
and ``frame_index``, and publishes them on that step's ``ModelRunnerOutput``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.config import (
    get_tt_supports_stage_outputs,
    resolve_tt_supports_stage_outputs,
)
from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.stage_outputs import (
    TTStageFrame,
    get_tt_stage_outputs,
)

EOS = 2150


class _StageModel:
    """Completes, in each decode, the frame whose codec-0 is the row's input token."""

    decode_input_update_contract = 1
    model_capabilities = {"supports_stage_outputs": True}

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
        multi_modal_kwargs={},
        stage_inputs=None,
        perform_device_sampling=False,
        prefill_empty_slots=[0],
    )

    TTModelRunner.submit_prefill(runner, model_input, [1])

    assert captured, "the prefill ran"


def test_the_sync_path_publishes_the_frame_of_a_step_that_samples_eos():
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

    def build(sampled, logprobs, req_ids=None, stage_outputs=None):
        # The real builder: it is what puts the frames on the output.
        return TTModelRunner._build_runner_output(
            SimpleNamespace(input_batch=runner.input_batch, _tt_committed_width=len),
            sampled,
            logprobs,
            stage_outputs=stage_outputs,
        )

    runner = SimpleNamespace(
        _apply_grammar_to_input=lambda model_input, grammar, lane_total: model_input,
        _sample_sync_forward=lambda fwd: ([torch.tensor([[EOS]])], []),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        apply_and_build_runner_output=build,
        input_batch=SimpleNamespace(req_ids=["a"], num_reqs=1),
        _num_speculative_tokens=0,
    )

    output = TTModelRunner._finish_front_packed_sync_impl(runner, None, fwd=fwd)

    assert output.sampled_token_ids == [[EOS]]
    assert get_tt_stage_outputs(output) == frames


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


def _config(output_tokens_per_step=1, speculative_config=None, max_num_seqs=1):
    return SimpleNamespace(
        additional_config={"_tt_output_tokens_per_step": output_tokens_per_step},
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
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
        (_config(max_num_seqs=2), False, "max_num_seqs=2"),
    ],
)
def test_stage_outputs_need_one_token_front_packed_steps(config, is_lane_mode, message):
    with pytest.raises(ValueError, match=message):
        resolve_tt_supports_stage_outputs(
            config, {"supports_stage_outputs": True}, _Model, is_lane_mode
        )


def test_stage_outputs_refuse_async_decode():
    """An overlapped steady step builds from host positions one step behind, so
    its stage frames would be tagged one frame low."""
    with pytest.raises(ValueError, match="supports_async_decode"):
        resolve_tt_supports_stage_outputs(
            _config(),
            {"supports_stage_outputs": True, "supports_async_decode": True},
            _Model,
            False,
        )


def test_a_non_boolean_declaration_is_refused():
    with pytest.raises(ValueError, match="as a boolean"):
        resolve_tt_supports_stage_outputs(
            _config(), {"supports_stage_outputs": "yes"}, _Model, False
        )
