# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The TT Omni AR worker and runner: Omni accepts them, and frames reach Omni's rows."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_omni")

import torch  # noqa: E402
import vllm.config  # noqa: E402,F401  # finish vLLM init before the plugin package
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput  # noqa: E402
from vllm_omni.core.sched.omni_ar_scheduler import (  # noqa: E402
    OmniARAsyncScheduler,
    OmniARScheduler,
)
from vllm_omni.data_entry_keys import unflatten_payload  # noqa: E402
from vllm_omni.engine.serialization import (  # noqa: E402
    serialize_additional_information,
)
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (  # noqa: E402
    _extract_last_frame,
    _should_append_async_frame,
)
from vllm_omni.outputs import OmniModelRunnerOutput  # noqa: E402
from vllm_omni.worker.omni_connector_validation import (  # noqa: E402
    validate_worker_omni_connector,
)

from vllm_tt_plugin.config import resolve_tt_scheduler_cls  # noqa: E402
from vllm_tt_plugin.model_runner import TTModelRunner  # noqa: E402
from vllm_tt_plugin.omni.model_runner import (  # noqa: E402
    TTOmniARModelRunner,
    to_omni_output,
)
from vllm_tt_plugin.omni.scheduler import TTOmniARScheduler  # noqa: E402
from vllm_tt_plugin.omni.worker import TTOmniARWorker  # noqa: E402
from vllm_tt_plugin.stage_outputs import (  # noqa: E402
    TTStageFrame,
    set_tt_stage_outputs,
)

FRAME = list(range(100, 116))


def _output(req_ids, frames=None):
    output = ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[7] for _ in req_ids],
    )
    set_tt_stage_outputs(output, frames)
    return output


def test_omni_accepts_the_worker_for_a_stage_with_a_next_stage():
    validate_worker_omni_connector(TTOmniARWorker, True)


@pytest.mark.parametrize("requested", [OmniARAsyncScheduler, OmniARScheduler])
def test_omnis_stage_scheduler_pairs_with_the_tt_one(requested):
    assert resolve_tt_scheduler_cls(requested, TTOmniARWorker, False) is (
        TTOmniARScheduler
    )


def test_frames_land_in_their_requests_rows():
    output = _output(["a", "b", "c"], {"c": TTStageFrame(4, FRAME)})

    converted = to_omni_output(output)

    assert isinstance(converted, OmniModelRunnerOutput)
    assert converted.sampled_token_ids == [[7], [7], [7]]
    assert converted.multimodal_outputs[:2] == [None, None]
    row = converted.multimodal_outputs[2]
    assert set(row) == {"codes.audio", "meta.codec_frame_valid"}, "flat, as on the wire"
    assert all(isinstance(value, torch.Tensor) for value in row.values())
    assert row["codes.audio"].tolist() == [FRAME]
    assert bool(row["meta.codec_frame_valid"])
    assert converted.inter_stage_outputs == converted.multimodal_outputs


def test_a_step_without_frames_has_no_stage_rows():
    converted = to_omni_output(_output(["a"]))

    assert converted.multimodal_outputs is None
    assert converted.inter_stage_outputs is None


def test_an_async_output_is_refused():
    class Pending(AsyncModelRunnerOutput):
        def get_output(self):
            raise AssertionError("not read")

    with pytest.raises(NotImplementedError, match="async_scheduling off"):
        to_omni_output(Pending())


def test_the_row_is_a_frame_the_qwen3_tts_chunker_appends():
    """The two checks vLLM-Omni's talker-to-code2wav chunker makes before it
    appends a row's frame for the next stage."""
    row = to_omni_output(_output(["a"], {"a": TTStageFrame(0, FRAME)}))
    # The chunk adapter unflattens a row before the chunker reads it.
    stage_output = unflatten_payload(row.inter_stage_outputs[0])

    frame = _extract_last_frame(stage_output)

    assert frame.tolist() == FRAME
    assert _should_append_async_frame(stage_output, frame)


def test_a_new_requests_payload_reaches_the_runner_as_a_dict(monkeypatch):
    seen = []
    monkeypatch.setattr(
        TTModelRunner,
        "_update_states",
        lambda self, scheduler_output: seen.extend(
            data.additional_information for data in scheduler_output.scheduled_new_reqs
        ),
    )
    info = {"task_type": ["Base"], "text": ["hi"], "ref_text": ["ref"]}
    new_request = SimpleNamespace(
        additional_information=serialize_additional_information(info),
        sampling_params=None,
    )

    TTOmniARModelRunner._update_states(
        _runner(vocab=3072), SimpleNamespace(scheduled_new_reqs=[new_request])
    )

    assert seen == [info]


def _runner(vocab):
    runner = object.__new__(TTOmniARModelRunner)
    runner.model_config = SimpleNamespace(get_vocab_size=lambda: vocab)
    return runner


def test_stop_ids_past_the_codec_vocabulary_are_dropped(monkeypatch):
    """min_tokens masks every stop id in the logits; the tokenizer's end-of-text
    (151645) is past the talker's 3072 ids and would index out of range."""
    from vllm import SamplingParams

    monkeypatch.setattr(TTModelRunner, "_update_states", lambda self, out: None)
    params = SamplingParams(min_tokens=2, stop_token_ids=[2150])
    params.update_from_generation_config({}, eos_token_id=151645)
    assert params.all_stop_token_ids == {2150, 151645}
    new_request = SimpleNamespace(additional_information=None, sampling_params=params)

    TTOmniARModelRunner._update_states(
        _runner(vocab=3072), SimpleNamespace(scheduled_new_reqs=[new_request])
    )

    assert params.all_stop_token_ids == {2150}
    assert params.stop_token_ids == [2150], "the stop condition itself is unchanged"


@pytest.mark.parametrize(
    "use_v2, async_scheduling, message",
    [(True, False, "model_runner: v1"), (False, True, "async_scheduling: false")],
)
def test_the_runner_refuses_omni_modes_it_does_not_serve(
    use_v2, async_scheduling, message
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_v2_model_runner=use_v2),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )

    with pytest.raises(ValueError, match=message):
        TTOmniARModelRunner(config)


def test_a_frame_tensor_round_trips_as_long():
    row = to_omni_output(
        _output(["a"], {"a": TTStageFrame(0, torch.tensor(FRAME))})
    ).multimodal_outputs[0]

    assert row["codes.audio"].dtype == torch.long


def test_a_row_survives_the_engine_core_wire():
    """The client decodes ``multimodal_output`` as ``dict[str, Tensor]``; a nested
    row fails there, after the engine has already run the step."""
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
    from vllm_omni.engine import OmniEngineCoreOutput

    row = to_omni_output(
        _output(["a"], {"a": TTStageFrame(0, FRAME)})
    ).multimodal_outputs[0]
    sent = OmniEngineCoreOutput(
        request_id="a", new_token_ids=[7], multimodal_output=row
    )

    received = MsgpackDecoder(OmniEngineCoreOutput).decode(
        MsgpackEncoder().encode(sent)
    )

    assert received.multimodal_output["codes.audio"].tolist() == [FRAME]
