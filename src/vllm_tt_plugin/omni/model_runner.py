# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The runner for a TT autoregressive vLLM-Omni stage.

``TTModelRunner`` does the work; this adds the two edges Omni has:

  * In: a new request's ``additional_information`` arrives serialized. It is decoded
    to the plain dict the stage model reads from ``stage_inputs``.
  * Out: the frames the model published (``stage_outputs``) become Omni's per-row
    ``multimodal_outputs`` and ``inter_stage_outputs``, in the shape vLLM-Omni's
    Qwen3-TTS talker emits, flattened to tensor values as Omni's wire format requires:
    ``{"codes.audio": [1, 16], "meta.codec_frame_valid": [1]}``. Omni's scheduler
    sends them on to the next stage, which unflattens them, or to the client when this
    stage is the last.

Synchronous scheduling only: under async scheduling a published output is still
amended after it is returned, which a converted copy would not see.
"""

from dataclasses import fields

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.stage_outputs import get_tt_stage_outputs


def omni_frame(data) -> dict:
    """One completed frame as one row of Omni's stage output."""
    return flatten_payload(
        {
            "codes": {"audio": torch.as_tensor(data, dtype=torch.long).reshape(1, -1)},
            # One flag per codes row, as Omni concatenates both along dim 0.
            "meta": {"codec_frame_valid": torch.ones(1, dtype=torch.int8)},
        }
    )


def to_omni_output(output: ModelRunnerOutput) -> OmniModelRunnerOutput:
    """The same step output with the published frames laid out by row."""
    if isinstance(output, AsyncModelRunnerOutput):
        raise NotImplementedError(
            "a TT vLLM-Omni AR stage runs with async_scheduling off; "
            "got an async output"
        )
    converted = OmniModelRunnerOutput(
        **{
            field.name: getattr(output, field.name)
            for field in fields(ModelRunnerOutput)
        }
    )
    frames = get_tt_stage_outputs(output)
    if frames:
        rows = [None] * len(output.req_ids)
        for req_id, frame in frames.items():
            rows[output.req_id_to_index[req_id]] = omni_frame(frame.data)
        converted.multimodal_outputs = rows
        converted.inter_stage_outputs = list(rows)
    return converted


class TTOmniARModelRunner(OmniConnectorModelRunnerMixin, TTModelRunner):
    def __init__(self, vllm_config, *args, **kwargs):
        if getattr(vllm_config.model_config, "use_v2_model_runner", False):
            raise ValueError(
                "a TT vLLM-Omni stage needs `model_runner: v1` in its deploy config"
            )
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError(
                "a TT vLLM-Omni AR stage needs `async_scheduling: false` in its "
                "deploy config"
            )
        super().__init__(vllm_config, *args, **kwargs)

    def _update_states(self, scheduler_output):
        vocab = self.model_config.get_vocab_size()
        for new_req_data in scheduler_output.scheduled_new_reqs:
            info = getattr(new_req_data, "additional_information", None)
            if info is not None and not isinstance(info, dict):
                new_req_data.additional_information = (
                    deserialize_additional_information(info)
                )
            # A stage's stop ids include the tokenizer's end-of-text, which lies past
            # a codec vocabulary; vLLM's min_tokens processor indexes the logits with
            # every stop id. An id the logits do not have is never sampled: drop it.
            params = new_req_data.sampling_params
            if params is not None:
                params._all_stop_token_ids = {
                    token for token in params.all_stop_token_ids if token < vocab
                }
        return super()._update_states(scheduler_output)

    def execute_model(self, scheduler_output):
        output = super().execute_model(scheduler_output)
        return None if output is None else to_omni_output(output)

    def sample_tokens(self, grammar_output):
        output = super().sample_tokens(grammar_output)
        return None if output is None else to_omni_output(output)
