# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM-Omni pipelines for bring-up on TT.

``tt_qwen3_tts_talker`` is vLLM-Omni's Qwen3-TTS pipeline cut to its first stage: the
talker is the final output, and the client gets the codec frames, not audio. It
exercises the TT AR stage end to end before a TT code2wav stage exists. A deploy
config selects it with ``pipeline: tt_qwen3_tts_talker``.
"""

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

TT_QWEN3_TTS_TALKER_PIPELINE = PipelineConfig(
    model_type="tt_qwen3_tts_talker",
    model_arch="Qwen3TTSTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="qwen3_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            sampling_constraints={"detokenize": False, "stop_token_ids": [2150]},
        ),
    ),
)


def register_tt_pipelines() -> None:
    register_pipeline(TT_QWEN3_TTS_TALKER_PIPELINE)
