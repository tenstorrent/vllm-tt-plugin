# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The PoC's Qwen3-TTS deploy config, merged with vLLM-Omni's own pipeline, gives each
stage the worker, scheduler, device and TT settings the TT stages need."""

from pathlib import Path

import pytest

pytest.importorskip("vllm_omni")

import vllm.config  # noqa: E402,F401  # finish vLLM init before the plugin package
from vllm_omni.config.stage_config import (  # noqa: E402
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler  # noqa: E402
from vllm_omni.core.sched.omni_generation_scheduler import (  # noqa: E402
    OmniGenerationScheduler,
)
from vllm_omni.model_executor.models.qwen3_tts.pipeline import (  # noqa: E402
    QWEN3_TTS_PIPELINE,
)

POC = Path(__file__).parents[2] / "poc" / "qwen3-tts-omni"
DEPLOY = POC / "deploy" / "tt_qwen3_tts.yaml"


def _path(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


@pytest.fixture
def stages():
    deploy = load_deploy_config(DEPLOY)
    # The TT stages refuse vLLM-Omni's second-generation runner.
    assert deploy.model_runner == "v1"
    merged = merge_pipeline_deploy(QWEN3_TTS_PIPELINE, deploy)
    return {stage.stage_id: stage for stage in merged}


def test_the_talker_runs_as_an_ar_stage_on_device_0(stages):
    talker = stages[0]
    assert talker.worker_type == "ar"
    # Async scheduling is off, so Omni names its sync AR scheduler, which the
    # plugin's pairing swaps for TTOmniARScheduler (test_scheduler.py).
    assert talker.scheduler_cls == _path(OmniARScheduler)
    assert talker.yaml_runtime["devices"] == "0"
    tt = talker.yaml_engine_args["additional_config"]["tt"]
    assert tt["trace_region_size"] == 90_000_000


def test_code2wav_runs_as_a_generation_stage_on_device_1(stages):
    code2wav = stages[1]
    assert code2wav.worker_type == "generation"
    assert code2wav.scheduler_cls == _path(OmniGenerationScheduler)
    assert code2wav.yaml_runtime["devices"] == "1"
    assert (
        code2wav.yaml_engine_args["additional_config"]["tt"]["l1_small_size"] == 65536
    )
    assert code2wav.input_sources == [0]
