# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Serving tasks follow the vLLM interfaces the model class implements.

vLLM builds the /v1/audio/transcriptions and /v1/realtime routes only for the
tasks the worker reports, so a speech-to-text model class must surface its
SupportsTranscription / SupportsRealtime interfaces here.
"""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.model_runner import TTModelRunner


def _tasks(**model_attrs):
    runner = SimpleNamespace(model=SimpleNamespace(**model_attrs))
    return TTModelRunner.get_supported_generation_tasks(runner)


def test_a_text_model_generates_only():
    assert _tasks() == ["generate"]


def test_a_transcription_model_also_transcribes():
    tasks = _tasks(supports_transcription=True, supports_transcription_only=False)
    assert tasks == ["generate", "transcription"]


def test_a_transcription_only_model_only_transcribes():
    tasks = _tasks(supports_transcription=True, supports_transcription_only=True)
    assert tasks == ["transcription"]


@pytest.mark.parametrize("transcribes", [False, True])
def test_a_realtime_model_adds_realtime(transcribes):
    tasks = _tasks(
        supports_realtime=True,
        supports_transcription=transcribes,
        supports_transcription_only=False,
    )
    assert tasks[-1] == "realtime"
    assert ("transcription" in tasks) is transcribes
