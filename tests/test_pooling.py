# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A TT pooling model gets each request's whole prompt and audio, and its output is
the request's pooler output."""

from types import SimpleNamespace

import pytest
import torch
from vllm.pooling_params import PoolingParams

from vllm_tt_plugin.pooling import execute_pooling_step, supported_pooling_tasks


class _Field:
    def __init__(self, data):
        self.data = data


class _TokenClassifier:
    is_pooling_model = True
    pooler = SimpleNamespace(get_supported_tasks=lambda: {"token_classify"})

    def __init__(self):
        self.calls = []

    def prefill_forward(self, tokens, prompt_lens=None, **kwargs):
        self.calls.append((tokens, prompt_lens, kwargs))
        return [torch.full((prompt_lens[0], 3), float(tokens[0, 0]))]


def _new_req(req_id, prompt, mel=None, computed=0):
    features = []
    if mel is not None:
        features = [SimpleNamespace(data={"input_audio_features": _Field(mel)})]
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=features,
        pooling_params=PoolingParams(task="token_classify"),
        num_computed_tokens=computed,
    )


def _step(new_reqs, scheduled=None, cached=()):
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=SimpleNamespace(req_ids=list(cached)),
        num_scheduled_tokens=scheduled
        or {r.req_id: len(r.prompt_token_ids) for r in new_reqs},
    )


def test_each_request_is_prefilled_alone_with_its_audio():
    model = _TokenClassifier()
    mel_a = torch.ones(128, 40)
    out = execute_pooling_step(
        model, _step([_new_req("a", [7, 8, 9], mel_a), _new_req("b", [5, 6])])
    )

    assert out.req_ids == ["a", "b"]
    assert out.req_id_to_index == {"a": 0, "b": 1}
    assert [o.shape for o in out.pooler_output] == [(3, 3), (2, 3)]
    assert out.pooler_output[1][0, 0] == 5
    tokens, lens, kwargs = model.calls[0]
    assert tokens.tolist() == [[7, 8, 9]] and lens == [3]
    assert kwargs["input_audio_features"] == [[mel_a]]
    assert model.calls[1][2] == {}


def test_a_partly_scheduled_prompt_is_rejected():
    with pytest.raises(ValueError, match="whole prompt"):
        execute_pooling_step(
            _TokenClassifier(), _step([_new_req("a", [1, 2, 3])], scheduled={"a": 2})
        )


def test_a_continuing_request_is_rejected():
    with pytest.raises(RuntimeError, match="first step"):
        execute_pooling_step(_TokenClassifier(), _step([], cached=["a"]))


def test_tasks_come_from_the_model_pooler():
    assert supported_pooling_tasks(_TokenClassifier()) == ["token_classify"]
    assert supported_pooling_tasks(object()) == []
