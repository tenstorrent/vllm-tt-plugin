# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A TT pooling model gets each request's whole prompt and audio in one prefill, and
vLLM's own pooler turns its output into the request's pooler output."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import PoolerConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.pooler.activations import PoolerClassify
from vllm.model_executor.layers.pooler.tokwise.heads import TokenClassifierPoolerHead
from vllm.model_executor.layers.pooler.tokwise.methods import get_tok_pooling_method
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.pooling_params import PoolingParams

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.pooling import supported_pooling_tasks

LOGITS = [1.0, 2.0, 3.0]


class _Field:
    def __init__(self, data):
        self.data = data


class _TokenClassifier:
    """Its prefill returns per-token logits; the pooler is upstream's for
    token_classify, as a model built with ``pooler_for_token_classify`` has."""

    is_pooling_model = True

    def __init__(self, pooling_type="ALL"):
        with set_current_vllm_config(VllmConfig()):
            self.pooler = TokenPooler(
                pooling=get_tok_pooling_method(pooling_type),
                head=TokenClassifierPoolerHead(
                    head_dtype=torch.float32, activation=PoolerClassify()
                ),
            )
        self.calls = []

    def prefill_forward(self, tokens, prompt_lens=None, **kwargs):
        self.calls.append((tokens, prompt_lens, kwargs))
        return [torch.tensor([LOGITS] * prompt_lens[0])]


class _PositionClassifier(_TokenClassifier):
    """Row j of its logits is [prompt[j], j, 0], so pooled rows show their source."""

    def prefill_forward(self, tokens, prompt_lens=None, **kwargs):
        (prompt,) = tokens.tolist()
        return [torch.tensor([[t, j, 0.0] for j, t in enumerate(prompt)])]


def _params(pooler_config=None, **kwargs):
    params = PoolingParams(task="token_classify", **kwargs)
    # As vLLM's frontend does before a request reaches the engine.
    params.verify(SimpleNamespace(pooler_config=pooler_config))
    return params


def _new_req(req_id, prompt, mel=None, params=None, computed=0):
    features = []
    if mel is not None:
        features = [SimpleNamespace(data={"input_audio_features": _Field(mel)})]
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=features,
        sampling_params=None,
        pooling_params=params or _params(),
        num_computed_tokens=computed,
    )


def _step(new_reqs, scheduled=None, cached=()):
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=SimpleNamespace(req_ids=list(cached)),
        num_scheduled_tokens=scheduled
        or {r.req_id: len(r.prompt_token_ids) for r in new_reqs},
    )


def _execute(model, scheduler_output):
    runner = SimpleNamespace(runs_pooling=True, model=model)
    return TTModelRunner.execute_model(runner, scheduler_output)


SOFTMAX = torch.softmax(torch.tensor(LOGITS), dim=-1)


@pytest.mark.parametrize(
    "params, expected",
    [
        (_params(), SOFTMAX),
        (_params(use_activation=True), SOFTMAX),
        (_params(use_activation=False), torch.tensor(LOGITS)),
    ],
    ids=["default", "activation", "raw logits"],
)
def test_the_runner_applies_the_requested_activation(params, expected):
    out = _execute(_TokenClassifier(), _step([_new_req("a", [7, 8], params=params)]))

    (pooled,) = out.pooler_output
    assert pooled.shape == (2, 3)
    torch.testing.assert_close(pooled, expected.expand(2, 3))


def test_each_request_is_prefilled_alone_with_its_audio():
    model = _TokenClassifier()
    mel_a = torch.ones(128, 40)
    out = _execute(
        model, _step([_new_req("a", [7, 8, 9], mel_a), _new_req("b", [5, 6])])
    )

    assert out.req_ids == ["a", "b"]
    assert out.req_id_to_index == {"a": 0, "b": 1}
    assert [o.shape for o in out.pooler_output] == [(3, 3), (2, 3)]
    tokens, lens, kwargs = model.calls[0]
    assert tokens.tolist() == [[7, 8, 9]] and lens == [3]
    assert kwargs["input_audio_features"] == [[mel_a]]
    assert model.calls[1][2] == {}


def test_step_pooling_keeps_the_rows_of_the_step_tag_in_request_order():
    step = PoolerConfig(tok_pooling_type="STEP", step_tag_id=8)
    reqs = [
        _new_req("a", [7, 8, 9, 8], params=_params(step, use_activation=False)),
        _new_req("b", [8, 5], params=_params(step, use_activation=False)),
    ]

    out = _execute(_PositionClassifier("STEP"), _step(reqs))

    assert out.req_ids == ["a", "b"]
    a, b = out.pooler_output
    torch.testing.assert_close(a, torch.tensor([[8.0, 1.0, 0.0], [8.0, 3.0, 0.0]]))
    torch.testing.assert_close(b, torch.tensor([[8.0, 0.0, 0.0]]))


def test_a_task_the_pooler_does_not_serve_is_rejected():
    with pytest.raises(ValueError, match="not one of the model's tasks"):
        _execute(
            _TokenClassifier(),
            _step([_new_req("a", [1, 2], params=PoolingParams(task="embed"))]),
        )


def test_a_partly_scheduled_prompt_is_rejected():
    with pytest.raises(ValueError, match="whole prompt"):
        _execute(
            _TokenClassifier(), _step([_new_req("a", [1, 2, 3])], scheduled={"a": 2})
        )


def test_a_continuing_request_is_rejected():
    with pytest.raises(RuntimeError, match="first step"):
        _execute(_TokenClassifier(), _step([], cached=["a"]))


def test_tasks_come_from_the_model_pooler():
    assert supported_pooling_tasks(_TokenClassifier()) == ["token_classify"]
    assert supported_pooling_tasks(object()) == []
