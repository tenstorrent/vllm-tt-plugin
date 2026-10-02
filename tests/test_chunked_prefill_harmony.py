# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Exercise actual recall requests/final scoring without a server or model."""

import asyncio
import re
from types import SimpleNamespace as NS

import pytest

from tests.tt import test_chunked_prefill as recall
from tests.tt.conftest import recall_reasoning_effort, recall_reasoning_token_budget
from tests.tt.utils import (
    RequestConfig,
    recalled_passphrase,
    run_concurrent_batch,
    send_request,
)

IDENTIFIERS = ["cobalt-heron-42", "cobalt-heron-256"] + [
    f"cobalt-heron-{rnd}-{i}" for rnd in range(3) for i in range(4)
]


class RecordingServer:
    def __init__(self, content=None, finish="stop", use_prompt_identifier=True):
        self.content = content
        self.finish = finish
        self.use_prompt_identifier = use_prompt_identifier
        self.requests = []
        self.closes = 0
        self.batch_sizes = []

    def get_async_client(self):
        batch_start = len(self.requests)

        async def create(*, chat=False, **body):
            self.requests.append((chat, body))
            prompt = body["messages"][0]["content"] if chat else body["prompt"]
            identifier = re.search(r"cobalt-heron-[\d-]+", prompt).group()
            content = (
                identifier.replace("-", "\u2011")
                if self.use_prompt_identifier
                else self.content
            )
            if chat:
                return NS(
                    choices=[
                        NS(
                            finish_reason=self.finish,
                            message=NS(content=content, reasoning=identifier),
                        )
                    ]
                )
            return NS(choices=[NS(text=identifier)])

        async def chat_create(**body):
            return await create(chat=True, **body)

        async def close():
            self.closes += 1
            self.batch_sizes.append(len(self.requests) - batch_start)

        return NS(
            completions=NS(create=create),
            chat=NS(completions=NS(create=chat_create)),
            close=close,
        )


def run_original14(server, effort, allowance):
    for test in (
        recall.test_a_solo_split_prefill_recalls_its_needle,
        recall.test_a_long_split_prefill_recalls_its_needle,
        recall.test_prefills_sharing_a_step_each_recall_their_own_needle,
    ):
        test(server, "reference", 8192, effort, allowance)


def test_supported_recall_preserves_all14_inputs_and_concurrency():
    legacy, supported = RecordingServer(), RecordingServer()
    run_original14(legacy, None, 1000)
    run_original14(supported, "low", 1000)
    assert len(legacy.requests) == len(supported.requests) == 14
    assert legacy.closes == supported.closes == 5  # 1/1/4/4/4 submitted groups
    assert legacy.batch_sizes == supported.batch_sizes == [1, 1, 4, 4, 4]
    for identifier, (raw_chat, raw), (chat, body) in zip(
        IDENTIFIERS, legacy.requests, supported.requests
    ):
        assert not raw_chat and chat
        assert body["messages"] == [{"role": "user", "content": raw["prompt"]}]
        assert raw["prompt"].count(identifier) == 1
        assert raw["max_tokens"] == 24 and body["max_tokens"] == 1024
        assert body["extra_body"] == {"reasoning_effort": "low"}
        assert raw["extra_body"] is None
        for key in (
            "model",
            "temperature",
            "top_p",
            "seed",
            "presence_penalty",
            "frequency_penalty",
        ):
            assert raw[key] == body[key]


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
@pytest.mark.parametrize("allowance", [0, 1000])
def test_explicit_effort_and_finite_allowance_reach_chat_request(effort, allowance):
    server = RecordingServer()
    config = RequestConfig(prompt="cobalt-heron-42", max_tokens=24, temperature=0)
    output = recall._run_recall_batch(server, "reference", [config], effort, allowance)
    assert recalled_passphrase(output[0], "cobalt-heron-42")
    assert server.requests[0][1]["max_tokens"] == 24 + allowance
    assert server.requests[0][1]["extra_body"] == {"reasoning_effort": effort}
    assert config.max_tokens == 24 and config.reasoning_effort is None


@pytest.mark.parametrize("finish", ["length", "tool_calls", None])
def test_expected_identifier_cannot_pass_without_normal_final_stop(finish):
    server = RecordingServer(finish=finish)
    with pytest.raises(AssertionError, match="completed final"):
        recall._run_recall_batch(
            server, "reference", [RequestConfig(prompt="cobalt-heron-42")], "low", 1000
        )


@pytest.mark.parametrize("content", [None, ""])
def test_analysis_identifier_is_not_a_final_answer(content):
    server = RecordingServer(content=content, use_prompt_identifier=False)
    with pytest.raises(AssertionError, match="no final content"):
        recall._run_recall_batch(
            server, "reference", [RequestConfig(prompt="cobalt-heron-42")], "low", 1000
        )


@pytest.mark.parametrize(
    "expected,other", [(a, b) for a in IDENTIFIERS for b in IDENTIFIERS if a != b]
)
def test_wrong_request_final_is_rejected_despite_correct_analysis(expected, other):
    server = RecordingServer(
        content=other.replace("-", "\u2011"), use_prompt_identifier=False
    )
    output = recall._run_recall_batch(
        server, "reference", [RequestConfig(prompt=expected)], "low", 1000
    )
    assert not recalled_passphrase(output[0], expected)


def test_chat_sender_defaults_remain_unchanged():
    server = RecordingServer()
    run_concurrent_batch(
        server, "reference", [RequestConfig(prompt="cobalt-heron-42")], use_chat=True
    )
    assert server.requests[0][1]["extra_body"] is None
    assert server.requests[0][1]["max_tokens"] == 10


def test_legacy_sender_rejects_an_explicit_chat_only_effort():
    server = RecordingServer()
    config = RequestConfig(prompt="cobalt-heron-42", reasoning_effort="low")
    with pytest.raises(ValueError, match="chat completions"):
        asyncio.run(send_request(server.get_async_client(), "reference", config))
    assert not server.requests


def test_recall_options_do_not_infer_a_model_or_change_other_budgets():
    options = {
        "--tt-recall-reasoning-effort": None,
        "--tt-recall-reasoning-token-budget": 0,
        "--tt-reasoning-token-budget": 1300,
    }
    request = NS(config=NS(getoption=options.__getitem__))
    assert recall_reasoning_effort.__wrapped__(request) is None
    assert recall_reasoning_token_budget.__wrapped__(request) == 0
    assert options["--tt-reasoning-token-budget"] == 1300


def test_negative_recall_allowance_is_rejected():
    request = NS(config=NS(getoption=lambda name: -1))
    with pytest.raises(pytest.UsageError, match="non-negative"):
        recall_reasoning_token_budget.__wrapped__(request)
