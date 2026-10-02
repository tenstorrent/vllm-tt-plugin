# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Exercise the real request builders and assertions without a model/server."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from tests.tt import test_host_only_params as host
from tests.tt import test_structured_output_dp1 as structured
from tests.tt.conftest import reasoning_token_budget


@pytest.mark.parametrize(
    "sender,base_cap,valid_content",
    [
        (structured._send_choice_request, 8, "red"),
        (structured._send_regex_request, 16, "LANE-0"),
        (structured._send_json_request, 64, '{"status":"ok","lane":0}'),
        (structured._send_plain_request, 16, "A short sentence."),
    ],
)
def test_reasoning_allowance_reaches_final_without_weakening_assertions(
    sender, base_cap, valid_content
):
    requests = []
    allowance = 512
    content = valid_content

    async def create(**kwargs):
        requests.append(kwargs)
        completed = kwargs["max_completion_tokens"] >= base_cap + allowance
        return NS(choices=[NS(message=NS(content=content if completed else None))])

    client = NS(chat=NS(completions=NS(create=create)))
    with pytest.raises(AssertionError):
        asyncio.run(sender(client, "reference", 0))
    asyncio.run(sender(client, "reference", 0, allowance))
    before, after = requests
    assert before["max_completion_tokens"] == base_cap
    assert after["max_completion_tokens"] == base_cap + allowance
    assert {k: v for k, v in before.items() if k != "max_completion_tokens"} == {
        k: v for k, v in after.items() if k != "max_completion_tokens"
    }

    content = ""
    with pytest.raises((AssertionError, ValueError)):
        asyncio.run(sender(client, "reference", 0, allowance))


def test_bad_word_allowance_preserves_masks_seeds_and_output_assertion(monkeypatch):
    seen = []
    result_text = "Greetings."

    def run_batch(server, model, configs, **kwargs):
        seen.append((configs, kwargs))
        return [result_text if c.max_tokens >= 612 else None for c in configs]

    monkeypatch.setattr(host, "run_concurrent_batch", run_batch)
    test = host.TestHostOnlyParameters().test_bad_words
    with pytest.raises(AssertionError, match="content is None"):
        test(None, "reference", 32, 0)
    test(None, "reference", 32, 512)
    before, before_kwargs = seen[0]
    after, after_kwargs = seen[1]
    assert before_kwargs == after_kwargs == {"use_chat": True}
    assert len(after) == 5
    for first, second in zip(before, after):
        assert first.max_tokens == 100
        assert second.max_tokens == 612
        assert {k: v for k, v in vars(first).items() if k != "max_tokens"} == {
            k: v for k, v in vars(second).items() if k != "max_tokens"
        }
    result_text = "Hello"
    with pytest.raises(AssertionError, match="bad_word"):
        test(None, "reference", 32, 512)


def test_negative_reasoning_allowance_is_rejected():
    request = NS(config=NS(getoption=lambda name: -1))
    with pytest.raises(pytest.UsageError, match="non-negative"):
        reasoning_token_budget.__wrapped__(request)


@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_mixed_batch_forwards_budget_to_every_request(batch_size):
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        body = kwargs.get("extra_body", {}).get("structured_outputs", {})
        if "choice" in body:
            content, answer_budget = "red", 8
        elif "regex" in body:
            content, answer_budget = "LANE-0", 16
        elif "json" in body:
            content, answer_budget = '{"status":"ok","lane":0}', 64
        else:
            content, answer_budget = "A short sentence.", 16
        if kwargs["max_completion_tokens"] < answer_budget + 512:
            content = None
        return NS(choices=[NS(message=NS(content=content))])

    client = NS(chat=NS(completions=NS(create=create)))
    server = NS(get_async_client=lambda: client)
    structured.test_dp1_full_capacity_mixes_structured_and_plain_requests_first_wave(
        server, "reference", batch_size, 512
    )
    assert len(requests) == min(batch_size, 32)
