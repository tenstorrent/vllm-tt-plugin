# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Token-level serving assertions must not request host logprob sampling."""

from types import SimpleNamespace

import pytest

from tests.tt.utils import (
    RequestConfig,
    assert_varied,
    run_concurrent_batch_native_tokens,
)


class RecordingServer:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.requests = []
        self.closes = 0

    def get_async_client(self):
        async def create(**body):
            self.requests.append(body)
            return SimpleNamespace(choices=[next(self.choices)])

        async def close():
            self.closes += 1

        return SimpleNamespace(completions=SimpleNamespace(create=create), close=close)


def test_native_tokens_keep_distinct_newline_tokens_without_requesting_logprobs():
    choices = [
        SimpleNamespace(text="\nA", token_ids=[198, 32]),
        SimpleNamespace(text="\n\n\nB", token_ids=[1358, 33]),
    ]
    server = RecordingServer(choices)
    config = RequestConfig(
        prompt="<question>Random letter a-z:</question><answer>",
        max_tokens=10,
        top_k=10,
        temperature=2.0,
    )
    results = run_concurrent_batch_native_tokens(server, "model", [config, config])

    assert results == [(198, 32), (1358, 33)]
    assert_varied([row[:1] for row in results], 2, "first token IDs differ")
    with pytest.raises(AssertionError):
        assert_varied([choice.text[:1] for choice in choices], 2, "lossy characters")
    assert not config.return_token_ids  # Caller-owned configs remain unchanged.
    assert server.closes == 1
    for body in server.requests:
        assert body["extra_body"] == {"top_k": 10, "return_token_ids": True}
        assert body["logprobs"] is None
        assert body["seed"] is None
        assert body["prompt"] == config.prompt
        assert body["max_tokens"] == 10
        assert body["temperature"] == 2.0
        assert body["top_p"] == 1.0


@pytest.mark.parametrize(
    "choice",
    [SimpleNamespace(text="\nA"), SimpleNamespace(text="\nA", token_ids=None)],
)
def test_native_tokens_fail_when_extension_is_missing(choice):
    server = RecordingServer([choice])
    with pytest.raises(ValueError, match="did not return requested token_ids"):
        run_concurrent_batch_native_tokens(server, "model", [RequestConfig("prompt")])
    assert server.closes == 1


def test_native_tokens_preserve_explicit_logprob_and_seed_parameters():
    server = RecordingServer([SimpleNamespace(text="a", token_ids=[64])])
    config = RequestConfig("prompt", logprobs=3, seed=42, top_k=5)
    assert run_concurrent_batch_native_tokens(server, "model", [config]) == [(64,)]
    assert server.requests[0]["logprobs"] == 3
    assert server.requests[0]["seed"] == 42
    assert server.requests[0]["extra_body"] == {"top_k": 5, "return_token_ids": True}
