# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Bad-word serving assertions use emitted IDs, not canonicalized output text."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from tests.tt import test_host_only_params as host
from tests.tt.utils import RequestConfig, send_chat_request


@pytest.mark.parametrize(
    "token_ids,text",
    [
        ([93383, 62], "_hello_"),
        ([45649, 62], "_hi_"),
        ([39, 4638], "Hello"),
        ([71, 4638], "hello"),
        ([900], "partial"),
        ([900, 0, 901], "nonadjacent"),
    ],
)
def test_legal_merged_alternate_or_partial_sequences_pass_without_retokenizing(
    monkeypatch, token_ids, text
):
    calls = []
    monkeypatch.setattr(
        host,
        "_served_bad_word_token_patterns",
        lambda *args: [[9419], [14556], [900, 901]],
    )

    def batch(server, model, configs, **kwargs):
        calls.append((configs, kwargs))
        return [
            NS(choices=[NS(message=NS(content=text), token_ids=token_ids)])
            for _ in configs
        ]

    monkeypatch.setattr(host, "run_concurrent_batch", batch)
    host.TestHostOnlyParameters().test_bad_words(None, "served", 32, 0)
    configs, kwargs = calls[0]
    assert kwargs == {"use_chat": True, "return_full_response": True}
    assert [c.seed for c in configs] == list(range(5))
    for config in configs:
        assert config.prompt == "Say hello to me"
        assert config.max_tokens == 100
        assert config.temperature == config.top_p == 1.0
        assert config.bad_words == ["hello", "Hello", "hi", "Hi", "hey", "Hey"]
        assert config.return_token_ids
        assert (
            config.logprobs is None
            and config.top_k is None
            and config.reasoning_effort is None
        )


@pytest.mark.parametrize(
    "token_ids", [[9419], [8, 9419], [900, 901], [900, 900, 901], [8, 900, 901, 9]]
)
def test_actual_forbidden_single_and_multi_token_sequences_fail(token_ids):
    with pytest.raises(AssertionError, match="Forbidden token sequence"):
        host._assert_no_bad_word_tokens(token_ids, [[9419], [900, 901]], 0)


@pytest.mark.parametrize(
    "token_ids", [None, [], [None], ["9419"], [True], [-1], (9419,)]
)
def test_missing_or_malformed_actual_token_ids_fail_closed(token_ids):
    with pytest.raises(AssertionError, match="output token IDs"):
        host._assert_no_bad_word_tokens(token_ids, [[9419]], 0)


def test_bad_word_patterns_match_upstream_prefix_and_duplicate_rules():
    mapping = {
        "a": [1],
        " a": [2],
        "b": [3, 4],
        " b": [5, 4],
        "c": [6],
        " c": [7, 8],
        "d": [9, 10],
        " d": [9, 11],
        "e": [12],
        " e": [],
    }
    calls = []

    def tokenize(text):
        calls.append(text)
        return mapping[text]

    assert host._bad_word_token_patterns(tokenize, ["a", "b", "c", "d", "e", "a"]) == [
        [1],
        [2],
        [3, 4],
        [5, 4],
        [6],
        [9, 10],
        [12],
    ]
    assert calls == ["a", " a", "b", " b", "c", " c", "d", " d", "e", " e"]
    assert host._bad_word_token_patterns(mapping.__getitem__, ["a", " a"]) == [
        [1],
        [2],
        [1],
        [2],
    ]


@pytest.mark.parametrize("bare", [None, [], "1", [None], [True], [-1], (1,)])
def test_bad_word_tokenization_fails_closed(bare):
    with pytest.raises(AssertionError, match="bad-word tokenization"):
        host._bad_word_token_patterns(lambda _: bare, ["a"])


def test_served_tokenization_uses_live_model_without_special_tokens(monkeypatch):
    requests = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, *, json):
            requests.append((url, json))
            return NS(
                raise_for_status=lambda: None,
                json=lambda: {"tokens": [2] if json["prompt"].startswith(" ") else [1]},
            )

    monkeypatch.setattr(host.httpx, "Client", Client)
    assert host._served_bad_word_token_patterns(
        NS(base_url="http://example"), "live-revision", ["hello"]
    ) == [[1], [2]]
    assert requests == [
        (
            "http://example/tokenize",
            {"model": "live-revision", "prompt": text, "add_special_tokens": False},
        )
        for text in ["hello", " hello"]
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_chat_request_output_id_opt_in_changes_only_response_metadata(enabled):
    seen = []
    response = NS(choices=[NS(message=NS(content="output"))])

    async def create(**kwargs):
        seen.append(kwargs)
        return response

    client = NS(chat=NS(completions=NS(create=create)))
    config = RequestConfig(
        prompt="Say hello to me",
        max_tokens=100,
        bad_words=["hello"],
        seed=3,
        return_token_ids=enabled,
    )
    assert (
        asyncio.run(
            send_chat_request(client, "served", config, return_full_response=True)
        )
        is response
    )
    request = seen[0]
    assert request["extra_body"] == (
        {"bad_words": ["hello"], "return_token_ids": True}
        if enabled
        else {"bad_words": ["hello"]}
    )
    assert request["max_tokens"] == 100 and request["seed"] == 3
    assert request["temperature"] == request["top_p"] == 1.0
    assert request["logprobs"] is False and request["top_logprobs"] is None
    assert "chat_template_kwargs" not in request["extra_body"]
