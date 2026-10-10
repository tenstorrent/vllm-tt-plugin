# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""First-token variety must distinguish IDs that share a decoded first space."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.tt import test_seeding_and_variety as live_tests
from tests.tt.utils import RequestConfig, send_request


@pytest.mark.parametrize("varied_first_ids", [False, True])
@pytest.mark.parametrize("method", ["in_batch", "between_batches", "topk", "seeding"])
def test_first_token_variety_preserves_threshold_with_space_prefixed_tokens(
    monkeypatch, method, varied_first_ids
):
    calls = []

    def token_batch(server, model, configs, *, return_full_response):
        assert return_full_response
        call = len(calls)
        calls.append(configs)
        result = []
        for row, config in enumerate(configs):
            assert config.return_token_ids
            assert config.logprobs is None
            assert not config.return_tokens_as_token_ids
            assert config.max_tokens == 10
            if method in ("in_batch", "between_batches"):
                assert config.prompt == "Random letter: "
                assert config.temperature == 2.0
                offset = row + call
            elif method == "topk":
                assert (
                    config.prompt == "<question>Random letter a-z:</question><answer>"
                )
                assert config.temperature in (0, 2.0)
                if config.temperature:
                    assert config.top_k == 10
                offset = row + call
            else:
                assert config.prompt == "Random word: "
                assert config.temperature in (0, 1.5)
                if config.temperature:
                    assert config.top_k == 50
                offset = (config.seed or 0) // 100
            if config.temperature == 0:
                result.append([900, 901])
            else:
                first = 100 + offset if varied_first_ids else 100
                # Every first token has a space-prefixed vocabulary spelling.
                # Later IDs differ even for a collapsed first-token sampler,
                # so the overall-variety check cannot mask the prefill gate.
                result.append([first, 200 + offset])
        spellings = {ids[0]: f" word_{ids[0]}" for ids in result}
        assert {spellings[ids[0]][:1] for ids in result} == {" "}
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(token_ids=ids, text=spellings[ids[0]])]
            )
            for ids in result
        ]

    monkeypatch.setattr(live_tests, "run_concurrent_batch", token_batch)
    test = live_tests.TestSeedingAndVariety()

    def invoke():
        if method == "in_batch":
            test.test_temperature_varied_in_batch(object(), "model", 7)
        elif method == "between_batches":
            test.test_temperature_varied_between_batches(object(), "model")
        elif method == "topk":
            test.test_topk(object(), "model", 15)
        else:
            test.test_seeding(object(), "model", 32)

    if varied_first_ids:
        invoke()
    else:
        with pytest.raises(AssertionError, match="[Ff]irst token"):
            invoke()
    assert (
        len(calls)
        == {
            "in_batch": 1,
            "between_batches": 10,
            "topk": 2,
            "seeding": 2,
        }[method]
    )
    assert all(
        len(configs)
        == {"in_batch": 7, "between_batches": 1, "topk": 15, "seeding": 32}[method]
        for configs in calls
    )


def test_first_token_variety_rejects_missing_ids_despite_varied_text(monkeypatch):
    monkeypatch.setattr(
        live_tests,
        "run_concurrent_batch",
        lambda server, model, configs, **kwargs: [
            SimpleNamespace(choices=[SimpleNamespace(text=f" distinct text {i}")])
            for i in range(len(configs))
        ],
    )
    with pytest.raises(AssertionError, match="exact integer token IDs"):
        live_tests.TestSeedingAndVariety().test_temperature_varied_in_batch(
            object(), "model", 7
        )


@pytest.mark.parametrize("return_token_ids", [False, True])
def test_token_id_metadata_preserves_logprobs_and_default_request(return_token_ids):
    response = object()
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return response

    client = SimpleNamespace(completions=SimpleNamespace(create=create))
    config = RequestConfig("Random letter: ", return_token_ids=return_token_ids)
    assert asyncio.run(send_request(client, "model", config, True)) is response
    assert len(calls) == 1
    assert calls[0]["logprobs"] is None
    assert calls[0]["extra_body"] == (
        {"return_token_ids": True} if return_token_ids else None
    )
