# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Concurrent requests that each take another device sampling feature, on one
speculating server.

On the Qwen3.8 MTP model (``supports_spec_device_mask``,
``supports_spec_device_logprobs``, ``supports_spec_device_penalties`` and
``supports_spec_device_top_p``) the requests below share ``argmax_ids``
verifies: a JSON request draws under a device token mask, a penalised one
under device penalties, a logprobs one returns the verify's payload, and a
request with top-k off and ``top_p`` below 1 draws from the nucleus under the
top-p certificate (docs/SPEC_DECODE_CONTRACT.md, sections 4j to 4m). A request
that combines a mask or a penalty with logprobs verifies in ``logits`` mode
instead, which every request sharing its steps would follow too, so it is
sent on its own. Each request is checked by what it asked for; on a server
without these capabilities the same requests take the host paths and must
pass as well.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.tt.utils import assert_deterministic_allow_near_tie

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["city", "facts"],
    "additionalProperties": False,
}
JSON_PROMPT = "Describe Vienna as JSON: its name and three facts about it."
STORY = "Random story: "
PENALISED = "List the colours of the rainbow, then list them again, twice:"
# The token ids an allowlisted top-p request may produce.
ALLOWED = list(range(0, 50000))


def _assert_schema(content: str) -> None:
    parsed = json.loads(content)
    assert set(parsed) == {"city", "facts"} and len(parsed["facts"]) == 3, content


def _json(client, model: str, **sampling):
    extra = {"structured_outputs": {"json": SCHEMA}}
    for key in ("top_k", "repetition_penalty"):
        if key in sampling:
            extra[key] = sampling.pop(key)
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": JSON_PROMPT}],
        max_completion_tokens=200,
        extra_body=extra,
        **sampling,
    )


def _complete(client, model: str, prompt: str, max_tokens: int, **sampling):
    extra = {
        key: sampling.pop(key)
        for key in ("top_k", "repetition_penalty", "allowed_token_ids")
        if key in sampling
    }
    extra["return_token_ids"] = True
    return client.completions.create(
        model=model, prompt=prompt, max_tokens=max_tokens, extra_body=extra, **sampling
    )


def _assert_logprobs_lead(response) -> None:
    """Each token of a greedy request is its own top-1 alternative with the same
    logprob; the alternatives are sorted, at most 0, and sum to at most 1."""
    logprobs = response.choices[0].logprobs
    assert logprobs is not None and logprobs.top_logprobs
    for j, (value, top) in enumerate(
        zip(logprobs.token_logprobs, logprobs.top_logprobs)
    ):
        ranked = sorted(top.values(), reverse=True)
        assert ranked[0] == pytest.approx(value, abs=1e-3), j
        assert all(v <= 1e-6 for v in ranked), (j, ranked)
        assert sum(pow(2.718281828459045, v) for v in ranked) <= 1.0 + 1e-3, (j, ranked)


def test_concurrent_json_penalty_logprobs_and_top_p_requests_each_hold(
    tt_server, tt_model_name, server_speculates
):
    if not server_speculates:
        pytest.skip("the device sampling features serve speculative verifies")
    client = tt_server.get_async_client()

    async def run():
        return await asyncio.gather(
            _json(client, tt_model_name, temperature=0),
            _json(client, tt_model_name, temperature=0.8, top_k=20, top_p=0.95, seed=3),
            _json(client, tt_model_name, temperature=0, repetition_penalty=1.2),
            _complete(
                client,
                tt_model_name,
                PENALISED,
                48,
                temperature=0,
                repetition_penalty=1.3,
            ),
            _complete(
                client,
                tt_model_name,
                "Count from 1 to 40: ",
                24,
                temperature=0,
                logprobs=5,
            ),
            _complete(
                client,
                tt_model_name,
                STORY,
                32,
                temperature=0.7,
                top_k=-1,
                top_p=0.9,
                seed=7,
            ),
            _complete(
                client,
                tt_model_name,
                STORY,
                32,
                temperature=0.7,
                top_k=-1,
                top_p=0.9,
                seed=7,
            ),
            _complete(
                client,
                tt_model_name,
                STORY,
                32,
                temperature=0.7,
                top_k=-1,
                top_p=0.9,
                seed=11,
                allowed_token_ids=ALLOWED,
            ),
        )

    responses = asyncio.run(run())
    json_greedy, json_sampled, json_penalised = responses[:3]
    penalised, logprobs, top_p_a, top_p_b, top_p_masked = responses[3:]
    for response in (json_greedy, json_sampled, json_penalised):
        _assert_schema(response.choices[0].message.content)
    _assert_logprobs_lead(logprobs)
    for response in (penalised, top_p_a, top_p_b, top_p_masked):
        assert response.choices[0].text, response
    assert_deterministic_allow_near_tie(
        [top_p_a.choices[0].text, top_p_b.choices[0].text],
        "Two seeded top-p requests with top-k off in one batch should match.",
    )
    token_ids = top_p_masked.choices[0].token_ids
    assert token_ids and set(token_ids) <= set(ALLOWED), token_ids
    # The penalised request alone, on the same path, commits the same tokens up
    # to a near tie: the verifies group the positions differently in a batch.
    sync = tt_server.get_client()
    alone = _complete(
        sync, tt_model_name, PENALISED, 48, temperature=0, repetition_penalty=1.3
    )
    again = _complete(
        sync, tt_model_name, PENALISED, 48, temperature=0, repetition_penalty=1.3
    )
    assert_deterministic_allow_near_tie(
        [penalised.choices[0].text, alone.choices[0].text, again.choices[0].text],
        "A greedy penalised request should commit the same tokens in a mixed "
        "batch and alone.",
        min_common_prefix=8,
    )


def test_a_mask_or_a_penalty_with_logprobs_takes_the_logits_path_and_holds(
    tt_server, tt_model_name, server_speculates
):
    """A JSON request and a penalised request that ask for logprobs verify in
    ``logits`` mode (the payload would describe their masked or penalized
    logits): the JSON stays valid, the logprobs come back, and the penalised
    request commits what its twin without logprobs commits, up to a near tie."""
    if not server_speculates:
        pytest.skip("the routing applies to speculative verifies")
    client = tt_server.get_client()
    with_logprobs = _json(
        client, tt_model_name, temperature=0, logprobs=True, top_logprobs=2
    )
    _assert_schema(with_logprobs.choices[0].message.content)
    assert with_logprobs.choices[0].logprobs.content
    twins = [
        _complete(
            client,
            tt_model_name,
            PENALISED,
            48,
            temperature=0,
            frequency_penalty=0.6,
            **({"logprobs": 2} if asks else {}),
        )
        for asks in (False, True, False)
    ]
    assert twins[1].choices[0].logprobs is not None
    assert_deterministic_allow_near_tie(
        [t.choices[0].text for t in twins],
        "A penalised request and its logprobs twin should commit the same tokens.",
        min_common_prefix=8,
    )
