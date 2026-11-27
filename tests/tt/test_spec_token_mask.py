# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""JSON-schema requests on a speculating server.

On a model declaring ``supports_spec_device_mask`` a greedy JSON request
verifies as an ``argmax_ids`` verify under a device token mask, while the same
request with logprobs verifies in ``logits`` mode and applies its grammar on the
host (docs/SPEC_DECODE_CONTRACT.md, sections 4h and 4j). Both must give valid
JSON, and greedy decoding must give the same text either way, up to a
near-tie: the two modes replay different verify traces, so a bf16 near-tie can
round the other way. Without the capability both requests take the ``logits``
path and the comparison holds trivially.
"""

from __future__ import annotations

import json

import pytest

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["topic", "detail"],
                "additionalProperties": False,
            },
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["city", "facts"],
    "additionalProperties": False,
}
PROMPT = "Describe Prague as JSON: its name and three facts about it."
# Raw logprob gap below which the first divergent token counts as a near-tie.
NEAR_TIE_NATS = 0.5


def _json_request(client, model: str, top_k: int | None = None, **sampling):
    extra = {"structured_outputs": {"json": SCHEMA}}
    if top_k is not None:
        extra["top_k"] = top_k
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=256,
        extra_body=extra,
        **sampling,
    )


def _assert_schema(content: str) -> None:
    parsed = json.loads(content)
    assert set(parsed) == {"city", "facts"}
    assert len(parsed["facts"]) == 3
    assert all(set(fact) == {"topic", "detail"} for fact in parsed["facts"])


def test_a_greedy_json_request_gives_its_logits_path_output(
    tt_server, tt_model_name, server_speculates
):
    if not server_speculates:
        pytest.skip("the token mask serves speculative verifies only")
    client = tt_server.get_client()
    masked = _json_request(client, tt_model_name, temperature=0)
    host = _json_request(
        client, tt_model_name, temperature=0, logprobs=True, top_logprobs=5
    )
    text, reference = masked.choices[0].message.content, host.choices[0].message.content
    _assert_schema(text)
    _assert_schema(reference)
    if text == reference:
        return
    # The first token of the logits-path output that the masked output does not
    # continue, and how close its runner-up was.
    produced = ""
    for entry in host.choices[0].logprobs.content:
        if not text.startswith(produced + entry.token):
            alternatives = sorted(
                (alt.logprob for alt in entry.top_logprobs), reverse=True
            )
            gap = alternatives[0] - alternatives[1] if len(alternatives) > 1 else 1e9
            assert gap < NEAR_TIE_NATS, (
                f"greedy JSON output differs from the logits path at {produced!r}, "
                f"where the logits path's top-2 raw logprob gap is {gap:.3f} nats:\n"
                f"masked: {text!r}\nlogits: {reference!r}"
            )
            return
        produced += entry.token
    pytest.fail(f"outputs differ only past the logits path's tokens: {text!r}")


def test_sampled_json_requests_stay_valid(tt_server, tt_model_name, server_speculates):
    if not server_speculates:
        pytest.skip("the token mask serves speculative verifies only")
    client = tt_server.get_client()
    for seed in range(3):
        # top_k within the device sampler's bound, so the step samples on the
        # device and verifies under the token mask.
        response = _json_request(
            client, tt_model_name, top_k=20, temperature=0.8, top_p=0.95, seed=seed
        )
        _assert_schema(response.choices[0].message.content)
