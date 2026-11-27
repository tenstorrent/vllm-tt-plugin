# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Live-server tests for vLLM's ``/v1/systemone`` structured decision API.

The endpoint sends each question as a ``max_tokens=1`` request whose
``SamplingParams.logprob_token_ids`` are the label tokens, and turns their
full-vocabulary logprobs into per-label probabilities. vLLM serves it only for
the architectures in its ``READ_STRATEGIES`` table and answers 501 otherwise,
so these tests skip on other models. A speculating launch refuses
``logprob_token_ids`` with 400, so they skip there too.
"""

import asyncio
import math

import httpx
import pytest

TIMEOUT_S = 300

TICKET = {"ticket": "My card was charged twice for one order."}
TICKET_QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "payments and refunds",
            "shipping": "deliveries",
            "security": "account access",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?",
        "criteria": {"true": "the account is locked or money is moving"},
    },
    "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}


def _decision_body(model, state, questions):
    return {
        "model": model,
        "state": state,
        "questions": questions,
        # Qwen3 templates otherwise open a thinking block before the label.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _post_decision(url, body):
    response = httpx.post(f"{url}/v1/systemone", json=body, timeout=TIMEOUT_S)
    assert response.status_code == 200, response.text
    return response.json()


def _assert_distribution(decision, name, num_labels):
    probabilities = decision["answers"][name]["probabilities"]
    assert len(probabilities) == num_labels, probabilities
    assert all(0.0 <= p <= 1.0 for p in probabilities.values()), probabilities
    assert math.fsum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    label_mass = decision["diagnostics"][name]["label_mass"]
    assert 0.0 < label_mass <= 1.0 + 1e-6, (name, label_mass)


@pytest.fixture(scope="module")
def systemone_url(tt_server_url, tt_model_name):
    probe = httpx.post(
        f"{tt_server_url}/v1/systemone",
        json=_decision_body(
            tt_model_name, TICKET, {"urgent": TICKET_QUESTIONS["urgent"]}
        ),
        timeout=TIMEOUT_S,
    )
    if probe.status_code == 501:
        pytest.skip(f"/v1/systemone is not served for {tt_model_name}")
    if (
        probe.status_code == 400
        and "Speculative decoding" in probe.text
        and "logprob_token_ids" in probe.text
    ):
        pytest.skip(
            f"this speculating launch of {tt_model_name} refuses logprob_token_ids"
        )
    return tt_server_url.rstrip("/")


def test_each_question_type_returns_a_label_distribution(systemone_url, tt_model_name):
    decision = _post_decision(
        systemone_url, _decision_body(tt_model_name, TICKET, TICKET_QUESTIONS)
    )
    _assert_distribution(decision, "team", 3)
    _assert_distribution(decision, "urgent", 2)
    _assert_distribution(decision, "tone", 3)
    assert decision["answers"]["team"]["choice"] == "billing", decision
    assert decision["diagnostics"]["team"]["argmax_is_label"], decision


def test_more_labels_than_max_logprobs(systemone_url, tt_model_name):
    """26 labels: ``logprob_token_ids`` is not capped by ``max_logprobs``."""
    criteria = {
        f"option_{c}": f"the word starts with the letter {c.upper()}"
        for c in "abcdefghijklmnopqrstuvwxyz"
    }
    question = {
        "letter": {
            "type": "choice",
            "instructions": "Which letter does the animal's name start with?",
            "criteria": criteria,
        }
    }
    decision = _post_decision(
        systemone_url,
        _decision_body(
            tt_model_name, "The animal in the photo caption is: zebra.", question
        ),
    )
    _assert_distribution(decision, "letter", 26)


def test_decisions_beside_top_n_completions(
    systemone_url, tt_model_name, max_batch_size
):
    """Decision rows (named label ids) and ``logprobs=5`` completion rows sent
    together. A completion row sampled in the same step as a decision row must
    still get its own top-5, not the decision row's label layout."""
    num_each = max(1, max_batch_size // 2)
    num_logprobs = 5

    async def decision(client):
        response = await client.post(
            f"{systemone_url}/v1/systemone",
            json=_decision_body(tt_model_name, TICKET, TICKET_QUESTIONS),
        )
        assert response.status_code == 200, response.text
        return response.json()

    async def completion(client, i):
        response = await client.post(
            f"{systemone_url}/v1/completions",
            json={
                "model": tt_model_name,
                "prompt": f"Count from {i}: ",
                "max_tokens": 8,
                "temperature": 0.0,
                "logprobs": num_logprobs,
                "return_tokens_as_token_ids": True,
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    async def run():
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            tasks = []
            for i in range(num_each):
                tasks.append(decision(client))
                tasks.append(completion(client, i))
            return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    decisions, completions = results[0::2], results[1::2]
    for d in decisions:
        _assert_distribution(d, "team", 3)
    for i, c in enumerate(completions):
        top_logprobs = c["choices"][0]["logprobs"]["top_logprobs"]
        assert top_logprobs, f"completion {i}: no top_logprobs"
        for j, position in enumerate(top_logprobs):
            assert num_logprobs <= len(position) <= num_logprobs + 1, (
                f"completion {i}, token {j}: {position}"
            )
            assert all(math.isfinite(lp) for lp in position.values()), (
                f"completion {i}, token {j}: {position}"
            )
