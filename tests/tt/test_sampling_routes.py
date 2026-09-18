# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Seeded request isolation across page growth and supported sampling routes."""

import asyncio

import pytest


@pytest.mark.parametrize("top_k", [20, 64, -1])
def test_seeded_page_growth_and_slot_reordering(tt_server, tt_model_name, top_k):
    # No logprobs or min_tokens: those would force host sampling and hide the
    # bounded top-k device path. ignore_eos only fixes the request length.
    requests = [
        {
            "model": tt_model_name,
            "prompt": f"Continue this story about explorer {i}: "
            + "The path crossed a quiet forest. " * (i + 1),
            "max_tokens": 129 + i,
            "temperature": 0.8,
            "seed": 2**40 + 97 * i,
            "extra_body": {"top_k": top_k, "ignore_eos": True},
        }
        for i in range(4)
    ]

    async def run(order):
        async with tt_server.get_async_client() as client:
            return await asyncio.gather(
                *(client.completions.create(**requests[i]) for i in order)
            )

    first = asyncio.run(run(range(4)))
    second = asyncio.run(run(reversed(range(4))))[::-1]
    for i, (a, b) in enumerate(zip(first, second)):
        assert a.usage.completion_tokens == b.usage.completion_tokens == 129 + i
        assert a.choices[0].text == b.choices[0].text
