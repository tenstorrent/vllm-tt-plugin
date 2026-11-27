# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Sampled requests with top-k off and top_p < 1.

vLLM turns top-k off for top_k -1 or 0. A request that sends no top_k gets
the checkpoint's ``generation_config.json`` default instead (top_k 20 on
Qwen3.8), so only a request that turns top-k off explicitly reaches this path.

On a speculating launch whose model declares ``supports_spec_device_top_p``
(the Qwen3.8 MTP model) such a request stays on device-sampled verifies: the
device draws from vLLM's nucleus when it proves the nucleus lies inside its
candidates, and draws the column again from the step's logits otherwise. On
any other launch the request samples on the host. Either way it must
complete, repeat under a seed and vary without one. Which path served it is in
the server log: the Qwen3.8 MTP model logs ``[spec_engine] top-p
certificate:`` with its count of such verifies.
"""

import pytest

from tests.tt.utils import (
    RequestConfig,
    assert_deterministic_allow_near_tie,
    assert_varied,
    run_concurrent_batch,
)

PROMPT = "Random story: "
TEMPERATURE = 0.7
TOP_P = 0.9


@pytest.mark.parametrize("top_k", [-1, 0])
def test_a_seeded_top_p_request_without_top_k_repeats(tt_server, tt_model_name, top_k):
    """A seeded request with top-k off and top_p 0.9 gives the same completion
    on each of four runs (one near-tie divergence excused, as for top-k 50)."""
    config = RequestConfig(
        prompt=PROMPT,
        max_tokens=32,
        temperature=TEMPERATURE,
        top_k=top_k,
        top_p=TOP_P,
        seed=7,
    )
    results = []
    for _ in range(4):
        results.extend(run_concurrent_batch(tt_server, tt_model_name, [config]))
    assert all(results), f"empty completions: {results}"
    assert_deterministic_allow_near_tie(
        results, "A seeded request with top-k off should repeat."
    )


def test_unseeded_top_p_requests_without_top_k_vary(tt_server, tt_model_name):
    """Eight concurrent unseeded requests with top-k off and top_p 0.9 complete
    and do not all give the same text."""
    configs = [
        RequestConfig(
            prompt=PROMPT,
            max_tokens=32,
            temperature=TEMPERATURE,
            top_k=-1,
            top_p=TOP_P,
        )
        for _ in range(8)
    ]
    results = run_concurrent_batch(tt_server, tt_model_name, configs)
    assert all(r for r in results), f"empty completions: {results}"
    assert_varied(results, 2, "Unseeded requests with top-k off should vary.")
