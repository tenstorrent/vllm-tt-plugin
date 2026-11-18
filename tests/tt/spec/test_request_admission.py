# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which request controls a speculating server admits, over HTTP.

``TTPlatform.validate_request`` decides this from the model's declared accept
modes. A launch whose model serves ``logits`` applies logprobs, structured
output, ``allowed_token_ids``, ``bad_words`` and ``min_tokens`` in its accept
walk and admits them. A launch whose model serves only ``argmax_ids`` cannot,
and refuses each with HTTP 400 rather than answering without it. ``logit_bias``
is refused on every speculating launch, by vLLM itself before the plugin.
"""

from __future__ import annotations

import pytest

CONTROLS = {
    "logprobs": {"logprobs": 1},
    "logprobs-0": {"logprobs": 0},
    "structured_outputs": {"structured_outputs": {"regex": "(2|pression)+"}},
    "allowed_token_ids": {"allowed_token_ids": [17, 4099]},
    "bad_words": {"bad_words": ["2pression"]},
    "min_tokens": {"min_tokens": 2, "stop_token_ids": [17]},
}


@pytest.mark.parametrize("name", list(CONTROLS))
def test_a_control_is_admitted_exactly_when_the_model_serves_logits(
    spec_server, spec_config, ascending_prompt, record, name
):
    result = spec_server.complete(
        ascending_prompt(16, start=3000), max_tokens=4, **CONTROLS[name]
    )
    record(status=result.status, body=result.body)
    if "logits" in spec_config.accept_modes:
        assert result.status == 200, result.body
        return
    assert result.status == 400, result.body
    assert "cannot serve" in str(result.body), result.body


def test_logit_bias_is_refused_on_every_speculating_launch(
    spec_server, ascending_prompt, record
):
    result = spec_server.complete(
        ascending_prompt(16, start=3100), max_tokens=4, logit_bias={"17": 1.0}
    )
    record(status=result.status, body=result.body)
    assert result.status == 400, result.body
