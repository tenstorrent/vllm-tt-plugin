# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Rows with top-k off and top_p below 1 on device verifies (``supports_spec_device_top_p``).

The engine-level behavior (such a row's steps are device-sampled verifies that
never chain and never have a prefill behind them) is covered by
``tests/test_mamba_spec_async.py``; these cases pin the declaration check and
the row test on their own.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.model_input import spec_verify_holds_top_p_rows
from vllm_tt_plugin.spec_admission import validate_spec_device_top_p
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS

VOCAB = 100


@pytest.mark.parametrize(
    "capabilities, match",
    [
        pytest.param({"supports_spec_device_top_p": 1}, "is a bool", id="not-bool"),
        pytest.param(
            {"supports_spec_device_top_p": True}, "spec_ids_are_sampled", id="no-draws"
        ),
    ],
)
def test_a_declaration_the_plugin_cannot_honor_raises(capabilities, match):
    with pytest.raises(ValueError, match=match):
        validate_spec_device_top_p(capabilities, "M")


def test_a_declaration_with_device_draws_is_accepted():
    validate_spec_device_top_p(
        {"supports_spec_device_top_p": True, "spec_ids_are_sampled": True}, "M"
    )
    validate_spec_device_top_p({}, "M")


def _verify(temperature, top_k, top_p, mode=ACCEPT_MODE_ARGMAX_IDS, device=True):
    return SimpleNamespace(
        spec_mode=mode,
        perform_device_sampling=device,
        tt_sampling_params=SimpleNamespace(
            temperature=torch.tensor(temperature),
            top_k=torch.tensor(top_k),
            top_p=torch.tensor(top_p),
        ),
        row_req_ids=[f"r{i}" for i in range(len(temperature))],
        live_rows=None,
    )


@pytest.mark.parametrize(
    "temperature, top_k, top_p, holds",
    [
        pytest.param([0.8, 0.0], [VOCAB, 1], [0.9, 1.0], True, id="top-k-off-top-p"),
        pytest.param([0.8], [VOCAB], [1.0], False, id="top-p-one"),
        pytest.param([0.8], [20], [0.9], False, id="top-k-on"),
        pytest.param([0.0], [VOCAB], [0.9], False, id="greedy"),
    ],
)
def test_a_verify_holds_a_top_p_row_with_top_k_off(temperature, top_k, top_p, holds):
    assert (
        spec_verify_holds_top_p_rows(_verify(temperature, top_k, top_p), VOCAB) is holds
    )


def test_only_a_device_sampled_argmax_ids_verify_draws_such_rows():
    row = ([0.8], [VOCAB], [0.9])
    assert not spec_verify_holds_top_p_rows(_verify(*row, device=False), VOCAB)
    assert not spec_verify_holds_top_p_rows(
        _verify(*row, mode=ACCEPT_MODE_LOGITS), VOCAB
    )
