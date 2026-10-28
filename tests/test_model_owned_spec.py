# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the model-owned speculative round's contract in the runner
(``config.MODEL_OWNED_SPEC``): the ragged output lists, the per-request state
update that appends every emitted token exactly once, the drafts handed to the
scheduler after a step, and the platform's acceptance of the config. No device,
no vLLM engine: fake runners as the other host-only tests use."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_tt_plugin.config import (
    MODEL_OWNED_SPEC,
    is_model_owned_spec,
    num_model_owned_spec_tokens,
)
from vllm_tt_plugin.model_runner import TTModelRunner, _ragged_token_lists


def test_ragged_token_lists():
    one = torch.tensor([[5], [7]], dtype=torch.int32)
    assert _ragged_token_lists(one, 2) == [[5], [7]]
    ragged = torch.tensor(
        [[1, 2, 3, 4], [9, -1, -1, -1], [6, 7, -1, -1]], dtype=torch.int32
    )
    assert _ragged_token_lists(ragged, 3) == [[1, 2, 3, 4], [9], [6, 7]]
    # a 1-D [n] vector (the prefill's device-sampled ids) is one token a row
    assert _ragged_token_lists(torch.tensor([3, 4], dtype=torch.int64), 2) == [[3], [4]]


def _runner(num_reqs, spec_k, max_len=64):
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=[f"r{i}" for i in range(num_reqs)],
        req_id_to_index={f"r{i}": i for i in range(num_reqs)},
        token_ids_cpu=np.zeros((num_reqs, max_len), dtype=np.int32),
        num_tokens=np.full(num_reqs, 10, dtype=np.int32),
        req_output_token_ids=[[] for _ in range(num_reqs)],
    )
    return SimpleNamespace(
        spec_k=spec_k,
        input_batch=batch,
        model_config=SimpleNamespace(max_model_len=max_len),
        requests={},
        _draft_token_ids=None,
        model=SimpleNamespace(take_drafts=lambda: {0: [11, 12, 13], 2: [21]}),
    )


def test_apply_ragged_tokens_to_state_appends_each_emitted_token_once():
    r = _runner(3, spec_k=3)
    ids = torch.tensor(
        [[1, 2, 3, 4], [9, -1, -1, -1], [6, 7, -1, -1]], dtype=torch.int32
    )
    TTModelRunner._apply_sampled_tokens_to_state(r, ids)
    assert r.input_batch.num_tokens.tolist() == [14, 11, 12]
    assert r.input_batch.token_ids_cpu[0, 10:14].tolist() == [1, 2, 3, 4]
    assert r.input_batch.token_ids_cpu[1, 10:12].tolist() == [9, 0]
    assert r.input_batch.token_ids_cpu[2, 10:12].tolist() == [6, 7]
    assert r.input_batch.req_output_token_ids == [[1, 2, 3, 4], [9], [6, 7]]
    out = TTModelRunner._build_runner_output(r, ids)
    assert out.sampled_token_ids == [[1, 2, 3, 4], [9], [6, 7]]
    assert out.req_ids == ["r0", "r1", "r2"]


def test_apply_one_token_unchanged_without_spec():
    r = _runner(2, spec_k=0)
    TTModelRunner._apply_sampled_tokens_to_state(
        r, torch.tensor([[5], [7]], dtype=torch.int32)
    )
    assert r.input_batch.num_tokens.tolist() == [11, 11]
    assert r.input_batch.req_output_token_ids == [[5], [7]]
    with pytest.raises(AssertionError):
        TTModelRunner._apply_sampled_tokens_to_state(
            r, torch.tensor([[5, 6], [7, -1]], dtype=torch.int32)
        )


def test_apply_captured_req_ids_path():
    r = _runner(2, spec_k=2)
    r.requests = {
        "r0": SimpleNamespace(output_token_ids=[]),
        "r1": SimpleNamespace(output_token_ids=[]),
    }
    ids = torch.tensor([[1, 2, 3], [4, -1, -1]], dtype=torch.int32)
    TTModelRunner._apply_sampled_tokens_to_state(r, ids, req_ids=["r0", "r1"])
    assert r.input_batch.num_tokens.tolist() == [13, 11]
    assert r.requests["r0"].output_token_ids == [1, 2, 3]
    assert r.requests["r1"].output_token_ids == [4]


def test_max_model_len_guard_counts_the_round():
    r = _runner(1, spec_k=3, max_len=12)
    with pytest.raises(AssertionError):
        TTModelRunner._apply_sampled_tokens_to_state(
            r, torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        )


def test_drafts_handed_once_with_an_entry_per_request():
    r = _runner(3, spec_k=3)
    TTModelRunner._record_drafts(r, is_decode=True)
    out = TTModelRunner.take_draft_token_ids(r)
    assert out.req_ids == ["r0", "r1", "r2"]
    assert out.draft_token_ids == [[11, 12, 13], [], [21]]
    assert TTModelRunner.take_draft_token_ids(r) is None  # taken once
    TTModelRunner._record_drafts(
        r, is_decode=False
    )  # a prefill step: no drafts, the placeholders cleared
    out = TTModelRunner.take_draft_token_ids(r)
    assert out.draft_token_ids == [[], [], []]


def test_model_owned_spec_config():
    cfg = SimpleNamespace(speculative_config=None, num_speculative_tokens=0)
    assert not is_model_owned_spec(cfg) and num_model_owned_spec_tokens(cfg) == 0
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="custom_class", model=MODEL_OWNED_SPEC
        ),
        num_speculative_tokens=3,
    )
    assert is_model_owned_spec(cfg) and num_model_owned_spec_tokens(cfg) == 3
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(method="ngram", model="ngram"),
        num_speculative_tokens=3,
    )
    assert not is_model_owned_spec(cfg) and num_model_owned_spec_tokens(cfg) == 0
