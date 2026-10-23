# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A row reused within one step keeps the new request's logits-processor state.

``TTModelRunner._update_states`` removes finished requests from the
front-packed ``InputBatch`` and places new ones into the freed rows in the same
step, before ``refresh_logitsprocs``. vLLM's builtin processors apply a batch
update's ``added`` entries before its ``removed`` ones, so a reused row that is
also still listed as removed has the new request's state set and then cleared.
min_tokens, logit_bias and min_p then go unapplied for the life of that
request. This drives ``InputBatch`` the way ``_update_states`` does and reads
the processors' effect on a row of logits.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils import torch_utils
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch

VOCAB = 16
BLOCK = 16
MAX_MODEL_LEN = 128


@pytest.fixture(autouse=True)
def _disable_pinned_memory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)


def _batch() -> InputBatch:
    config = SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    return InputBatch(
        max_num_reqs=4,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        logitsprocs=build_logitsprocs(
            config, torch.device("cpu"), is_pin_memory=False, is_pooling_model=False
        ),
    )


def _request(req_id: str, **params) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(**params),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[],
    )


def _processed(batch: InputBatch) -> torch.Tensor:
    """Uniform logits after every builtin processor, one row per request."""
    logits = torch.zeros(batch.num_reqs, VOCAB)
    procs = batch.sampling.logitsprocs
    for processor in (*procs.non_argmax_invariant, *procs.argmax_invariant):
        logits = processor.apply(logits)
    return logits


@pytest.mark.parametrize(
    "params, check",
    [
        (
            {"min_tokens": 4, "stop_token_ids": [5]},
            lambda row: row[5] == float("-inf"),
        ),
        ({"logit_bias": {7: 3.0}}, lambda row: row[7] == 3.0),
    ],
    ids=["min_tokens", "logit_bias"],
)
def test_a_request_placed_in_a_row_freed_this_step_keeps_its_processor_state(
    params, check
):
    batch = _batch()
    batch.add_request(_request("old", temperature=0.0))
    batch.add_request(_request("kept", temperature=0.0))
    batch.refresh_logitsprocs()

    # The order ``_update_states`` uses: remove, place into the freed row,
    # condense, then refresh.
    freed = batch.remove_request("old")
    batch.add_request(_request("new", temperature=1.0, **params), freed)
    batch.condense([])
    batch.refresh_logitsprocs()

    assert batch.req_id_to_index["new"] == freed
    row = _processed(batch)[freed]
    assert check(row), row


def test_min_p_survives_a_row_reuse_too():
    """min_p is argmax invariant, so it is the other processor list."""
    batch = _batch()
    batch.add_request(_request("old", temperature=1.0, min_p=0.3))
    batch.refresh_logitsprocs()
    freed = batch.remove_request("old")
    batch.add_request(_request("new", temperature=1.0, min_p=0.5), freed)
    batch.condense([])
    batch.refresh_logitsprocs()

    logits = torch.zeros(batch.num_reqs, VOCAB)
    logits[freed, 0] = 4.0
    for processor in batch.sampling.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)
    # Every other token has probability far below half the top one's.
    assert bool((logits[freed, 1:] == float("-inf")).all()), logits[freed]
