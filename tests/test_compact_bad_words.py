# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.sample.sampler import Sampler

from tests.test_lane_input_batch import (
    BLOCK,
    MAX_MODEL_LEN,
    _make_logitsprocs,
    _make_req,
    _plan,
    _step_output,
)
from tests.test_lane_input_batch import (
    _disable_pinned_memory as _disable_pinned_memory,
)
from vllm_tt_plugin.host_sampler import create_host_sampler
from vllm_tt_plugin.input_batch import TTLaneInputBatch
from vllm_tt_plugin.model_input import TTCompactedHostLogits


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("output_mode", ["prefill", "decode", "compact_readback"])
def test_compact_bad_words_preserves_retained_output_history(resumed, output_mode):
    vocab = 4096
    batch = TTLaneInputBatch(
        num_lanes=2,
        per_lane=4,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * 8,
        vocab_size=vocab,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        logitsprocs=_make_logitsprocs(8, with_custom=False),
    )
    target = _make_req("target", [1, 2], [7], {"temperature": 1.0}, seed=42)
    target.sampling_params._bad_words_token_ids = [[7, 8]]
    idle = _make_req(
        "idle", [3], [7], {"temperature": 1.0, "presence_penalty": 0.5}, seed=99
    )
    idle.sampling_params._bad_words_token_ids = [[7, 9]]
    requests = {"target": target, "idle": idle}
    batch.add_request_to_row(target, 6)
    batch.add_request_to_row(idle, 1)

    if resumed:
        batch.apply_step_plan(
            _step_output(preempted=["target"]),
            _plan({"idle": 1}, is_decode=True, capacity=8, num_lanes=2),
            requests,
            {},
        )
        assert requests["target"] is target
        assert target.output_token_ids == [7]
        assert "target" not in batch.req_id_to_index
        cached = CachedRequestData(
            req_ids=["target"],
            resumed_req_ids={"target"},
            new_token_ids=[[]],
            all_token_ids={"target": [1, 2, 7]},
            new_block_ids=[([7],)],
            num_computed_tokens=[0],
            num_output_tokens=[1],
        )
        batch.apply_step_plan(
            _step_output(cached=cached),
            _plan({"target": 6}, capacity=8, num_lanes=2),
            requests,
            {},
        )
        assert requests["target"] is target
        assert batch.req_id_to_index == {"idle": 1, "target": 6}

    batch.refresh_logitsprocs()
    assert batch.can_compact_host_sampling()
    rows = [6]
    is_decode = output_mode != "prefill"
    logits = torch.full((8 if is_decode else 1, vocab), -torch.inf)
    logits[:, 8], logits[:, 9] = 100.0, 90.0
    initial_rng = target.generator.get_state().clone()
    idle_rng = idle.generator.get_state().clone()
    full_metadata = batch.build_merged_sampling_metadata(rows)
    assert not full_metadata.no_penalties  # The idle row has a penalty.
    full_logits = batch._host_logits(logits.clone(), rows, is_decode, 8)
    expected = Sampler()(logits=full_logits, sampling_metadata=full_metadata)
    assert expected.sampled_token_ids[6].item() == 9
    expected_rng = target.generator.get_state().clone()
    target.generator.set_state(initial_rng)

    sampler = create_host_sampler()
    seen = []

    def sample(*, logits, sampling_metadata):
        seen.append(sampling_metadata)
        return sampler(logits=logits, sampling_metadata=sampling_metadata)

    model_output = logits.clone()
    if output_mode == "compact_readback":
        model_output = TTCompactedHostLogits(logits[rows].clone(), tuple(rows))
    try:
        actual, _ = batch.extract_output(
            SimpleNamespace(host_sampler=sample),
            model_output,
            None,
            SimpleNamespace(
                perform_device_sampling=False,
                grammar_bitmask=[None],
                intermediate_prefill_mask=None,
            ),
            rows,
            is_decode,
        )
        assert actual.item() == 9  # Token 8 completes the forbidden [7, 8].
        assert seen[0].no_penalties
        assert seen[0].prompt_token_ids is None
        assert seen[0].output_token_ids == [[7]]
        assert seen[0].bad_words_token_ids == {0: [[7, 8]]}
        assert torch.equal(target.generator.get_state(), expected_rng)
        assert torch.equal(idle.generator.get_state(), idle_rng)
        # The vocabulary and complete seeded batch must enter the optimized path.
        assert sampler.topk_topp_sampler._sampling_pool is not None
    finally:
        pool = sampler.topk_topp_sampler._sampling_pool
        if pool is not None:
            pool.shutdown()
