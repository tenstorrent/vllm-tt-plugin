# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""An asynchronous runner over ``SampledTarget`` whose readback is held open.

Shared by the sampled spec tests that drive a ``logits`` verify through the
deferred path: ``DeferredSampledTarget`` fills its verify logits only when a
test releases them.
"""

from __future__ import annotations

import threading

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.sample.sampler import Sampler

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.model_runner import TTModelRunner

from . import test_spec_async as async_harness
from .sampled_target import SAMPLED_VOCAB_SIZE, SampledTarget, base_token

DRAFT_LEN = async_harness.DRAFT_LEN
PROMPT_LEN = async_harness.PROMPT_LEN


class DeferredSampledTarget(SampledTarget):
    """``SampledTarget`` whose verify logits are readable only once released."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[tuple[torch.Tensor, torch.Tensor, threading.Event]] = []
        self.reads = 0
        self.wait_failure: BaseException | None = None

    def read_decode_output(self, tt_out, async_read: bool = False):
        self.reads += 1
        # NaN until released, so a walk that read the buffer early could not
        # produce a valid sample from it.
        buffer = torch.full_like(tt_out, float("nan"))
        event = threading.Event()
        self.pending.append((buffer, tt_out.clone(), event))
        return buffer, [event]

    def release(self) -> None:
        buffer, answer, event = self.pending.pop(0)
        buffer.copy_(answer)
        if self.wait_failure is not None:
            event.tt_test_failure = self.wait_failure
            self.wait_failure = None
        event.set()


def _runner(model: DeferredSampledTarget):
    runner = async_harness._runner(model)
    runner.vocab_size = SAMPLED_VOCAB_SIZE
    # An ordinary decode on a narrow-capable model samples on the host.
    runner.host_sampler = Sampler()
    runner._spec_accept_modes = type(model).accept_modes
    # The real builder: the walk samples under these tensors.
    del runner._sampling_params_for_padded_decode
    runner._sampling_params_for_padded_decode = (
        TTModelRunner._sampling_params_for_padded_decode.__get__(runner)
    )
    return runner


def _rebuild_batch(runner) -> None:
    """The harness batch was sized for the greedy target's vocabulary."""
    from vllm_tt_plugin.input_batch import InputBatch

    runner.input_batch = InputBatch(
        max_num_reqs=async_harness.MAX_NUM_REQS,
        max_model_len=async_harness.MAX_MODEL_LEN,
        max_num_batched_tokens=async_harness.MAX_MODEL_LEN,
        vocab_size=SAMPLED_VOCAB_SIZE,
        block_sizes=[async_harness.BLOCK_SIZE],
        kernel_block_sizes=[async_harness.BLOCK_SIZE],
    )


def _sampled_runner():
    model = DeferredSampledTarget()
    runner = _runner(model)
    _rebuild_batch(runner)
    return model, runner


def _admit(runner, *specs: tuple[str, int, SamplingParams]) -> None:
    """Admit requests through ``_update_states``, which seeds their generators.

    All in one scheduler output: a running request that a step does not
    schedule leaves the persistent batch.
    """
    output = async_harness._scheduler_output(runner)
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=[
                (first_token + index) % SAMPLED_VOCAB_SIZE
                for index in range(PROMPT_LEN)
            ],
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=PROMPT_LEN,
            lora_request=None,
        )
        for req_id, first_token, params in specs
    ]
    output.num_scheduled_tokens = {req_id: 1 for req_id, _, _ in specs}
    output.total_num_scheduled_tokens = len(specs)
    runner._update_states(output)


def _drafts(runner, req_id: str) -> list[int]:
    token, position = async_harness._tail(runner, req_id)
    drafts = []
    for _ in range(DRAFT_LEN):
        token = base_token(token, position)
        position += 1
        drafts.append(token)
    return drafts
