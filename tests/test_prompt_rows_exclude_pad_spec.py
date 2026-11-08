# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A WAITING request whose prefix-cache hit leaves exactly one uncached prompt
token is padded by vLLM to ``1 + num_spec_tokens`` scheduled tokens with
``[-1] * num_spec_tokens`` placeholder drafts (scheduler.py, "Pad new decode
requests to uniform spec decoding size"); vLLM rolls the rejected placeholders
back after the step. The runner's prefill row must end at the prompt, not
``num_spec_tokens`` positions past it (tt-metal qwen36 spec engine crash
``slot 1: start_pos 1409 != prompt length 1412``, 2026-10-19).
"""

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

from tests.spec.test_spec_async import (
    BLOCK_SIZE,
    DRAFT_LEN,
    DeferredVerifyTarget,
    _runner,
)
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

PROMPT = BLOCK_SIZE + 1  # one full block cached, one prompt token left


def _chunk_len(num_scheduled_tokens, scheduled_drafts):
    # Imported here so the build_model_input test below still runs (and fails on
    # its own assertion) on a runner without the helper.
    from vllm_tt_plugin.model_runner import _prompt_chunk_len

    return _prompt_chunk_len(num_scheduled_tokens, scheduled_drafts)


def test_padded_new_request_computes_one_prompt_token():
    assert _chunk_len(1 + DRAFT_LEN, [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN) == 1


def test_rows_without_drafts_are_unchanged():
    assert _chunk_len(1409, None) == 1409
    assert _chunk_len(7, []) == 7


def test_real_drafts_on_a_prompt_row_raise():
    with pytest.raises(ValueError, match="real draft tokens"):
        _chunk_len(1 + DRAFT_LEN, [5, 6, 7])


def test_placeholders_without_a_prompt_token_raise():
    with pytest.raises(ValueError, match="no prompt token left"):
        _chunk_len(DRAFT_LEN, [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN)


def _padded_new_request(req_id: str) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(range(1000, 1000 + PROMPT)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=PROMPT - 1,
        lora_request=None,
    )


def _padded_scheduler_output(req: NewRequestData) -> SchedulerOutput:
    """vLLM waiting-loop output for the padded request: 1 + K tokens, K placeholders."""
    out = SchedulerOutput.make_empty()
    out.scheduled_new_reqs = [req]
    out.scheduled_cached_reqs = CachedRequestData.make_empty()
    out.num_scheduled_tokens = {req.req_id: 1 + DRAFT_LEN}
    out.total_num_scheduled_tokens = 1 + DRAFT_LEN
    drafts = [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN
    out.scheduled_spec_decode_tokens = {req.req_id: drafts}
    return out


def test_build_model_input_prefill_row_ends_at_the_prompt():
    """The prefill of the padded request computes position PROMPT-1 only.

    prompt_lens is the prompt length, the token slice stops at the prompt, and the
    chunk is final (no intermediate-prefill mask).
    """
    runner = _runner(DeferredVerifyTarget())
    out = _padded_scheduler_output(_padded_new_request("pad"))
    runner._update_states(out)
    model_input = runner.build_model_input(out, None)
    assert model_input is not None and model_input.prompt_lens is not None
    assert model_input.input_positions.tolist() == [PROMPT - 1]
    assert model_input.prompt_lens.tolist() == [PROMPT], model_input.prompt_lens
    assert tuple(model_input.input_tokens.shape) == (1, PROMPT)
    assert model_input.intermediate_prefill_mask.tolist() == [False]
