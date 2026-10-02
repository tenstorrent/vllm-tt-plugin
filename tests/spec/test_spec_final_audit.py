# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from tests.spec.test_spec_async import (
    DeferredVerifyTarget,
    _accept_everything,
    _admit,
    _drain,
    _runner,
    _submit_step,
)
from tests.spec.test_spec_async import (
    wait_for_released_reads as wait_for_released_reads,
)
from tests.test_block_scheduler import _request, _runner_output, _scheduler
from vllm_tt_plugin.scheduler import get_tt_forced_reset_discard_counts


def _chunked_scheduler():
    scheduler = _scheduler(output_width=1, async_scheduling=True)
    scheduler.max_num_scheduled_tokens = 16
    scheduler.scheduler_config.enable_chunked_prefill = True
    request = _request(32)
    scheduler.add_request(request)
    return scheduler, request


def test_forced_reset_counts_the_final_prefill_frame():
    scheduler, request = _chunked_scheduler()
    intermediate = scheduler.schedule()
    assert request.is_prefill_chunk
    scheduler.update_from_output(intermediate, _runner_output(intermediate, []))

    final = scheduler.schedule()
    assert not request.is_prefill_chunk
    assert request.num_output_placeholders == 1
    scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = scheduler.schedule()

    assert request.async_tokens_to_discard == 1
    assert get_tt_forced_reset_discard_counts(resumed) == {request.request_id: 1}
    scheduler.update_from_output(final, _runner_output(final, [7]))
    assert request.async_tokens_to_discard == 0


def test_intermediate_prefill_does_not_consume_a_later_output_frame():
    scheduler, request = _chunked_scheduler()
    intermediate = scheduler.schedule()
    assert request.is_prefill_chunk
    final = scheduler.schedule()
    assert not request.is_prefill_chunk

    scheduler.update_from_output(intermediate, _runner_output(intermediate, []))
    assert scheduler._pending_async_output_frames == {request.request_id: 1}
    scheduler.update_from_output(final, _runner_output(final, [7]))
    assert scheduler._pending_async_output_frames == {}


def test_length_clipped_verify_keeps_submitted_committed_positions():
    model = DeferredVerifyTarget()
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    _admit(runner, ("a", 11))
    runner.model_config.max_model_len = 5

    step = _submit_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    model.release()
    assert len(step.get_output().sampled_token_ids[0]) == 1
    _drain(runner, "a")

    assert model.proposal_positions[-1][0].tolist() == [4, 5, 6, 7]
