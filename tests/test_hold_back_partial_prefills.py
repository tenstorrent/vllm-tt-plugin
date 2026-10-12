"""TTScheduler._hold_back_partial_prefills: with a token budget that holds a whole prompt (max_num_batched_tokens >= max_model_len) a prefill step takes the longest FIFO prefix of
waiting prompts that fits whole, so no prompt is split across steps (a model whose prefill state is per slot restarts a split prompt from position 0)."""

import types

from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_tt_plugin.scheduler import TTScheduler


def _req(n, computed=0):
    return types.SimpleNamespace(num_tokens=n, num_computed_tokens=computed, name=n)


def _run(budget, waiting, partial=(), max_model_len=131584):
    s = types.SimpleNamespace(
        policy=SchedulingPolicy.FCFS,
        max_num_scheduled_tokens=budget,
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=max_model_len)),
    )
    s.waiting = create_request_queue(s.policy)
    for r in waiting:
        s.waiting.add_request(r)
    held = TTScheduler._hold_back_partial_prefills(s, list(partial))
    return [r.name for r in s.waiting], [r.name for r in held]


def test_one_whole_prompt_per_step_when_each_fills_the_budget():
    assert _run(131584, [_req(131076)] * 3) == ([131076], [131076, 131076])


def test_prompts_that_fit_together_are_admitted_together():
    assert _run(131584, [_req(65540)] * 4) == ([65540, 65540], [65540, 65540])


def test_fifo_order_is_kept():
    assert _run(131584, [_req(65540), _req(4000), _req(90000), _req(100)]) == ([65540, 4000], [90000, 100])


def test_first_request_is_always_admitted():
    assert _run(1000, [_req(5000), _req(10)], max_model_len=500) == ([5000], [10])


def test_untouched_when_the_budget_cannot_hold_a_whole_prompt():
    assert _run(67584, [_req(65540), _req(65540)]) == ([65540, 65540], [])


def test_running_partial_prefills_use_the_budget_first():
    assert _run(131584, [_req(60000), _req(60000)], partial=[_req(100000)]) == ([60000], [60000])
