# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-lane draft length by lane load, on a MiMo-shaped lane launch.

MiMo-V2.6-Flash verifies a lane's candidate tokens one per decode row
(``lanes_per_request == 1 + K``), so a lane of ``per_lane`` rows verifies at
most ``per_lane`` candidate tokens in one pass, and its drafter drafts by the
lane's load: the full K for a request alone, one draft each for two, none for
three or more. Its plan says the same through ``k_by_rows``, which lane mode
reads per lane. The plugin keeps every lane's verify inside one pass: the cap
for the lane's request count, then trimming the drafts a lane's requests
carry until the lane fits, such as the K drafts a request carried into a step
where a second request joined its lane.

The layout is two lanes of four slots, MiMo's per-lane batch, driven through
``test_spec_lane_async``'s engine, synchronously and asynchronously.
"""

from __future__ import annotations

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_tt_plugin.input_batch import TTLaneInputBatch
from vllm_tt_plugin.spec_accept import accept_speculated_tokens

from .lane_device_target import greedy_chain, lane_device_target
from .sampled_target import SAMPLED_VOCAB_SIZE, rule_logits
from .test_spec_lane_async import (
    BLOCK_SIZE,
    MAX_MODEL_LEN,
    PROMPT_LEN,
    K,
    LaneEngine,
    chain_counters,
    expected_greedy,
    lane_runner,
)

NUM_LANES = 2
PER_LANE = 4
MIMO_K_BY_ROWS = ((1, K), (2, 1), (PER_LANE, 1))


def mimo_target(**knobs):
    knobs = {
        "draft_policy": "lane_load",
        "k_by_rows": MIMO_K_BY_ROWS,
        "lanes_per_request": 1 + K,
        **knobs,
    }
    return lane_device_target(**knobs)()


def mimo_engine(model, *, async_scheduling: bool = True) -> LaneEngine:
    runner = lane_runner(
        model,
        async_scheduling=async_scheduling,
        num_lanes=NUM_LANES,
        per_lane=PER_LANE,
    )
    return LaneEngine(runner, depth=2 if async_scheduling else 1)


def assert_one_pass(model) -> None:
    """Every verify fit each lane's ``per_lane`` rows."""
    for call in model.verify_calls:
        assert all(rows <= PER_LANE for rows in call["lane_rows"].values()), call[
            "lane_rows"
        ]
        assert set(model.passes(call).values()) <= {1}


def live_valid(call) -> dict[int, int]:
    """Each slot one verify carries with a position, which the device decodes,
    and the drafts it verified."""
    return {
        row: int(call["num_valid_drafts"][row])
        for row in range(int(call["positions"].shape[0]))
        if int(call["positions"][row, 0]) >= 0
    }


def first_output(engine, req_id) -> list[int]:
    """A request's tokens after its prompt: the prefill's, then its decodes'."""
    first = [engine.first[req_id]] if req_id in engine.prefilled else []
    return first + engine.outputs[req_id]


# region The lane budget


def lane_batch() -> TTLaneInputBatch:
    return TTLaneInputBatch(
        num_lanes=NUM_LANES,
        per_lane=PER_LANE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=64,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )


@pytest.mark.parametrize(
    "k_by_rows, lanes_per_request, carried, expected",
    [
        # Alone: the full K, one pass of 1 + K rows.
        (MIMO_K_BY_ROWS, 1 + K, {0: K}, {0: K}),
        # A request joins one that carried K: the cap for two keeps one draft.
        (MIMO_K_BY_ROWS, 1 + K, {0: K, 1: 0}, {0: 1, 1: 0}),
        # Two that drafted one each: four rows, one pass.
        (MIMO_K_BY_ROWS, 1 + K, {0: 1, 1: 1}, {0: 1, 1: 1}),
        # Three: the cap allows one each, the lane fits one draft in total,
        # taken from the highest slot first among equals.
        (MIMO_K_BY_ROWS, 1 + K, {0: 1, 1: 1, 2: 1}, {0: 1, 1: 0, 2: 0}),
        (MIMO_K_BY_ROWS, 1 + K, {0: 1, 1: 1, 2: 1, 3: 1}, dict.fromkeys(range(4), 0)),
        # No cap mapping: the fit alone trims, the largest carrier first.
        ((), 1 + K, {0: K, 1: 0}, {0: 2, 1: 0}),
        ((), 1 + K, {0: K, 1: K}, {0: 1, 1: 1}),
        # A plan whose block rides its slot is only capped, per lane: lane 1
        # holds one request and keeps K whatever lane 0 holds.
        (
            MIMO_K_BY_ROWS,
            1,
            {0: K, 1: K, 2: K, 4: K},
            {0: 1, 1: 1, 2: 1, 4: K},
        ),
        ((), 1, {0: K, 1: K}, {0: K, 1: K}),
    ],
)
def test_a_lane_s_budget_is_its_cap_then_its_rows(
    k_by_rows, lanes_per_request, carried, expected
):
    plan = lane_device_target(
        k_by_rows=k_by_rows, lanes_per_request=lanes_per_request
    ).spec_plan(None, NUM_LANES * PER_LANE, K)
    assert lane_batch().lane_draft_budget(plan, K, carried) == expected


def test_a_slot_the_step_leaves_out_still_fills_a_row():
    """A partial prefill or a finishing request holding slot 1 is decoded
    with the lane: ``a`` alone is capped at K, and fits K - 1 beside it."""
    plan = lane_device_target(
        k_by_rows=MIMO_K_BY_ROWS, lanes_per_request=1 + K
    ).spec_plan(None, NUM_LANES * PER_LANE, K)
    batch = lane_batch()
    assert batch.lane_draft_budget(plan, K, {0: K}, riders=[1]) == {0: K - 1}
    # A rider in another lane takes nothing from this one.
    assert batch.lane_draft_budget(plan, K, {0: K}, riders=[5]) == {0: K}


THREE_BESIDE_A_PAIR = {0: 0, 1: 0, 2: 0, 4: 1, 5: 1}


@pytest.mark.parametrize(
    "k_by_rows, all_or_none, carried, expected",
    [
        # Lane 0 holds three requests the drafter drafted nothing for, lane 1
        # a pair with one draft each: on a chaining launch nobody verifies.
        (
            MIMO_K_BY_ROWS,
            True,
            THREE_BESIDE_A_PAIR,
            dict.fromkeys(THREE_BESIDE_A_PAIR, 0),
        ),
        # A lane of three keeping one draft still leaves two slots without.
        (
            MIMO_K_BY_ROWS,
            True,
            {0: 1, 1: 1, 2: 1, 4: 1, 5: 1},
            dict.fromkeys(THREE_BESIDE_A_PAIR, 0),
        ),
        # Every slot keeps a draft: the step verifies.
        (MIMO_K_BY_ROWS, True, {0: 1, 1: 1, 4: K}, {0: 1, 1: 1, 4: K}),
        # Without chaining, or without a load-sized plan, the mixed verify stays.
        (MIMO_K_BY_ROWS, False, THREE_BESIDE_A_PAIR, THREE_BESIDE_A_PAIR),
        ((), True, {0: 0, 1: 0, 4: K}, {0: 0, 1: 0, 4: K}),
    ],
)
def test_a_chaining_launch_verifies_only_when_every_slot_keeps_a_draft(
    k_by_rows, all_or_none, carried, expected
):
    plan = lane_device_target(k_by_rows=k_by_rows, lanes_per_request=1 + K).spec_plan(
        None, NUM_LANES * PER_LANE, K
    )
    budget = lane_batch().lane_draft_budget(plan, K, carried, all_or_none=all_or_none)
    assert budget == expected


# endregion The lane budget


# region Joins, pairs and crowds


@pytest.mark.parametrize("async_scheduling", [True, False], ids=["async", "sync"])
@pytest.mark.parametrize("k_by_rows", [MIMO_K_BY_ROWS, ()], ids=["mimo", "no-cap"])
def test_a_request_joining_a_speculating_lane_keeps_its_verify_one_pass(
    async_scheduling, k_by_rows
):
    """``a`` speculates alone in lane 0 with K drafts; ``b`` joins it. The K
    drafts ``a`` carried into the first shared step are trimmed so the lane
    fits its four rows, rather than five rows verified in two passes, and
    both requests emit the argmax chain. A chaining launch whose plan sizes
    drafts by load runs that step, where ``b`` carries none, as the ordinary
    decode instead (``TTLaneInputBatch.lane_draft_budget``)."""
    model = mimo_target(k_by_rows=k_by_rows)
    engine = mimo_engine(model, async_scheduling=async_scheduling)
    prompt_a = engine.add("a", 0, 1)
    engine.run(6)
    assert any(live_valid(call) == {0: K} for call in model.verify_calls)
    joined_at = len(model.calls)
    prompt_b = engine.prefill("b", 1, 9)
    engine.run(10)
    engine.drain()

    assert_one_pass(model)
    shared = [
        live_valid(call)
        for call in model.calls[joined_at:]
        if call["spec_mode"] is not None and 1 in live_valid(call)
    ]
    assert shared, "no verify carried both requests"
    # The first shared verify: what ``a`` carried in, trimmed to fit.
    trimmed = {0: 1 if k_by_rows else 2, 1: 0}
    if async_scheduling and k_by_rows:
        assert trimmed not in shared
    else:
        assert shared[0] == trimmed
    # Then the drafter drafts one each for the pair.
    assert {0: 1, 1: 1} in shared
    produced = engine.outputs["a"]
    assert produced == expected_greedy(prompt_a, len(produced))
    produced_b = first_output(engine, "b")
    assert produced_b == greedy_chain(prompt_b[-1], PROMPT_LEN - 1, len(produced_b))


def test_two_requests_in_a_lane_speculate_one_draft_each_and_chain():
    """Lane 0 holds ``a`` and ``b``, lane 1 ``c`` alone. The pair verifies one
    draft each, ``c`` K, all in one pass per lane, and every verify after the
    first chains onto the one in flight with both of lane 0's requests as
    chain rows."""
    model = mimo_target()
    engine = mimo_engine(model)
    prompts = {
        "a": engine.add("a", 0, 1),
        "b": engine.add("b", 1, 9),
        "c": engine.add("c", 4, 17),
    }
    engine.run(16)
    engine.drain()

    assert_one_pass(model)
    chained, breaks = chain_counters(engine.runner)
    assert chained >= 12
    assert "draft_count" not in breaks
    for call in model.chained_calls:
        assert call["chain_rows"].nonzero().flatten().tolist() == [0, 1, 4]
        assert live_valid(call) == {0: 1, 1: 1, 4: K}
        assert call["lane_rows"] == {0: 4, 1: 1 + K}
    for req_id, tokens in prompts.items():
        produced = engine.outputs[req_id]
        assert produced == expected_greedy(tokens, len(produced))
        assert len(produced) > 16, "no draft was ever accepted"


@pytest.mark.parametrize("async_scheduling", [True, False], ids=["async", "sync"])
def test_a_lane_of_three_runs_plain(async_scheduling):
    """Three requests in lane 0: the drafter drafts nothing for them. Alone,
    every step is the ordinary decode. Beside ``d`` speculating alone in lane
    1, they ride its verifies with nothing to verify, three rows in one pass;
    on a chaining launch such a verify could not be chained onto, and the
    step stays the ordinary decode (``TTLaneInputBatch.lane_draft_budget``)."""
    model = mimo_target()
    engine = mimo_engine(model, async_scheduling=async_scheduling)
    prompts = {
        "a": engine.add("a", 0, 1),
        "b": engine.add("b", 1, 9),
        "c": engine.add("c", 2, 17),
    }
    engine.run(6)
    assert model.verify_calls == []
    prompts["d"] = engine.prefill("d", 4, 25)
    engine.run(10)
    engine.drain()

    assert_one_pass(model)
    if async_scheduling:
        assert model.verify_calls == []
    else:
        assert any(live_valid(call).get(4) == K for call in model.verify_calls)
    for call in model.verify_calls:
        valid = live_valid(call)
        assert [valid[row] for row in (0, 1, 2)] == [0, 0, 0]
        assert call["lane_rows"][0] == 3
    for req_id in "abc":
        produced = engine.outputs[req_id]
        assert produced == expected_greedy(prompts[req_id], len(produced))
    produced_d = first_output(engine, "d")
    assert produced_d == greedy_chain(prompts["d"][-1], PROMPT_LEN - 1, len(produced_d))


# endregion Joins, pairs and crowds


# region Transitions


def finish_soon(engine, req_id: str) -> None:
    """Finish ``req_id`` ten tokens on, through its ``max_tokens``."""
    engine.max_tokens[req_id] = len(engine.history(req_id)) + 10
    engine.stop_after[req_id] = len(engine.outputs[req_id]) + 10


def on_submit(record):
    """A ``run_transitions`` hook calling ``record(runner, model_input)`` at
    each decode submission, before the step draws anything."""

    def hook(engine):
        runner = engine.runner
        original = runner.async_decode.submit_decode

        def submit(model_input, **kwargs):
            record(runner, model_input)
            return original(model_input, **kwargs)

        runner.async_decode.submit_decode = submit

    return hook


def scheduled_slots(model_input) -> dict[int, str]:
    """The slots a decode schedules, and their requests."""
    return {
        row: req_id
        for row, req_id in enumerate(model_input.row_req_ids or ())
        if req_id is not None
    }


def run_transitions(async_scheduling: bool, params=None, hook=None):
    """Lane 0 goes 1 -> 2 -> 1 -> 3 -> 2 requests while ``c`` speculates alone
    in lane 1. ``params`` maps a request to its ``SamplingParams``, and
    ``hook`` sees the engine before the first step. Returns the model, the
    engine and every request's prompt."""
    params = params or {}
    model = mimo_target()
    engine = mimo_engine(model, async_scheduling=async_scheduling)
    if hook is not None:
        hook(engine)

    def kwargs(req_id):
        return {"params": params[req_id]} if req_id in params else {}

    prompts = {
        "a": engine.add("a", 0, 1, **kwargs("a")),
        "c": engine.add("c", 4, 17, **kwargs("c")),
    }
    engine.run(6)
    prompts["b"] = engine.prefill("b", 1, 9, **kwargs("b"))
    engine.run(8)
    finish_soon(engine, "b")
    engine.run(14)
    assert "b" in engine.done
    prompts["d"] = engine.prefill("d", 2, 25, **kwargs("d"))
    prompts["e"] = engine.prefill("e", 3, 33, **kwargs("e"))
    engine.run(8)
    finish_soon(engine, "d")
    engine.run(14)
    engine.drain()
    assert "d" in engine.done
    return model, engine, prompts


def test_lane_transitions_emit_the_synchronous_tokens():
    """Token for token, greedy, asynchronous against synchronous, through
    every lane load, with every verify one pass per lane."""
    sync_model, sync_engine, prompts = run_transitions(False)
    model, engine, _ = run_transitions(True)

    for run_model in (sync_model, model):
        assert_one_pass(run_model)
        lane_0 = [
            [row for row in live_valid(call) if row < PER_LANE]
            for call in run_model.verify_calls
        ]
        assert {1, 2, 3} <= {len(rows) for rows in lane_0}
    assert chain_counters(engine.runner)[0] > 0
    for req_id, prompt in prompts.items():
        sync_tokens = first_output(sync_engine, req_id)
        async_tokens = first_output(engine, req_id)
        if req_id in sync_engine.prefilled:
            expected = greedy_chain(prompt[-1], PROMPT_LEN - 1, len(async_tokens))
        else:
            expected = expected_greedy(prompt, len(async_tokens))
        assert async_tokens == expected
        n = min(len(sync_tokens), len(async_tokens))
        assert n > 0
        assert async_tokens[:n] == sync_tokens[:n]


def test_sampled_verifies_through_lane_transitions_are_exact():
    """Seeded sampled requests through the same transitions: every fused
    verify commits exactly what the ``logits`` walk commits over the block the
    device verified, from the generator states before the step, so the
    uniforms were drawn for the trimmed draft counts."""
    params = {
        req_id: SamplingParams(temperature=1.0, top_p=0.95, seed=seed, ignore_eos=True)
        for seed, req_id in enumerate("abcde", start=3)
    }
    # Per submission: each slot's generator state, and the slots it schedules.
    snapshots: list[dict[int, torch.Tensor]] = []
    scheduled: list[dict[int, str]] = []

    def record(runner, model_input):
        generators = runner.lane_batch.sampling.generators
        snapshots.append({slot: g.get_state() for slot, g in generators.items()})
        scheduled.append(scheduled_slots(model_input))

    model, _, _ = run_transitions(True, params, on_submit(record))

    assert len(snapshots) == len(model.calls)
    checked = 0
    for index, call in enumerate(model.calls):
        if call["spec_mode"] != "fused_sample":
            continue
        states = snapshots[index]
        for row in scheduled[index]:
            valid = int(call["num_valid_drafts"][row])
            generator = torch.Generator(device="cpu")
            generator.set_state(states[row])
            tokens = call["device_tokens"][row : row + 1]
            positions = call["device_positions"][row : row + 1]
            result = accept_speculated_tokens(
                target_logits=rule_logits(tokens, positions),
                draft_token_ids=tokens[:, 1:],
                num_valid_drafts=torch.tensor([valid], dtype=torch.int32),
                temperature=torch.tensor([1.0]),
                top_k=torch.tensor([SAMPLED_VOCAB_SIZE]),
                top_p=torch.tensor([0.95]),
                generators={0: generator},
            )
            count = int(result.accepted_counts[0])
            assert int(call["accepted"][row]) == count
            assert (
                call["committed"][row, :count].tolist()
                == result.committed_token_ids[0, :count].tolist()
            )
            if index + 1 < len(snapshots) and row in snapshots[index + 1]:
                assert generator.get_state().equal(snapshots[index + 1][row])
            checked += 1
    assert checked >= 30
    assert_one_pass(model)


# endregion Transitions


# region Speculative metrics


@pytest.mark.parametrize("async_scheduling", [True, False], ids=["async", "sync"])
def test_spec_metrics_count_only_the_drafts_a_verify_carried(async_scheduling):
    """Through the transitions, trims included, the drafted total is exactly
    what the verifies carried for the requests whose output vLLM observed,
    synchronously (where the scheduler offered the untrimmed proposals) and
    asynchronously. A request that stops on a token of a verify in flight has
    the next verify's output for it thrown away, and none of its drafts
    observed."""
    scheduled: list[dict[int, str]] = []
    model, engine, _ = run_transitions(
        async_scheduling,
        hook=on_submit(
            lambda _, model_input: scheduled.append(scheduled_slots(model_input))
        ),
    )
    assert len(scheduled) == len(model.calls) == len(engine.observed)
    drafted = 0
    for call, rows, observed in zip(model.calls, scheduled, engine.observed):
        if call["spec_mode"] is None:
            continue
        for row, req_id in rows.items():
            if observed.get(req_id):
                drafted += int(call["num_valid_drafts"][row])
    stats = engine.stats.pending_spec_decoding_stats
    assert stats is not None
    assert stats.num_draft_tokens == drafted
    assert 0 < stats.num_accepted_tokens < drafted


# endregion Speculative metrics
