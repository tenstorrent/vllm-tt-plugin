# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for per-request device state slots.

A persistent-batch ROW is not stable for a request: ``_update_states`` evicts a
running request the step does not schedule (every prefill step does) and re-adds it
at whatever row is free, and ``condense`` moves rows down when a request finishes.
Device state indexed by slot (Qwen3.6 GDN recurrent+conv, the per-slot seed RNG, the
decode trace's token/position buffers) does not follow, so
``_alloc_prefill_state_slots`` and ``_decode_state_slot_remap`` say where each
request's state is, and ``_release_dead_state_slots`` says when a request stops
owning one. No device execution: all three are pure index bookkeeping, run here
against a fake runner.
"""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.model_runner import TTModelRunner

SLOTS = 8


def _runner(slots=SLOTS):
    """Fake runner: the state-slot map, the live-request set and the slot capacity."""
    return SimpleNamespace(
        tt_per_lane_max_num_seqs=slots,
        _req_state_slot={},
        _pending_state_slot_settle=None,
        requests={},
    )


def _prefill(runner, row_req_ids):
    out = TTModelRunner._alloc_prefill_state_slots(runner, list(row_req_ids))
    runner.requests.update(dict.fromkeys(row_req_ids))
    return out


def _decode(runner, row_req_ids):
    remap = TTModelRunner._decode_state_slot_remap(runner, list(row_req_ids))
    TTModelRunner.note_decode_state_slots_settled(runner)
    return None if remap is None else remap.tolist()


def _scheduler_output(*, preempted=None, scheduled=("KEEP",)):
    """Just the SchedulerOutput fields ``_update_states`` reads on a quiet step."""
    return SimpleNamespace(
        finished_req_ids=[],
        preempted_req_ids=preempted,
        free_encoder_mm_hashes=[],
        num_scheduled_tokens=dict.fromkeys(scheduled, 1),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )


def _gather(state, remap):
    """What the device does with a remap: row ``i`` reads slot ``remap[i]``."""
    return list(state) if remap is None else [state[s] for s in remap]


def _assert_state_found(runner, state):
    """The invariant: a request's recorded slot is where its state actually sits."""
    for req_id in runner.requests:
        slot = runner._req_state_slot[req_id]
        assert state[slot] == req_id, (
            f"{req_id} thinks its state is in slot {slot}, which holds "
            f"{state[slot]!r} (device state: {state})"
        )


def _release(runner, finished=(), preempted=None):
    """One release pass. ``preempted`` defaults to None, the value a scheduler that
    preempted nothing reports."""
    TTModelRunner._release_dead_state_slots(
        runner,
        SimpleNamespace(finished_req_ids=set(finished), preempted_req_ids=preempted),
    )


def test_state_follows_the_request_across_row_moves():
    """The full lifecycle: fresh prefill, eviction, return at a new row, re-prefill."""
    r = _runner()
    # Empty server: fresh slots equal fresh rows, so no state has to move.
    assert _prefill(r, ["A"]) == [0]
    assert _decode(r, ["A"]) is None  # identity -> consumers skip the gather

    # THE BUG. A is live but unscheduled, so vLLM gives its row 0 to a new request.
    # A prefill into slot 0 destroys A's recurrent state and A then emits garbage.
    incoming = [f"B{i}" for i in range(7)]
    slots = _prefill(r, incoming)
    assert 0 not in slots, f"prefill took A's live slot: {slots}"
    assert len(set(slots)) == 7 and all(0 < s < SLOTS for s in slots)

    # vLLM re-adds A at the first free row (7). The remap must fetch A's state from
    # the slot it sits in, and every other row from its own.
    rows = incoming + ["A"]
    remap = _decode(r, rows)
    assert remap is not None, "a returning request needs its state moved"
    assert len(remap) == SLOTS and sorted(remap) == list(range(SLOTS)), (
        "must be a permutation"
    )
    assert remap[7] == 0, f"row 7 (A) must read A's slot 0, got {remap[7]}"
    for row, s in enumerate(slots):
        assert remap[row] == s, (
            f"row {row} must read {rows[row]}'s slot {s}, got {remap[row]}"
        )
    # State now sits at each request's row, so the next step is free again.
    assert _decode(r, rows) is None

    # A preempted request is re-prefilled while it still owns its slot: it must keep
    # that slot, not move and leave the old one stranded.
    assert _prefill(r, ["B0"]) == [0], (
        "a re-prefilled live request must keep its own slot"
    )


def test_accepting_the_decode_remap_advances_the_ownership_map():
    """An accepted decode records the post-gather layout.

    Building only proposes the move; the helper also models successful submission.
    A raised decode must leave the ownership map at its pre-gather layout.
    """
    r = _runner()
    r._req_state_slot.update({"A": 3, "B": 1})
    r.requests.update(dict.fromkeys(["A", "B"]))

    assert _decode(r, ["A", "B"]) == [3, 1, 0, 2, 4, 5, 6, 7]
    assert r._req_state_slot["A"] == 0 and r._req_state_slot["B"] == 1


def test_release_keeps_a_merely_unscheduled_request():
    """Every prefill step leaves the whole decode batch unscheduled, and that state is
    still live. The release predicate must not read "unscheduled" as "dead"."""
    r = _runner()
    assert _prefill(r, ["A"]) == [0]

    _release(r)
    assert r._req_state_slot == {"A": 0}
    assert _prefill(r, ["B"]) == [1], "a live request's slot must not be reused"

    _release(r, finished={"A"})
    assert r._req_state_slot == {"B": 1}


def test_preemption_frees_the_slot_a_later_prefill_needs():
    """A preempted request's state is dead: ``_preempt_request`` freed its KV and reset
    ``num_computed_tokens``, so the resume re-prefills from zero and rewrites the slot.
    Holding it only shrinks capacity, and the shortfall is a hard error."""
    r = _runner()
    rows = [f"r{i}" for i in range(SLOTS)]
    assert _prefill(r, rows) == list(range(SLOTS))
    assert _decode(r, rows) is None

    _release(r, preempted={"r7"})
    assert "r7" not in r._req_state_slot
    # The cached request state stays: the resume needs it, and it is what keeps the
    # stale slot inside ``held``.
    assert "r7" in r.requests

    # Under ``--scheduling-policy priority`` the preempted request is re-queued by
    # priority, not to the front, so a higher-priority arrival prefills ahead of it.
    # Its slot must be free or the prefill has nowhere to go.
    assert _prefill(r, ["r8"]) == [7]


def test_a_resumed_preempted_request_gets_a_slot_and_a_valid_remap():
    """The resume is an ordinary prefill: it takes whatever slot is free and the next
    decode step gathers its state to its row."""
    r = _runner()
    rows = [f"r{i}" for i in range(SLOTS)]
    _prefill(r, rows)
    _decode(r, rows)
    _release(r, preempted={"r7"})
    _prefill(r, ["r8"])

    # r7 can resume only once the batch has room: the scheduler caps running at the
    # slot count and a preempted request does not count against it.
    _release(r, finished={"r0"})
    r.requests.pop("r0")
    assert _prefill(r, ["r7"]) == [0], "the resume takes the slot the finish freed"

    # vLLM re-adds both newcomers at the free rows, so the decode rows no longer match
    # the slots.
    decode_rows = [f"r{i}" for i in range(1, 7)] + ["r8", "r7"]
    remap = _decode(r, decode_rows)
    assert remap == [1, 2, 3, 4, 5, 6, 7, 0]
    assert sorted(remap) == list(range(SLOTS)), "must be a permutation"
    assert _decode(r, decode_rows) is None


def test_remap_carries_off_batch_state():
    """A non-identity remap permutes ALL slots, live off-batch holders' included."""
    r = _runner()
    state: list[str | None] = [None] * SLOTS
    for row, slot in enumerate(_prefill(r, ["A", "B"])):
        state[slot] = ["A", "B"][row]
    assert r._req_state_slot == {"A": 0, "B": 1}

    # Only B decodes; pulling it to row 0 pushes live off-batch A out of slot 0.
    remap = _decode(r, ["B"])
    assert remap is not None and remap[0] == 1, f"row 0 must read B's slot 1: {remap}"
    state = _gather(state, remap)
    assert state[0] == "B"
    assert state[1] == "A", "A's state was displaced by the gather"
    _assert_state_found(r, state)

    # A is rescheduled: its recorded slot must be the one it landed in.
    remap = _decode(r, ["B", "A"])
    state = _gather(state, remap)
    _assert_state_found(r, state)
    assert state[:2] == ["B", "A"], f"state must sit at each request's row: {state}"


def test_preemption_releases_its_state_slot():
    """The wiring: ``_update_states`` must actually call the release pass. The tests
    above drive ``_release_dead_state_slots`` directly and would not notice its loss."""
    r = _runner()
    r.encoder_cache = {}
    r._decode_layout_changed_since_last_decode = False
    r.input_batch = SimpleNamespace(
        req_id_to_index={"KEEP": 0}, refresh_logitsprocs=lambda: None
    )
    r._release_dead_state_slots = lambda so: _release(
        r, finished=so.finished_req_ids, preempted=so.preempted_req_ids
    )
    released: list[int] = []
    r.model = SimpleNamespace(release_request=released.append)
    r._release_model_request = lambda req_id: TTModelRunner._release_model_request(
        r, req_id
    )
    r._req_state_slot.update({"P": 0, "KEEP": 1})
    r.requests.update(dict.fromkeys(["P", "KEEP"]))

    TTModelRunner._update_states(r, _scheduler_output(preempted={"P"}))

    assert r._req_state_slot == {"KEEP": 1}, "only the preempted request releases"
    assert released == [0], "the model releases the preempted request's slot"
    assert "P" in r.requests, "the request is still live, it just re-prefills"

    # Which is the point: the freed slot is available to the incoming prefill.
    assert _prefill(r, ["NEW"]) == [0]


def test_slot_exhaustion_fails_instead_of_guessing():
    """Exhaustion means the map has stopped describing the device, and it is the only
    record of slot ownership. Guessing returns plausible, wrong text. A raise, not an
    assert, so the diagnostic survives ``python -O``."""
    r = _runner(slots=2)
    _prefill(r, ["A", "B"])  # both slots held by live requests
    with pytest.raises(RuntimeError, match="no free device state slot"):
        _prefill(r, ["C"])

    # Over-capacity is the scheduler's decision to make, not this function's.
    with pytest.raises(RuntimeError, match="exceed the 2 device state slots"):
        _prefill(_runner(slots=2), ["A", "B", "C"])


def test_more_decode_rows_than_slots_raises():
    """Truncating to the slot width would silently drop C's state instead of saying
    the batch cannot be described."""
    r = _runner(slots=2)
    r._req_state_slot.update({"A": 1, "B": 0})
    assert _decode(r, ["A", "B"]) == [1, 0], "at capacity is still fine"
    with pytest.raises(RuntimeError, match="3 decode row"):
        _decode(r, ["A", "B", "C"])


def test_a_clean_map_is_silent_and_a_broken_one_raises():
    """A duplicated source slot would make a device gather read one slot twice.
    Refusing sends no gather at all, which corrupts every off-row request instead."""
    # Steady state: everyone already sits at their own row, so nothing moves.
    r = _runner()
    _prefill(r, ["X", "Y"])
    assert _decode(r, ["X", "Y"]) is None

    # A duplicate is an impossible state, and Z's entry says who else is affected.
    r._req_state_slot.update({"X": 3, "Y": 3, "Z": 5})
    with pytest.raises(RuntimeError, match="not a permutation") as exc:
        _decode(r, ["X", "Y"])
    assert "duplicated=[3]" in str(exc.value)
    assert "'Z': 5" in str(exc.value), "the whole map is the diagnostic"
    # It fails before writing, so off-batch entries like Z are left alone.
    assert r._req_state_slot == {"X": 3, "Y": 3, "Z": 5}


def test_a_decoding_request_without_a_slot_raises():
    """Inventing ownership records a second request at a slot a live one owns. It is
    the hole both corruptions travel through."""
    r = _runner()
    _prefill(r, ["A"])
    with pytest.raises(RuntimeError, match="'GHOST' has no device state slot"):
        _decode(r, ["A", "GHOST"])
