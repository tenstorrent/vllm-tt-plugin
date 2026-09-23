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
        _pending_state_slot_moves=None,
        requests={},
    )


class _SessionModel:
    """A model holding ONE B=1 session, keyed on the slot it was armed at.

    Mirrors ``Gemma4DFlashForCausalLM``: arming records the owner slot, a release
    for a different slot is ignored so a live owner is not torn down, and the
    runner reports every gather so the owner slot follows its state.
    """

    def __init__(self):
        self.owner_slot = None
        self.session_of = None  # which req_id the live session belongs to
        self.moves_seen: list[dict[int, int]] = []

    def arm(self, req_id, slot):
        self.owner_slot = int(slot)
        self.session_of = req_id

    def note_state_slots_moved(self, moves):
        self.moves_seen.append(dict(moves))
        if self.owner_slot is not None and int(self.owner_slot) in moves:
            self.owner_slot = int(moves[int(self.owner_slot)])

    def release_request(self, row):
        if (
            self.owner_slot is not None
            and row is not None
            and int(row) != int(self.owner_slot)
        ):
            return
        self.owner_slot = None
        self.session_of = None


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


def test_stateless_model_does_not_retain_slots_across_scheduler_steps():
    """Queued work may exceed max_num_seqs even though each scheduled step cannot.

    A stateless paged model therefore uses row-local prefill slots and its page
    tables, not the persistent map required by recurrent models.
    """
    r = _runner(slots=2)
    r.requires_persistent_request_state_slots = False
    r._req_state_slot.update({"A": 0, "B": 1})
    r.requests.update(dict.fromkeys(["A", "B", "C", "D"]))

    prefill_slots, remap = TTModelRunner._state_slot_inputs(
        r, ["C", "D"], is_prompt=True
    )
    assert prefill_slots == [0, 1]
    assert remap is None
    assert r._req_state_slot == {"A": 0, "B": 1}, "stateless path must not mutate"

    prefill_slots, remap = TTModelRunner._state_slot_inputs(
        r, ["C", "D"], is_prompt=False
    )
    assert prefill_slots is None and remap is None


def test_stateless_model_accepts_new_prefill_wave_after_full_prior_wave():
    """A completed B32 wave cannot consume the next wave's row-local slots."""
    r = _runner(slots=32)
    r.requires_persistent_request_state_slots = False
    prior = [f"old-{i}" for i in range(32)]
    incoming = [f"new-{i}" for i in range(31)]
    r._req_state_slot.update({req_id: i for i, req_id in enumerate(prior)})
    r.requests.update(dict.fromkeys([*prior, *incoming]))

    prefill_slots, remap = TTModelRunner._state_slot_inputs(r, incoming, is_prompt=True)

    assert prefill_slots == list(range(31))
    assert remap is None
    assert r._req_state_slot == dict(zip(prior, range(32), strict=True))


def test_persistent_model_keeps_request_owned_slot_mapping():
    r = _runner(slots=2)
    r.requires_persistent_request_state_slots = True

    prefill_slots, remap = TTModelRunner._state_slot_inputs(
        r, ["A", "B"], is_prompt=True
    )
    assert prefill_slots == [0, 1] and remap is None
    r.requests.update(dict.fromkeys(["A", "B"]))

    prefill_slots, remap = TTModelRunner._state_slot_inputs(
        r, ["B", "A"], is_prompt=False
    )
    assert prefill_slots is None
    assert remap.tolist()[:2] == [1, 0]


def test_missing_capability_keeps_persistent_state_slots():
    """Legacy runners without normalized setup state remain conservative."""
    r = _runner(slots=2)

    prefill_slots, remap = TTModelRunner._state_slot_inputs(
        r, ["A", "B"], is_prompt=True
    )
    assert prefill_slots == [0, 1] and remap is None
    assert r._req_state_slot == {"A": 0, "B": 1}


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


def test_cancelling_an_older_request_keeps_a_reused_slots_live_session():
    """Slot movement + slot REUSE + cancelling the older request (#118 finding 1).

    Releasing by the slot a request was PREFILLED into is not enough, because
    slots are reused. Victor's sequence:

      1. X prefills into slot 0, A into slot 1.
      2. X is cancelled; its slot records are dropped and slot 0 frees.
      3. A is gathered 1 -> 0.
      4. B prefills into the now-free slot 1 and ARMS the session there.
      5. A is cancelled before B ever decodes.
      6. B must still own its session -- otherwise its next solo decode returns
         one token against a reserved block and the scheduler rejects the width.

    With release-by-prefill-slot, step 5 released A as slot 1, matched B's owner
    slot and cleared B. The fix is that the model is told about step 3's move and
    the release uses the CURRENT slot.
    """
    r = _runner()
    model = _SessionModel()
    r.model = model

    assert _prefill(r, ["X", "A"]) == [0, 1]
    state = ["X", "A"] + [None] * (SLOTS - 2)

    # A is the solo owner at this point; arm it where it was prefilled.
    model.arm("A", r._req_state_slot["A"])
    assert model.owner_slot == 1

    # 2. X is cancelled. A is not the released slot, so its session survives.
    TTModelRunner._release_model_request(r, "X")
    _release(r, finished=["X"])
    r.requests.pop("X")
    assert model.session_of == "A", "releasing X must not clear A's session"

    # 3. A decodes alone and is gathered into row 0.
    remap = _decode(r, ["A"])
    state = _gather(state, remap)
    _assert_state_found(r, state)
    assert r._req_state_slot["A"] == 0, "A moved to row 0"
    assert model.owner_slot == 0, (
        "the model was told about the gather, so its owner slot follows A"
    )

    # 4. B takes the freed slot 1 and arms the session there.
    assert _prefill(r, ["B"]) == [1]
    state[1] = "B"
    model.arm("B", r._req_state_slot["B"])
    assert model.owner_slot == 1 and model.session_of == "B"

    # 5. A is cancelled. A's PREFILL slot was 1 -- which is now B's owner slot.
    TTModelRunner._release_model_request(r, "A")
    _release(r, finished=["A"])
    r.requests.pop("A")

    # 6. The live owner keeps its session.
    assert model.session_of == "B", (
        "cancelling A cleared B's session: the release identified B by a slot A "
        "merely used to be prefilled into"
    )
    assert model.owner_slot == 1, "B still owns slot 1"
    _assert_state_found(r, state)

    # And B's own release still works.
    TTModelRunner._release_model_request(r, "B")
    assert model.session_of is None, "B's own release clears B"


def test_the_model_is_told_about_a_gather_only_once_it_is_accepted():
    """A refused decode never moved anything, so the model must not be told.

    ``note_decode_state_slots_settled`` is the commit point for the plugin's own
    ownership map; the model's owner slot has to move on exactly the same event
    or the two describe different permutations.
    """
    r = _runner()
    model = _SessionModel()
    r.model = model
    _prefill(r, ["A", "B"])
    model.arm("B", r._req_state_slot["B"])
    assert model.owner_slot == 1

    # Build a remap that moves B, then DROP it without settling.
    remap = TTModelRunner._decode_state_slot_remap(r, ["B"])
    assert remap is not None, "B at slot 1 decoding at row 0 is a real move"
    assert model.moves_seen == [], "nothing is reported before the decode is accepted"
    assert model.owner_slot == 1, "and the owner slot has not moved"

    # Now accept it.
    TTModelRunner.note_decode_state_slots_settled(r)
    # The gather is a whole permutation, not just the batch row: B's slot 1 goes
    # to row 0 and A's slot 0 goes to row 1, so BOTH are reported. That is why
    # the mapping is handed over at once -- walking the pairs in sequence would
    # move an owner twice.
    assert model.moves_seen == [{1: 0, 0: 1}], "the accepted gather is reported once"
    assert model.owner_slot == 0


def test_identity_gathers_report_nothing_to_the_model():
    """The steady state moves nothing, so it must not churn the model's owner."""
    r = _runner()
    model = _SessionModel()
    r.model = model
    _prefill(r, ["A", "B"])
    model.arm("A", 0)
    assert _decode(r, ["A", "B"]) is None, "already in place"
    assert model.moves_seen == []
    assert model.owner_slot == 0


def test_a_model_without_the_move_hook_still_releases():
    """The hook is optional on the runner side (the platform requires it only for
    multi-sequence block-output models), so a model without it must not crash."""
    r = _runner()
    released: list[int] = []
    r.model = SimpleNamespace(release_request=released.append)
    _prefill(r, ["A", "B"])
    _decode(r, ["B"])  # a real gather, with nothing to notify
    TTModelRunner._release_model_request(r, "B")
    assert released == [0], "released by B's current slot"
