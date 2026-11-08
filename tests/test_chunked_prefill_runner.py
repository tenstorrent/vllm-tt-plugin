# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runner side of chunked prefill on a stateful model.

- ``intermediate_prefill_mask`` reaches ``prefill_forward`` only for models that
  declare ``accepts_intermediate_prefill_mask``.
- A continuation chunk keeps the device state slot of its first chunk, though
  its persistent-batch row changed.
"""

from vllm.v1.core.sched.output import CachedRequestData

from tests.test_stage_inputs import (
    _empty_cached,
    _new_request,
    _run_prefill,
    _runner,
    _step,
)
from vllm_tt_plugin.model_runner import TTModelRunner

PROMPT_LEN = 4


def _continuation(req_id: str, computed: int) -> CachedRequestData:
    """A running partial prefill: its next chunk starts at ``computed``."""
    return CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[computed],
        num_output_tokens=[0],
    )


def _with_capabilities(runner, capabilities):
    runner.model.model_capabilities = capabilities
    return runner


def test_mask_reaches_a_model_that_declares_the_capability():
    runner = _with_capabilities(
        _runner(supports_stage_inputs=False),
        {"accepts_intermediate_prefill_mask": True},
    )

    kwargs = _run_prefill(
        runner,
        _step(
            new=[_new_request("a"), _new_request("b")],
            scheduled={"a": 2, "b": PROMPT_LEN},
        ),
    )

    by_row = dict(zip(runner.input_batch.req_ids, kwargs["intermediate_prefill_mask"]))
    assert by_row == {"a": True, "b": False}
    assert all(isinstance(x, bool) for x in kwargs["intermediate_prefill_mask"])


def test_final_chunks_get_an_all_false_mask():
    runner = _with_capabilities(
        _runner(supports_stage_inputs=False),
        {"accepts_intermediate_prefill_mask": True},
    )
    kwargs = _run_prefill(
        runner, _step(new=[_new_request("a")], scheduled={"a": PROMPT_LEN})
    )
    assert kwargs["intermediate_prefill_mask"] == [False]


def test_a_model_without_the_capability_gets_no_mask_keyword():
    for capabilities in ({}, {"accepts_intermediate_prefill_mask": False}, None):
        runner = _with_capabilities(_runner(supports_stage_inputs=False), capabilities)
        kwargs = _run_prefill(
            runner, _step(new=[_new_request("a")], scheduled={"a": 2})
        )
        assert "intermediate_prefill_mask" not in kwargs


def test_a_continuation_chunk_keeps_its_slot_when_its_row_changes():
    runner = _runner(supports_stage_inputs=False)

    # Chunk 1 of "a" (row 0, slot 0); "b" prefills in full next to it (row 1).
    kwargs = _run_prefill(
        runner,
        _step(
            new=[_new_request("a"), _new_request("b")],
            scheduled={"a": 2, "b": PROMPT_LEN},
        ),
    )
    assert kwargs["empty_slots"] == [0, 1]
    assert runner._req_state_slot == {"a": 0, "b": 1}

    # "b" is done prefilling. The step that prefills "c" does not schedule "a",
    # so the persistent batch drops it and "c" takes row 0.
    kwargs = _run_prefill(runner, _step(new=[_new_request("c")], scheduled={"c": 2}))
    assert runner.input_batch.req_id_to_index.get("a") is None
    assert kwargs["empty_slots"] == [2], "slots 0 and 1 belong to live requests"

    # Both partials continue; "a" re-enters the batch behind "c", at another
    # row than its first chunk's, and each keeps the slot of its first chunk.
    both = CachedRequestData(
        req_ids=["c", "a"],
        resumed_req_ids=set(),
        new_token_ids=[[], []],
        all_token_ids={},
        new_block_ids=[None, None],
        num_computed_tokens=[2, 2],
        num_output_tokens=[0, 0],
    )
    kwargs = _run_prefill(runner, _step(cached=both, scheduled={"c": 2, "a": 2}))
    index = runner.input_batch.req_id_to_index
    assert index["a"] != 0
    by_row = dict(zip(runner.input_batch.req_ids, kwargs["empty_slots"]))
    assert by_row == {"c": 2, "a": 0}
    assert runner._req_state_slot == {"a": 0, "b": 1, "c": 2}


def test_a_new_request_never_takes_a_continuations_row_as_its_slot():
    r = _runner(supports_stage_inputs=False)
    r._req_state_slot.update({"a": 1})
    r.requests.update(dict.fromkeys(["a", "n"]))

    # "n" sits at row 1, which is "a"'s slot: it must go elsewhere.
    slots = TTModelRunner._alloc_prefill_state_slots(r, ["x", "n", "a"])

    assert slots[2] == 1
    assert len(set(slots)) == 3
    assert slots[1] != 1


def test_a_continuation_colliding_with_a_live_holder_fails_loudly():
    import pytest

    r = _runner(supports_stage_inputs=False)
    r._req_state_slot.update({"a": 2, "decoder": 2})
    r.requests.update(dict.fromkeys(["a", "decoder"]))

    with pytest.raises(RuntimeError, match="continuation request"):
        TTModelRunner._alloc_prefill_state_slots(r, ["a"])


def test_release_paths_follow_the_kept_slot():
    runner = _runner(supports_stage_inputs=False)
    _run_prefill(runner, _step(new=[_new_request("a")], scheduled={"a": 2}))
    _run_prefill(
        runner,
        _step(cached=_continuation("a", 2), scheduled={"a": PROMPT_LEN - 2}),
    )
    assert runner._req_state_slot == {"a": 0}

    runner._update_states(_step(finished={"a"}))
    assert runner.model.released == [0]
    assert runner._req_state_slot == {}
    assert _empty_cached().req_ids == []
