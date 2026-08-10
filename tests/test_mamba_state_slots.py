# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for scheduler-owned TT GDN state."""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.input_batch import _lane_gdn_state_indices
from vllm_tt_plugin.model_runner import TTModelRunner


def _runner(capacity=8):
    return SimpleNamespace(
        tt_per_lane_max_num_seqs=capacity,
        _req_state_slot={},
        _req_state_owner={},
        requests={},
    )


def _prefill(runner, req_ids, owners):
    runner.requests.update(dict.fromkeys(req_ids))
    return TTModelRunner._managed_state_slots(
        runner,
        list(req_ids),
        list(owners),
        is_prompt=True,
    )


def _decode(runner, req_ids, owners):
    return TTModelRunner._managed_state_slots(
        runner,
        list(req_ids),
        list(owners),
        is_prompt=False,
    )


def test_state_follows_owner_without_permuting_the_decode_batch():
    r = _runner()
    assert _prefill(r, ["A"], [(100,)]) == [0]

    incoming = [f"B{i}" for i in range(7)]
    incoming_owners = [(200 + i,) for i in range(7)]
    assert _prefill(r, incoming, incoming_owners) == list(range(1, 8))

    req_ids = incoming + ["A"]
    owners = incoming_owners + [(100,)]
    assert _decode(r, req_ids, owners) == [1, 2, 3, 4, 5, 6, 7, 0]
    assert _decode(r, list(reversed(req_ids)), list(reversed(owners))) == [
        0,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
    ]


def test_unscheduled_state_is_held_until_finish_or_preemption():
    r = _runner(capacity=2)
    assert _prefill(r, ["A", "B"], [(10,), (11,)]) == [0, 1]
    assert _decode(r, ["B"], [(11,)]) == [1]
    with pytest.raises(RuntimeError, match="no compact GDN state slot"):
        _prefill(r, ["C"], [(12,)])

    r.requests.pop("A")
    r._req_state_slot.pop("A")
    r._req_state_owner.pop("A")
    assert _prefill(r, ["C"], [(12,)]) == [0]


def test_owner_change_requires_rebuilding_prefill():
    r = _runner()
    assert _prefill(r, ["A"], [(10,)]) == [0]
    with pytest.raises(RuntimeError, match="changed Mamba ownership"):
        _decode(r, ["A"], [(99,)])
    assert _prefill(r, ["A"], [(99,)]) == [0]
    assert _decode(r, ["A"], [(99,)]) == [0]


def test_lane_unscheduled_occupied_rows_use_dummy_state():
    # Rows 0 and 2 may both be occupied, but only row 2 runs this step.
    assert _lane_gdn_state_indices(4, (2,)).tolist() == [4, 5, 2, 7]
