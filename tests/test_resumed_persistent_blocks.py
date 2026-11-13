# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Forced-reset resumes must replace persistent physical block-table rows."""

import json
import threading
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import CachedRequestData

from tests.test_stage_inputs import _new_request, _runner, _step
from vllm_tt_plugin.input_batch import InputBatch


def _blocks(first, groups, count=3):
    return tuple(
        list(range(first + 100 * group, first + 100 * group + count))
        for group in range(groups)
    )


def _cached(runner, target, blocks, *, resumed=False, computed=0):
    ids = [target, "peer"]
    return CachedRequestData(
        req_ids=ids,
        resumed_req_ids={target} if resumed else set(),
        new_token_ids=[[], []],
        all_token_ids={},
        new_block_ids=[blocks, None],
        num_computed_tokens=[computed, 1],
        num_output_tokens=[len(runner.requests[rid].output_token_ids) for rid in ids],
    )


def _snapshot(runner, req_id, label):
    row = runner.input_batch.req_id_to_index[req_id]
    cached = [list(ids) for ids in runner.requests[req_id].block_ids]
    tables = runner.input_batch.block_tables_for_rows([row], 32)
    live = [
        int(table.num_blocks_per_row[row])
        for table in runner.input_batch.block_table.block_tables
    ]
    return {
        "label": label,
        "row": row,
        "cached_blocks": cached,
        "persistent_blocks": [
            table[0, :count].tolist() for table, count in zip(tables, live)
        ],
        "live_counts": live,
        # Check the actual tables passed to a model, including padding.
        "padded_tables": [table[0].tolist() for table in tables],
    }


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("row_retained", [True, False])
def test_repeated_resumes_replace_block_rows_but_decode_growth_appends(
    groups, row_retained
):
    runner = _runner(supports_stage_inputs=False)
    runner.input_batch = InputBatch(
        max_num_reqs=4,
        max_model_len=512,
        max_num_batched_tokens=512,
        vocab_size=64,
        block_sizes=[16] * groups,
        kernel_block_sizes=[16] * groups,
    )
    target, peer = _new_request("target"), _new_request("peer")
    target.prompt_token_ids = peer.prompt_token_ids = list(range(32))
    target.block_ids, peer.block_ids = _blocks(11, groups), _blocks(81, groups)
    runner._update_states(
        _step(new=[target, peer], scheduled={"target": 16, "peer": 16})
    )
    # Model slots exist before a forced preemption. No device calls are needed.
    runner._alloc_prefill_state_slots(["target", "peer"])
    initial_peer = _snapshot(runner, "peer", "initial peer")
    snapshots = []
    peer_snapshots = []

    for cycle, first in enumerate((31, 51)):
        if not row_retained:
            # Ordinary preemption removes the request for at least one step.
            runner._update_states(_step(preempted={"target"}, scheduled={"peer": 1}))
            assert "target" not in runner.input_batch.req_id_to_index
        else:
            # Forced reset can report preemption and resume in the same output:
            # the target stays scheduled, so the unscheduled-row sweep keeps it.
            assert "target" in runner.input_batch.req_id_to_index
        fresh_blocks = _blocks(first, groups)
        runner._update_states(
            _step(
                cached=_cached(runner, "target", fresh_blocks, resumed=True),
                scheduled={"target": 16, "peer": 1},
                preempted={"target"} if row_retained else (),
            )
        )
        snapshots.append(_snapshot(runner, "target", f"resume {cycle}"))
        peer_snapshots.append(_snapshot(runner, "peer", f"peer after resume {cycle}"))
        # Ordinary growth of a running request still appends, rather than
        # replacing the pages just assigned at resume.
        runner._update_states(
            _step(
                cached=_cached(
                    runner, "target", _blocks(first + 3, groups, count=1), computed=32
                ),
                scheduled={"target": 1, "peer": 1},
            )
        )
        snapshots.append(_snapshot(runner, "target", f"decode growth {cycle}"))
        peer_snapshots.append(_snapshot(runner, "peer", f"peer after growth {cycle}"))

    # Finishing and admitting a new request through the reused row must still
    # discard the old table; this is the existing successful placement path.
    replacement = _new_request("replacement")
    replacement.block_ids = _blocks(71, groups)
    runner._update_states(
        _step(
            new=[replacement],
            finished={"target"},
            scheduled={"replacement": 4, "peer": 1},
        )
    )
    snapshots.append(_snapshot(runner, "replacement", "finished row reused"))
    peer_snapshots.append(_snapshot(runner, "peer", "peer after row reuse"))

    print(
        json.dumps(
            {
                "row_retained": row_retained,
                "groups": groups,
                "snapshots": snapshots,
                "peer_snapshots": peer_snapshots,
            }
        )
    )
    for observed in peer_snapshots:
        assert observed["cached_blocks"] == initial_peer["cached_blocks"]
        assert observed["persistent_blocks"] == initial_peer["persistent_blocks"]
        assert observed["live_counts"] == initial_peer["live_counts"]
    assert snapshots[-1]["persistent_blocks"] == snapshots[-1]["cached_blocks"]
    # Cached request state and actual physical table must agree after every
    # transition, including a retained-row resume. Assert after all transitions
    # so a failing run records repeated accumulation, growth and peer behavior.
    for observed in snapshots:
        assert observed["persistent_blocks"] == observed["cached_blocks"], observed
        for ids, count, padded in zip(
            observed["cached_blocks"],
            observed["live_counts"],
            observed["padded_tables"],
        ):
            assert count == len(ids)
            assert padded == ids + [0] * (32 - len(ids))


@pytest.mark.parametrize("discard_count", [0, 1])
def test_retained_resume_preserves_committed_history_and_pending_suffix_rules(
    discard_count,
):
    from tests.test_decode_reload_contract import _completed_step
    from vllm_tt_plugin.async_decode import TTAsyncDecodeController
    from vllm_tt_plugin.scheduler import set_tt_forced_reset_discard_counts

    runner = _runner(supports_stage_inputs=False)
    request, peer = _new_request("request"), _new_request("peer")
    request.block_ids, peer.block_ids = ([11],), ([81],)
    runner._update_states(
        _step(new=[request, peer], scheduled={"request": 4, "peer": 4})
    )
    runner._alloc_prefill_state_slots(["request", "peer"])
    runner._apply_sampled_tokens_to_state(torch.tensor([[9]]), req_ids=["request"])
    row = runner.input_batch.req_id_to_index["request"]
    output_list = runner.requests["request"].output_token_ids
    assert output_list == [9]
    assert runner.input_batch.req_output_token_ids[row] is output_list
    runner._decode_layout_changed_since_last_decode = False
    step = _step(
        cached=_cached(runner, "request", ([31],), resumed=True),
        scheduled={"request": 5, "peer": 1},
        preempted={"request"},
    )
    # Older completed results remain valid; only the reset's newest suffix is
    # stale. Updating the block row must not reset or duplicate token metadata.
    step.scheduled_cached_reqs.num_output_tokens[0] = 3 - discard_count
    runner._update_states(step)
    assert runner.input_batch.req_id_to_index["request"] == row
    assert runner.input_batch.req_output_token_ids[row] is output_list
    assert output_list == [9]
    assert int(runner.input_batch.num_tokens[row]) == 5
    assert int(runner.input_batch.num_computed_tokens_cpu[row]) == 0
    assert runner._decode_layout_changed_since_last_decode
    runner._steady_decode_lock = threading.Lock()
    runner._completed_decode_steps = deque()
    runner._pending_async_steps = deque()
    runner._pending_async_overlap_ok = deque()
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    controller = TTAsyncDecodeController(runner)
    outputs = [
        SimpleNamespace(req_id_to_index={"request": 0}, sampled_token_ids=[[token]])
        for token in (10, 11)
    ]
    runner._completed_decode_steps.extend(
        _completed_step(token, output) for token, output in zip((10, 11), outputs)
    )
    set_tt_forced_reset_discard_counts(step, {"request": discard_count})
    controller.apply_completed_decode_steps_before_build(step)
    expected = [9, 10, 11] if discard_count == 0 else [9, 10]
    assert output_list == expected
    assert runner.input_batch.req_output_token_ids[row] is output_list
    assert int(runner.input_batch.num_tokens[row]) == 4 + len(expected)
    assert (
        runner.input_batch.token_ids_cpu[row, 4 : 4 + len(expected)].tolist()
        == expected
    )
    assert [output.sampled_token_ids for output in outputs] == [[[10]], [[11]]]
    assert _snapshot(runner, "request", "after apply")["persistent_blocks"] == [[31]]
    assert _snapshot(runner, "peer", "peer")["persistent_blocks"] == [[81]]
    assert runner.requests["peer"].output_token_ids == []
