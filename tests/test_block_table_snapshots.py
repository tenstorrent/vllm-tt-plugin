# SPDX-License-Identifier: Apache-2.0
"""Host-only immutable page-table snapshot and reload contract coverage."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch, TTLaneInputBatch
from vllm_tt_plugin.model_input import TTSamplingParams


def _batch():
    batch = InputBatch(max_num_reqs=4, max_model_len=128, max_num_batched_tokens=128,
                       vocab_size=32, block_sizes=[16, 32], kernel_block_sizes=[16, 32])
    for row in range(3):
        batch.add_request(_request(str(row), 10 + row * 10))
    return batch


def _request(name, block):
    return CachedRequestState(req_id=name, prompt_token_ids=[1, 2, 3], mm_features=None,
                              sampling_params=SamplingParams(temperature=0), generator=None,
                              block_ids=([block, block + 1], [block + 2]),
                              num_computed_tokens=3, output_token_ids=[4])


def test_final_shape_cache_hits_without_any_materialization(monkeypatch):
    batch = _batch()
    token, tables = batch.block_table_snapshot([0, 1], 10, 4)
    assert tables[0].shape == (4, 10)
    assert tables[0][0].tolist() == [10, 11] + [0] * 8
    assert tables[1][1].tolist() == [22] + [0] * 9
    assert not bool(tables[0][2:].any())
    assert not bool(tables[1][2:].any())
    monkeypatch.setattr(batch, 'block_tables_for_rows', Mock(side_effect=AssertionError('materialized')))
    with monkeypatch.context() as m:
        for name in ('arange', 'cat', 'zeros', 'equal'):
            m.setattr(torch, name, Mock(side_effect=AssertionError(name)))
        m.setattr(torch.Tensor, 'clone', Mock(side_effect=AssertionError('clone')))
        for _ in range(5):
            same_token, same_tables = batch.block_table_snapshot([0, 1], 10, 4)
            assert same_token == token
            assert same_tables is tables


@pytest.mark.parametrize('growth', [None, ([], []), ([], [35]), ([35], [])])
def test_empty_appends_keep_revision_real_growth_replaces_snapshot(growth):
    batch = _batch()
    before_token, before = batch.block_table_snapshot([0], 8, 4)
    frozen = tuple(t.clone() for t in before)
    batch.append_block_ids(growth, 0)
    after_token, after = batch.block_table_snapshot([0], 8, 4)
    changed = growth is not None and any(growth)
    assert (after_token != before_token) is bool(changed)
    assert (after is before) is not bool(changed)
    assert all(torch.equal(old, saved) for old, saved in zip(before, frozen))
    if changed:
        group = 0 if growth[0] else 1
        assert after[group][0, 2 if group == 0 else 1].item() == 35


def test_remove_reuse_and_condense_invalidate_and_preserve_old_snapshot():
    batch = _batch()
    token, old = batch.block_table_snapshot([0, 1, 2], 8, 4)
    saved = tuple(t.clone() for t in old)
    assert batch.remove_request('1') == 1
    removed_revision = batch._block_table_revision
    assert removed_revision > token[2]
    batch.add_request(_request('replacement', 44), req_index=1)
    replacement_token, replacement = batch.block_table_snapshot([0, 1, 2], 8, 4)
    assert replacement_token != token
    assert replacement[0][1, :2].tolist() == [44, 45]
    assert all(torch.equal(t, s) for t, s in zip(old, saved))
    batch.remove_request('0')
    revision_before_move = batch._block_table_revision
    batch.condense([0])
    assert batch._block_table_revision > revision_before_move
    moved_token, moved = batch.block_table_snapshot([0, 1], 8, 4)
    assert moved_token != replacement_token
    assert moved[0][0, :2].tolist() == [30, 31]
    assert not bool(moved[0][2:].any())


def test_order_width_padding_and_replaced_storage_are_part_of_key():
    batch = _batch()
    first, _ = batch.block_table_snapshot([0, 1], 8, 4)
    reverse, tables = batch.block_table_snapshot([1, 0], 8, 4)
    assert first != reverse
    assert tables[0][0, :2].tolist() == [20, 21]
    wide, _ = batch.block_table_snapshot([1, 0], 9, 4)
    compact, _ = batch.block_table_snapshot([1, 0], 9, 2)
    assert wide != reverse and compact != wide
    batch.block_table = _batch().block_table
    replaced, _ = batch.block_table_snapshot([1, 0], 9, 2)
    assert replaced != compact
    with pytest.raises(ValueError):
        batch.block_table_snapshot([0, 1], 8, 1)


def test_direct_and_lane_views_remain_fresh_and_independently_mutable():
    batch = _batch()
    _, snapshot = batch.block_table_snapshot([0, 1], 8, 4)
    lane = TTLaneInputBatch.slot_block_tables(batch, [1], True, 4, 8)
    assert not bool(lane[0][0].any())
    assert snapshot[0][0, 0].item() == 10
    direct = batch.block_tables_for_rows([0], 8)
    direct[0].zero_()
    assert snapshot[0][0, 0].item() == 10
    # Direct callers of the uncached helper observe raw table mutations too.
    batch.block_table.block_tables[0].get_cpu_tensor()[0, 0] = 99
    assert batch.block_tables_for_rows([0], 8)[0][0, 0].item() == 99


def _input(version, tables, *, device=True, changed=False):
    return SimpleNamespace(
        input_tokens=torch.zeros(1, 1, dtype=torch.int32), input_positions=torch.zeros(1, dtype=torch.int32),
        block_tables=tables[0], block_tables_per_group=list(tables), block_tables_per_layer=None,
        block_table_snapshot_version=version, unpadded_batch_size=1,
        tt_sampling_params=TTSamplingParams(temperature=torch.ones(1), top_k=torch.tensor([32]),
                                            top_p=torch.ones(1), seed=torch.tensor([42]),
                                            presence_penalty=torch.zeros(1), frequency_penalty=torch.zeros(1),
                                            repetition_penalty=torch.ones(1), num_logprobs=torch.tensor([-2]),
                                            enable_log_probs=torch.zeros(1, dtype=torch.bool)),
        perform_device_sampling=device, prompt_tokens=None, output_tokens=None,
        decode_layout_changed=changed, num_valid_drafts=None, accepted_counts=None,
        spec_mode=None, slot_remap=None,
    )


def _controller():
    calls = []
    class Model:
        decode_input_update_contract = 1
        model_capabilities = {'supports_async_decode': True}
        fail = False
        def decode_forward(self, **kw):
            calls.append(kw)
            if self.fail:
                raise RuntimeError('refused submit')
            return torch.zeros(1, 1)
    runner = SimpleNamespace(model=Model(), trace_mode='decode_only', kv_caches=object(),
                             request_specific_rope=False, supports_stage_outputs=False,
                             note_decode_layout_consumed=lambda: None,
                             note_decode_state_slots_settled=lambda: None)
    return TTAsyncDecodeController(runner), calls


def test_versioned_steady_and_host_reload_do_not_equal_or_clone(monkeypatch):
    batch = _batch()
    version, tables = batch.block_table_snapshot([0], 8, 4)
    controller, _ = _controller()
    model_input = _input(version, tables)
    controller.commit_decode_submission(model_input, controller.plan_decode_reload(model_input))
    assert controller._submitted_page_tables[0] is tables[0]
    with monkeypatch.context() as m:
        m.setattr(torch, 'equal', Mock(side_effect=AssertionError('equal')))
        m.setattr(torch.Tensor, 'clone', Mock(side_effect=AssertionError('clone')))
        for host in (False, False, True, True):
            model_input.perform_device_sampling = not host
            plan = controller.plan_decode_reload(model_input)
            assert plan.reload_inputs is host
            assert not plan.reload_page_table
            controller.commit_decode_submission(model_input, plan)


def test_growth_commits_version_only_after_accepted_submission():
    batch = _batch()
    first, tables = batch.block_table_snapshot([0], 8, 1)
    controller, calls = _controller()
    controller.submit_decode(_input(first, tables, changed=True), read_from_device=True)
    batch.append_block_ids(([55], []), 0)
    second, grown = batch.block_table_snapshot([0], 8, 1)
    grown_input = _input(second, grown)
    controller.runner.model.fail = True
    with pytest.raises(RuntimeError, match='refused submit'):
        controller.submit_decode(grown_input, read_from_device=True)
    assert controller._submitted_page_table_version == first
    assert controller._submitted_page_tables[0] is tables[0]
    controller.runner.model.fail = False
    controller.submit_decode(grown_input, read_from_device=True)
    assert controller._submitted_page_table_version == second
    assert calls[-1]['reload_page_table']
    assert not calls[-1]['reload_inputs']
    assert not calls[-1]['reset_sampling_state']
    controller.submit_decode(grown_input, read_from_device=True)
    assert not calls[-1]['reload_page_table']
    controller.note_prefill_submitted()
    assert controller.plan_decode_reload(grown_input).reload_inputs


def test_unversioned_in_place_mutation_and_mode_transitions_keep_defensive_copy():
    controller, _ = _controller()
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    direct = _input(None, [table])
    controller.commit_decode_submission(direct, controller.plan_decode_reload(direct))
    assert controller._submitted_page_tables[0] is not table
    table[0, 1] = 42
    assert controller.plan_decode_reload(direct).reload_page_table
    controller.commit_decode_submission(direct, controller.plan_decode_reload(direct))
    versioned = _input(('new_snapshot',), [table.clone()])
    assert controller.plan_decode_reload(versioned).reload_page_table
    controller.commit_decode_submission(versioned, controller.plan_decode_reload(versioned))
    assert controller.plan_decode_reload(direct).reload_page_table


@pytest.mark.parametrize('path', ['runner', 'lane'])
@pytest.mark.parametrize('growth', [None, ([], []), ([], [77])])
def test_scheduler_update_paths_invalidate_only_for_nonempty_growth(path, growth):
    from vllm_tt_plugin.model_runner import TTModelRunner
    batch = _batch()
    requests = {str(row): _request(str(row), 10 + row * 10) for row in range(3)}
    version, before = batch.block_table_snapshot([0, 1, 2], 8, 4)
    output = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set(),
                             free_encoder_mm_hashes=[], scheduled_new_reqs=[],
                             num_scheduled_tokens={str(row): 1 for row in range(3)},
                             scheduled_cached_reqs=SimpleNamespace(
                                 req_ids=['0', '1', '2'], num_computed_tokens=[4, 4, 4],
                                 new_block_ids=[growth, None, None], resumed_req_ids=set()))
    if path == 'runner':
        runner = SimpleNamespace(input_batch=batch, requests=requests, encoder_cache={},
                                  _release_dead_state_slots=lambda _: None,
                                  _release_model_request=lambda _: None,
                                  _decode_layout_changed_since_last_decode=False)
        TTModelRunner._update_states(runner, output)
    else:
        # The steady lane update reaches the same wrapper without placements;
        # full lane admission/removal coverage lives in test_lane_input_batch.
        TTLaneInputBatch.apply_step_plan(batch, output, SimpleNamespace(req_id_to_row={}), requests, {})
    after_version, after = batch.block_table_snapshot([0, 1, 2], 8, 4)
    assert (after_version != version) is (growth is not None and bool(any(growth)))
    if growth is not None and any(growth):
        assert after[1][0, 1].item() == 77
        assert before[1][0, 1].item() == 0
