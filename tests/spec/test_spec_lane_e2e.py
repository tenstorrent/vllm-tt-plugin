# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The greedy speculative loop over a lane batch, driven end to end on the host.

``test_spec_greedy_e2e.py`` drives the front-packed runner. These drive the lane
step's own path: ``TTLaneInputBatch`` builds the candidate block over its slot
grid, ``_finish_lane_sync`` routes a verify to the walk, commit and proposal, and
an ordinary decode inside the speculating launch proposes after its commit. The
model is ``FakeSpecModel`` drafting through ``custom_class``, the only method a
lane launch admits, with synchronous scheduling.

The layout is two lanes of two slots. A request sits at its stable slot and the
other slots are gaps, so a defect that read a gap as a request, shifted a row
onto a packed index, or let a gap's position turn into 0..K changes an
assertion.
"""

import inspect
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import TTLaneInputBatch
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS

from .fake_spec_model import FAKE_VOCAB_SIZE, FakeSpecModel, make_fake_spec_model

BLOCK_SIZE = 16
MAX_MODEL_LEN = 128
NUM_LANES = 2
PER_LANE = 2
CAPACITY = NUM_LANES * PER_LANE
PROMPT_LEN = 4
DRAFT_LEN = 3

# region Test helpers


def _runner(model: FakeSpecModel, *, narrow: bool = False) -> SimpleNamespace:
    """A runner fake over a real lane batch, carrying what the lane step reads."""
    batch = TTLaneInputBatch(
        num_lanes=NUM_LANES,
        per_lane=PER_LANE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=FAKE_VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    runner = SimpleNamespace(
        model=model,
        input_batch=batch,
        lane_batch=batch,
        requests={},
        kv_caches=object(),
        trace_mode="decode_only",
        request_specific_rope=False,
        supports_stage_outputs=False,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=DRAFT_LEN,
        async_decode_scheduling=False,
        _spec_plan=None,
        _spec_method="custom_class",
        _spec_supports_narrow_decode=narrow,
        _spec_drafts_from_model=True,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        # A gap's request id is None, which the commit looks up here.
        _req_state_slot={},
        _ngram_proposer=None,
        tt_per_lane_max_num_seqs=PER_LANE,
        tt_data_parallel_size=NUM_LANES,
        vocab_size=FAKE_VOCAB_SIZE,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        supports_stage_inputs=False,
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=MAX_MODEL_LEN
        ),
        check_perform_device_sampling=lambda **_: False,
        host_sampler=Sampler(),
        _block_tables_per_layer=lambda _: None,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
        _spec_row_state=TTModelRunner._spec_row_state,
        _spec_candidate_block=TTModelRunner._spec_candidate_block,
        _committed_positions=TTModelRunner._committed_positions,
    )
    # Every remaining method comes from the real class, each descriptor with its
    # own binding (see test_spec_greedy_e2e._runner).
    for name, member in vars(TTModelRunner).items():
        if hasattr(runner, name):
            continue
        if isinstance(member, staticmethod):
            setattr(runner, name, member.__func__)
        elif isinstance(member, classmethod):
            setattr(runner, name, member.__func__.__get__(TTModelRunner))
        elif inspect.isfunction(member):
            setattr(runner, name, member.__get__(runner))
    return runner


def _add_request(runner: SimpleNamespace, req_id: str, row: int, first_token: int):
    """A fully prefilled request at its stable slot ``row``."""
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([row + 1],),
        num_computed_tokens=PROMPT_LEN,
        output_token_ids=[],
    )
    runner.requests[req_id] = request
    runner.lane_batch.add_request_to_row(request, row)


def _last_token(runner: SimpleNamespace, req_id: str) -> int:
    batch = runner.lane_batch
    row = batch.req_id_to_index[req_id]
    return int(batch.token_ids_cpu[row, batch.num_tokens[row] - 1])


def _step(
    runner: SimpleNamespace,
    drafts: dict[str, list[int]] | None = None,
    scheduled: list[str] | None = None,
):
    """One synchronous lane decode step over ``scheduled``, every live request
    by default."""
    batch = runner.lane_batch
    rows = {batch.req_ids[r]: r for r in batch.occupied_rows()}
    if scheduled is not None:
        rows = {req_id: rows[req_id] for req_id in scheduled}
    for req_id, row in rows.items():
        # Stands in for ``_update_states``: the computed count covers what the
        # previous step committed.
        batch.num_computed_tokens_cpu[row] = batch.num_tokens[row]
    req_ids = list(rows)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_spec_decode_tokens = dict(drafts or {})
    scheduler_output.num_scheduled_tokens = dict.fromkeys(req_ids, 1)
    scheduler_output.total_num_scheduled_tokens = len(req_ids)
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in req_ids],
        num_computed_tokens=[int(batch.num_tokens[r]) for r in rows.values()],
        num_output_tokens=[0 for _ in req_ids],
    )
    scheduled_rows = tuple(sorted(rows.values()))
    plan = TTStepPlan(
        is_decode=True,
        capacity=CAPACITY,
        scheduled_req_ids=tuple(batch.req_ids[r] for r in scheduled_rows),
        scheduled_rows=scheduled_rows,
        input_rows=scheduled_rows,
        req_id_to_row=dict(rows),
        batch_size_per_dp=tuple(
            sum(1 for r in scheduled_rows if r // PER_LANE == lane)
            for lane in range(NUM_LANES)
        ),
        prefill_empty_slots=None,
    )
    model_input = batch.build_model_input(runner, scheduler_output, None, plan)
    sampling_rows = list(scheduled_rows) if model_input.spec_mode is None else None
    controller = TTAsyncDecodeController(runner)
    submission = controller.submit_decode(
        model_input, read_from_device=True, sampling_rows=sampling_rows
    )
    finalized = controller.finalize_decode(submission, sampling_rows=sampling_rows)
    output = TTModelRunner._finish_lane_sync(
        runner,
        None,
        tt_out=finalized.tt_out,
        tt_log_probs=None,
        model_input=model_input,
        scheduled_rows=list(scheduled_rows),
        is_decode=True,
        lane_total=CAPACITY,
        decode_submission=submission,
    )
    return model_input, output


def _committed(output) -> dict[str, list[int]]:
    return dict(zip(output.req_ids, output.sampled_token_ids, strict=True))


# endregion Test helpers


def test_a_lane_verify_builds_the_block_on_the_slot_grid():
    """Each request's columns sit on its own slot; a gap carries no position."""
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "a", row=0, first_token=1)
    _add_request(runner, "b", row=3, first_token=40)
    last = _last_token(runner, "a")
    drafts = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    model_input, _ = _step(runner, drafts={"a": drafts})

    assert model_input.spec_mode == ACCEPT_MODE_ARGMAX_IDS
    assert model_input.row_req_ids == ["a", None, None, "b"]
    assert model_input.num_valid_drafts.tolist() == [DRAFT_LEN, 0, 0, 0]
    start = PROMPT_LEN - 1
    assert model_input.input_positions[0].tolist() == list(
        range(start, start + DRAFT_LEN + 1)
    )
    assert model_input.input_tokens[0, 1:].tolist() == drafts
    # Gaps and the draftless row's draft columns stay at -1: a gap's -1 must
    # not become 0..K.
    for gap in (1, 2):
        assert model_input.input_positions[gap].tolist() == [-1] * (DRAFT_LEN + 1)
    assert model_input.input_positions[3].tolist() == [start] + [-1] * DRAFT_LEN
    assert model.verify_calls[-1]["rows"] == CAPACITY


def test_a_lane_verify_commits_per_request_and_skips_the_gaps():
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "a", row=0, first_token=1)
    _add_request(runner, "b", row=3, first_token=40)
    last_a, last_b = _last_token(runner, "a"), _last_token(runner, "b")
    drafts = [(last_a + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    _, output = _step(runner, drafts={"a": drafts})

    committed = _committed(output)
    assert list(committed) == ["a", "b"]
    assert committed["a"][:DRAFT_LEN] == drafts
    assert len(committed["a"]) == DRAFT_LEN + 1
    assert committed["b"] == [(last_b + 1) % FAKE_VOCAB_SIZE]
    assert runner._req_accepted_counts == {"a": DRAFT_LEN + 1, "b": 1}
    batch = runner.lane_batch
    assert int(batch.num_tokens[0]) == PROMPT_LEN + DRAFT_LEN + 1
    assert int(batch.num_tokens[3]) == PROMPT_LEN + 1
    # The gaps took nothing.
    assert int(batch.num_tokens[1]) == int(batch.num_tokens[2]) == 0


def test_a_lane_verify_proposes_for_the_live_slots_only():
    model = FakeSpecModel()
    runner = _runner(model)
    _add_request(runner, "a", row=0, first_token=1)
    _add_request(runner, "b", row=3, first_token=40)
    last = _last_token(runner, "a")
    drafts = [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]

    _step(runner, drafts={"a": drafts})

    call = model.propose_calls[-1]
    # The drafter's rows are the verify's: every slot, gaps included.
    assert call["rows"] == CAPACITY
    assert call["hidden"] is model.verify_hidden
    for gap in (1, 2):
        assert call["committed_positions"][gap].tolist() == [-1] * (DRAFT_LEN + 1)
    assert set(runner._proposed_draft_token_ids) == {"a", "b"}
    new_last = _last_token(runner, "a")
    assert runner._proposed_draft_token_ids["a"] == [
        (new_last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)
    ]


def test_a_draftless_lane_step_runs_the_ordinary_decode_and_proposes():
    """With narrow decode, a step with no drafts is the plain call, then a proposal."""
    model = make_fake_spec_model(supports_narrow_decode=True)()
    runner = _runner(model, narrow=True)
    _add_request(runner, "a", row=1, first_token=1)
    _add_request(runner, "b", row=2, first_token=40)
    last_a, last_b = _last_token(runner, "a"), _last_token(runner, "b")

    model_input, output = _step(runner)

    assert model_input.spec_mode is None
    assert model.plain_calls and not model.verify_calls
    assert _committed(output) == {
        "a": [(last_a + 1) % FAKE_VOCAB_SIZE],
        "b": [(last_b + 1) % FAKE_VOCAB_SIZE],
    }
    call = model.propose_calls[-1]
    assert call["hidden_was_none"]
    # The committed tokens went back to the slots they were sampled at.
    assert int(call["committed_tokens"][1, 0]) == (last_a + 1) % FAKE_VOCAB_SIZE
    assert int(call["committed_tokens"][2, 0]) == (last_b + 1) % FAKE_VOCAB_SIZE
    for gap in (0, 3):
        assert int(call["committed_positions"][gap, 0]) == -1
    assert set(runner._proposed_draft_token_ids) == {"a", "b"}


@pytest.mark.parametrize("narrow", [False, True], ids=["verify", "ordinary"])
def test_a_slot_the_step_leaves_out_takes_nothing(narrow):
    """A request holding its slot through a step that skips it gets no token.

    A decode step the coordinator interleaves into a run of prefills leaves a
    partial prefill in its slot. The device still runs that slot, but the
    commit, the recorded count and the proposal key on the verify's row
    identity, so that slot must read as unowned there, as a gap does.
    """
    model = make_fake_spec_model(supports_narrow_decode=narrow)()
    runner = _runner(model, narrow=narrow)
    _add_request(runner, "a", row=0, first_token=1)
    _add_request(runner, "p", row=2, first_token=40)
    batch = runner.lane_batch
    length = int(batch.num_tokens[2])
    last = _last_token(runner, "a")
    drafts = {} if narrow else {"a": [last + 1 + j for j in range(DRAFT_LEN)]}

    model_input, output = _step(runner, drafts=drafts, scheduled=["a"])

    assert int(batch.num_tokens[2]) == length
    assert runner.requests["p"].output_token_ids == []
    assert "p" not in runner._req_accepted_counts
    assert set(runner._proposed_draft_token_ids) == {"a"}
    assert list(_committed(output)) == ["a"]
    assert model_input.row_req_ids == ["a", None, None, None]


def _run_loop(model: FakeSpecModel, steps: int):
    """Feed each step's proposal back as the next step's drafts, as the engine
    does through ``take_draft_token_ids``. Returns the starts and, per request,
    the tokens each step committed."""
    runner = _runner(model, narrow=True)
    _add_request(runner, "a", row=0, first_token=1)
    _add_request(runner, "b", row=3, first_token=40)
    starts = {req: _last_token(runner, req) for req in ("a", "b")}
    per_step: dict[str, list[list[int]]] = {"a": [], "b": []}
    drafts: dict[str, list[int]] = {}
    for _ in range(steps):
        _, output = _step(runner, drafts=drafts)
        for req_id, tokens in _committed(output).items():
            per_step[req_id].append(list(tokens))
        drafts = dict(runner._proposed_draft_token_ids)
        runner._proposed_draft_token_ids = {}
    return starts, per_step


def test_the_lane_loop_commits_what_plain_decoding_would():
    """With every draft accepted, the speculating loop commits exactly the
    tokens the model chooses one at a time, several per step."""
    model = make_fake_spec_model(supports_narrow_decode=True)()
    starts, per_step = _run_loop(model, steps=4)

    for req_id, start in starts.items():
        produced = [token for step in per_step[req_id] for token in step]
        assert produced == [
            (start + 1 + i) % FAKE_VOCAB_SIZE for i in range(len(produced))
        ]
        # The first step has no drafts yet; every later one commits K + 1.
        assert [len(step) for step in per_step[req_id]] == [1] + [DRAFT_LEN + 1] * 3


@pytest.mark.parametrize("accept_depth", [0, 1, 2])
def test_the_lane_loop_commits_the_accepted_prefix_and_a_correction(accept_depth):
    model = make_fake_spec_model(
        accept_depth=accept_depth, supports_narrow_decode=True
    )()
    _, per_step = _run_loop(model, steps=4)

    for req_id in ("a", "b"):
        assert [len(step) for step in per_step[req_id]] == [1] + [accept_depth + 1] * 3
