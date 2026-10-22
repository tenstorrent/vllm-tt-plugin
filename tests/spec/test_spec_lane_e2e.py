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

The last region drives sampled requests through the same step against
``SampledTarget``, whose distribution does not depend on what was drafted:
a lane verify asks for ``logits`` when a scheduled request needs its target
distribution, and each request commits exactly what the host walk draws for
its slot from its own generator.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    SchedulerOutput,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import TTLaneInputBatch
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_accept import accept_speculated_tokens
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS

from . import test_spec_sampled_controls_runner as controls
from . import test_spec_sampled_runner as sync_harness
from .fake_spec_model import FAKE_VOCAB_SIZE, FakeSpecModel, make_fake_spec_model
from .sampled_target import rule_logits

BLOCK_SIZE = 16
MAX_MODEL_LEN = 128
NUM_LANES = 2
PER_LANE = 2
CAPACITY = NUM_LANES * PER_LANE
PROMPT_LEN = 4
DRAFT_LEN = 3

# region Test helpers


def _runner(model: FakeSpecModel, *, narrow: bool = False) -> SimpleNamespace:
    """A runner fake over a real lane batch, carrying what the lane step reads.

    The accept modes are the model's own, as ``TTModelRunner`` reads them off
    its ``SpecPlan``.
    """
    batch = TTLaneInputBatch(
        num_lanes=NUM_LANES,
        per_lane=PER_LANE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=model.vocab_size,
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
        _spec_accept_modes=tuple(model.accept_modes),
        _spec_drafts_from_model=True,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        # A gap's request id is None, which the commit looks up here.
        _req_state_slot={},
        _ngram_proposer=None,
        tt_per_lane_max_num_seqs=PER_LANE,
        tt_data_parallel_size=NUM_LANES,
        vocab_size=model.vocab_size,
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


def _add_request(
    runner: SimpleNamespace,
    req_id: str,
    row: int,
    first_token: int,
    params: SamplingParams | None = None,
):
    """A fully prefilled request at its stable slot ``row``, greedy by default,
    with its own generator if it is seeded."""
    params = params or SamplingParams(temperature=0.0)
    generator = None
    if params.sampling_type == SamplingType.RANDOM_SEED:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(params.seed)
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
        mm_features=None,
        sampling_params=params,
        generator=generator,
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


def _tail(runner: SimpleNamespace, req_id: str) -> tuple[int, int]:
    """A request's last token and its position."""
    batch = runner.lane_batch
    row = batch.req_id_to_index[req_id]
    length = int(batch.num_tokens[row])
    return int(batch.token_ids_cpu[row, length - 1]), length - 1


def _step(
    runner: SimpleNamespace,
    drafts: dict[str, list[int]] | None = None,
    scheduled: list[str] | None = None,
    grammar: GrammarOutput | None = None,
):
    """One synchronous lane decode step over ``scheduled``, every live request
    by default, sampled with the sample-time ``grammar``."""
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
        grammar,
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


# region Sampled requests (logits mode)

# MiMo-V2.6-Flash's default sampling: temperature 1, top-p 0.95, no top-k.
SAMPLED = SamplingParams(temperature=1.0, top_p=0.95, seed=7, ignore_eos=True)


def _drafting_target(**knobs):
    """``SampledTarget`` with its own drafter, the rule's argmax chain."""
    return type("LaneDraftingTarget", (sync_harness._DraftingTarget,), knobs)()


def _generator_copy(runner: SimpleNamespace, req_id: str) -> torch.Generator:
    copy = torch.Generator(device="cpu")
    copy.set_state(runner.requests[req_id].generator.get_state())
    return copy


def _walked_directly(
    model_input, row: int, generator: torch.Generator, params: SamplingParams
) -> list[int]:
    """What ``accept_speculated_tokens`` commits for one slot on its own.

    The target's logits are recomputed from the block the step submitted,
    because ``SampledTarget`` answers by that rule alone.
    """
    logits = rule_logits(model_input.input_tokens, model_input.input_positions)
    result = accept_speculated_tokens(
        target_logits=logits[row : row + 1],
        draft_token_ids=model_input.draft_token_ids[row : row + 1],
        num_valid_drafts=model_input.num_valid_drafts[row : row + 1],
        temperature=torch.tensor([params.temperature]),
        top_p=torch.tensor([params.top_p]),
        generators={0: generator},
    )
    count = int(result.accepted_counts[0])
    return result.committed_token_ids[0, :count].tolist()


def _proposals(runner: SimpleNamespace) -> dict[str, list[int]]:
    """Hand the step's proposals on as the next step's drafts, as the engine
    does through ``take_draft_token_ids``."""
    drafts = dict(runner._proposed_draft_token_ids)
    runner._proposed_draft_token_ids = {}
    return drafts


def test_a_sampled_lane_request_speculates_through_the_logits_walk():
    """The drafter drafts for a sampled request, the verify asks for logits,
    and every step commits what the walk draws from the request's generator.

    The request sits alone in lane 1, the gaps around it, as a MiMo lane
    holding one request does.
    """
    model = _drafting_target()
    runner = _runner(model)
    _add_request(runner, "s", row=3, first_token=5, params=SAMPLED)
    drafts: dict[str, list[int]] = {}
    widths = []
    for _ in range(12):
        reference = _generator_copy(runner, "s")
        model_input, output = _step(runner, drafts=drafts)

        assert model_input.spec_mode == ACCEPT_MODE_LOGITS
        assert not model_input.perform_device_sampling
        assert model_input.row_req_ids == [None, None, None, "s"]
        assert model_input.generators_list[0] == {3: runner.requests["s"].generator}
        committed = _committed(output)
        assert committed == {"s": _walked_directly(model_input, 3, reference, SAMPLED)}
        assert runner.requests["s"].generator.get_state().equal(reference.get_state())
        widths.append(len(committed["s"]))
        drafts = _proposals(runner)
        assert set(drafts) == {"s"}

    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}
    assert max(widths) > 1, "no draft was ever accepted"
    assert min(widths[1:]) < DRAFT_LEN + 1, "no draft was ever rejected"


def test_a_greedy_lane_step_stays_argmax_ids():
    """Ids are enough while every scheduled request is greedy, on a model
    serving both modes, and the step reads no generator."""
    model = _drafting_target()
    runner = _runner(model)
    _add_request(runner, "g", row=0, first_token=1)
    _add_request(runner, "h", row=2, first_token=9)
    _step(runner)
    tails = {req_id: _tail(runner, req_id) for req_id in ("g", "h")}

    model_input, output = _step(runner, drafts=_proposals(runner))

    assert model_input.spec_mode == ACCEPT_MODE_ARGMAX_IDS
    assert model.verify_modes == [ACCEPT_MODE_ARGMAX_IDS] * 2
    assert model_input.generators_list == [{}]
    assert model_input.spec_min_tokens is None
    assert _committed(output) == {
        req_id: sync_harness._base_chain(*tail, DRAFT_LEN + 1)
        for req_id, tail in tails.items()
    }


def test_an_argmax_only_model_never_speculates_a_sampled_lane_request():
    """Without ``logits`` no verify can sample the request, so none is sent.

    Every step the sampled request shares runs as the ordinary decode, and the
    drafter's proposal for it is withheld, while its greedy neighbor in the
    other lane is still offered drafts.
    """
    model = _drafting_target(
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS,), supports_narrow_decode=True
    )
    runner = _runner(model, narrow=True)
    _add_request(runner, "s", row=1, first_token=5, params=SAMPLED)
    _add_request(runner, "g", row=2, first_token=9)
    drafts: dict[str, list[int]] = {}
    for _ in range(4):
        model_input, _ = _step(runner, drafts=drafts)
        assert model_input.spec_mode is None
        drafts = _proposals(runner)
        assert set(drafts) == {"g"}

    assert model.verify_modes == []
    assert model.plain_calls == 4


def test_mixed_lanes_verify_in_logits_mode():
    """One sampled lane makes the whole step logits; each lane gets its own walk.

    The greedy lane commits its exact argmax chain, every draft accepted; the
    sampled lane commits what its own generator draws. A seeded request the
    step leaves out holds its slot without drawing or committing.
    """
    model = _drafting_target()
    runner = _runner(model)
    _add_request(runner, "g", row=0, first_token=1)
    _add_request(runner, "s", row=2, first_token=5, params=SAMPLED)
    _add_request(
        runner,
        "p",
        row=3,
        first_token=20,
        params=SamplingParams(temperature=1.0, seed=99),
    )
    _step(runner, scheduled=["g", "s"])
    left_out = runner.requests["p"].generator.get_state()
    length = int(runner.lane_batch.num_tokens[3])
    greedy_tail = _tail(runner, "g")
    reference = _generator_copy(runner, "s")

    model_input, output = _step(runner, drafts=_proposals(runner), scheduled=["g", "s"])

    assert model_input.spec_mode == ACCEPT_MODE_LOGITS
    assert model_input.row_req_ids == ["g", None, "s", None]
    assert model_input.num_valid_drafts.tolist() == [DRAFT_LEN, 0, DRAFT_LEN, 0]
    assert _committed(output) == {
        "g": sync_harness._base_chain(*greedy_tail, DRAFT_LEN + 1),
        "s": _walked_directly(model_input, 2, reference, SAMPLED),
    }
    assert runner.requests["p"].generator.get_state().equal(left_out)
    assert int(runner.lane_batch.num_tokens[3]) == length


def test_a_lane_logits_verify_publishes_each_request_its_own_logprobs():
    """The walk's logprobs follow the scheduled slots, gaps skipped.

    Asked for by the request in the second lane only, so a walk or a cut that
    read the slot grid as packed rows would hand them to the wrong request or
    misalign the token counts.
    """
    model = _drafting_target()
    runner = _runner(model)
    _add_request(runner, "g", row=1, first_token=1)
    _add_request(
        runner,
        "l",
        row=2,
        first_token=5,
        params=SamplingParams(temperature=1.0, seed=3, logprobs=2),
    )
    _step(runner)

    model_input, output = _step(runner, drafts=_proposals(runner))

    assert model_input.spec_mode == ACCEPT_MODE_LOGITS
    assert model_input.max_num_logprobs == [2]
    committed = _committed(output)
    logprobs = output.logprobs
    starts = logprobs.cu_num_generated_tokens
    assert starts == [0, len(committed["g"]), len(committed["g"]) + len(committed["l"])]
    # Column 0 is each published token's own id.
    assert logprobs.logprob_token_ids[:, 0].tolist() == committed["g"] + committed["l"]


def test_a_left_out_request_s_logprobs_do_not_reach_an_ids_verify():
    """The mode is chosen over the scheduled slots, and so are the logprobs.

    A partial prefill that asked for logprobs holds its slot through a decode
    step whose scheduled requests are greedy. That step verifies in
    ``argmax_ids``, which carries no logprobs, and its request reads none.
    """
    model = _drafting_target()
    runner = _runner(model)
    _add_request(runner, "g", row=0, first_token=1)
    _add_request(runner, "p", row=2, first_token=9, params=SamplingParams(logprobs=1))
    _step(runner, scheduled=["g"])

    model_input, output = _step(runner, drafts=_proposals(runner), scheduled=["g"])

    assert model_input.spec_mode == ACCEPT_MODE_ARGMAX_IDS
    assert model_input.max_num_logprobs == [None]
    assert list(_committed(output)) == ["g"]
    assert output.logprobs is None


def _lane_grammar_output(
    runner: SimpleNamespace, scheduled_drafts: dict[str, list[int]]
) -> GrammarOutput | None:
    """What ``StructuredOutputManager`` hands the step: 1 + len(drafts) rows
    per structured request."""
    structured = [
        req_id
        for req_id in runner.lane_batch.req_ids
        if req_id is not None
        and runner.requests[req_id].sampling_params.structured_outputs is not None
    ]
    if not structured:
        return None
    rows = [
        controls._scheduler_rows(
            list(runner.requests[req_id].output_token_ids),
            scheduled_drafts.get(req_id, []),
        )
        for req_id in structured
    ]
    return GrammarOutput(structured, torch.cat(rows).numpy())


PENALTIES = {
    "presence_penalty": 0.7,
    "frequency_penalty": 0.3,
    "repetition_penalty": 1.5,
}


@pytest.mark.parametrize(
    "case",
    ["penalties", "allowed", "bad-words", "min-tokens", "logprobs", "structured"],
)
def test_a_lane_request_with_a_control_speculates_to_the_ordinary_output(case):
    """Every control the walk applies reaches it from the lane slot grid.

    Greedy, so the unspeculated front-packed arm is an exact reference, ids and
    logprobs both. The request sits at the last slot behind a greedy neighbor
    and two gaps, so a control read at its slot rather than at its live row,
    or not captured at all, changes its output. Its drafts are the reference's
    own continuation, so every filter and penalty has to read them as history.
    """
    if case == "penalties":
        params = SamplingParams(temperature=0.0, ignore_eos=True, **PENALTIES)
    else:
        params = controls._params(case, temperature=0.0)
    reference, reference_logprobs = controls._run_plain(params, 30)
    # The drafter runs, as it does on every lane verify, but each step
    # verifies the drafts the test schedules.
    model = _drafting_target()
    runner = _runner(model)
    _add_request(runner, "n", row=0, first_token=21)
    _add_request(runner, "c", row=3, first_token=1, params=params)
    emitted: list[int] = []
    logprobs: list = []
    while len(emitted) < 24:
        drafts = {
            "n": sync_harness._base_chain(*_tail(runner, "n"), DRAFT_LEN),
            "c": reference[len(emitted) : len(emitted) + DRAFT_LEN],
        }
        if params.structured_outputs is not None:
            drafts["c"] = controls._validated(emitted, drafts["c"])
        _, output = _step(
            runner, drafts=drafts, grammar=_lane_grammar_output(runner, drafts)
        )
        committed = _committed(output)
        emitted.extend(committed["c"])
        logprobs.extend(controls._logprobs_by_request(output, committed).get("c", []))

    assert emitted[:24] == reference[:24]
    assert logprobs[:24] == reference_logprobs[:24]
    assert ACCEPT_MODE_LOGITS in model.verify_modes


# endregion Sampled requests (logits mode)
