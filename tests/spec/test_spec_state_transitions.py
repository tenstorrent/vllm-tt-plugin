# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative state survives the request lifecycle, or is dropped with it.

Speculation adds two pieces of per-request state to the runner:
``_req_accepted_counts``, which tells the next verify which candidate slot each
row continued from, and ``_proposed_draft_token_ids``, which holds a proposal
until the engine collects it. Both are keyed by request id while the persistent
batch is keyed by row, and the rows move: a completion frees one, a preemption
releases one, and ``InputBatch.condense`` slides a later request down into a
freed index.

Every other test in this directory skips that machinery. They stand in for
``_update_states`` by writing ``num_computed_tokens_cpu`` directly, which keeps
them cheap and leaves the whole lifecycle untested: a stale accepted count
would make a reused row continue from a dead request's candidate state, and a
stale proposal would report drafts for a request that no longer exists.

These tests drive the real ``TTModelRunner._update_states`` with real
``SchedulerOutput`` values, then run a real speculative step across the
transition and check what the surviving request committed.
"""

import inspect
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward

from .deterministic_target import (
    TARGET_VOCAB_SIZE,
    DeterministicTarget,
    continuation,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 256
MAX_NUM_REQS = 4
PROMPT_LEN = 4
DRAFT_LEN = 3


def _runner() -> SimpleNamespace:
    """A speculating runner fake whose lifecycle methods are the real ones."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=TARGET_VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    released: list[int] = []
    runner = SimpleNamespace(
        model=DeterministicTarget(),
        input_batch=batch,
        requests={},
        encoder_cache={},
        kv_caches=object(),
        trace_mode="decode_only",
        request_specific_rope=False,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _num_speculative_tokens=DRAFT_LEN,
        # The drafts come from each test, so no proposer runs.
        _spec_method=None,
        _spec_supports_narrow_decode=False,
        _spec_drafts_from_model=False,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _ngram_proposer=None,
        _req_state_slot={},
        # What the runner told the model to release, so a test can see it.
        released_slots=released,
        vllm_config=SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
        ),
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=MAX_MODEL_LEN
        ),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
        _spec_row_state=TTModelRunner._spec_row_state,
        _spec_candidate_block=TTModelRunner._spec_candidate_block,
        _committed_positions=TTModelRunner._committed_positions,
    )
    runner.model.release_request = released.append
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


def _new_request(req_id: str, first_token: int) -> NewRequestData:
    """A request as the scheduler first hands it over, prompt already computed.

    ``num_computed_tokens`` of ``PROMPT_LEN`` stands for a prefill that already
    ran, so the next step for this request is a decode. That keeps these tests
    on the lifecycle rather than on prefill.
    """
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=PROMPT_LEN,
        lora_request=None,
    )


def _empty_cached() -> CachedRequestData:
    return CachedRequestData(
        req_ids=[],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )


def _cached_for(runner, req_ids, resumed=()) -> CachedRequestData:
    """A resumed request carries fresh blocks; a running one carries none.

    ``apply_cached_req_state_update`` requires that of a resume, because the
    scheduler freed the request's blocks when it preempted the request and
    allocates new ones to restart it.
    """
    return CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(resumed),
        new_token_ids=[[] for _ in req_ids],
        all_token_ids={},
        new_block_ids=[
            ([1],) if req_id in set(resumed) else None for req_id in req_ids
        ],
        num_computed_tokens=[
            int(runner.requests[req_id].num_computed_tokens) for req_id in req_ids
        ],
        num_output_tokens=[
            len(runner.requests[req_id].output_token_ids) for req_id in req_ids
        ],
    )


def _scheduler_output(
    *,
    new=(),
    cached=None,
    scheduled=(),
    finished=(),
    preempted=(),
    drafts=None,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = list(new)
    output.scheduled_cached_reqs = cached if cached is not None else _empty_cached()
    output.num_scheduled_tokens = {req_id: 1 for req_id in scheduled}
    output.total_num_scheduled_tokens = len(output.num_scheduled_tokens)
    output.finished_req_ids = set(finished)
    output.preempted_req_ids = set(preempted)
    output.scheduled_spec_decode_tokens = dict(drafts or {})
    return output


def _admit(runner, *specs) -> None:
    """Put requests into the batch the way the scheduler does."""
    runner._update_states(
        _scheduler_output(
            new=[_new_request(req_id, first_token) for req_id, first_token in specs],
            scheduled=[req_id for req_id, _ in specs],
        )
    )


def _tail(runner, req_id) -> tuple[int, int]:
    row = runner.input_batch.req_id_to_index[req_id]
    length = int(runner.input_batch.num_tokens[row])
    return int(runner.input_batch.token_ids_cpu[row, length - 1]), length - 1


def _decode_step(runner, *req_ids, drafts=None):
    """One real speculative step over the rows the batch now holds."""
    scheduler_output = _scheduler_output(
        cached=_cached_for(runner, req_ids),
        scheduled=req_ids,
        drafts=drafts,
    )
    runner._update_states(scheduler_output)
    for req_id in req_ids:
        row = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)
    submission = TTAsyncDecodeController(runner).submit_decode(
        model_input, read_from_device=True
    )
    return runner._finish_front_packed_sync(
        None,
        fwd=_SyncForward(
            tt_out=submission.tt_out,
            tt_log_probs=None,
            sampling_params=model_input.tt_sampling_params,
            model_input=model_input,
            batch_size_per_dp=[len(req_ids)],
            perform_device_sampling=False,
            is_decode=True,
            spec_hidden=submission.spec_hidden,
        ),
    )


def _accept_everything(runner, req_id):
    """The drafts this target will accept, from that row's own tail."""
    token, position = _tail(runner, req_id)
    return continuation(token, position, DRAFT_LEN)


# region Completion


def test_a_completed_request_leaves_no_speculative_state_behind():
    """Its accepted count, its row and its device slot all go.

    A surviving accepted count is not inert: the next request to take that id
    would continue from a candidate slot chosen for a request that is gone.
    """
    runner = _runner()
    _admit(runner, ("a", 11), ("b", 211))
    _decode_step(
        runner,
        "a",
        "b",
        drafts={
            "a": _accept_everything(runner, "a"),
            "b": _accept_everything(runner, "b"),
        },
    )
    assert runner._req_accepted_counts["a"] == DRAFT_LEN + 1
    runner._req_state_slot["a"] = 0

    # "a" finishes; "b" keeps decoding.
    _decode_step(runner, "b", drafts={"b": _accept_everything(runner, "b")})
    runner._update_states(
        _scheduler_output(
            cached=_cached_for(runner, ["b"]), scheduled=["b"], finished=["a"]
        )
    )

    assert "a" not in runner.requests
    assert "a" not in runner.input_batch.req_id_to_index
    assert "a" not in runner._req_state_slot
    assert 0 in runner.released_slots
    # The accepted count is swept the next time inputs are built, which is
    # where the runner next looks at the map.
    _decode_step(runner, "b", drafts={"b": _accept_everything(runner, "b")})
    assert "a" not in runner._req_accepted_counts


def test_a_request_s_first_speculative_step_is_told_it_continued_from_one():
    """No recorded count means one committed token, which is the prefill's.

    The default is not free: a model reads the count to pick the candidate
    state slot its previous step committed from, so a first step told anything
    but one reads a slot nothing wrote.
    """
    runner = _runner()
    _admit(runner, ("a", 11))

    _decode_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})

    assert runner.model.verify_calls[0]["accepted_counts"].tolist()[0] == 1


def test_a_request_that_finishes_with_a_proposal_in_hand_reports_no_drafts():
    """The engine collects after the step, and the request may be gone by then."""
    runner = _runner()
    _admit(runner, ("a", 11))
    runner._proposed_draft_token_ids["a"] = [1, 2, 3]

    runner._update_states(_scheduler_output(finished=["a"]))

    assert runner.take_draft_token_ids() is None


# endregion Completion

# region Preemption


def test_a_preempted_request_loses_its_candidate_state():
    """Preemption frees the KV blocks, so no candidate slot survives it.

    A resume re-prefills the prompt and every generated token. Continuing from
    the old accepted count would select a candidate slot the model has since
    overwritten, and the row would carry on from a token the request never
    emitted.
    """
    runner = _runner()
    _admit(runner, ("a", 11), ("b", 211))
    _decode_step(
        runner,
        "a",
        "b",
        drafts={
            "a": _accept_everything(runner, "a"),
            "b": _accept_everything(runner, "b"),
        },
    )
    assert runner._req_accepted_counts["a"] == DRAFT_LEN + 1
    runner._req_state_slot["a"] = 0

    # "a" is preempted: still known, but out of the batch and off the device.
    runner._update_states(
        _scheduler_output(
            cached=_cached_for(runner, ["b"]), scheduled=["b"], preempted=["a"]
        )
    )

    assert "a" in runner.requests, "a preempted request keeps its cached state"
    assert "a" not in runner.input_batch.req_id_to_index
    assert "a" not in runner._req_state_slot
    assert 0 in runner.released_slots


def test_a_resumed_request_speculates_from_a_reset_count():
    """The prefill that resumes it drops the count, so the next verify sees 1."""
    runner = _runner()
    _admit(runner, ("a", 11))
    _decode_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    assert runner._req_accepted_counts["a"] == DRAFT_LEN + 1

    runner._update_states(_scheduler_output(preempted=["a"]))
    # The scheduler resumes it as a cached request with its prompt to recompute.
    runner.requests["a"].num_computed_tokens = 0
    resumed = _scheduler_output(
        cached=_cached_for(runner, ["a"], resumed=["a"]),
        scheduled=["a"],
    )
    runner._update_states(resumed)
    # A prefill step: what the scheduler schedules for a resumed request.
    TTModelRunner._prepare_model_inputs(runner, resumed, None)

    assert "a" not in runner._req_accepted_counts


# endregion Preemption

# region Compaction and row reuse


def test_a_condensed_row_keeps_its_own_speculative_state():
    """The survivor slides down a row and continues its own sequence.

    ``condense`` moves a request to a different index, and both speculative
    maps are keyed by request id rather than by row, which is what makes the
    move safe. This is the test that says so: the survivor's next committed
    tokens have to follow its own history, not the history of the request that
    used to hold its new row.
    """
    runner = _runner()
    _admit(runner, ("a", 11), ("b", 211), ("c", 111))
    rows = {req: runner.input_batch.req_id_to_index[req] for req in ("a", "b", "c")}
    assert rows == {"a": 0, "b": 1, "c": 2}
    _decode_step(
        runner,
        "a",
        "b",
        "c",
        drafts={req: _accept_everything(runner, req) for req in ("a", "b", "c")},
    )
    counts_before = dict(runner._req_accepted_counts)

    # "a" and "b" finish. "c" is the only request left and condense moves it.
    runner._update_states(
        _scheduler_output(
            cached=_cached_for(runner, ["c"]), scheduled=["c"], finished=["a", "b"]
        )
    )

    assert runner.input_batch.req_id_to_index["c"] == 0, "condense did not move it"
    assert runner._req_accepted_counts["c"] == counts_before["c"]

    # And the step after the move continues "c" and nothing else.
    expected = continuation(*_tail(runner, "c"), DRAFT_LEN + 1)
    output = _decode_step(runner, "c", drafts={"c": expected[:DRAFT_LEN]})
    assert output.req_ids == ["c"]
    assert output.sampled_token_ids[0] == expected


def test_a_reused_row_inherits_nothing_from_the_request_that_left_it():
    """A new request taking a freed index starts from its own prompt.

    The row's token history, its accepted count and any pending proposal all
    belonged to a request that is gone. Anything inherited here would make the
    new request's first committed token a continuation of the old one's text.
    """
    runner = _runner()
    _admit(runner, ("a", 11))
    _decode_step(runner, "a", drafts={"a": _accept_everything(runner, "a")})
    runner._proposed_draft_token_ids["a"] = [1, 2, 3]
    assert runner.input_batch.req_id_to_index["a"] == 0

    runner._update_states(_scheduler_output(finished=["a"]))
    # A different request takes the freed index.
    _admit(runner, ("z", 151))

    assert runner.input_batch.req_id_to_index["z"] == 0
    assert "z" not in runner._req_accepted_counts
    assert "z" not in runner._proposed_draft_token_ids
    assert runner.take_draft_token_ids() is None

    # Its first speculative step continues its own prompt.
    expected = continuation(*_tail(runner, "z"), DRAFT_LEN + 1)
    output = _decode_step(runner, "z", drafts={"z": expected[:DRAFT_LEN]})

    assert output.sampled_token_ids[0] == expected
    assert runner.requests["z"].output_token_ids == expected
    # And the verify was told this row continued from one token, its prefill's,
    # rather than from the candidate state its predecessor had reached. A model
    # selects its candidate slot from this, so a stale or defaulted-wrong count
    # reads the wrong state while the committed tokens still look right.
    assert int(runner.model.verify_calls[-1]["accepted_counts"][0]) == 1


def test_two_survivors_keep_their_counts_apart_across_a_condense():
    """A mixed batch where the moved rows accepted different amounts.

    One row accepted everything and one rejected at its first draft, so their
    counts differ, and then the row between them finishes. If the maps followed
    rows rather than request ids, the two would swap what the next verify is
    told about them.
    """
    runner = _runner()
    _admit(runner, ("a", 11), ("b", 211), ("c", 111))
    good = _accept_everything(runner, "a")
    wrong_first = _accept_everything(runner, "c")
    wrong_first[0] = (wrong_first[0] + 1) % TARGET_VOCAB_SIZE
    _decode_step(
        runner,
        "a",
        "b",
        "c",
        drafts={
            "a": good,
            "b": _accept_everything(runner, "b"),
            "c": wrong_first,
        },
    )

    assert runner._req_accepted_counts["a"] == DRAFT_LEN + 1
    assert runner._req_accepted_counts["c"] == 1

    runner._update_states(
        _scheduler_output(
            cached=_cached_for(runner, ["a", "c"]),
            scheduled=["a", "c"],
            finished=["b"],
        )
    )

    assert runner.input_batch.req_id_to_index == {"a": 0, "c": 1}
    assert runner._req_accepted_counts["a"] == DRAFT_LEN + 1
    assert runner._req_accepted_counts["c"] == 1

    # Each continues its own sequence after the move.
    expected = {
        req: continuation(*_tail(runner, req), DRAFT_LEN + 1) for req in ("a", "c")
    }
    output = _decode_step(
        runner,
        "a",
        "c",
        drafts={req: expected[req][:DRAFT_LEN] for req in ("a", "c")},
    )

    committed = dict(zip(output.req_ids, output.sampled_token_ids))
    assert committed["a"] == expected["a"]
    assert committed["c"] == expected["c"]
    # What the verify was told, per row, which is what a model selects its
    # candidate state slot with and what the committed tokens cannot reveal.
    call = runner.model.verify_calls[-1]
    assert int(call["accepted_counts"][0]) == DRAFT_LEN + 1
    assert int(call["accepted_counts"][1]) == 1


def test_the_accepted_count_reaches_the_verify_after_a_condense():
    """What the model is told, not only what the map holds.

    The count crosses to the model as ``accepted_counts``, per row, in the
    order the batch now holds. A condense that left the map right and the wire
    order wrong would still commit the right tokens this step and select the
    wrong candidate state on the device.
    """
    runner = _runner()
    _admit(runner, ("a", 11), ("b", 211), ("c", 111))
    wrong_first = _accept_everything(runner, "a")
    wrong_first[0] = (wrong_first[0] + 1) % TARGET_VOCAB_SIZE
    _decode_step(
        runner,
        "a",
        "b",
        "c",
        drafts={
            "a": wrong_first,
            "b": _accept_everything(runner, "b"),
            "c": _accept_everything(runner, "c"),
        },
    )
    assert runner._req_accepted_counts == {
        "a": 1,
        "b": DRAFT_LEN + 1,
        "c": DRAFT_LEN + 1,
    }

    runner._update_states(
        _scheduler_output(
            cached=_cached_for(runner, ["a", "c"]),
            scheduled=["a", "c"],
            finished=["b"],
        )
    )
    _decode_step(
        runner,
        "a",
        "c",
        drafts={req: _accept_everything(runner, req) for req in ("a", "c")},
    )

    call = runner.model.verify_calls[-1]
    rows = runner.input_batch.req_id_to_index
    assert rows == {"a": 0, "c": 1}
    # Row order is the batch's, and each row carries its own request's count
    # from the step before: "a" rejected its first draft and committed one
    # token, "c" accepted everything and committed 1+K.
    assert int(call["accepted_counts"][rows["a"]]) == 1
    assert int(call["accepted_counts"][rows["c"]]) == DRAFT_LEN + 1
    assert int(call["num_valid_drafts"][rows["a"]]) == DRAFT_LEN
    assert int(call["num_valid_drafts"][rows["c"]]) == DRAFT_LEN


# endregion Compaction and row reuse
