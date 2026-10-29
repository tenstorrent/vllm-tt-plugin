# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device penalties of the fused verify, on the host.

A ``fused_sample`` model that declares ``supports_spec_device_penalties`` keeps
each penalty row's output counts and prompt mask on the device, applies the
penalties inside the step and counts every commit itself. The plugin sends the
per-row parameters, and the token lists of a row whose device state is not
current (``spec_device_penalties.SpecPenaltyTracker``). These tests pin the
plugin side:

- the predicates and the mode choice, with the capability on and off;
- the ``penalties`` entry of ``spec_sampling`` (parameters, a greedy penalty row
  sent as sampled with top_k 1 and top_p 0, the rebuild lists, no draw for an
  all-greedy step);
- the tracker: when a row is rebuilt (join, resume, slot change, a non-fused
  step, a finish) and that a chained step never rebuilds;
- the chain blockers;
- end to end on the chain fake, whose "device" keeps the counts the way the real
  one does: at every step that is not chained they equal the request's output.

The capability is one runner attribute (``_spec_device_penalties``), which
``load_model`` sets from ``model_capabilities``; the default is False and then
nothing changes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.model_runner import TTModelRunner, fused_spec_sampling
from vllm_tt_plugin.spec_accept import params_fused_servable, params_support_servable
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    PLACEHOLDER_TOKEN_ID,
)
from vllm_tt_plugin.spec_device_penalties import (
    SpecPenaltyTracker,
    device_penalties,
    device_penalty_stats,
    non_neutral_rows,
    reset_device_penalty_stats,
)

from .sampled_target import SAMPLED_VOCAB_SIZE
from .test_spec_chain_async import (
    GREEDY,
    SAMPLED,
    Engine,
    _controller,
    _decode_so,
    _make_runner,
)
from .test_spec_chain_async import K as CHAIN_K
from .test_spec_device_logprobs_plugin import (
    LogprobsTarget,
    _caps,
    _first_step_build,
    _params,
    _sampling_runner,
)
from .test_spec_sampled_runner import _add_request as _add_sampled_request
from .test_spec_sampled_runner import _argmax_drafts, _build

V = SAMPLED_VOCAB_SIZE
CAND = 8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (
        "VLLM_TT_SPEC_FUSED",
        "VLLM_TT_SPEC_FUSED_SAMPLED",
        "VLLM_TT_SPEC_CHAIN",
        "VLLM_TT_SPEC_CHAIN_SAMPLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")
    reset_device_penalty_stats()


def _pen_caps(runner, penalties: bool):
    """What ``load_model`` does with the capability, on a runner double."""
    runner._spec_device_penalties = penalties
    runner._spec_penalty_tracker = SpecPenaltyTracker()
    return runner


# region H7: the predicate matrix


@pytest.mark.parametrize(
    "kwargs, penalized, cap, expected",
    [
        (dict(), False, False, True),
        (dict(), False, True, True),
        (dict(), True, False, False),  # a penalty needs the capability
        (dict(), True, True, True),
        (dict(temperature=1.0, top_k=5), True, True, True),
        (dict(temperature=1.0, top_k=CAND), True, True, False),  # the other rules hold
        (
            dict(temperature=1.0, top_p=0.9),
            True,
            True,
            False,
        ),  # top_k off needs device_top_p
        (dict(logprobs=3), True, True, False),  # raw logprobs come before penalties
        (dict(logprobs=3), False, True, False),  # and no logprobs capability here
        (dict(allowed_token_ids=[1, 2]), True, True, False),
        (dict(min_tokens=4, stop_token_ids=[7]), True, True, False),
    ],
)
def test_params_fused_servable_with_penalties(kwargs, penalized, cap, expected):
    got = params_fused_servable(
        _params(**kwargs),
        0,
        CAND,
        V,
        device_logprobs=False,
        device_top_p=False,
        penalized=penalized,
        device_penalties=cap,
    )
    assert got is expected


def test_with_the_capability_off_the_predicate_is_the_old_one_over_a_grid():
    for temperature in (0.0, 0.7):
        for top_k in (0, 1, CAND - 1, CAND, V + 5):
            for top_p in (1.0, 0.8):
                for logprobs in (None, 0, 5, -1):
                    for lp_cap in (False, True):
                        for tp_cap in (False, True):
                            params = _params(
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                logprobs=logprobs,
                            )
                            old = params_fused_servable(
                                params,
                                0,
                                CAND,
                                V,
                                device_logprobs=lp_cap,
                                device_top_p=tp_cap,
                            )
                            new = params_fused_servable(
                                params,
                                0,
                                CAND,
                                V,
                                device_logprobs=lp_cap,
                                device_top_p=tp_cap,
                                penalized=False,
                                device_penalties=True,
                            )
                            assert old == new
                            assert not params_fused_servable(
                                params,
                                0,
                                CAND,
                                V,
                                device_logprobs=lp_cap,
                                device_top_p=tp_cap,
                                penalized=True,
                                device_penalties=False,
                            )
    # the support rule does not read penalties at all (the runner does)
    assert params_support_servable(_params(), 0, CAND)


# endregion

# region H7: the mode the runner asks for


def _modes(runner, rows):
    return runner._spec_mode_for_step(
        rows, torch.full((len(rows),), CHAIN_K, dtype=torch.int32)
    )


def _two_rows(runner, params, first=GREEDY):
    engine = Engine(runner)
    engine.add("a", 1, first)
    engine.add("p", 40, params)
    return ["a", "p"]


def _pen_runner(penalties=True, logprobs=False, top_p=False):
    runner = _make_runner(LogprobsTarget())
    _caps(runner, logprobs, top_p)
    return _pen_caps(runner, penalties)


@pytest.mark.parametrize(
    "params, caps, expected",
    [
        (dict(presence_penalty=0.5), True, ACCEPT_MODE_FUSED_SAMPLE),
        (dict(frequency_penalty=0.3), True, ACCEPT_MODE_FUSED_SAMPLE),
        (dict(repetition_penalty=1.2), True, ACCEPT_MODE_FUSED_SAMPLE),
        (
            dict(repetition_penalty=1.2, temperature=0.8, top_k=5),
            True,
            ACCEPT_MODE_FUSED_SAMPLE,
        ),
        # without the capability the row needs the host target distribution, as before
        (dict(presence_penalty=0.5), False, ACCEPT_MODE_LOGITS),
        (
            dict(repetition_penalty=1.2, temperature=0.8, top_k=5),
            False,
            ACCEPT_MODE_LOGITS,
        ),
        # a penalty and logprobs together are not served
        (dict(presence_penalty=0.5, logprobs=3), True, ACCEPT_MODE_LOGITS),
        # a penalty with a control the device cannot apply
        (
            dict(presence_penalty=0.5, allowed_token_ids=[1, 2]),
            True,
            ACCEPT_MODE_LOGITS,
        ),
    ],
)
def test_the_mode_follows_the_penalty_row_and_the_capability(params, caps, expected):
    runner = _pen_runner(penalties=caps, logprobs=True)
    rows = _two_rows(runner, _params(**params))
    assert _modes(runner, rows) == expected


def test_a_penalty_row_beside_a_logprobs_row_is_served_only_without_a_shared_row():
    runner = _pen_runner(logprobs=True)
    engine = Engine(runner)
    engine.add("l", 1, _params(logprobs=5))
    engine.add("p", 40, _params(presence_penalty=0.5))
    assert _modes(runner, ["l", "p"]) == ACCEPT_MODE_FUSED_SAMPLE


# endregion

# region The sampling inputs


def _join_build(runner, *req_ids):
    """The first build of requests just added: their device state does not exist yet."""
    return _build(
        runner, *req_ids, drafts={r: _argmax_drafts(runner, r) for r in req_ids}
    )


def _pen_sampling_runner(penalties=True):
    model, runner = _sampling_runner()
    _pen_caps(runner, penalties)
    return model, runner


def _sampling(runner, mi, tracker=True):
    return fused_spec_sampling(
        mi,
        runner._spec_plan,
        vocab_size=runner.vocab_size,
        device_top_p=runner._spec_device_top_p,
        penalty_tracker=runner._spec_penalty_tracker if tracker else None,
    )


def test_a_penalty_row_is_sent_with_its_parameters_and_a_greedy_one_as_sampled():
    model, runner = _pen_sampling_runner()
    _add_sampled_request(runner, "g", GREEDY, first_token=1)
    _add_sampled_request(
        runner,
        "p",
        _params(presence_penalty=0.5, frequency_penalty=0.25, repetition_penalty=1.5),
        first_token=9,
    )
    _add_sampled_request(
        runner,
        "s",
        _params(temperature=0.9, top_k=6, presence_penalty=0.1),
        first_token=17,
    )
    mi = _join_build(runner, "g", "p", "s")
    assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    ss = _sampling(runner, mi)
    rows = int(mi.input_tokens.shape[0])
    pen = ss["penalties"]
    assert set(pen) == {"repetition", "frequency", "presence", "rebuild"}
    for name in ("repetition", "frequency", "presence"):
        assert pen[name].dtype == torch.float32 and pen[name].shape == (rows,)
    assert pen["repetition"].tolist()[:3] == [1.0, 1.5, 1.0]
    assert pen["frequency"].tolist()[:3] == [0.0, 0.25, 0.0]
    assert pen["presence"].tolist()[:3] == pytest.approx([0.0, 0.5, 0.1])
    # padding rows are neutral
    assert pen["repetition"].tolist()[3:] == [1.0] * (rows - 3)
    # the greedy penalty row is a sampled row keeping one candidate
    assert ss["greedy"].tolist()[:3] == [True, False, False]
    assert ss["top_k"].tolist()[:3] == [1, 1, 6]
    assert ss["top_p"][1].item() == 0.0 and ss["top_p"][0].item() == 1.0
    assert ss["temperature"][1].item() == 1.0 and ss["temperature"][0].item() == 0.0
    # both penalty rows are rebuilt on their first step, with the request's host lists
    assert sorted(pen["rebuild"]) == [1, 2]
    prompt, out = pen["rebuild"][1]
    row = runner.input_batch.req_id_to_index["p"]
    n = int(runner.input_batch.num_tokens[row])
    assert prompt.dtype == out.dtype == torch.int32 and prompt.dim() == out.dim() == 1
    assert out.tolist() == runner.requests["p"].output_token_ids
    assert prompt.tolist() == runner.requests["p"].prompt_token_ids
    assert int(prompt.numel()) + int(out.numel()) == n
    # the next build of the same rows (a step later) carries no rebuild
    mi2 = _first_step_build(runner, "g", "p", "s")
    assert "rebuild" not in _sampling(runner, mi2)["penalties"]


def test_an_all_greedy_step_with_a_penalty_row_sends_the_dict_and_draws_nothing():
    model, runner = _pen_sampling_runner()
    _add_sampled_request(runner, "g", GREEDY, first_token=1)
    _add_sampled_request(runner, "p", _params(repetition_penalty=1.3), first_token=9)
    mi = _join_build(runner, "g", "p")
    state = torch.get_rng_state()
    ss = _sampling(runner, mi)
    assert torch.equal(torch.get_rng_state(), state), "no sampled row: nothing is drawn"
    assert ss is not None and not ss["uniforms"].any()
    assert ss["greedy"].tolist()[:2] == [True, False]


def test_without_a_penalty_row_or_without_the_tracker_nothing_new_is_sent():
    model, runner = _pen_sampling_runner()
    _add_sampled_request(runner, "g", GREEDY, first_token=1)
    _add_sampled_request(runner, "s", _params(temperature=0.9, top_k=6), first_token=9)
    mi = _first_step_build(runner, "g", "s")
    ss = _sampling(runner, mi)
    assert "penalties" not in ss
    assert device_penalty_stats()["steps"] == 0
    # a step with penalties but no tracker (capability off) builds as before
    model2, runner2 = _pen_sampling_runner(penalties=False)
    _add_sampled_request(runner2, "g", GREEDY, first_token=1)
    _add_sampled_request(
        runner2,
        "p",
        _params(temperature=0.9, top_k=6, presence_penalty=0.5),
        first_token=9,
    )
    mi2 = _first_step_build(runner2, "g", "p")
    ss2 = _sampling(runner2, mi2, tracker=False)
    assert "penalties" not in ss2 and ss2["top_k"].tolist()[:2] == [1, 6]


# endregion

# region The tracker: when a row is rebuilt


def _rows(n, *, rep=None, freq=None, pres=None):
    def vec(values, neutral):
        return torch.tensor(
            [
                neutral if values is None or i >= len(values) else values[i]
                for i in range(n)
            ],
            dtype=torch.float32,
        )

    return vec(rep, 1.0), vec(freq, 0.0), vec(pres, 0.0)


def _call(tracker, ids, rep, freq, pres, *, chained=False, rows=4):
    prompts = torch.tensor(
        [[1, 2, -1], [3, -1, -1], [4, 5, 6], [7, -1, -1]], dtype=torch.int32
    )
    outputs = torch.tensor(
        [[8, 8, 9], [10, -1, -1], [11, 12, -1], [13, -1, -1]], dtype=torch.int32
    )
    return device_penalties(
        tracker, ids, rep, freq, pres, prompts, outputs, chained=chained, rows=rows
    )


def test_a_row_is_rebuilt_on_join_then_followed_without_a_rebuild():
    tracker = SpecPenaltyTracker()
    rep, freq, pres = _rows(3, rep=[1.2, 1.0, 1.0], pres=[0.0, 0.0, 0.5])
    first = _call(tracker, ["a", "b", "c"], rep, freq, pres)
    assert sorted(first["rebuild"]) == [0, 2]
    assert first["rebuild"][0][0].tolist() == [1, 2] and first["rebuild"][0][
        1
    ].tolist() == [8, 8, 9]
    assert first["rebuild"][2][0].tolist() == [4, 5, 6]
    assert first["repetition"].tolist() == pytest.approx([1.2, 1.0, 1.0, 1.0])
    assert first["presence"].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0])
    second = _call(tracker, ["a", "b", "c"], rep, freq, pres)
    assert "rebuild" not in second
    assert device_penalty_stats() == {"steps": 2, "rebuilds": 2}


def test_a_chained_step_never_rebuilds_and_refuses_a_row_that_is_not_current():
    tracker = SpecPenaltyTracker()
    rep, freq, pres = _rows(2, rep=[1.2, 1.1])
    _call(tracker, ["a", "b"], rep, freq, pres)
    chained = _call(tracker, ["a", "b"], rep, freq, pres, chained=True)
    assert "rebuild" not in chained
    # the batch condensed: "b" now sits in slot 0 of a chained step; the model's
    # rows follow the requests, so no rebuild is needed
    again = _call(tracker, ["b"], rep[1:], freq[1:], pres[1:], chained=True)
    assert "rebuild" not in again
    tracker.invalidate(["b"])
    with pytest.raises(RuntimeError, match="not current"):
        _call(tracker, ["b"], rep[1:], freq[1:], pres[1:], chained=True)


@pytest.mark.parametrize("event", ["slot_change", "other_step", "finish", "preempt"])
def test_the_events_that_make_a_row_stale_rebuild_it_on_the_next_step(event):
    tracker = SpecPenaltyTracker()
    rep, freq, pres = _rows(2, rep=[1.2, 1.1])
    _call(tracker, ["a", "b"], rep, freq, pres)
    ids = ["a", "b"]
    if (
        event == "slot_change"
    ):  # "a" finished, "b" moved to slot 0 on a step that is not chained
        ids, rep, freq, pres = ["b"], rep[1:], freq[1:], pres[1:]
        want = [0]
    else:
        tracker.invalidate(["a"])  # what a logits step, a finish or a preemption does
        want = [0]
    step = _call(tracker, ids, rep, freq, pres)
    assert sorted(step["rebuild"]) == want


def test_neutral_rows_need_no_entry_at_all():
    tracker = SpecPenaltyTracker()
    rep, freq, pres = _rows(2)
    assert _call(tracker, ["a", "b"], rep, freq, pres) is None
    assert device_penalty_stats()["steps"] == 0
    assert not bool(non_neutral_rows(rep, freq, pres).any())
    # a row that turned neutral by the plugin's own values stays out of the payload
    rep, freq, pres = _rows(2, freq=[0.0, 0.2])
    payload = _call(tracker, ["a", "b"], rep, freq, pres)
    assert sorted(payload["rebuild"]) == [1]


def test_a_rebuild_without_the_token_lists_fails_loudly():
    tracker = SpecPenaltyTracker()
    rep, freq, pres = _rows(1, rep=[1.2])
    with pytest.raises(RuntimeError, match="needs its prompt and output tokens"):
        device_penalties(
            tracker, ["a"], rep, freq, pres, None, None, chained=False, rows=2
        )


# endregion

# region The runner's lifecycle hooks


def test_a_non_fused_decode_step_invalidates_the_penalty_rows(monkeypatch):
    runner = _pen_runner(penalties=True)
    engine = Engine(runner)
    engine.add("p", 40, _params(presence_penalty=0.5))
    runner._spec_penalty_tracker.mark_current("p", 0)
    assert runner._spec_penalty_tracker.is_current("p")
    # a request that asks for logprobs beside it makes the verify a logits step
    engine.add("l", 1, _params(min_tokens=14, max_tokens=16, stop_token_ids=[7]))
    so = _decode_so(runner, ["p", "l"])
    mi = TTModelRunner._prepare_model_inputs(runner, so, None)
    assert mi.spec_mode != ACCEPT_MODE_FUSED_SAMPLE
    assert not runner._spec_penalty_tracker.is_current("p")


def test_a_finished_or_preempted_request_is_forgotten():
    runner = _pen_runner(penalties=True)
    runner._req_state_slot = {"p": 0}
    runner._spec_penalty_tracker.mark_current("p", 0)
    TTModelRunner._release_model_request(runner, "p")
    assert not runner._spec_penalty_tracker.is_current("p")


def _load(model_capabilities):
    runner = SimpleNamespace(
        load_config=None,
        vllm_config=None,
        model_config=None,
        supports_device_grammar=False,
        _spec_device_logprobs=False,
        _spec_device_top_p=False,
        _spec_device_penalties=False,
    )
    model = SimpleNamespace(model_capabilities=model_capabilities)
    import vllm_tt_plugin.model_runner as mr

    real = mr.TTModelLoader
    mr.TTModelLoader = lambda _config: SimpleNamespace(load_model=lambda **_: model)
    try:
        TTModelRunner.load_model(runner)
    finally:
        mr.TTModelLoader = real
    return runner


def test_the_capability_is_read_at_load_and_defaults_to_false():
    assert (
        _load({"supports_spec_device_penalties": True})._spec_device_penalties is True
    )
    assert _load({})._spec_device_penalties is False
    only_lp = _load({"supports_spec_device_logprobs": True})
    assert only_lp._spec_device_penalties is False


# endregion

# region The chain blockers


def _blocker(params, *, cap=True, current=True, rows=("a", "p")):
    runner = _pen_runner(penalties=cap)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("p", 40, params)
    if current:
        runner._spec_penalty_tracker.mark_current("p", 1)
    so = _decode_so(runner, list(rows))
    return runner._chain_rows_blocker(list(rows), so, 1)[0]


def test_penalty_rows_chain_only_with_the_capability_and_a_current_state(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    assert _blocker(_params(presence_penalty=0.5)) is None
    assert _blocker(_params(repetition_penalty=1.2, temperature=0.8, top_k=5)) is None
    assert _blocker(_params(presence_penalty=0.5), cap=False) == "penalties"
    assert _blocker(_params(presence_penalty=0.5, logprobs=2)) == "penalties"
    assert _blocker(_params(presence_penalty=0.5), current=False) == "pen_state"
    # the other blockers keep their order and names
    assert (
        _blocker(_params(presence_penalty=0.5, allowed_token_ids=[1, 2])) == "penalties"
    )


# endregion

# region End to end on the chain fake: the model's counts follow the request


class PenaltyTarget(LogprobsTarget):
    """The logprobs chain target with a model-side penalty state (the real one's).

    ``counts[slot]`` is the list of output ids the device counts hold. A ``rebuild``
    entry replaces a slot's list from the plugin's lists; every other fused step
    adds its committed tokens; a slot remap moves the state with the request; a
    step that is not fused clears it (the real model never sees those commits, and
    a later fused step must be sent a rebuild). At every step that is not chained,
    each penalty row's state must equal the request's committed output, which
    ``truth`` reads off the runner."""

    model_capabilities = {
        **LogprobsTarget.model_capabilities,
        "supports_spec_device_penalties": True,
    }

    def __init__(self):
        super().__init__()
        self.counts: dict[int, list[int]] = {}
        self.truth = None  # slot -> committed output of the request in that slot
        self.checked = 0
        self.rebuilt: list[tuple[int, ...]] = []
        self.pen_steps = 0
        self.chained_pen_steps = 0

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        ss = kwargs.get("spec_sampling")
        pen = None
        if ss is not None and "penalties" in ss:
            ss = dict(ss)
            pen = ss.pop("penalties")
            kwargs["spec_sampling"] = ss
        remap = kwargs.get("slot_remap")
        if remap is not None:
            self.counts = {
                i: self.counts[int(old)]
                for i, old in enumerate(remap)
                if int(old) in self.counts
            }
        if spec_mode != ACCEPT_MODE_FUSED_SAMPLE:
            self.counts = {}  # the commit of this step is invisible to the device state
        live = []
        if pen is not None:
            live = (
                non_neutral_rows(pen["repetition"], pen["frequency"], pen["presence"])
                .nonzero()
                .reshape(-1)
                .tolist()
            )
            rebuild = pen.get("rebuild") or {}
            self.rebuilt.append(tuple(sorted(rebuild)))
            for slot, (prompt, out) in rebuild.items():
                self.counts[int(slot)] = out.tolist()
            chained = kwargs.get("chain_rows") is not None
            for slot in live:
                assert slot in self.counts, (
                    f"penalty row {slot} sent with no state and no rebuild"
                )
                if not chained:
                    assert self.counts[slot] == self.truth(slot), (
                        slot,
                        self.counts[slot],
                        self.truth(slot),
                    )
                    self.checked += 1
            self.pen_steps += 1
            self.chained_pen_steps += int(chained)
        out = super().decode_forward(tokens, start_pos, spec_mode, **kwargs)
        if live:
            handle = getattr(out, "handle", None)
            readout = handle._readout if handle is not None else out
            committed = (
                readout.accepted_token_ids
                if hasattr(readout, "accepted_token_ids")
                else readout[0]
            )
            counts = (
                readout.accepted_counts
                if hasattr(readout, "accepted_counts")
                else readout[1]
            )
            for slot in live:
                n = int(counts[slot])
                self.counts[slot] = self.counts[slot] + [
                    int(t) for t in committed[slot, :n]
                ]
        return out


def _truth(runner):
    def truth(slot):
        req_id = runner.input_batch.req_ids[slot]
        return list(runner.requests[req_id].output_token_ids)

    return truth


def _run(monkeypatch, *, steps=16, chain=True):
    if chain:
        monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
        monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    model = PenaltyTarget()
    runner = _make_runner(model)
    _caps(runner, False, False)
    _pen_caps(runner, True)
    model.truth = _truth(runner)
    engine = Engine(runner)
    engine.add("p0", 1, _params(presence_penalty=0.5))
    engine.add("p1", 9, _params(repetition_penalty=1.2, temperature=0.8, top_k=5))
    engine.add("n", 17, GREEDY)
    for i in range(steps):
        if i == 5:
            engine.add("p2", 25, _params(frequency_penalty=0.3))  # a join
        if i == 7:
            engine.stop_after["p0"] = (
                len(engine.outputs["p0"]) + 1
            )  # a finish: the batch condenses
        engine.iterate()
    engine.finish()
    return model, runner, engine


def test_the_model_state_follows_the_request_through_joins_finishes_and_chains(
    monkeypatch,
):
    model, runner, engine = _run(monkeypatch)
    assert model.checked >= 4, "the state check ran on too few steps"
    ctrl = _controller(runner)
    assert ctrl._chained_submissions > 0, dict(ctrl._chain_breaks)
    assert model.chained_pen_steps > 0, "no penalty step was chained"
    assert ctrl._chain_breaks["penalties"] == 0 and ctrl._chain_breaks["pen_state"] == 0
    # every rebuild is a join (p0 and p1 first, p2 at step 5) or a slot change
    # after the finish; never a chained step
    stats = device_penalty_stats()
    assert stats["steps"] == model.pen_steps
    assert 3 <= stats["rebuilds"] <= 6, stats
    assert any(model.rebuilt), "no rebuild reached the model"
    assert len(engine.outputs["p2"]) > 0 and len(engine.outputs["p1"]) > 0


def test_the_sync_path_keeps_the_same_invariant(monkeypatch):
    model, runner, engine = _run(monkeypatch, chain=False)
    assert model.chained_pen_steps == 0 and model.checked >= 10
    # one rebuild per join, one per row that changed slot, none otherwise
    assert device_penalty_stats()["rebuilds"] <= 6


def test_a_logits_step_in_between_makes_the_next_fused_step_rebuild(monkeypatch):
    model = PenaltyTarget()
    runner = _make_runner(model)
    _caps(runner, False, False)
    _pen_caps(runner, True)
    model.truth = _truth(runner)
    engine = Engine(runner)
    engine.add("p0", 1, _params(presence_penalty=0.5))
    for _ in range(3):
        engine.iterate()
    engine.finish()
    base = device_penalty_stats()["rebuilds"]
    assert base == 1
    # a request that needs full logits joins: its steps run in logits mode and
    # commit tokens the device never counts
    engine.add("l", 5, _params(min_tokens=14, max_tokens=16, stop_token_ids=[7]))
    for _ in range(2):
        engine.iterate()
    engine.finish()
    assert not runner._spec_penalty_tracker.is_current("p0")
    engine.stop_after["l"] = len(engine.outputs["l"]) + 1
    for _ in range(4):
        engine.iterate()
    engine.finish()
    assert device_penalty_stats()["rebuilds"] == base + 1
    assert model.checked >= 3


# endregion

# region Capability off: exactly the old behaviour


def test_with_the_capability_off_a_penalty_run_never_reaches_the_fused_path(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    model = PenaltyTarget()
    runner = _make_runner(model)
    _caps(runner, False, False)
    _pen_caps(runner, False)
    model.truth = _truth(runner)
    engine = Engine(runner)
    engine.add("p0", 1, _params(presence_penalty=0.5))
    engine.add("n", 17, SAMPLED)
    for _ in range(6):
        engine.iterate()
    engine.finish()
    assert model.pen_steps == 0 and not any(model.rebuilt)
    assert device_penalty_stats() == {"steps": 0, "rebuilds": 0}
    assert _controller(runner)._chained_submissions == 0


def test_the_placeholder_token_is_not_a_penalty_id():
    assert PLACEHOLDER_TOKEN_ID == -1  # the plugin's filter of -1 padding relies on it


# endregion
