# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device logprobs and the device top-p certificate of the fused verify, on the host.

A ``fused_sample`` model that declares ``supports_spec_device_logprobs`` returns,
for a step with a logprobs row, ``SpecCandidates`` (the top ``M`` raw logits per
position and the raw ``logsumexp``). The plugin builds vLLM's ``LogprobsTensors``
from them. A model that declares ``supports_spec_device_top_p`` takes a sampled
row with ``top_k`` off and ``top_p`` below 1 as ``top_k`` 0, on a synchronous
step. These tests pin the plugin side:

- the predicates and the mode choice, with the capabilities on and off;
- the sampling inputs (``top_k`` 0, ``num_logprobs``, no global draw for an
  all-greedy logprobs step);
- the logprobs themselves against the host path on the same logits;
- the contract checks;
- the chain blockers and the ``defer_read`` rule, end to end on the chain fake.

The capabilities are two runner attributes (``_spec_device_logprobs``,
``_spec_device_top_p``), which ``load_model`` sets from ``model_capabilities``.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.model_runner import (
    TTModelRunner,
    fused_spec_sampling,
    spec_sampling_has_certificate_row,
)
from vllm_tt_plugin.spec_accept import (
    committed_logprobs,
    params_fused_servable,
    params_support_servable,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    ACCEPT_MODE_TOPK,
    MAX_DEVICE_LOGPROBS,
    FusedReadout,
    SpecCandidates,
    VerifyOutput,
)
from vllm_tt_plugin.spec_device_logprobs import (
    device_logprob_stats,
    device_spec_logprobs,
)

from .fake_spec_model import FakeFusedHandle
from .sampled_target import SAMPLED_VOCAB_SIZE, rule_logits
from .test_spec_chain_async import (
    GREEDY,
    SAMPLED,
    ChainSampledTarget,
    Engine,
    _controller,
    _decode_so,
    _make_runner,
)
from .test_spec_chain_async import K as CHAIN_K
from .test_spec_device_topk import _topk_runner, exact_support
from .test_spec_fused import _FusedTopkTarget
from .test_spec_sampled_runner import GREEDY as SR_GREEDY
from .test_spec_sampled_runner import (
    _add_request as _add_sampled_request,
)
from .test_spec_sampled_runner import (
    _argmax_drafts,
    _build,
    _step,
)

M = 32  # candidates per position the fake model returns


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


def _caps(runner, logprobs: bool, top_p: bool, mode: str = "raw_logprobs"):
    """What ``load_model`` does with the model's capabilities, on a runner double."""
    runner._spec_device_logprobs = logprobs
    runner._spec_device_top_p = top_p
    runner.model_config.logprobs_mode = mode
    return runner


def _params(temperature=0.0, top_k=0, top_p=1.0, logprobs=None, **kw):
    return SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        logprobs=logprobs,
        ignore_eos=True,
        **kw,
    )


# region The fake model: the chain target plus a candidates payload


def _candidates(logits: torch.Tensor, m: int = M) -> SpecCandidates:
    values, ids = exact_support(logits, m)
    return SpecCandidates(
        ids=ids,
        values=values.to(torch.float32),
        lse=torch.logsumexp(logits.to(torch.float32), dim=-1),
    )


class LogprobsTarget(ChainSampledTarget):
    """The sampled chain target, returning ``SpecCandidates`` when a row asks.

    Both capabilities are declared. A top_k 0 row (a certificate row) is
    accepted on the 7-candidate support the way the other sampled rows are:
    the certificate arithmetic is the model's, not under test here."""

    model_capabilities = {
        **ChainSampledTarget.model_capabilities,
        "supports_spec_device_logprobs": True,
        "supports_spec_device_top_p": True,
    }

    def __init__(self):
        super().__init__()
        self.candidate_steps = 0
        self.sampling_log: list[dict] = []
        self._pending = None

    @staticmethod
    def _wants(ss) -> bool:
        return (
            ss is not None
            and "num_logprobs" in ss
            and bool((ss["num_logprobs"] >= 0).any())
        )

    def _as_support_sampling(self, ss):
        if ss is None:
            return None
        top_k = torch.where(
            (~ss["greedy"]) & (ss["top_k"] <= 0),
            torch.full_like(ss["top_k"], self.topk_candidates - 1),
            ss["top_k"],
        )
        return {**ss, "top_k": top_k}

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        ss = kwargs.get("spec_sampling")
        if spec_mode == ACCEPT_MODE_FUSED_SAMPLE:
            self.sampling_log.append(
                {"ss": ss, "defer_read": bool(kwargs.get("defer_read"))}
            )
        kwargs["spec_sampling"] = self._as_support_sampling(ss)
        if kwargs["spec_sampling"] is None:
            del kwargs["spec_sampling"]
        out = super().decode_forward(tokens, start_pos, spec_mode, **kwargs)
        if (
            spec_mode == ACCEPT_MODE_FUSED_SAMPLE
            and not kwargs.get("defer_read")
            and self._wants(ss)
        ):
            self.candidate_steps += 1
            out = dataclasses.replace(
                out, candidates=_candidates(rule_logits(tokens, start_pos))
            )
        return out

    def _chain_accept(self, tokens, p0, num_valid, spec_sampling, kwargs):
        columns = torch.arange(tokens.shape[1], dtype=torch.int32).unsqueeze(0)
        positions = p0.unsqueeze(1) + columns
        positions = torch.where(
            (columns > num_valid.unsqueeze(1)) | (p0 < 0).unsqueeze(1),
            torch.full_like(positions, -1),
            positions,
        )
        self._pending = (
            _candidates(rule_logits(tokens, positions))
            if self._wants(spec_sampling)
            else None
        )
        return super()._chain_accept(
            tokens, p0, num_valid, self._as_support_sampling(spec_sampling), kwargs
        )

    def _chain_forward(self, *args, **kwargs):
        out = super()._chain_forward(*args, **kwargs)
        if self._pending is not None:
            self.candidate_steps += 1
            handle: FakeFusedHandle = out.handle
            old = handle._readout
            handle._readout = FusedReadout(
                old.accepted_token_ids,
                old.accepted_counts,
                old.p0,
                candidates=self._pending,
            )
        self._pending = None
        return out


class LogprobsEngine(Engine):
    """The chain engine that also keeps each published token's logprobs."""

    def __init__(self, runner, **kw):
        super().__init__(runner, **kw)
        self.logprobs: dict[str, list[tuple]] = {}

    def update_from_output(self, out, k):
        lp = out.logprobs
        if lp is not None:
            cu = lp.cu_num_generated_tokens
            for index, req_id in enumerate(out.req_ids):
                for i in range(cu[index], cu[index + 1]):
                    self.logprobs.setdefault(req_id, []).append(
                        (
                            lp.logprob_token_ids[i],
                            lp.logprobs[i],
                            int(lp.sampled_token_ranks[i]),
                        )
                    )
        super().update_from_output(out, k)


def _lp_runner(model=None, logprobs=True, top_p=True):
    runner = _make_runner(model or LogprobsTarget())
    return _caps(runner, logprobs, top_p)


# endregion

# region H7: the predicate matrix


V = SAMPLED_VOCAB_SIZE
CAND = 8


@pytest.mark.parametrize(
    "kwargs, logprobs_cap, top_p_cap, expected",
    [
        # greedy rows: always, logprobs only with the capability and N in 0..20
        (dict(), False, False, True),
        (dict(logprobs=0), True, False, True),
        (dict(logprobs=5), True, False, True),
        (dict(logprobs=MAX_DEVICE_LOGPROBS), True, False, True),
        (dict(logprobs=MAX_DEVICE_LOGPROBS + 1), True, False, False),
        (dict(logprobs=-1), True, False, False),
        (dict(logprobs=5), False, True, False),
        # sampled rows: top_k in 1..M-1 as the topk mode, any top_p
        (dict(temperature=1.0, top_k=5), False, False, True),
        (dict(temperature=1.0, top_k=CAND - 1, top_p=0.9), False, False, True),
        (dict(temperature=1.0, top_k=CAND), False, False, False),
        (dict(temperature=1.0, top_k=CAND), True, True, False),
        (dict(temperature=1.0, top_k=V - 1, top_p=0.9), True, True, False),
        # top_k off: only with the capability and top_p below 1
        (dict(temperature=1.0, top_p=0.9), False, False, False),
        (dict(temperature=1.0, top_p=0.9), True, False, False),
        (dict(temperature=1.0, top_p=0.9), False, True, True),
        (dict(temperature=1.0, top_p=1.0), False, True, False),
        (dict(temperature=1.0, top_k=V, top_p=0.9), False, True, True),
        (dict(temperature=1.0, top_k=-1, top_p=0.9), False, True, True),
        # logprobs and a certificate row together
        (dict(temperature=1.0, top_p=0.9, logprobs=3), True, True, True),
        (dict(temperature=1.0, top_p=0.9, logprobs=3), False, True, False),
        # the controls that mask tokens outside the candidates
        (dict(allowed_token_ids=[1, 2]), True, True, False),
        (
            dict(bad_words=["x"]),
            True,
            True,
            True,
        ),  # no tokenizer here: the words stay unset
        (dict(min_tokens=4, stop_token_ids=[7]), True, True, False),
    ],
)
def test_params_fused_servable_matrix(kwargs, logprobs_cap, top_p_cap, expected):
    params = _params(**kwargs)
    got = params_fused_servable(
        params,
        0,
        CAND,
        V,
        device_logprobs=logprobs_cap,
        device_top_p=top_p_cap,
    )
    assert got is expected


def test_a_structured_row_is_servable_only_inside_its_open_reasoning_phase():
    params = _params(structured_outputs=StructuredOutputsParams(json_object=True))
    assert not params_fused_servable(params, 0, CAND, V, device_logprobs=True)
    assert params_fused_servable(
        params, 0, CAND, V, device_logprobs=True, grammar_ok=True
    )
    with_logprobs = _params(
        logprobs=5, structured_outputs=StructuredOutputsParams(json_object=True)
    )
    assert params_fused_servable(
        with_logprobs, 0, CAND, V, device_logprobs=True, grammar_ok=True
    )


def test_with_both_capabilities_off_the_predicate_is_the_support_predicate():
    """H11 for the predicate: every row of a grid gets the old answer."""
    for temperature in (0.0, 0.7):
        for top_k in (-1, 0, 1, 5, CAND - 1, CAND, V - 1, V, V + 3):
            for top_p in (1.0, 0.9):
                for logprobs in (None, 0, 5, 21, -1):
                    for min_tokens in (0, 3):
                        params = _params(
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            logprobs=logprobs,
                            min_tokens=min_tokens,
                            stop_token_ids=[9] if min_tokens else None,
                        )
                        for output_len in (0, 5):
                            old = params_support_servable(params, output_len, CAND)
                            new = params_fused_servable(params, output_len, CAND, V)
                            assert new == old, (params, output_len)


# endregion

# region H7: the mode the runner asks for


def _modes(runner, rows):
    """The mode for rows that Engine.add built, with every row's drafts valid."""
    return runner._spec_mode_for_step(
        rows, torch.full((len(rows),), CHAIN_K, dtype=torch.int32)
    )


def _two_rows(runner, params):
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("x", 40, params)
    return ["a", "x"]


@pytest.mark.parametrize(
    "params, caps, expected",
    [
        # logprobs row
        (_params(logprobs=5), (True, False), ACCEPT_MODE_FUSED_SAMPLE),
        (_params(logprobs=5), (False, True), ACCEPT_MODE_LOGITS),
        (_params(logprobs=5), (False, False), ACCEPT_MODE_LOGITS),
        (_params(logprobs=21), (True, True), ACCEPT_MODE_LOGITS),
        (_params(logprobs=-1), (True, True), ACCEPT_MODE_LOGITS),
        (_params(logprobs=5, presence_penalty=0.5), (True, True), ACCEPT_MODE_LOGITS),
        (
            _params(logprobs=5, allowed_token_ids=[1, 2]),
            (True, True),
            ACCEPT_MODE_LOGITS,
        ),
        # sampled logprobs row inside the topk support
        (
            _params(temperature=1.0, top_k=5, logprobs=2),
            (True, False),
            ACCEPT_MODE_FUSED_SAMPLE,
        ),
        # certificate row
        (_params(temperature=1.0, top_p=0.9), (False, True), ACCEPT_MODE_FUSED_SAMPLE),
        (_params(temperature=1.0, top_p=0.9), (True, False), ACCEPT_MODE_LOGITS),
        (_params(temperature=1.0, top_p=1.0), (True, True), ACCEPT_MODE_LOGITS),
        (
            _params(temperature=1.0, top_k=20, top_p=0.9),
            (True, True),
            ACCEPT_MODE_LOGITS,
        ),
        (
            _params(temperature=1.0, top_p=0.9, presence_penalty=0.5),
            (False, True),
            ACCEPT_MODE_LOGITS,
        ),
        (
            _params(temperature=1.0, top_p=0.9, logprobs=3),
            (True, True),
            ACCEPT_MODE_FUSED_SAMPLE,
        ),
        (
            _params(temperature=1.0, top_p=0.9, logprobs=3),
            (False, True),
            ACCEPT_MODE_LOGITS,
        ),
    ],
)
def test_the_mode_follows_the_row_and_the_capabilities(params, caps, expected):
    runner = _lp_runner()
    _caps(runner, *caps)
    rows = _two_rows(runner, params)
    assert _modes(runner, rows) == expected


@pytest.mark.parametrize("mode", ["processed_logprobs", "processed_logits"])
def test_only_the_raw_logprobs_mode_is_served(mode):
    runner = _lp_runner()
    _caps(runner, True, True, mode)
    rows = _two_rows(runner, _params(logprobs=5))
    assert _modes(runner, rows) == ACCEPT_MODE_LOGITS
    # a row without logprobs is unaffected by the mode
    runner2 = _lp_runner()
    _caps(runner2, True, True, mode)
    assert (
        _modes(runner2, _two_rows(runner2, _params(temperature=1.0, top_k=5)))
        == ACCEPT_MODE_FUSED_SAMPLE
    )


def test_a_penalty_row_next_to_logprobs_rows_keeps_the_logits_step():
    runner = _lp_runner()
    engine = Engine(runner)
    engine.add("a", 1, _params(logprobs=5))
    engine.add("p", 40, _params(presence_penalty=0.5))
    assert _modes(runner, ["a", "p"]) == ACCEPT_MODE_LOGITS


# endregion

# region H7: the sampling inputs


class _CertificateTarget(_FusedTopkTarget):
    """The fused topk target, taking a top_k 0 row on its support."""

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        ss = kwargs.get("spec_sampling")
        if ss is not None:
            top_k = torch.where(
                (~ss["greedy"]) & (ss["top_k"] <= 0),
                torch.full_like(ss["top_k"], self.topk_candidates - 1),
                ss["top_k"],
            )
            kwargs["spec_sampling"] = {**ss, "top_k": top_k}
        out = super().decode_forward(tokens, start_pos, spec_mode, **kwargs)
        if spec_mode == ACCEPT_MODE_FUSED_SAMPLE and LogprobsTarget._wants(ss):
            out = dataclasses.replace(
                out, candidates=_candidates(rule_logits(tokens, start_pos))
            )
        return out


def _sampling_runner(logprobs=True, top_p=True):
    model, runner = _topk_runner(_CertificateTarget())
    _caps(runner, logprobs, top_p)
    return model, runner


def _first_step_build(runner, *req_ids):
    _step(runner, *req_ids)
    return _build(
        runner, *req_ids, drafts={r: _argmax_drafts(runner, r) for r in req_ids}
    )


def _sampling_inputs(runner, mi):
    return fused_spec_sampling(
        mi,
        runner._spec_plan,
        vocab_size=runner.vocab_size,
        device_top_p=runner._spec_device_top_p,
    )


def test_a_certificate_row_is_sent_with_top_k_0():
    model, runner = _sampling_runner()
    _add_sampled_request(runner, "g", SR_GREEDY, first_token=1)
    _add_sampled_request(
        runner, "c", _params(temperature=0.9, top_p=0.9), first_token=9
    )
    _add_sampled_request(runner, "s", _params(temperature=0.9, top_k=6), first_token=17)
    # the input batch holds the off top_k as the vocabulary size
    assert (
        int(runner.input_batch.sampling.top_k[runner.input_batch.req_id_to_index["c"]])
        == V
    )
    mi = _first_step_build(runner, "g", "c", "s")
    assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    ss = _sampling_inputs(runner, mi)
    rows = int(mi.input_tokens.shape[0])
    assert ss["top_k"].dtype == torch.int32 and ss["top_k"].shape == (rows,)
    assert ss["top_k"].tolist()[:3] == [1, 0, 6]
    assert ss["top_p"][1].item() == pytest.approx(0.9)
    assert ss["greedy"].tolist()[:3] == [True, False, False]
    assert "num_logprobs" not in ss
    assert spec_sampling_has_certificate_row(ss)


def test_without_the_capability_the_off_top_k_is_still_refused_as_before():
    model, runner = _sampling_runner(top_p=False)
    _add_sampled_request(
        runner, "c", _params(temperature=0.9, top_p=0.9), first_token=9
    )
    mi = _build(runner, "c")
    assert mi.spec_mode != ACCEPT_MODE_FUSED_SAMPLE
    with pytest.raises(RuntimeError, match="top_k is outside"):
        fused_spec_sampling(mi, runner._spec_plan, vocab_size=runner.vocab_size)


def test_a_top_p_1_row_never_reaches_the_certificate():
    model, runner = _sampling_runner()
    _add_sampled_request(runner, "c", _params(temperature=0.9), first_token=9)
    mi = _build(runner, "c")
    with pytest.raises(RuntimeError, match="top_p 1"):
        _sampling_inputs(runner, mi)


def test_the_num_logprobs_key_carries_each_rows_n_and_minus_2_elsewhere():
    model, runner = _sampling_runner()
    _add_sampled_request(runner, "a", _params(logprobs=5), first_token=1)
    _add_sampled_request(runner, "b", _params(temperature=0.9, top_k=6), first_token=9)
    _add_sampled_request(runner, "c", _params(logprobs=0), first_token=17)
    mi = _first_step_build(runner, "a", "b", "c")
    assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    ss = _sampling_inputs(runner, mi)
    rows = int(mi.input_tokens.shape[0])
    assert ss["num_logprobs"].dtype == torch.int32 and ss["num_logprobs"].shape == (
        rows,
    )
    assert ss["num_logprobs"].tolist() == [5, -2, 0] + [-2] * (rows - 3)


def test_an_all_greedy_logprobs_step_sends_the_dict_and_draws_nothing():
    model, runner = _sampling_runner()
    _add_sampled_request(runner, "a", _params(logprobs=5), first_token=1)
    _add_sampled_request(runner, "b", _params(logprobs=0), first_token=9)
    mi = _first_step_build(runner, "a", "b")
    assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    state = torch.get_rng_state()
    ss = _sampling_inputs(runner, mi)
    assert torch.equal(torch.get_rng_state(), state), "an all-greedy step must not draw"
    rows, width = (int(s) for s in mi.input_tokens.shape)
    assert ss is not None and ss["greedy"].all() and not ss["uniforms"].any()
    assert (
        ss["uniforms"].shape == (rows, width + 1)
        and ss["uniforms"].dtype == torch.float32
    )
    assert ss["num_logprobs"].tolist()[:2] == [5, 0]
    assert not spec_sampling_has_certificate_row(ss)


def test_an_all_greedy_step_without_logprobs_sends_nothing():
    model, runner = _sampling_runner()
    _add_sampled_request(runner, "a", SR_GREEDY, first_token=1)
    mi = _first_step_build(runner, "a")
    assert _sampling_inputs(runner, mi) is None


def test_a_sampled_step_draws_as_before_with_a_logprobs_row_in_it():
    model, runner = _sampling_runner()
    _add_sampled_request(runner, "a", _params(logprobs=2), first_token=1)
    _add_sampled_request(runner, "b", _params(temperature=0.9, top_k=6), first_token=9)
    mi = _first_step_build(runner, "a", "b")
    state = torch.get_rng_state()
    ss = _sampling_inputs(runner, mi)
    assert not torch.equal(torch.get_rng_state(), state)
    assert ss["uniforms"][1].any()


# endregion

# region H8: LogprobsTensors from the candidates, against the host path


def _random_logits(seed, rows, width, vocab=200, ties=False):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, width, vocab, generator=g) * 3
    if ties:
        logits = logits.to(torch.bfloat16).to(torch.float32)
        flat = logits.reshape(-1, vocab)
        for row in range(flat.shape[0]):
            top = torch.topk(flat[row], 12).indices
            flat[row, top[2:5]] = flat[row, top[2]].clone()
    return logits


def _committed(logits, counts, seed):
    """A committed token per column, drawn from the 6 largest logits of its column."""
    g = torch.Generator().manual_seed(seed)
    rows, width, _ = logits.shape
    top = torch.topk(logits, 6, dim=-1).indices
    pick = torch.randint(0, 6, (rows, width), generator=g)
    chosen = top.gather(2, pick.unsqueeze(2)).squeeze(2).to(torch.int32)
    columns = torch.arange(width).unsqueeze(0)
    return torch.where(
        columns < counts.unsqueeze(1), chosen, torch.full_like(chosen, -1)
    )


@pytest.mark.parametrize("n", [0, 5, 20])
@pytest.mark.parametrize("ties", [False, True])
def test_the_device_logprobs_equal_the_host_path_on_the_same_logits(n, ties):
    rows, width = 4, 4
    logits = _random_logits(7 + n, rows, width, ties=ties)
    counts = torch.tensor([1, 4, 2, 3], dtype=torch.int32)
    committed = _committed(logits, counts, 3)
    host = committed_logprobs(logits, committed, counts, n)
    before = device_logprob_stats()
    device = device_spec_logprobs(_candidates(logits), committed, counts, n)

    assert (
        device.cu_num_generated_tokens
        == host.cu_num_generated_tokens
        == [0, 1, 5, 7, 10]
    )
    assert device.logprob_token_ids.shape == host.logprob_token_ids.shape == (10, n + 1)
    assert device.logprob_token_ids.dtype == host.logprob_token_ids.dtype
    assert device.logprobs.dtype == host.logprobs.dtype == torch.float32
    assert device.selected_token_ranks.dtype == host.selected_token_ranks.dtype
    torch.testing.assert_close(device.logprobs, host.logprobs, atol=1e-5, rtol=0)
    assert torch.equal(device.selected_token_ranks, host.selected_token_ranks)
    assert torch.equal(device.logprob_token_ids[:, 0], host.logprob_token_ids[:, 0])
    if not ties:
        assert torch.equal(device.logprob_token_ids, host.logprob_token_ids)
    assert _stats_delta(before) == (1, 0, 0)


def _stats_delta(before):
    """(steps, rank lower bounds, missing ids) counted since ``before``."""
    now = device_logprob_stats()
    return tuple(
        now[k] - before[k] for k in ("steps", "rank_lower_bound", "missing_id")
    )


def test_a_rank_that_may_be_a_lower_bound_is_counted_and_a_missing_id_too():
    vocab = 40
    logits = torch.linspace(0, -10, vocab).reshape(1, 1, vocab).repeat(2, 2, 1)
    logits[
        :, :, 8:
    ] = -3.0  # a tie group that starts inside the 8 candidates and runs past them
    cand = _candidates(logits, m=8)
    counts = torch.tensor([2, 1], dtype=torch.int32)
    committed = torch.tensor([[7, 0], [30, -1]], dtype=torch.int32)
    # (0, 0) id 7 ties the last candidate; (0, 1) id 0 leads; (1, 0) id 30 is none
    before = device_logprob_stats()
    tensors = device_spec_logprobs(cand, committed, counts, 3)
    assert _stats_delta(before) == (1, 1, 1)
    lse = cand.lse
    assert tensors.selected_token_ranks.tolist() == [8, 1, 8 + 1]
    assert tensors.logprobs[2, 0].item() == pytest.approx(
        (cand.values[1, 0, -1] - lse[1, 0]).item()
    )
    # only the asking rows are counted
    before = device_logprob_stats()
    device_spec_logprobs(cand, committed, counts, 3, torch.tensor([True, False]))
    assert _stats_delta(before) == (1, 1, 0)


def test_too_few_candidates_or_the_wrong_block_is_an_error():
    logits = _random_logits(1, 2, 3, vocab=60)
    counts = torch.ones(2, dtype=torch.int32)
    committed = torch.zeros(2, 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="too few"):
        device_spec_logprobs(_candidates(logits, m=4), committed, counts, 5)
    with pytest.raises(ValueError, match="committed block"):
        device_spec_logprobs(
            _candidates(logits[:1]),
            committed,
            counts,
            1,
        )


def test_the_published_prefix_cut_works_on_the_device_tensors():
    rows, width = 3, 3
    logits = _random_logits(5, rows, width)
    counts = torch.tensor([3, 2, 3], dtype=torch.int32)
    committed = _committed(logits, counts, 1)
    tensors = device_spec_logprobs(_candidates(logits), committed, counts, 2)
    prefixes = [
        [int(t) for t in committed[0, :2]],
        [int(t) for t in committed[1, :2]],
        [int(t) for t in committed[2, :1]],
    ]
    cut = TTModelRunner.spec_committed_logprobs(tensors, prefixes)
    assert cut.cu_num_generated_tokens == [0, 2, 4, 5]
    assert cut.logprob_token_ids[:, 0].tolist() == [
        *prefixes[0],
        *prefixes[1],
        *prefixes[2],
    ]
    host = committed_logprobs(logits, committed, counts, 2)
    expected = TTModelRunner.spec_committed_logprobs(host, prefixes)
    torch.testing.assert_close(
        torch.as_tensor(cut.logprobs),
        torch.as_tensor(expected.logprobs),
        atol=1e-5,
        rtol=0,
    )


# endregion

# region H9: the contract


def _good(b=2, t=3, m=6):
    return dict(
        ids=torch.zeros(b, t, m, dtype=torch.int32),
        values=torch.zeros(b, t, m),
        lse=torch.zeros(b, t),
    )


def test_spec_candidates_accepts_the_contract_shapes_and_both_id_widths():
    SpecCandidates(**_good())
    SpecCandidates(**{**_good(), "ids": torch.zeros(2, 3, 6, dtype=torch.int64)})


@pytest.mark.parametrize(
    "change, error, text",
    [
        (dict(ids=torch.zeros(2, 3, 6)), ValueError, "ids must be int32 or int64"),
        (
            dict(values=torch.zeros(2, 3, 6, dtype=torch.bfloat16)),
            ValueError,
            "values must be float32",
        ),
        (
            dict(lse=torch.zeros(2, 3, dtype=torch.float64)),
            ValueError,
            "lse must be float32",
        ),
        (
            dict(ids=torch.zeros(2, 3, dtype=torch.int32)),
            ValueError,
            r"ids must be \[B, 1\+K, M\]",
        ),
        (dict(values=torch.zeros(2, 3, 5)), ValueError, "values must match ids"),
        (dict(lse=torch.zeros(2, 4)), ValueError, "lse must be"),
        (dict(lse=[0.0]), TypeError, "host tensor"),
    ],
)
def test_spec_candidates_rejects_the_wrong_shape_or_dtype(change, error, text):
    with pytest.raises(error, match=text):
        SpecCandidates(**{**_good(), **change})


def test_the_candidates_field_is_checked_on_the_verify_output_and_the_readout():
    ids = torch.zeros(2, 3, dtype=torch.int32)
    counts = torch.ones(2, dtype=torch.int32)
    cand = SpecCandidates(**_good())
    out = VerifyOutput(
        spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
        accepted_token_ids=ids,
        accepted_counts=counts,
        candidates=cand,
    )
    assert out.candidates is cand
    assert (
        VerifyOutput(spec_mode=ACCEPT_MODE_FUSED_SAMPLE, handle=object()).candidates
        is None
    )
    with pytest.raises(TypeError, match="must be a SpecCandidates"):
        VerifyOutput(
            spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
            accepted_token_ids=ids,
            accepted_counts=counts,
            candidates="x",
        )
    with pytest.raises(ValueError, match="only defined for spec_mode"):
        VerifyOutput(
            spec_mode=ACCEPT_MODE_LOGITS, logits=torch.zeros(2, 3, 5), candidates=cand
        )
    assert FusedReadout(ids, counts, counts, cand).candidates is cand
    assert FusedReadout(ids, counts, counts).candidates is None
    with pytest.raises(TypeError, match="must be a SpecCandidates"):
        FusedReadout(ids, counts, counts, candidates=object())


class _NoCandidates(_FusedTopkTarget):
    """A model that is asked for logprobs and does not return the candidates."""


class _WrongShape(_FusedTopkTarget):
    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        out = super().decode_forward(tokens, start_pos, spec_mode, **kwargs)
        if (
            spec_mode == ACCEPT_MODE_FUSED_SAMPLE
            and kwargs.get("spec_sampling") is not None
        ):
            logits = rule_logits(tokens, start_pos)
            return dataclasses.replace(out, candidates=_candidates(logits[:, :-1]))
        return out


@pytest.mark.parametrize(
    "target, error, text",
    [
        (_NoCandidates, RuntimeError, "returned no candidates"),
        (_WrongShape, ValueError, "must return candidates"),
    ],
)
def test_a_logprobs_step_without_usable_candidates_fails_loudly(target, error, text):
    model, runner = _topk_runner(target())
    _caps(runner, True, True)
    _add_sampled_request(runner, "a", _params(logprobs=3), first_token=1)
    with pytest.raises(error, match=text):
        _step(runner, "a")


# endregion

# region The runner reads the capabilities from the model


def _load(model_capabilities):
    runner = SimpleNamespace(
        load_config=None,
        vllm_config=None,
        model_config=None,
        supports_device_grammar=False,
        _spec_device_logprobs=False,
        _spec_device_top_p=False,
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


def test_the_capabilities_are_read_at_load_and_default_to_false():
    both = _load(
        {"supports_spec_device_logprobs": True, "supports_spec_device_top_p": True}
    )
    assert both._spec_device_logprobs is True and both._spec_device_top_p is True
    only_lp = _load({"supports_spec_device_logprobs": True})
    assert only_lp._spec_device_logprobs is True and only_lp._spec_device_top_p is False
    none = _load({})
    assert none._spec_device_logprobs is False and none._spec_device_top_p is False


# endregion

# region End to end on the chain fake: host-path values, blockers, defer_read


def _lp_run(monkeypatch, *, device, chain=True, steps=14):
    """Greedy rows asking N = 0, 5, 20 and one asking none, over a chained run."""
    if chain:
        monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
        # a logprobs step carries spec_sampling: it chains as a sampled step does
        monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    model = LogprobsTarget()
    runner = _lp_runner(model, logprobs=device, top_p=device)
    engine = LogprobsEngine(runner)
    engine.add("g0", 1, _params(logprobs=0))
    engine.add("g5", 9, _params(logprobs=5))
    engine.add("g20", 17, _params(logprobs=20))
    engine.add("gn", 25, GREEDY)
    for _ in range(steps):
        engine.iterate()
    engine.finish()
    return model, runner, engine


def test_logprobs_rows_chain_and_equal_the_host_path(monkeypatch):
    before = device_logprob_stats()
    dev_model, dev_runner, dev = _lp_run(monkeypatch, device=True)
    host_model, host_runner, host = _lp_run(monkeypatch, device=False)

    # the device run served the logprobs rows on fused steps and chained them
    assert dev_model.candidate_steps >= 5
    assert _controller(dev_runner)._chained_submissions > 0, dict(
        _controller(dev_runner)._chain_breaks
    )
    assert _controller(dev_runner)._chain_breaks["logprobs"] == 0
    assert device_logprob_stats()["missing_id"] == before["missing_id"]
    assert device_logprob_stats()["steps"] > before["steps"]
    # the host run could not: every logprobs row is a logits step
    assert _controller(host_runner)._chained_submissions == 0
    assert host_model.candidate_steps == 0

    assert dev.outputs == host.outputs
    for req_id in ("g0", "g5", "g20"):
        got, want = dev.logprobs[req_id], host.logprobs[req_id]
        assert len(got) == len(want) == len(dev.outputs[req_id])
        for (g_ids, g_lp, g_rank), (w_ids, w_lp, w_rank) in zip(got, want):
            assert (
                len(g_ids) == len(w_ids) == 21
            )  # the step's largest N, as on the host path
            assert g_ids[0] == w_ids[0] and g_rank == w_rank
            torch.testing.assert_close(
                torch.as_tensor(g_lp), torch.as_tensor(w_lp), atol=1e-5, rtol=0
            )
            finite = [i for i, v in enumerate(w_lp) if v > -1e9]
            assert [g_ids[i] for i in finite] == [w_ids[i] for i in finite]


def test_the_sync_path_equals_the_host_path_too(monkeypatch):
    dev_model, dev_runner, dev = _lp_run(monkeypatch, device=True, chain=False)
    host_model, _, host = _lp_run(monkeypatch, device=False, chain=False)
    assert dev_model.candidate_steps >= 5
    assert dev.outputs == host.outputs
    for req_id in ("g0", "g5", "g20"):
        for (g_ids, g_lp, g_rank), (w_ids, w_lp, w_rank) in zip(
            dev.logprobs[req_id], host.logprobs[req_id]
        ):
            assert g_ids[0] == w_ids[0] and g_rank == w_rank
            torch.testing.assert_close(
                torch.as_tensor(g_lp), torch.as_tensor(w_lp), atol=1e-5, rtol=0
            )


def test_the_blocker_reasons_for_logprobs_and_certificate_rows(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")

    def blocker(params, caps, rows=("a", "x")):
        runner = _lp_runner()
        _caps(runner, *caps)
        engine = Engine(runner)
        engine.add("a", 1)
        engine.add("x", 40, params)
        so = _decode_so(runner, list(rows))
        return runner._chain_rows_blocker(list(rows), so, 1)[0]

    # a servable logprobs row stays chainable
    assert blocker(_params(logprobs=5), (True, True)) is None
    assert blocker(_params(temperature=1.0, top_k=5, logprobs=2), (True, True)) is None
    # one the model cannot answer is a "logprobs" blocker, as before
    assert blocker(_params(logprobs=5), (False, False)) == "logprobs"
    assert blocker(_params(logprobs=21), (True, True)) == "logprobs"
    assert blocker(_params(logprobs=-1), (True, True)) == "logprobs"
    # a certificate row must be synchronous
    assert blocker(_params(temperature=1.0, top_p=0.9), (True, True)) == "device_top_p"
    assert (
        blocker(_params(temperature=1.0, top_p=0.9, logprobs=3), (True, True))
        == "device_top_p"
    )
    # without the capability it is not a fused row at all
    assert blocker(_params(temperature=1.0, top_p=0.9), (True, False)) == "not_fused"
    # the other blockers keep their order and names
    assert blocker(_params(presence_penalty=0.5), (True, True)) == "penalties"


def test_a_certificate_step_is_sync_and_logprobs_steps_stay_deferred(monkeypatch):
    before = device_logprob_stats()
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    model = LogprobsTarget()
    runner = _lp_runner(model)
    engine = LogprobsEngine(runner)
    engine.add("g", 1, _params(logprobs=5))
    engine.add("s", 9, SAMPLED)
    for i in range(10):
        if i == 5:
            engine.add("c", 17, _params(temperature=0.8, top_p=0.9))
        engine.iterate()
    engine.finish()

    is_cert = [spec_sampling_has_certificate_row(e["ss"]) for e in model.sampling_log]
    cert_steps = [e for e, cert in zip(model.sampling_log, is_cert) if cert]
    other = [e for e, cert in zip(model.sampling_log, is_cert) if not cert]
    assert cert_steps, "the certificate row never reached the model"
    assert all(not e["defer_read"] for e in cert_steps)
    assert any(e["defer_read"] for e in other), (
        "steps before the certificate row still defer"
    )
    # the certificate row went to the model as top_k 0, never as the vocabulary size
    assert all(
        int(e["ss"]["top_k"].max()) < SAMPLED_VOCAB_SIZE
        for e in model.sampling_log
        if e["ss"] is not None
    )
    # the logprobs row kept its values through the sync steps
    assert len(engine.logprobs["g"]) == len(engine.outputs["g"])
    assert device_logprob_stats()["missing_id"] == before["missing_id"]


def test_a_grammar_row_in_its_thinking_phase_and_a_logprobs_row_share_a_fused_step():
    from .test_spec_grammar_think import (
        OPEN,
        _add_structured,
        _mode,
    )
    from .test_spec_grammar_think import (
        _runner as _grammar_runner,
    )
    from .test_spec_greedy_e2e import DRAFT_LEN, _add_request

    def runner_with_topk():
        runner = _grammar_runner()
        plan = type(runner.model).spec_plan(None, 1, DRAFT_LEN)
        modes = (*plan.accept_modes, ACCEPT_MODE_TOPK)
        runner._spec_plan = dataclasses.replace(
            plan, accept_modes=modes, topk_candidates=8
        )
        runner._spec_accept_modes = modes
        runner.vocab_size = 512
        return runner

    runner = runner_with_topk()
    _caps(runner, True, False)
    _add_structured(runner, "s", OPEN)
    _add_request(runner, "n", first_token=40)
    runner.requests["n"].sampling_params = _params(logprobs=5)

    assert _mode(runner, ["s", "n"], {"s": [3, 4, 5]}) == ACCEPT_MODE_FUSED_SAMPLE
    # with the capability off the logprobs row alone asks for logits
    _caps(runner, False, False)
    assert _mode(runner, ["s", "n"], {"s": [3, 4, 5]}) == ACCEPT_MODE_LOGITS
    # and the logprobs row of the grammar row itself is served while its phase is open
    _caps(runner, True, False)
    runner.requests["s"].sampling_params.logprobs = 3
    assert _mode(runner, ["s", "n"], {"s": [3, 4, 5]}) == ACCEPT_MODE_FUSED_SAMPLE
    # an end id among the drafts still makes the whole step a logits step
    assert _mode(runner, ["s", "n"], {"s": [501, 4, 5]}) == ACCEPT_MODE_LOGITS


# endregion

# region H11: the capabilities off is the old behaviour


def test_with_the_capabilities_off_nothing_new_is_sent_and_the_old_modes_stay(
    monkeypatch,
):
    before = device_logprob_stats()
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    model = LogprobsTarget()
    runner = _lp_runner(model, logprobs=False, top_p=False)
    engine = LogprobsEngine(runner)
    engine.add("g", 1)
    engine.add("s", 9, SAMPLED)
    engine.add("l", 17, _params(logprobs=3))  # a logits step from here on
    engine.add("c", 25, _params(temperature=0.8, top_p=0.9))
    for _ in range(6):
        engine.iterate()
    engine.finish()
    assert model.candidate_steps == 0
    assert all(
        e["ss"] is None or "num_logprobs" not in e["ss"] for e in model.sampling_log
    )
    # the logits mode carried the logprobs, as it always has
    assert len(engine.logprobs["l"]) == len(engine.outputs["l"])
    assert device_logprob_stats() == before
    assert _controller(runner)._chained_submissions == 0


# endregion
