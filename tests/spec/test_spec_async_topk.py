# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A ``topk`` verify through ``finalize_decode``, synchronously and deferred.

``topk`` is the one accept mode ``submit_decode`` does not unwrap to a tensor:
the walk reads three fields and the lazy ``fetch_logits``, so the
``VerifyOutput`` itself travels to it. The other spec harnesses never pass it
through ``finalize_decode``: the synchronous one builds ``_SyncForward`` from
the submission directly, and the device test drives the model class without
the runner. The server does pass it through, in ``_forward_with_model_input``
and in ``complete_spec_decode_step``, and that is where ``finalize_decode``
once took it for an unread device output and handed it to the model's
``process_decode_output_host``, which iterates a plain decode's per-rank
outputs and cannot iterate a ``VerifyOutput``.

These tests take both routes with a model whose conversion hook refuses
everything: its plain decode already answers on the host, so nothing of its
may legitimately reach that hook.
"""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest
import torch
from vllm.sampling_params import SamplingParams

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_LOGITS,
    ACCEPT_MODE_TOPK,
    VerifyOutput,
)

from . import test_spec_async as async_harness
from . import test_spec_async_sampled as deferred_harness
from . import test_spec_sampled_runner as sync_harness
from .test_spec_device_topk import _TopkTarget

# The held-readback fixture, applied to every test here as it is there. The
# synchronous tests submit no read events, so it never waits for them.
wait_for_released_reads = async_harness.wait_for_released_reads

DRAFT_LEN = async_harness.DRAFT_LEN

# top_k below the fake's ``topk_candidates`` of 8, so the support serves it.
SEEDED = SamplingParams(temperature=1.0, top_k=3, seed=41, ignore_eos=True)


@pytest.fixture
def topk_on(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")
    monkeypatch.delenv("VLLM_TT_SPEC_FAST_TOPK", raising=False)
    monkeypatch.delenv("VLLM_TT_SPEC_TIE_EXACT", raising=False)


class _ConvertingTopkTarget(_TopkTarget):
    """``_TopkTarget`` with the base generator's host conversion hook.

    ``tt_transformers`` generators define ``process_decode_output_host`` to
    convert a plain decode's device output, one per data-parallel rank. This
    fake's plain decode already answers with host logits, so the hook has no
    legitimate caller here: whatever reaches it is a misrouted output, and it
    says so instead of converting.
    """

    def __init__(self) -> None:
        super().__init__()
        self.host_conversions: list[str] = []

    def process_decode_output_host(self, tt_out, is_tokens=False, **kwargs):
        self.host_conversions.append(type(tt_out).__name__)
        raise TypeError(
            f"process_decode_output_host was handed {type(tt_out).__name__}, "
            "which is not a plain decode's device output"
        )


class _DeferredTopkTarget(_ConvertingTopkTarget):
    """``_ConvertingTopkTarget`` whose readback is held until released.

    ``read_decode_output`` is handed what ``submit_decode`` unwrapped: a tensor
    for a ``logits`` verify or a plain decode, the ``VerifyOutput`` itself for
    ``topk``. Every tensor is replaced by a buffer filled only on release, NaN
    or -1 until then, so a walk that read early could not find a valid answer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[
            tuple[list[tuple[torch.Tensor, torch.Tensor]], threading.Event]
        ] = []
        self.reads = 0

    @staticmethod
    def _hold(answer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fill = float("nan") if answer.is_floating_point() else -1
        return torch.full_like(answer, fill), answer.clone()

    def read_decode_output(self, tt_out, async_read: bool = False):
        self.reads += 1
        event = threading.Event()
        if isinstance(tt_out, VerifyOutput):
            held = {
                name: self._hold(getattr(tt_out, name))
                for name in ("argmax_ids", "topk_values", "topk_ids")
            }
            self.pending.append((list(held.values()), event))
            buffers = {name: buffer for name, (buffer, _) in held.items()}
            return replace(tt_out, **buffers), [event]
        buffer, answer = self._hold(tt_out)
        self.pending.append(([(buffer, answer)], event))
        return buffer, [event]

    def release(self) -> None:
        copies, event = self.pending.pop(0)
        for buffer, answer in copies:
            buffer.copy_(answer)
        event.set()


def _sync_runner(model: _ConvertingTopkTarget | None = None):
    """The synchronous runner, with the controller the server submits through."""
    model = model or _ConvertingTopkTarget()
    runner = sync_harness._runner(model)
    runner._spec_plan = type(model).spec_plan(
        None, sync_harness.MAX_NUM_REQS, DRAFT_LEN
    )
    runner.async_decode = TTAsyncDecodeController(runner)
    return model, runner


def _server_step(runner, req_id: str) -> list[int]:
    """One verify the way the server runs it, ``finalize_decode`` included.

    ``_forward_with_model_input`` submits and finalizes, as
    ``execute_model`` does on a synchronous launch, and the front-packed tail
    walks, commits and publishes; the harness's own ``_step`` skips the
    finalize, which is the point of taking this route instead.
    """
    drafts = {req_id: sync_harness._argmax_drafts(runner, req_id)}
    model_input = sync_harness._build(runner, req_id, drafts=drafts)
    fwd = runner._forward_with_model_input(model_input)
    output = runner._finish_front_packed_sync(None, fwd=fwd)
    return list(output.sampled_token_ids[output.req_id_to_index[req_id]])


def _serve(runner, req_id: str, count: int) -> list[int]:
    while len(sync_harness._output(runner, req_id)) < count:
        _server_step(runner, req_id)
    return sync_harness._output(runner, req_id)[:count]


def _async_runner():
    model = _DeferredTopkTarget()
    runner = deferred_harness._runner(model)
    deferred_harness._rebuild_batch(runner)
    runner._spec_plan = type(model).spec_plan(
        None, async_harness.MAX_NUM_REQS, DRAFT_LEN
    )
    return model, runner


def _generator_state(runner, req_id: str) -> torch.Tensor:
    return deferred_harness._generator(runner, req_id).get_state().clone()


# region The synchronous route: _forward_with_model_input


def test_a_topk_verify_finalizes_as_its_verify_output(topk_on, monkeypatch):
    """The server's synchronous route, in both modes, from one seed.

    ``finalize_decode`` hands the ``VerifyOutput`` to the walk as it is, never
    to the model's conversion hook, and the seeded stream repeats.
    """
    streams = []
    for _ in range(2):
        model, runner = _sync_runner()
        sync_harness._add_request(runner, "s", SEEDED)
        streams.append(_serve(runner, "s", 30))
        assert set(model.verify_modes) == {ACCEPT_MODE_TOPK}
        assert model.host_conversions == []
    assert streams[0] == streams[1]

    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "0")
    model, runner = _sync_runner()
    sync_harness._add_request(runner, "s", SEEDED)
    assert len(_serve(runner, "s", 30)) == 30

    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}
    assert model.host_conversions == []


class _DeviceHandle:
    """Stands for a ttnn tensor a model forgot to read back."""


def test_a_topk_verify_with_unread_fields_is_refused_before_any_conversion(
    topk_on,
):
    """A field left on the device is named, not handed to the model's hook.

    ``process_decode_output_host`` converts a plain decode's output and knows
    nothing of a ``VerifyOutput``; the refusal happens in ``finalize_decode``,
    where the field and its type are still visible.
    """

    class _UnreadTopkTarget(_ConvertingTopkTarget):
        def answer(self, spec_mode: str, logits: torch.Tensor) -> VerifyOutput:
            out = super().answer(spec_mode, logits)
            if spec_mode == ACCEPT_MODE_TOPK:
                out = replace(out, topk_values=_DeviceHandle())
            return out

    model, runner = _sync_runner(_UnreadTopkTarget())
    sync_harness._add_request(runner, "s", SEEDED)
    with pytest.raises(TypeError, match=r"topk_values.*_DeviceHandle"):
        _server_step(runner, "s")
    assert model.host_conversions == []


# endregion

# region The asynchronous route: complete_spec_decode_step


def test_a_held_topk_verify_is_walked_only_once_its_readback_completes(topk_on):
    """The deferred route reads the ``VerifyOutput`` back through the model.

    ``read_decode_output`` is handed the ``VerifyOutput`` and returns one whose
    fields are host buffers; ``finalize_decode`` waits for them and hands the
    output to the walk, which moves the generator only then.
    """
    model, runner = _async_runner()
    deferred_harness._admit(runner, ("s", 5, SEEDED))
    untouched = _generator_state(runner, "s")

    wrapper = async_harness._submit_step(
        runner, "s", drafts={"s": deferred_harness._drafts(runner, "s")}
    )

    assert model.verify_modes == [ACCEPT_MODE_TOPK]
    assert model.reads == 1 and len(model.pending) == 1
    assert not wrapper.is_resolved()
    assert torch.equal(_generator_state(runner, "s"), untouched)

    model.release()
    output = wrapper.get_output()

    assert model.host_conversions == []
    assert not torch.equal(_generator_state(runner, "s"), untouched)
    committed = output.sampled_token_ids[output.req_id_to_index["s"]]
    assert 1 <= len(committed) <= DRAFT_LEN + 1
    assert runner.requests["s"].output_token_ids == []

    async_harness._drain(runner, "s")
    assert runner.requests["s"].output_token_ids == committed


def test_a_deferred_topk_verify_commits_what_the_synchronous_route_commits(
    topk_on,
):
    """Same seed, same drafts, same tokens, whichever route finalizes."""
    params = SamplingParams(temperature=1.1, top_k=5, seed=19, ignore_eos=True)

    model, runner = _async_runner()
    deferred_harness._admit(runner, ("s", 7, params))
    while len(runner.requests["s"].output_token_ids) < 30:
        wrapper = async_harness._submit_step(
            runner, "s", drafts={"s": deferred_harness._drafts(runner, "s")}
        )
        model.release()
        wrapper.get_output()
        async_harness._drain(runner, "s")
    deferred = list(runner.requests["s"].output_token_ids)[:30]
    assert set(model.verify_modes) == {ACCEPT_MODE_TOPK}
    assert model.host_conversions == []

    sync_model, sync_runner = _sync_runner()
    sync_harness._add_request(sync_runner, "s", params, first_token=7)
    immediate = _serve(sync_runner, "s", 30)
    assert set(sync_model.verify_modes) == {ACCEPT_MODE_TOPK}

    assert deferred == immediate


# endregion
