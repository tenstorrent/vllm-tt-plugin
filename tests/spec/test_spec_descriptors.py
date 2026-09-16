# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The speculative methods are reachable through ordinary attribute lookup.

Every other test in this directory drives the runner through a
``SimpleNamespace`` carrying hand-picked attributes, which is what makes those
tests cheap. It also makes them blind to one whole class of error: a
``SimpleNamespace`` binds nothing, so copying a class function onto one hides a
missing ``@staticmethod``, and applying ``__get__`` by hand hides a spurious
one. Either mistake leaves every fake-runner test green and every real launch
raising ``TypeError`` on its first speculative step.

So these tests reach the methods the way production does: off a real
``TTModelRunner`` instance, and off a real ``TTWorker``. They build the
instances with ``__new__`` and set only the attributes the method under test
reads, because a fully constructed runner needs a mesh device.
"""

import pytest
import torch

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID
from vllm_tt_plugin.worker import TTWorker


def _bare_runner() -> TTModelRunner:
    """A runner with no device, for methods that touch none."""
    return TTModelRunner.__new__(TTModelRunner)


def test_take_draft_token_ids_is_callable_on_an_instance():
    """vLLM reaches this through the worker on every speculative step."""
    runner = _bare_runner()
    runner._proposed_draft_token_ids = {"a": [11, 12], "b": [21]}
    runner.requests = {"a": object(), "b": object()}

    drafts = runner.take_draft_token_ids()

    assert drafts is not None
    assert sorted(drafts.req_ids) == ["a", "b"]
    assert dict(zip(drafts.req_ids, drafts.draft_token_ids))["a"] == [11, 12]
    # Handed over once: the scheduler has stored them by now.
    assert runner.take_draft_token_ids() is None


def test_take_draft_token_ids_reports_nothing_when_nothing_was_proposed():
    runner = _bare_runner()
    runner._proposed_draft_token_ids = {}
    runner.requests = {}

    assert runner.take_draft_token_ids() is None


def test_take_draft_token_ids_drops_a_request_the_runner_no_longer_holds():
    """A request can finish in the step that proposed for it.

    The scheduler tolerates a draft for a request it has finished, but the
    runner has no business reporting one for a request it has forgotten, and a
    consumer that trusted the report would look the request up and fail.
    """
    runner = _bare_runner()
    runner._proposed_draft_token_ids = {"live": [11], "finished": [21]}
    runner.requests = {"live": object()}

    drafts = runner.take_draft_token_ids()

    assert drafts is not None
    assert drafts.req_ids == ["live"]
    assert drafts.draft_token_ids == [[11]]


def test_take_draft_token_ids_reports_nothing_when_every_request_finished():
    runner = _bare_runner()
    runner._proposed_draft_token_ids = {"finished": [21]}
    runner.requests = {}

    assert runner.take_draft_token_ids() is None


def test_the_worker_delegates_take_draft_token_ids_to_its_runner():
    """``EngineCore`` calls the worker, not the runner, so the hop must work."""
    worker = TTWorker.__new__(TTWorker)
    runner = _bare_runner()
    runner._proposed_draft_token_ids = {"r": [7]}
    runner.requests = {"r": object()}
    worker.model_runner = runner

    drafts = worker.take_draft_token_ids()

    assert drafts is not None
    assert drafts.req_ids == ["r"]
    assert drafts.draft_token_ids == [[7]]
    assert worker.take_draft_token_ids() is None


def test_the_candidate_block_builder_is_callable_on_an_instance():
    """``_prepare_model_inputs`` calls this as ``self._spec_candidate_block``.

    It takes no runner state, so it is a static method, and reaching it through
    an instance must not bind one: an extra argument would land in ``drafts``.
    """
    runner = _bare_runner()

    tokens, positions = runner._spec_candidate_block(
        torch.tensor([[11, 12]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[5]], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
    )

    assert tokens.tolist() == [[5, 11, PLACEHOLDER_TOKEN_ID]]
    assert positions.tolist() == [[7, 8, -1]]


def test_the_row_state_gatherer_is_callable_on_an_instance():
    runner = _bare_runner()

    drafts, num_valid, counts = runner._spec_row_state(
        {"r": 3}, {"r": [11, 12]}, ["r"], 3
    )

    assert drafts.tolist() == [[11, 12, PLACEHOLDER_TOKEN_ID]]
    assert num_valid.tolist() == [2]
    assert counts.tolist() == [3]


def test_a_speculative_step_missing_its_side_tensors_raises_by_name():
    """The builder sets all three together, so one missing is a runner bug.

    Raised rather than asserted: ``python -O`` strips assertions, and the
    indexing below would then read whichever tensor was absent.
    """
    from types import SimpleNamespace

    runner = _bare_runner()
    fwd = SimpleNamespace(
        tt_out=torch.zeros(1, 4, dtype=torch.int32),
        model_input=SimpleNamespace(
            row_req_ids=["r"],
            draft_token_ids=None,
            num_valid_drafts=torch.zeros(1, dtype=torch.int32),
            spec_mode="argmax_ids",
        ),
    )

    with pytest.raises(RuntimeError, match="draft_token_ids"):
        runner._finish_spec_decode(fwd)


def test_the_committed_position_builder_is_callable_on_an_instance():
    """``_propose_model_drafts`` calls this as ``self._committed_positions``.

    It takes no runner state, so it is a static method, and reaching it
    through an instance must not bind one: an extra argument would land in
    ``width`` and the drafter would be told the wrong positions.
    """
    runner = _bare_runner()

    positions = runner._committed_positions(
        torch.tensor([[4, 5, 6]], dtype=torch.int32), 3
    )

    assert positions.tolist() == [[5, 6, 7]]


def test_the_model_drafter_call_is_callable_on_an_instance():
    """``_finish_spec_decode`` calls this as ``self._propose_model_drafts``.

    It reads runner state, so it is an instance method. Reached off a real
    instance here, because a ``SimpleNamespace`` carrying a copied function
    would hide a spurious ``@staticmethod`` on it.
    """
    from types import SimpleNamespace

    import vllm_tt_plugin.spec_decode as spec_decode

    runner = _bare_runner()
    runner._num_speculative_tokens = 2
    runner.input_batch = SimpleNamespace(
        vocab_size=64, num_tokens=torch.tensor([8, 8], dtype=torch.int32)
    )
    runner.model_config = SimpleNamespace(max_model_len=128)
    runner.model = SimpleNamespace(
        propose_draft_tokens=lambda *a, **k: spec_decode.DraftOutput(
            draft_token_ids=torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)
        )
    )
    runner._proposed_draft_token_ids = {}

    runner._propose_model_drafts(
        torch.tensor([[5, 6, 7], [5, 6, 7]], dtype=torch.int32),
        torch.tensor([2, 2], dtype=torch.int32),
        SimpleNamespace(
            input_positions=torch.tensor([[3, 4, 5], [3, 4, 5]], dtype=torch.int32)
        ),
        None,
        ["a", "b"],
    )

    assert runner._proposed_draft_token_ids == {"a": [11, 12], "b": [21, 22]}
