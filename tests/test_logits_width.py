# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the logits-shape guard on the host-sampling paths.

A TT model returns host logits as ``[rows, 1, vocab]``, and host sampling
reads token ids straight out of the last dimension. A model adapter that folds
one row of a wide vocabulary into several narrow rows returns a tensor the
sampler still indexes without complaint, so the step argmaxes over a slice of
the vocabulary and answers with tokens that were never compared against the
rest of it. ``TTModelRunner._check_host_logits`` refuses such a tensor on both
host-sampling paths: ``_get_output_tokens`` (decode and prefill) and
``TTLaneInputBatch.extract_output`` (lane decode and lane prefill).

No device execution: the model output is a plain tensor and the host sampler
is a fake that records the logits it is handed.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.input_batch import TTLaneInputBatch
from vllm_tt_plugin.model_runner import TTModelRunner

VOCAB = 256
# The device batch a decode reads back, and the width one row of VOCAB
# collapses to when it is viewed as that many rows.
SLOT_ROWS = 32
FOLDED_WIDTH = VOCAB // SLOT_ROWS
BLOCK = 16
MAX_MODEL_LEN = 256


class FakeTTGenerator:
    """Stands in for the TT model whose adapter shapes the returned logits."""


def _capturing_host_sampler(captured: dict):
    """Fake host sampler: records the logits it sees, returns per-row argmax."""

    def sampler(logits, sampling_metadata):
        captured["logits"] = logits
        return SimpleNamespace(
            sampled_token_ids=logits.argmax(dim=-1), logprobs_tensors=None
        )

    return sampler


def _runner(captured: dict, *, vocab: int = VOCAB) -> SimpleNamespace:
    runner = SimpleNamespace(
        _num_speculative_tokens=0,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        tt_per_lane_max_num_seqs=SLOT_ROWS,
        vocab_size=vocab,
        model=FakeTTGenerator(),
        host_sampler=_capturing_host_sampler(captured),
    )
    # The guard under test, bound to the fake runner.
    runner._check_host_logits = MethodType(TTModelRunner._check_host_logits, runner)
    return runner


def _sampling_params(rows: int) -> SimpleNamespace:
    """Greedy, penalty-free params wide enough for ``rows`` live rows."""
    return SimpleNamespace(
        temperature=torch.zeros(rows),
        top_k=torch.zeros(rows, dtype=torch.int32),
        top_p=torch.ones(rows),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.zeros(rows, dtype=torch.int64),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )


def _model_input() -> SimpleNamespace:
    return SimpleNamespace(
        intermediate_prefill_mask=None,
        grammar_bitmask=[None],
        output_tokens=None,
        prompt_tokens=None,
        max_num_logprobs=[None],
        allowed_token_ids_mask_list=[None],
        bad_words_token_ids_list=[None],
        logitsprocs_list=[None],
        generators_list=[{}],
    )


def _extract(
    tt_out: torch.Tensor,
    *,
    live_rows: int = 1,
    device_sampling: bool = False,
    is_decode: bool = True,
    captured: dict | None = None,
):
    return TTModelRunner._get_output_tokens(
        _runner(captured if captured is not None else {}),
        tt_out,
        None,
        _sampling_params(live_rows),
        _model_input(),
        [live_rows],
        perform_device_sampling=device_sampling,
        is_decode=is_decode,
    )


def _lane_batch(per_lane: int = SLOT_ROWS) -> TTLaneInputBatch:
    return TTLaneInputBatch(
        num_lanes=1,
        per_lane=per_lane,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * per_lane,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
    )


# region Malformed vocabulary width


def test_decode_host_sampling_rejects_narrow_vocabulary():
    """One live row, one returned row, a vocabulary slice instead of the
    vocabulary."""
    tt_out = torch.zeros((1, 1, FOLDED_WIDTH), dtype=torch.float32)

    with pytest.raises(ValueError) as excinfo:
        _extract(tt_out)

    message = str(excinfo.value)
    assert "FakeTTGenerator returned decode logits" in message
    assert f"last dimension is {FOLDED_WIDTH}" in message
    assert f"vocabulary is {VOCAB} wide" in message
    assert "adapter" in message


def test_decode_host_sampling_rejects_row_folded_vocabulary():
    """The observed device failure: one row of logits read back as the full
    device batch, so the width is the vocabulary divided by that batch."""
    tt_out = torch.zeros((SLOT_ROWS, 1, FOLDED_WIDTH), dtype=torch.float32)

    with pytest.raises(ValueError) as excinfo:
        _extract(tt_out)

    message = str(excinfo.value)
    assert f"shape ({SLOT_ROWS}, 1, {FOLDED_WIDTH})" in message
    assert "for 1 live row(s)" in message
    assert f"last dimension is {FOLDED_WIDTH}" in message
    assert f"vocabulary is {VOCAB} wide" in message


def test_prefill_host_sampling_rejects_narrow_vocabulary():
    tt_out = torch.zeros((1, 1, FOLDED_WIDTH), dtype=torch.float32)

    with pytest.raises(ValueError, match="returned prefill logits"):
        _extract(tt_out, is_decode=False)


def test_host_sampling_rejects_fewer_rows_than_the_step_reads():
    """The width is right but the tensor has no row for the second request."""
    tt_out = torch.zeros((1, 1, VOCAB), dtype=torch.float32)

    with pytest.raises(ValueError) as excinfo:
        _extract(tt_out, live_rows=2)

    message = str(excinfo.value)
    assert "it holds 1 rows, fewer than the 2 rows the step reads" in message


def test_lane_host_sampling_rejects_narrow_vocabulary():
    batch = _lane_batch()
    tt_out = torch.zeros((SLOT_ROWS, FOLDED_WIDTH), dtype=torch.float32)

    with pytest.raises(ValueError, match="returned lane decode logits"):
        batch.extract_output(
            _runner({}),
            tt_out,
            None,
            SimpleNamespace(perform_device_sampling=False, grammar_bitmask=[None]),
            scheduled_rows=[0],
            is_decode=True,
        )


# endregion Malformed vocabulary width

# region Well-formed output


def test_decode_host_sampling_accepts_full_vocabulary():
    captured: dict = {}
    tt_out = torch.full((1, 1, VOCAB), -10.0)
    tt_out[0, 0, 7] = 5.0

    sampled, logprobs = _extract(tt_out, captured=captured)

    assert captured["logits"].shape == (1, VOCAB)
    assert sampled[0].tolist() == [[7]]
    assert logprobs == [None]


def test_device_sampling_reads_token_ids_without_the_logits_guard():
    """Device sampling returns token ids, not logits, so a last dimension that
    is not the vocabulary is the normal case and must still sample."""
    tt_out = torch.arange(SLOT_ROWS, dtype=torch.int32)

    sampled, logprobs = _extract(tt_out, device_sampling=True)

    assert sampled[0].tolist() == [[0]]
    assert logprobs == [None]


# endregion Well-formed output
