# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A target model whose next token is independent of what was drafted.

``FakeSpecModel`` cannot settle whether speculation is lossless, because its
verify returns each draft unchanged wherever it agrees: what it "chooses" at a
candidate position is the draft that was offered there, so a speculated run and
an unspeculated run of it produce their tokens by different rules and any
equality between them is partly circular.

This model chooses by a rule that never looks at the drafts:

    next(token, position) = (token * 31 + position * 7 + 11) % vocab_size

Two calls compute the same rule.

``decode_forward`` without ``spec_mode`` is an ordinary decode: it returns
``[B, 1, V]`` logits whose argmax per row is ``next(token, position)`` for that
row's single input token, so the plugin's ordinary host-sampling tail commits
exactly that token. A run of those steps is the reference sequence.

``decode_forward`` with ``spec_mode`` is a verify over the ``[B, 1+K]``
candidate block. The contract's return column ``j`` is the model's choice at
candidate position ``j``, which is the token that follows the block's input
column ``j``, so the answer is ``next`` applied to that column and its
position, elementwise:

    argmax_ids[:, j] = next(tokens[:, j], positions[:, j])

Nothing in that reads ``num_valid_drafts``, the accept depth, or whether a
column holds a real committed token or a draft. A draft is therefore accepted
exactly when it happens to equal what this model would have chosen anyway,
which is what greedy acceptance means, and a wrong draft is rejected at its own
position with the model's own choice committed in its place.

Because the rule is first order, the whole reference sequence is also
computable in closed form from the last committed token, which lets a test
check both runs against a third, independent expectation.
"""

import torch

from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    DRAFTER_STATE_INTERNAL,
    SpecPlan,
    VerifyOutput,
    check_spec_side_tensors,
)

TARGET_VOCAB_SIZE = 512


def next_token(token: int, position: int) -> int:
    """The rule, in host Python, for a test to predict against."""
    return (int(token) * 31 + int(position) * 7 + 11) % TARGET_VOCAB_SIZE


def continuation(token: int, position: int, count: int) -> list[int]:
    """The ``count`` tokens this model emits after ``token`` at ``position``.

    Each emitted token sits one position later than the one it followed, which
    is what makes the reference sequence predictable without running anything.
    """
    emitted: list[int] = []
    for _ in range(count):
        token = next_token(token, position)
        position += 1
        emitted.append(token)
    return emitted


class DeterministicTarget:
    """Serves both an ordinary decode and a verify, by one rule."""

    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": [],
    }

    vocab_size = TARGET_VOCAB_SIZE

    def __init__(self) -> None:
        self.verify_calls: list[dict] = []
        self.plain_calls = 0

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        del vllm_config, max_num_seqs
        return SpecPlan(
            effective_k=requested_k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=False,
        )

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        """One decode step, or one verify when the runner asks for one."""
        if spec_mode is None:
            return self._plain(tokens, start_pos)
        if spec_mode != ACCEPT_MODE_ARGMAX_IDS:
            raise NotImplementedError(
                f"DeterministicTarget serves {ACCEPT_MODE_ARGMAX_IDS!r}, asked "
                f"for {spec_mode!r}"
            )
        rows, width = tokens.shape
        check_spec_side_tensors(
            kwargs["num_valid_drafts"],
            kwargs["accepted_counts"],
            rows,
            width - 1,
        )
        self.verify_calls.append(
            {
                "tokens": tokens.clone(),
                "positions": start_pos.clone(),
                "num_valid_drafts": kwargs["num_valid_drafts"].clone(),
                # Recorded because it is the one input this model does not
                # use: a real model selects its candidate state slot from it,
                # so a test has to assert what crossed the boundary rather
                # than infer it from the committed tokens.
                "accepted_counts": kwargs["accepted_counts"].clone(),
            }
        )
        return VerifyOutput(
            spec_mode=ACCEPT_MODE_ARGMAX_IDS,
            argmax_ids=self._choice(tokens, start_pos),
        )

    def _plain(self, tokens, start_pos):
        """``[B, 1, V]`` logits, argmax at this row's own next token.

        The ordinary decode path host-samples an argmax, so a one-hot row is
        the way to make it commit a chosen token. Positions arrive 1-D here,
        which is the plain call's own shape.
        """
        self.plain_calls += 1
        rows = tokens.shape[0]
        positions = start_pos.reshape(rows, -1)[:, :1]
        choice = self._choice(tokens.reshape(rows, -1)[:, :1], positions)
        logits = torch.zeros(rows, 1, self.vocab_size)
        logits.scatter_(2, choice.to(torch.int64).unsqueeze(2)[:, :1, :], 1.0)
        return logits

    def _choice(self, tokens, positions):
        """``next`` applied elementwise, which is the whole model."""
        ids = tokens.to(torch.int64) * 31 + positions.to(torch.int64) * 7 + 11
        return (ids % self.vocab_size).to(torch.int32)
