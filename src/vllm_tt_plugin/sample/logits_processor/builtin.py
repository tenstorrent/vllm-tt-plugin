# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Built-in host logits processors: min_p, logit_bias and min_tokens.

Adapted from vLLM v0.31.0 ``vllm/v1/sample/logits_processor/builtin.py``, with
all state on the host.
"""

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch
from vllm import SamplingParams

from vllm_tt_plugin.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

T = TypeVar("T")


class MinPLogitsProcessor(LogitsProcessor):
    def __init__(self, max_num_reqs: int):
        self.min_p_count: int = 0
        self.min_p_cpu_tensor = torch.zeros((max_num_reqs,), dtype=torch.float32)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()
        # The rows of the current batch, shaped (batch size, 1).
        self.min_p: torch.Tensor = self.min_p_cpu_tensor[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling."""
        return True

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            min_p_before = self.min_p_cpu[index]
            if min_p_before != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
                if min_p and not min_p_before:
                    self.min_p_count += 1
                elif not min_p and min_p_before:
                    self.min_p_count -= 1

        if self.min_p_count:
            # Process removed requests.
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b).
            for adx, bdx, direct in batch_update.moved:
                min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
                if min_p_a != min_p_b:
                    needs_update = True
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if min_p_a:
                        self.min_p_cpu[adx] = 0
                    if min_p_b:
                        self.min_p_count -= 1

        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_cpu_tensor[:size].unsqueeze(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
        )

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.biases, batch_update, lambda params, _, __: params.logit_bias or None
        )

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = torch.tensor(biases, dtype=torch.float32)
            self.logits_slice = (
                torch.tensor(reqs, dtype=torch.int32),
                torch.tensor(tok_ids, dtype=torch.int32),
            )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):
    def __init__(self):
        # index -> (min_toks, output_token_ids, stop_token_ids, uses_structured_output)
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int], bool]] = {}

        # (req_idx_tensor, stop_token_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
        )

        # Structured-output stop-token logits that may need to be restored.
        self.restore_logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
        )

        self.neg_inf_tensor = torch.tensor(-float("inf"), dtype=torch.float32)

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int], bool] | None:
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return (
            min_tokens,
            output_tok_ids,
            params.all_stop_token_ids,
            params.structured_outputs is not None,
        )

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            restore_reqs: list[int] = []
            restore_tok_ids: list[int] = []
            for req, (
                _,
                _,
                stop_tok_ids,
                uses_structured_output,
            ) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)
                if uses_structured_output:
                    restore_reqs.extend([req] * len(stop_tok_ids))
                    restore_tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                torch.tensor(reqs, dtype=torch.int32),
                torch.tensor(tok_ids, dtype=torch.int32),
            )
            self.restore_logits_slice = (
                torch.tensor(restore_reqs, dtype=torch.int32),
                torch.tensor(restore_tok_ids, dtype=torch.int32),
            )

    def _mask_stop_token_logits(
        self,
        logits: torch.Tensor,
        logits_slice: tuple[torch.Tensor, torch.Tensor],
        restore_logits_slice: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        restore_rows, restore_toks = restore_logits_slice
        # Advanced indexing already returns a copy, so this survives the
        # in-place masking below.
        stop_logits = logits[restore_logits_slice]
        logits.index_put_(logits_slice, self.neg_inf_tensor)

        # Stop-token entries for the same row are emitted together, so this
        # checks each row once without sorting the index tensor.
        unique_rows, inverse_indices = torch.unique_consecutive(
            restore_rows, return_inverse=True
        )
        row_needs_restore = torch.isneginf(logits[unique_rows]).all(dim=-1)
        restore_mask = row_needs_restore[inverse_indices] & torch.isfinite(stop_logits)
        restore_slice = (
            restore_rows[restore_mask],
            restore_toks[restore_mask],
        )
        logits.index_put_(restore_slice, stop_logits[restore_mask])

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit stop tokens for requests which have not reached min length.
            if self.restore_logits_slice[0].numel() == 0:
                logits.index_put_(self.logits_slice, self.neg_inf_tensor)
            else:
                self._mask_stop_token_logits(
                    logits, self.logits_slice, self.restore_logits_slice
                )
        return logits


def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    """Utility function to update dict state for sparse LogitsProcessors."""
    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Process removed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Process moved requests, unidirectional (a->b) and
        # swapped (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
