# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Batch-update collection and the set of loaded host logits processors.

Adapted from vLLM v0.31.0 ``vllm/v1/sample/logits_processor/state.py``.
"""

from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING

from vllm_tt_plugin.sample.logits_processor.interface import (
    AddedRequest,
    BatchUpdate,
    MovedRequest,
    RemovedRequest,
)

if TYPE_CHECKING:
    from vllm_tt_plugin.sample.logits_processor.custom import CustomLogitsProcessors
    from vllm_tt_plugin.sample.logits_processor.interface import LogitsProcessor


class BatchUpdateBuilder:
    """Collects one step's persistent-batch row changes for the processors."""

    def __init__(self) -> None:
        self._removed: list[RemovedRequest] = []
        self.added: list[AddedRequest] = []
        self.moved: list[MovedRequest] = []

    def removed_append(self, index: int) -> None:
        """Register the removal of a request from the persistent batch."""
        self._removed.append(index)

    def get_and_reset(self, batch_size: int) -> BatchUpdate | None:
        """Return this step's changes and start collecting the next step's.

        Args:
          batch_size: current persistent batch size

        Returns:
          Frozen batch update; ``None`` if nothing changed

        """
        if not any((self._removed, self.moved, self.added)):
            return None
        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=self._removed,
            moved=self.moved,
            added=self.added,
        )
        self._removed = []
        self.moved = []
        self.added = []
        return batch_update


class LogitsProcessors:
    """The host sampler's loaded logits processors.

    ``custom`` holds the operator's custom processors. The sampler applies it
    after the penalties, and the built-in processors at the points their
    argmax invariance allows.
    """

    def __init__(
        self,
        logitsprocs: Iterable["LogitsProcessor"] | None = None,
        custom: "CustomLogitsProcessors | None" = None,
    ) -> None:
        self.argmax_invariant: list[LogitsProcessor] = []
        self.non_argmax_invariant: list[LogitsProcessor] = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)
        self.custom = custom

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        """Iterator over all logits processors."""
        custom = () if self.custom is None else (self.custom,)
        return chain(self.argmax_invariant, self.non_argmax_invariant, custom)
