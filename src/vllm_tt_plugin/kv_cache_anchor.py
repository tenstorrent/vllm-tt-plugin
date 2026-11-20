# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Sliding-window KV blocks that keep a prompt's cache-hit anchor until the request ends.

A prefix hit on a sliding-window group needs the window that ends at the hit boundary. The next turn of a
conversation hits at the end of the previous prompt, but the stock manager frees that window as soon as a
long reply moves past it, so the blocks are the first ones evicted while the agent waits for a tool.
``AnchoredSlidingWindowManager`` keeps them (cached, never attended) until the request finishes. A model opts
in by returning ``AnchoredSlidingWindowSpec`` from ``get_kv_cache_spec``.
"""

from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec


class AnchoredSlidingWindowManager(SlidingWindowManager):
    def _anchor_blocks(self, num_prompt_tokens: int) -> tuple[int, int]:
        """Block range of the window ending at the last prefix-hit boundary of the prompt."""
        alignment = self.cache_hit_alignment_tokens
        end = (num_prompt_tokens - 1) // alignment * alignment // self.block_size
        return max(0, end - cdiv(self.sliding_window - 1, self.block_size)), end

    def _remove_blocks_in_range(self, request_id: str, first_block: int, last_block: int) -> None:
        """As the stock manager, but the freed blocks go to the front of the eviction queue (reused first)."""
        if request_id not in self.req_to_blocks or first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))
        freed = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        evictable = []
        for block in reversed(freed):  # the oldest positions are reused first
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                evictable.append(block)
        self.block_pool.free_block_queue.prepend_n(evictable)

    def remove_skipped_blocks(
        self, request_id: str, processed_computed_tokens: int, num_prompt_tokens: int | None = None
    ) -> None:
        if num_prompt_tokens is None or request_id not in self.req_to_blocks:
            return super().remove_skipped_blocks(request_id, processed_computed_tokens, num_prompt_tokens)
        skipped = min(
            self.get_num_skipped_tokens(processed_computed_tokens) // self.block_size,
            len(self.req_to_blocks[request_id]),
        )
        first, last = self._anchor_blocks(num_prompt_tokens)
        self._remove_blocks_in_range(request_id, last, skipped)
        self._remove_blocks_in_range(request_id, 0, min(first, skipped))


@register_kv_cache_spec(manager_class=AnchoredSlidingWindowManager, uniform_type_base_spec=SlidingWindowSpec)
@dataclass(frozen=True, kw_only=True)
class AnchoredSlidingWindowSpec(SlidingWindowSpec):
    def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
        # The retained anchor is held on top of the recycled window.
        anchor = cdiv(self.sliding_window - 1, self.block_size)
        return super().max_admission_blocks_per_request(max_in_flight_tokens, max_model_len) + anchor
