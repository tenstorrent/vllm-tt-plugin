# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Allocate only the live sliding window for a whole-prompt prefill.

Opt-in decoders compute prompt attention from temporary K/V, so only the
final window must persist. Full-attention allocation and decode eviction
remain standard vLLM behavior. External prefill chunks are unsupported.
"""

from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import (
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


def validate_whole_prompt_cache(config):
    scheduler = config.scheduler_config
    parallel = config.parallel_config
    unsupported = {
        "chunked prefill": scheduler.enable_chunked_prefill,
        "long-prefill threshold": scheduler.long_prefill_token_threshold,
        "prefix caching": config.cache_config.enable_prefix_caching,
        "disabled hybrid cache manager": scheduler.disable_hybrid_kv_cache_manager,
        "KV transfer": config.kv_transfer_config,
        "speculative decoding": config.speculative_config,
        "decode context parallelism": parallel.decode_context_parallel_size != 1,
        "prefill context parallelism": parallel.prefill_context_parallel_size != 1,
    }
    failures = [name for name, enabled in unsupported.items() if enabled]
    if failures:
        raise ValueError(
            "Whole-prompt sliding cache cannot use: " + ", ".join(failures)
        )


@dataclass(frozen=True, kw_only=True)
class WholePromptSlidingWindowSpec(SlidingWindowSpec):
    def max_admission_blocks_per_request(self, max_in_flight_tokens, max_model_len):
        # Prefill consumes temporary K/V; only its tail enters the persistent pool.
        tokens = min(self.sliding_window, max_model_len)
        return cdiv(tokens + self.block_size - 1, self.block_size)

    def max_memory_usage_bytes(self, vllm_config):
        validate_whole_prompt_cache(vllm_config)
        return (
            self.max_admission_blocks_per_request(
                vllm_config.max_in_flight_tokens, vllm_config.model_config.max_model_len
            )
            * self.page_size_bytes
        )


class WholePromptSlidingWindowManager(SlidingWindowManager):
    def __init__(self, kv_cache_spec, **kwargs):
        super().__init__(kv_cache_spec, **kwargs)
        if self.enable_caching or self.dcp_world_size != 1 or self.pcp_world_size != 1:
            raise ValueError(
                "Whole-prompt sliding cache requires uncached, whole-prompt TP prefill"
            )

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        if not self.req_to_blocks.get(request_id):
            if (
                new_computed_blocks
                or total_computed_tokens
                or num_local_computed_tokens
                or num_tokens != num_tokens_main_model
            ):
                raise ValueError(
                    "Fresh whole-prompt prefill must start at zero "
                    "without external cache/lookahead"
                )
            skipped = self.get_num_skipped_tokens(num_tokens) // self.block_size
            return cdiv(num_tokens, self.block_size) - skipped
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        blocks = self.req_to_blocks[request_id]
        if not blocks:
            if num_tokens != num_tokens_main_model:
                raise ValueError(
                    "Whole-prompt sliding cache does not support lookahead"
                )
            skipped = self.get_num_skipped_tokens(num_tokens) // self.block_size
            blocks.extend([self._null_block] * skipped)
        return super().allocate_new_blocks(
            request_id, num_tokens, num_tokens_main_model
        )


def register_whole_prompt_cache():
    KVCacheSpecRegistry.register(
        WholePromptSlidingWindowSpec,
        WholePromptSlidingWindowManager,
        uniform_type_base_spec=WholePromptSlidingWindowSpec,
    )
