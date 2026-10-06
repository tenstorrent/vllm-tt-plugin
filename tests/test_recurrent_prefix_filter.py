# SPDX-License-Identifier: Apache-2.0
"""The filter runs against a real KVCacheManager, because its contract is upstream's.

``RecurrentPrefixCache`` is tested on its own; what is left to prove is the wrapping: that the
tuple upstream returns is understood, that a trimmed hit is still something ``allocate_slots``
accepts, and that the two block sizes in play are not confused. A model with one KV cache group
hides that last distinction entirely — hashes are indexed at the gcd of the group block sizes and
a hit is reported at their lcm — so one group is not enough to test against.
"""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_tt_plugin.scheduler import install_recurrent_prefix_filter

PROMPT_TOKENS = 128
# The chain root is random per process unless PYTHONHASHSEED is set, so it is seeded once here:
# re-seeding between requests would give two identical prompts different hashes and no hit.
_HASH_FN = get_hash_fn_by_name("sha256")
init_none_hash(_HASH_FN)


def _manager(group_block_sizes, hash_block_size, scheduler_block_size, enable_caching=True):
    groups = [
        KVCacheGroupSpec(
            layer_names=[f"layer{i}"],
            kv_cache_spec=FullAttentionSpec(
                block_size=size, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
            ),
        )
        for i, size in enumerate(group_block_sizes)
    ]
    return KVCacheManager(
        kv_cache_config=KVCacheConfig(num_blocks=512, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=4096,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        enable_caching=enable_caching,
    )


def _request(request_id, hash_block_size, tokens=PROMPT_TOKENS):
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(1, tokens + 1)),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, _HASH_FN),
    )


def _warm(manager, hash_block_size):
    """Admit one request and cache its blocks, so later requests have a prefix to hit."""
    request = _request("warming", hash_block_size)
    blocks, computed, _ = manager.get_computed_blocks(request)
    manager.allocate_slots(request, PROMPT_TOKENS, computed, blocks)
    manager.cache_blocks(request, PROMPT_TOKENS)


@pytest.fixture
def uniform():
    """One group: the shape the Qwen3.8 model runs in today."""
    manager = _manager([16], hash_block_size=16, scheduler_block_size=16)
    _warm(manager, 16)
    return manager


class TestAgainstUpstream:
    def test_upstream_reports_the_whole_cached_prefix(self, uniform):
        # The baseline the filter has to cap. vLLM recomputes the last token, so the hit stops
        # one block short of the 128-token prompt.
        _, computed, _ = uniform.get_computed_blocks(_request("plain", 16))
        assert computed == 112

    def test_an_empty_index_turns_a_hit_into_a_full_prefill(self, uniform):
        install_recurrent_prefix_filter(
            uniform, hash_block_size=16, alignment_tokens=16, capacity=4
        )
        blocks, computed, boundary = uniform.get_computed_blocks(_request("cold", 16))
        assert (computed, boundary) == (0, 0)
        assert [len(group) for group in blocks.blocks] == [0]

    def test_a_hit_is_capped_to_the_prefix_a_snapshot_backs(self, uniform):
        cache = install_recurrent_prefix_filter(
            uniform, hash_block_size=16, alignment_tokens=16, capacity=4
        )
        request = _request("capped", 16)
        cache.remember(request.block_hashes[1], handle=7, tokens=32)
        blocks, computed, boundary = uniform.get_computed_blocks(request)
        assert (computed, boundary) == (32, 0)
        assert [len(group) for group in blocks.blocks] == [2]

    def test_a_fully_backed_prefix_passes_through_untouched(self, uniform):
        cache = install_recurrent_prefix_filter(
            uniform, hash_block_size=16, alignment_tokens=16, capacity=4
        )
        request = _request("whole", 16)
        cache.remember(request.block_hashes[6], handle=1, tokens=112)
        _, computed, _ = uniform.get_computed_blocks(request)
        assert computed == 112

    def test_a_trimmed_hit_is_still_allocatable(self, uniform):
        # The failure this guards against is silent at the trim and fatal at admission: a hit
        # that is not block-aligned is rejected by allocate_slots, not by the filter.
        cache = install_recurrent_prefix_filter(
            uniform, hash_block_size=16, alignment_tokens=16, capacity=4
        )
        request = _request("allocating", 16)
        cache.remember(request.block_hashes[2], handle=3, tokens=48)
        blocks, computed, _ = uniform.get_computed_blocks(request)
        assert uniform.allocate_slots(request, PROMPT_TOKENS - computed, computed, blocks)

    def test_a_miss_is_left_exactly_as_upstream_returned_it(self):
        manager = _manager([16], hash_block_size=16, scheduler_block_size=16)
        install_recurrent_prefix_filter(
            manager, hash_block_size=16, alignment_tokens=16, capacity=4
        )
        blocks, computed, boundary = manager.get_computed_blocks(_request("first", 16))
        assert (computed, boundary) == (0, 0)
        assert blocks is manager.empty_kv_cache_blocks


class TestTwoBlockSizes:
    """Hashes are indexed at the gcd of the group block sizes; a hit lands on their lcm."""

    HASH, ALIGN = 16, 32

    @pytest.fixture
    def hybrid(self):
        manager = _manager([16, 32], hash_block_size=self.HASH, scheduler_block_size=self.ALIGN)
        _warm(manager, self.HASH)
        return manager

    def test_each_group_is_trimmed_in_its_own_block_size(self, hybrid):
        cache = install_recurrent_prefix_filter(
            hybrid, hash_block_size=self.HASH, alignment_tokens=self.ALIGN, capacity=4
        )
        request = _request("hybrid", self.HASH)
        # Two 16-token hashes stand for the 32 tokens one scheduler block covers.
        cache.remember(request.block_hashes[1], handle=5, tokens=32)
        blocks, computed, _ = hybrid.get_computed_blocks(request)
        assert computed == 32
        # The same 32 tokens: two blocks in the 16-sized group, one in the 32-sized group.
        assert [len(group) for group in blocks.blocks] == [2, 1]
        assert hybrid.allocate_slots(request, PROMPT_TOKENS - computed, computed, blocks)

    def test_a_prefix_held_off_the_alignment_gives_back_the_remainder(self, hybrid):
        # A snapshot can only stand for whole scheduler blocks, so a prefix held at 48 tokens
        # serves 32. Rounding up would hand the model a summary it never computed.
        cache = install_recurrent_prefix_filter(
            hybrid, hash_block_size=self.HASH, alignment_tokens=self.ALIGN, capacity=4
        )
        request = _request("ragged", self.HASH)
        cache.remember(request.block_hashes[2], handle=6, tokens=48)
        blocks, computed, _ = hybrid.get_computed_blocks(request)
        assert computed == 32
        assert [len(group) for group in blocks.blocks] == [2, 1]
        assert hybrid.allocate_slots(request, PROMPT_TOKENS - computed, computed, blocks)


class TestWhenItStaysOut:
    @staticmethod
    def _is_wrapped(manager):
        # The wrapper is installed as an instance attribute shadowing the class method, so its
        # absence from the instance dict is what says the manager was left alone.
        return "get_computed_blocks" in vars(manager)

    def test_a_model_declaring_no_snapshots_leaves_the_manager_alone(self):
        manager = _manager([16], hash_block_size=16, scheduler_block_size=16)
        assert (
            install_recurrent_prefix_filter(
                manager, hash_block_size=16, alignment_tokens=16, capacity=0
            )
            is None
        )
        assert not self._is_wrapped(manager)

    def test_caching_turned_off_leaves_the_manager_alone(self):
        manager = _manager([16], hash_block_size=16, scheduler_block_size=16, enable_caching=False)
        assert (
            install_recurrent_prefix_filter(
                manager, hash_block_size=16, alignment_tokens=16, capacity=4
            )
            is None
        )
        assert not self._is_wrapped(manager)

    def test_a_declared_capacity_does_install_the_wrapper(self):
        # The control for the two above: without it they would pass on a filter that never
        # installs anything at all.
        manager = _manager([16], hash_block_size=16, scheduler_block_size=16)
        # An empty index is falsy, so this has to be an identity check.
        assert (
            install_recurrent_prefix_filter(
                manager, hash_block_size=16, alignment_tokens=16, capacity=4
            )
            is not None
        )
        assert self._is_wrapped(manager)

    def test_no_manager_at_all_is_not_an_error(self):
        assert (
            install_recurrent_prefix_filter(
                None, hash_block_size=16, alignment_tokens=16, capacity=4
            )
            is None
        )
