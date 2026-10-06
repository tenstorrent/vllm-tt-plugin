# SPDX-License-Identifier: Apache-2.0
"""The prefix index decides what a hybrid model can actually start from.

Reporting a prefix as servable when no recurrent state backs it is the one failure that cannot be
recovered: the scheduler drops those tokens from the request, so the model is asked to continue
from a summary that was never computed and has no way to say so.
"""

import pytest

from vllm_tt_plugin.recurrent_prefix import RecurrentPrefixCache

BLOCK = 32
# One hash per block, standing in for vLLM's content hashes.
HASHES = [f"h{i}" for i in range(8)]


@pytest.fixture
def cache():
    return RecurrentPrefixCache(block_size=BLOCK, capacity=4)


def test_an_empty_index_serves_nothing(cache):
    assert cache.servable_tokens(HASHES, 8 * BLOCK) == 0
    assert cache.handle_for(HASHES, BLOCK) is None


def test_a_held_prefix_is_served_at_its_own_length(cache):
    cache.remember(HASHES[1], handle=7, tokens=2 * BLOCK)
    assert cache.servable_tokens(HASHES, 8 * BLOCK) == 2 * BLOCK
    assert cache.handle_for(HASHES, 2 * BLOCK) == 7


def test_the_longest_held_prefix_wins(cache):
    cache.remember(HASHES[0], handle=1, tokens=BLOCK)
    cache.remember(HASHES[3], handle=2, tokens=4 * BLOCK)
    assert cache.servable_tokens(HASHES, 8 * BLOCK) == 4 * BLOCK
    assert cache.handle_for(HASHES, 4 * BLOCK) == 2


def test_the_limit_caps_what_is_offered(cache):
    cache.remember(HASHES[3], handle=2, tokens=4 * BLOCK)
    cache.remember(HASHES[0], handle=1, tokens=BLOCK)
    # vLLM recomputes the final token, so the caller's limit can fall below a held prefix.
    assert cache.servable_tokens(HASHES, 3 * BLOCK) == BLOCK


def test_a_different_prompt_is_not_served(cache):
    cache.remember(HASHES[1], handle=7, tokens=2 * BLOCK)
    assert cache.servable_tokens(["x0", "x1", "x2"], 8 * BLOCK) == 0


def test_a_shared_head_is_served_only_as_far_as_it_is_shared(cache):
    cache.remember(HASHES[1], handle=7, tokens=2 * BLOCK)
    cache.remember(HASHES[3], handle=8, tokens=4 * BLOCK)
    diverging = [HASHES[0], HASHES[1], "other2", "other3"]
    assert cache.servable_tokens(diverging, 8 * BLOCK) == 2 * BLOCK


def test_a_partial_block_is_never_offered(cache):
    cache.remember(HASHES[0], handle=1, tokens=BLOCK)
    assert cache.servable_tokens(HASHES, BLOCK + 1) == BLOCK
    assert cache.servable_tokens(HASHES, BLOCK - 1) == 0
    assert cache.handle_for(HASHES, BLOCK + 7) is None


def test_a_saved_prefix_must_be_whole_blocks(cache):
    with pytest.raises(ValueError):
        cache.remember(HASHES[0], handle=1, tokens=BLOCK + 1)


def test_the_least_recently_used_entry_is_displaced_and_handed_back(cache):
    for i in range(4):
        assert cache.remember(HASHES[i], handle=i, tokens=(i + 1) * BLOCK) is None
    # Capacity is 4, so a fifth entry displaces the oldest and names its handle for freeing.
    assert cache.remember(HASHES[4], handle=99, tokens=5 * BLOCK) == 0
    assert len(cache) == 4


def test_serving_a_prefix_keeps_it_from_being_displaced(cache):
    for i in range(4):
        cache.remember(HASHES[i], handle=i, tokens=(i + 1) * BLOCK)
    # Touch the oldest, so the next eviction takes the second oldest instead.
    assert cache.servable_tokens([HASHES[0]], 8 * BLOCK) == BLOCK
    assert cache.remember(HASHES[4], handle=99, tokens=5 * BLOCK) == 1


def test_replacing_a_prefix_hands_back_the_handle_it_replaced(cache):
    cache.remember(HASHES[0], handle=1, tokens=BLOCK)
    # Two handles for one prefix would strand state nothing can ask for.
    assert cache.remember(HASHES[0], handle=2, tokens=BLOCK) == 1
    assert len(cache) == 1
    assert cache.handle_for(HASHES, BLOCK) == 2


def test_a_forgotten_handle_is_no_longer_served(cache):
    cache.remember(HASHES[1], handle=7, tokens=2 * BLOCK)
    cache.forget_handle(7)
    assert cache.servable_tokens(HASHES, 8 * BLOCK) == 0


def test_a_zero_capacity_index_serves_nothing_and_frees_what_it_is_given():
    disabled = RecurrentPrefixCache(block_size=BLOCK, capacity=0)
    assert disabled.remember(HASHES[0], handle=1, tokens=BLOCK) is None
    assert disabled.servable_tokens(HASHES, 8 * BLOCK) == 0


def test_a_nonsense_configuration_is_refused():
    for block, capacity in ((0, 4), (-1, 4), (BLOCK, -1)):
        with pytest.raises(ValueError):
            RecurrentPrefixCache(block_size=block, capacity=capacity)


class TestCapacityPlumbing:
    """The model's snapshot capacity reaches the scheduler through the config.

    Platform-derived, like the other underscore-prefixed keys: it is written from what the model
    declares, never read from operator input, so an --additional-config entry of the same name
    cannot talk the scheduler into offering prefixes the model cannot serve.
    """

    @staticmethod
    def _config():
        from types import SimpleNamespace

        return SimpleNamespace(additional_config=None)

    def test_a_model_that_declares_nothing_offers_no_prefixes(self):
        from vllm_tt_plugin.config import get_tt_recurrent_prefix_capacity

        assert get_tt_recurrent_prefix_capacity(self._config()) == 0

    def test_a_declared_capacity_survives_the_handoff(self):
        from vllm_tt_plugin.config import (
            get_tt_recurrent_prefix_capacity,
            store_tt_recurrent_prefix_capacity,
        )

        config = self._config()
        store_tt_recurrent_prefix_capacity(config, 8)
        assert get_tt_recurrent_prefix_capacity(config) == 8

    def test_a_nonsense_capacity_reads_as_unsupported_rather_than_raising(self):
        from vllm_tt_plugin.config import get_tt_recurrent_prefix_capacity

        config = self._config()
        config.additional_config = {"_tt_recurrent_prefix_capacity": "lots"}
        # Reporting a hit the model cannot serve is unrecoverable, so an unreadable value has to
        # fail closed rather than propagate.
        assert get_tt_recurrent_prefix_capacity(config) == 0

    def test_a_negative_capacity_reads_as_unsupported(self):
        from vllm_tt_plugin.config import (
            get_tt_recurrent_prefix_capacity,
            store_tt_recurrent_prefix_capacity,
        )

        config = self._config()
        store_tt_recurrent_prefix_capacity(config, -3)
        assert get_tt_recurrent_prefix_capacity(config) == 0
