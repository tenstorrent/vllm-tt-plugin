# SPDX-License-Identifier: Apache-2.0
"""Handles travelling between the scheduler, which owns the index, and the runner, which owns
the bytes.

Each direction is one attribute on a plain dataclass that is pickled across the executor
boundary. What matters here is the bookkeeping either side of it: a hit must name the snapshot it
was admitted on, and a snapshot the index stops naming must reach the runner to be freed, or the
bounded store fills with state nothing can ask for.
"""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.recurrent_prefix import RecurrentPrefixCache, rows_worth_snapshotting
from vllm_tt_plugin.scheduler import (
    get_tt_recurrent_free,
    get_tt_recurrent_restore,
    get_tt_recurrent_saved,
    set_tt_recurrent_free,
    set_tt_recurrent_restore,
    set_tt_recurrent_saved,
)

BLOCK = 32
HASHES = [f"h{i}" for i in range(8)]


class TestTheChannel:
    """Both carriers are plain dataclasses, so the signal rides an attribute on the instance."""

    @staticmethod
    def _carrier():
        class Carrier:  # a dataclass stand-in: anything with a __dict__
            pass

        return Carrier()

    def test_nothing_is_attached_when_there_is_nothing_to_say(self):
        for setter, getter, empty in (
            (set_tt_recurrent_restore, get_tt_recurrent_restore, {}),
            (set_tt_recurrent_free, get_tt_recurrent_free, []),
            (set_tt_recurrent_saved, get_tt_recurrent_saved, []),
        ):
            carrier = self._carrier()
            setter(carrier, empty)
            assert getter(carrier) == empty
            assert not vars(carrier)

    def test_an_absent_signal_reads_as_empty_rather_than_raising(self):
        # Every step the feature is off, and every step before the first hit, takes this path.
        carrier = self._carrier()
        assert get_tt_recurrent_restore(carrier) == {}
        assert get_tt_recurrent_free(carrier) == []
        assert get_tt_recurrent_saved(carrier) == []

    def test_each_signal_survives_the_executor_boundary(self):
        # The real carriers, because the whole scheme rests on vLLM keeping these as plain
        # dataclasses pickled whole: a msgspec schema would drop an attribute it has no field
        # for, silently, and the handles would simply never arrive.
        import pickle

        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.outputs import ModelRunnerOutput

        to_runner = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
        set_tt_recurrent_restore(to_runner, {"a": 1})
        set_tt_recurrent_free(to_runner, [4, 5])
        revived = pickle.loads(pickle.dumps(to_runner))
        assert get_tt_recurrent_restore(revived) == {"a": 1}
        assert get_tt_recurrent_free(revived) == [4, 5]

        to_scheduler = ModelRunnerOutput(req_ids=[], req_id_to_index={})
        set_tt_recurrent_saved(to_scheduler, [("a", 64, 9)])
        assert get_tt_recurrent_saved(pickle.loads(pickle.dumps(to_scheduler))) == [
            ("a", 64, 9)
        ]

    def test_the_carrier_keeps_its_own_copy(self):
        carrier = self._carrier()
        handles = {"a": 1}
        set_tt_recurrent_restore(carrier, handles)
        handles["b"] = 2
        assert get_tt_recurrent_restore(carrier) == {"a": 1}


class TestWhichRowsAreWorthKeeping:
    def test_a_prompt_finishing_on_a_boundary_is_kept(self):
        assert rows_worth_snapshotting(["a"], [3], [64], [64], BLOCK) == [("a", 3, 64)]

    def test_a_prompt_finishing_off_the_boundary_is_not(self):
        # The state is sequential, so it cannot be wound back to 64; keeping it under the hash
        # for 64 tokens would name a summary of 70.
        assert rows_worth_snapshotting(["a"], [3], [70], [70], BLOCK) == []

    def test_a_partial_prefill_is_not_kept(self):
        # Block-aligned but the prompt continues, so a later chunk still has to run in this slot.
        assert rows_worth_snapshotting(["a"], [3], [64], [128], BLOCK) == []

    def test_an_empty_prefill_is_not_kept(self):
        assert rows_worth_snapshotting(["a"], [0], [0], [0], BLOCK) == []

    def test_rows_are_judged_one_at_a_time(self):
        kept = rows_worth_snapshotting(
            ["a", "b", "c"], [0, 1, 2], [64, 70, 96], [64, 70, 96], BLOCK
        )
        assert kept == [("a", 0, 64), ("c", 2, 96)]


class TestSchedulerBookkeeping:
    """``_annotate_recurrent_prefix`` and ``_record_recurrent_prefix_saves`` without a scheduler.

    Both are bound methods over three attributes, so they are exercised against a stand-in: a
    real TTScheduler needs a model, and what is under test is the bookkeeping, not scheduling.
    """

    @staticmethod
    def _scheduler(cache=None):
        from vllm_tt_plugin.scheduler import TTScheduler

        scheduler = SimpleNamespace(
            _recurrent_prefix=cache,
            _pending_recurrent_frees=[],
            hash_block_size=BLOCK,
            requests={},
        )
        for name in (
            "_annotate_recurrent_prefix",
            "_take_recurrent_frees",
            "_record_recurrent_prefix_saves",
        ):
            setattr(
                scheduler, name, getattr(TTScheduler, name).__get__(scheduler, SimpleNamespace)
            )
        return scheduler

    @staticmethod
    def _output(new_reqs=()):
        return SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id=rid, num_computed_tokens=n) for rid, n in new_reqs
            ]
        )

    def test_a_step_with_the_feature_off_annotates_nothing(self):
        scheduler = self._scheduler(cache=None)
        out = self._output([("a", 64)])
        scheduler._annotate_recurrent_prefix(out)
        assert get_tt_recurrent_restore(out) == {}

    def test_a_cold_request_needs_no_restore(self):
        scheduler = self._scheduler(RecurrentPrefixCache(BLOCK, 4))
        out = self._output([("a", 0)])
        scheduler._annotate_recurrent_prefix(out)
        assert get_tt_recurrent_restore(out) == {}

    def test_an_admitted_hit_names_the_snapshot_it_rests_on(self):
        cache = RecurrentPrefixCache(BLOCK, 4)
        cache.remember(HASHES[1], handle=7, tokens=2 * BLOCK)
        scheduler = self._scheduler(cache)
        scheduler.requests["a"] = SimpleNamespace(block_hashes=HASHES)
        out = self._output([("a", 2 * BLOCK)])
        scheduler._annotate_recurrent_prefix(out)
        assert get_tt_recurrent_restore(out) == {"a": 7}

    def test_a_hit_with_no_snapshot_behind_it_is_refused_loudly(self):
        # Unreachable while the filter is the only source of hits. If it ever happens the
        # request's prefix tokens are already gone, so continuing would answer from state
        # nobody computed.
        scheduler = self._scheduler(RecurrentPrefixCache(BLOCK, 4))
        scheduler.requests["a"] = SimpleNamespace(block_hashes=HASHES)
        with pytest.raises(RuntimeError, match="no recurrent snapshot"):
            scheduler._annotate_recurrent_prefix(self._output([("a", 2 * BLOCK)]))

    def test_a_saved_snapshot_becomes_servable(self):
        cache = RecurrentPrefixCache(BLOCK, 4)
        scheduler = self._scheduler(cache)
        scheduler.requests["a"] = SimpleNamespace(block_hashes=HASHES)
        scheduler._record_recurrent_prefix_saves(
            SimpleNamespace(_tt_recurrent_saved=[("a", 2 * BLOCK, 7)])
        )
        assert cache.servable_tokens(HASHES, 8 * BLOCK) == 2 * BLOCK
        assert cache.handle_for(HASHES, 2 * BLOCK) == 7

    def test_a_displaced_snapshot_is_queued_for_the_runner_to_free(self):
        cache = RecurrentPrefixCache(BLOCK, capacity=1)
        scheduler = self._scheduler(cache)
        scheduler.requests["a"] = SimpleNamespace(block_hashes=HASHES)
        scheduler.requests["b"] = SimpleNamespace(block_hashes=["z0", "z1", "z2"])
        scheduler._record_recurrent_prefix_saves(
            SimpleNamespace(_tt_recurrent_saved=[("a", BLOCK, 1)])
        )
        scheduler._record_recurrent_prefix_saves(
            SimpleNamespace(_tt_recurrent_saved=[("b", BLOCK, 2)])
        )
        assert scheduler._pending_recurrent_frees == [1]

    def test_a_snapshot_for_a_request_that_is_already_gone_is_freed_not_indexed(self):
        cache = RecurrentPrefixCache(BLOCK, 4)
        scheduler = self._scheduler(cache)
        scheduler._record_recurrent_prefix_saves(
            SimpleNamespace(_tt_recurrent_saved=[("vanished", BLOCK, 9)])
        )
        assert scheduler._pending_recurrent_frees == [9]
        assert len(cache) == 0

    def test_queued_frees_ride_the_next_step_and_only_once(self):
        cache = RecurrentPrefixCache(BLOCK, 4)
        scheduler = self._scheduler(cache)
        scheduler._pending_recurrent_frees = [4, 5]
        out = self._output()
        scheduler._annotate_recurrent_prefix(out)
        assert get_tt_recurrent_free(out) == [4, 5]
        later = self._output()
        scheduler._annotate_recurrent_prefix(later)
        assert get_tt_recurrent_free(later) == []
