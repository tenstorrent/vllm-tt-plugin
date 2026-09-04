# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for the determinism assertion helpers in tests/tt/utils.py.

These need no server and no device: they pin the divergence-index reporting and
the slot-count gate, which is the part that decides whether a Galaxy sampling run
is called a near-tie or a regression.
"""

import warnings

import pytest

from tests.tt.utils import (
    assert_deterministic_allow_near_tie,
    first_divergence,
)


class TestFirstDivergence:
    def test_identical_sequences_have_no_divergence(self):
        assert first_divergence([1, 2, 3], [1, 2, 3]) is None

    def test_reports_index_of_first_differing_token(self):
        assert first_divergence([1, 2, 3, 4], [1, 2, 9, 9]) == 2

    def test_prefix_counts_as_divergence_at_the_shorter_length(self):
        assert first_divergence([1, 2, 3], [1, 2]) == 2

    def test_text_falls_back_to_word_units(self):
        # " My friend asked ..." vs " My friend recently ..." parts at unit 2.
        a = " My friend asked me if"
        b = " My friend recently discovered he"
        assert first_divergence(a, b) == 2


class TestAssertDeterministicAllowNearTie:
    def test_all_identical_passes_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert_deterministic_allow_near_tie([[1, 2]] * 8, "x")

    def test_single_late_deviating_slot_warns_but_passes(self):
        results = [[1, 2, 3]] * 7 + [[1, 2, 9]]
        with pytest.warns(UserWarning, match="1/8 slot"):
            assert_deterministic_allow_near_tie(results, "x")

    def test_two_deviating_slots_fail(self):
        results = [[1, 2, 3]] * 6 + [[1, 2, 9]] * 2
        with pytest.raises(AssertionError, match="2 slots deviate"):
            assert_deterministic_allow_near_tie(results, "x")

    def test_correlated_flips_are_counted_per_slot_not_per_alternative(self):
        """The CI signature: 28 identical, 4 identical-but-different.

        Only two distinct strings, but four deviating slots. Counting distinct
        alternatives would tolerate this (and would tolerate a 16/16 split);
        counting slots does not.
        """
        results = [[1, 2, 3]] * 28 + [[1, 2, 9]] * 4
        with pytest.raises(AssertionError) as excinfo:
            assert_deterministic_allow_near_tie(results, "x")
        message = str(excinfo.value)
        assert "4/32 slot" in message
        assert "index 2 x4" in message

    def test_divergence_index_distinguishes_first_token_from_mid_stream(self):
        early = [[1, 2, 3]] * 6 + [[9, 2, 3]] * 2
        late = [[1, 2, 3]] * 6 + [[1, 2, 9]] * 2
        with pytest.raises(AssertionError, match=r"index 0 x2"):
            assert_deterministic_allow_near_tie(early, "x")
        with pytest.raises(AssertionError, match=r"index 2 x2"):
            assert_deterministic_allow_near_tie(late, "x")

    def test_accepts_text_results(self):
        results = [" My friend asked me"] * 5 + [" My friend recently found"]
        with pytest.warns(UserWarning, match="index 2"):
            assert_deterministic_allow_near_tie(results, "x")


class TestFailureBranchIsReachableAtTwoResults:
    """Review item 1: with two results the outlier count is capped at one, so a
    naive `len(outliers) <= max_outlier_slots` gate can never fail. Every 2-element
    assertion in the suite depended on this branch being reachable."""

    def test_two_wholly_different_results_fail(self):
        with pytest.raises(AssertionError, match="no strict majority"):
            assert_deterministic_allow_near_tie(["aaa", "bbb"], "x")

    def test_two_results_diverging_late_still_fail(self):
        with pytest.raises(AssertionError, match="no strict majority"):
            assert_deterministic_allow_near_tie([[1, 2, 3], [1, 2, 9]], "x")

    def test_two_identical_results_pass(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert_deterministic_allow_near_tie([[1, 2, 3]] * 2, "x")

    def test_even_split_has_no_majority_and_fails(self):
        results = [[1, 2, 3]] * 4 + [[1, 2, 9]] * 4
        with pytest.raises(AssertionError, match="no strict majority"):
            assert_deterministic_allow_near_tie(results, "x")


class TestPerResultPrefixGate:
    """Review item 2: the tolerance must also live inside a single result, not only
    in the count of deviating slots."""

    def test_divergence_at_token_zero_never_excused(self):
        results = [[1, 2, 3]] * 7 + [[9, 9, 9]]
        with pytest.raises(AssertionError, match="before the 1-unit shared prefix"):
            assert_deterministic_allow_near_tie(results, "x")

    def test_longer_prefix_requirement_is_configurable(self):
        results = [[1, 2, 3, 4]] * 7 + [[1, 2, 9, 9]]
        with pytest.warns(UserWarning):
            assert_deterministic_allow_near_tie(results, "x")
        with pytest.raises(AssertionError, match="before the 3-unit shared prefix"):
            assert_deterministic_allow_near_tie(results, "x", min_common_prefix=3)

    def test_slot_threshold_is_configurable(self):
        results = [[1, 2, 3]] * 6 + [[1, 2, 9]] * 2
        with pytest.warns(UserWarning):
            assert_deterministic_allow_near_tie(results, "x", max_outlier_slots=2)
