# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lane scheduling policies are ``--additional-config '{"tt": {...}}'`` settings."""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.config import (
    get_tt_lane_prefill_policy,
    get_tt_lane_prefix_affinity,
)


def _config(tt=None, *, prefix_caching=True):
    return SimpleNamespace(
        additional_config={"tt": dict(tt or {})},
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
    )


def test_prefix_affinity_defaults_follow_prefix_caching():
    assert get_tt_lane_prefix_affinity(_config()) == (256, 4)
    # Without a prefix cache there is nothing for a lane to stay close to.
    assert get_tt_lane_prefix_affinity(_config(prefix_caching=False)) == (0, 4)
    assert get_tt_lane_prefix_affinity(SimpleNamespace(additional_config={})) == (0, 4)


def test_prefix_affinity_settings_are_read_and_validated():
    tt = {"lane_prefix_affinity_tokens": 64, "lane_prefix_affinity_slack": 0}
    assert get_tt_lane_prefix_affinity(_config(tt)) == (64, 0)
    assert get_tt_lane_prefix_affinity(_config({"lane_prefix_affinity_tokens": 0})) == (
        0,
        4,
    )
    for bad in (
        {"lane_prefix_affinity_tokens": -1},
        {"lane_prefix_affinity_slack": "4"},
    ):
        with pytest.raises(ValueError, match="must be an integer >= 0"):
            get_tt_lane_prefix_affinity(_config(bad))


def test_lane_prefill_policy_defaults_to_no_cap_and_no_hold():
    assert get_tt_lane_prefill_policy(_config()) == (0, 0.0)
    assert get_tt_lane_prefill_policy(SimpleNamespace(additional_config={})) == (0, 0.0)


def test_lane_prefill_policy_settings_are_read_and_validated():
    tt = {"lane_prefill_admissions_per_step": 1, "lane_prefill_gather_ms": 150}
    assert get_tt_lane_prefill_policy(_config(tt)) == (1, 150.0)
    assert get_tt_lane_prefill_policy(_config({"lane_prefill_gather_ms": 12.5})) == (
        0,
        12.5,
    )
    with pytest.raises(ValueError, match="must be an integer >= 0"):
        get_tt_lane_prefill_policy(_config({"lane_prefill_admissions_per_step": True}))
    with pytest.raises(ValueError, match="must be a number >= 0"):
        get_tt_lane_prefill_policy(_config({"lane_prefill_gather_ms": -1}))
