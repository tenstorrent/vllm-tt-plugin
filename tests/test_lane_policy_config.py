# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lane scheduling policies are ``--additional-config '{"tt": {...}}'`` settings."""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.config import (
    get_tt_lane_prefill_gather_ms,
    get_tt_lane_prefill_group,
    get_tt_lane_prefix_affinity,
    store_tt_lane_prefill_group,
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


def test_lane_prefill_gather_defaults_to_no_hold():
    assert get_tt_lane_prefill_gather_ms(_config()) == 0.0
    assert get_tt_lane_prefill_gather_ms(SimpleNamespace(additional_config={})) == 0.0


def test_lane_prefill_gather_is_read_and_validated():
    assert (
        get_tt_lane_prefill_gather_ms(_config({"lane_prefill_gather_ms": 150})) == 150.0
    )
    assert (
        get_tt_lane_prefill_gather_ms(_config({"lane_prefill_gather_ms": 12.5})) == 12.5
    )
    with pytest.raises(ValueError, match="must be a number >= 0"):
        get_tt_lane_prefill_gather_ms(_config({"lane_prefill_gather_ms": -1}))


def test_lane_prefill_group_is_a_recorded_model_declaration_not_a_setting():
    """The group behaviour comes from the model's capability (recorded by the
    fold), never from the user's tt config."""
    config = _config({"lane_prefill_group": True})
    assert get_tt_lane_prefill_group(config) is False
    store_tt_lane_prefill_group(config, True)
    assert get_tt_lane_prefill_group(config) is True
    store_tt_lane_prefill_group(config, False)
    assert get_tt_lane_prefill_group(config) is False
    assert get_tt_lane_prefill_group(SimpleNamespace(additional_config={})) is False
