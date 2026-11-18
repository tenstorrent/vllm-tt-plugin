# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.


def test_plugin_tile_matches_ttnn():
    """Pin the admission constant to tt-metal's real tile dimension."""
    # Imported here, not at module scope, so an import error fails only this
    # test instead of stopping collection of every file in tests/tt.
    # `vllm` must load before `vllm_tt_plugin.platform`. If the plugin module
    # is the first to import vLLM, vLLM's package init resolves
    # `current_platform`, which loads `TTPlatform` from the half-initialized
    # `vllm_tt_plugin.platform` and fails with
    # "cannot import name 'current_platform'".
    import ttnn
    import vllm  # noqa: F401

    from vllm_tt_plugin.platform import _TT_TOKEN_TILE_SIZE

    assert _TT_TOKEN_TILE_SIZE == ttnn.TILE_SIZE == 32
