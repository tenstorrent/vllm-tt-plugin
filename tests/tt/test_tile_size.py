# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
import ttnn

from vllm_tt_plugin.platform import _TT_TOKEN_TILE_SIZE


def test_plugin_tile_matches_ttnn():
    """Pin the admission constant to tt-metal's real tile dimension."""
    assert _TT_TOKEN_TILE_SIZE == ttnn.TILE_SIZE == 32
