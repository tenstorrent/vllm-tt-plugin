# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Keeps module unit tests independent of host engine fixture initialization."""

import pytest


@pytest.fixture(autouse=True)
def reset_tt_platform_class_state() -> None:
    """Overrides the root engine/patch reset fixture for isolated unit tests."""
    return None
