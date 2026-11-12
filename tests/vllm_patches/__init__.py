# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests of the vLLM patch stack in patches/vllm-<version>/ (see its README).

They run against the installed, patched vLLM, so they skip when the installed
vLLM has no patch series (vllm-release-watch's latest-vLLM leg).
"""

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VLLM_VERSION = importlib.metadata.version("vllm").split("+")[0]
PATCH_DIR = REPO / "patches" / f"vllm-{VLLM_VERSION}"
SERIES_FILE = PATCH_DIR / "series"
SITE = Path(list(importlib.util.find_spec("vllm").submodule_search_locations)[0]).parent


def series() -> list[Path]:
    """The patches the installer applies, in order."""
    if not SERIES_FILE.exists():
        return []
    entries = []
    for line in SERIES_FILE.read_text().splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            entries.append(PATCH_DIR / entry)
    return entries


requires_patched_vllm = pytest.mark.skipif(
    not series(), reason=f"no vLLM patch series for the installed vLLM {VLLM_VERSION}"
)
