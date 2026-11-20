# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""patches/apply-vllm-patches.sh, which docs/install-vllm-tt.sh runs after
installing vLLM, and the stack it leaves in the installed vLLM."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tests.vllm_patches import (
    REPO,
    SITE,
    VLLM_VERSION,
    requires_patched_vllm,
    series,
)

pytestmark = requires_patched_vllm

_HELPER = REPO / "patches" / "apply-vllm-patches.sh"


def _patched_paths() -> list[str]:
    paths = set()
    for patch in series():
        text = patch.read_text()
        paths.update(re.findall(r"^\+\+\+ b/(\S+)", text, flags=re.MULTILINE))
    return sorted(paths)


def _patch(site: Path, patch: Path, *args: str) -> bool:
    cmd = ["patch", "-p1", "-d", str(site), "-i", str(patch), "--fuzz=0"]
    cmd += ["--forward", "--silent", "--batch", *args]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _copy_site(dst: Path) -> None:
    """Copy every file the stack touches, laid out like an install."""
    for rel in _patched_paths():
        if (SITE / rel).exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SITE / rel, dst / rel)
    (dst / "vllm" / "__init__.py").touch()
    dist_info = dst / f"vllm-{VLLM_VERSION}+empty.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: vllm\nVersion: {VLLM_VERSION}+empty\n"
    )


def _unpatch(site: Path) -> None:
    """Reverse the whole stack, top first; it must be fully applied."""
    for patch in reversed(series()):
        assert _patch(site, patch, "--reverse"), f"{patch.name} does not reverse"


def _run_helper(site: Path) -> subprocess.CompletedProcess:
    # The copied tree and its dist-info shadow the installed vLLM for the helper.
    env = {**os.environ, "PYTHONPATH": str(site)}
    return subprocess.run(
        ["bash", str(_HELPER), sys.executable],
        env=env,
        capture_output=True,
        text=True,
    )


def test_installed_vllm_has_the_whole_stack(tmp_path):
    _copy_site(tmp_path)
    _unpatch(tmp_path)  # fails unless every patch is applied, in order
    for patch in series():
        assert _patch(tmp_path, patch), f"{patch.name} does not re-apply"


def test_helper_applies_the_stack_once_and_rejects_a_conflict(tmp_path):
    patches = series()
    _copy_site(tmp_path)
    _unpatch(tmp_path)

    result = _run_helper(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count(": applied to") == len(patches)

    result = _run_helper(tmp_path)
    assert result.returncode == 0, result.stderr
    assert ": applied to" not in result.stderr
    assert result.stderr.count("already applied") == len(patches)

    # Only the first patch in place: the helper applies the rest.
    _unpatch(tmp_path)
    assert _patch(tmp_path, patches[0])
    result = _run_helper(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("already applied") == 1
    assert result.stderr.count(": applied to") == len(patches) - 1

    # Edit a line the first patch needs, unpatched: nothing applies.
    _unpatch(tmp_path)
    target = tmp_path / "vllm" / "tool_parsers" / "__init__.py"
    source = target.read_text()
    assert '"qwen3_engine_tool_parser",\n        "Qwen3EngineToolParser"' in source
    target.write_text(source.replace('"mimo": (', '"mimo_local_edit": ('))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.py")}
    result = _run_helper(tmp_path)
    assert result.returncode != 0
    assert "does not apply" in result.stderr
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.py")} == before
