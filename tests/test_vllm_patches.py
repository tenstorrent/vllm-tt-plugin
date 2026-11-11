# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""patches/apply-vllm-patches.sh, which docs/install-vllm-tt.sh runs after
installing vLLM, and the patched vLLM it leaves behind."""

import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HELPER = _REPO / "patches" / "apply-vllm-patches.sh"
_VERSION = importlib.metadata.version("vllm").split("+")[0]
_PATCHES = sorted((_REPO / "patches" / f"vllm-{_VERSION}").glob("*.patch"))
_SITE = Path(
    list(importlib.util.find_spec("vllm").submodule_search_locations)[0]
).parent

pytestmark = pytest.mark.skipif(
    not _PATCHES, reason=f"no patches for the installed vLLM {_VERSION}"
)


def _patched_paths(patch: Path) -> list[str]:
    return re.findall(r"^\+\+\+ b/(\S+)", patch.read_text(), flags=re.MULTILINE)


def _reverse_applies(patch: Path, site: Path) -> bool:
    cmd = ["patch", "-p1", "-d", str(site), "-i", str(patch), "--fuzz=0"]
    cmd += ["--reverse", "--forward", "--dry-run", "--silent", "--batch"]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _run_helper(site: Path) -> subprocess.CompletedProcess:
    # The copied tree and its dist-info shadow the installed vLLM for the helper.
    env = {**os.environ, "PYTHONPATH": str(site)}
    return subprocess.run(
        ["bash", str(_HELPER), sys.executable],
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("patch", _PATCHES, ids=lambda p: p.name)
def test_installed_vllm_is_patched(patch):
    assert _reverse_applies(patch, _SITE), (
        f"{patch.name} is not applied to {_SITE}; re-run docs/install-vllm-tt.sh"
    )


def test_mimo_tool_parser_uses_mimo_structural_tag():
    from vllm.tool_parsers import ToolParserManager

    parser = ToolParserManager.get_tool_parser("mimo")
    assert parser.__name__ == "MiMoToolParser"
    assert parser.structural_tag_model == "mimo"


def _unpatch(site: Path) -> None:
    for patch in reversed(_PATCHES):
        if _reverse_applies(patch, site):
            cmd = ["patch", "-p1", "-d", str(site), "-i", str(patch)]
            subprocess.run(cmd + ["--reverse", "--silent", "--batch"], check=True)


def test_helper_applies_once_and_rejects_a_conflict(tmp_path):
    # A pristine copy of every file the patches touch, laid out like an install.
    for patch in _PATCHES:
        for rel in _patched_paths(patch):
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_SITE / rel, dst)
    (tmp_path / "vllm" / "__init__.py").touch()
    dist_info = tmp_path / f"vllm-{_VERSION}+empty.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: vllm\nVersion: {_VERSION}+empty\n"
    )
    _unpatch(tmp_path)

    result = _run_helper(tmp_path)
    assert result.returncode == 0, result.stderr
    assert ": applied to" in result.stderr
    assert all(_reverse_applies(p, tmp_path) for p in _PATCHES)

    result = _run_helper(tmp_path)
    assert result.returncode == 0, result.stderr
    assert ": applied to" not in result.stderr
    assert "already applied" in result.stderr

    # Edit a line the patch needs, unpatched: it applies in neither direction.
    _unpatch(tmp_path)
    target = tmp_path / "vllm" / "tool_parsers" / "__init__.py"
    source = target.read_text()
    assert '"mimo"' in source
    target.write_text(source.replace('"mimo"', '"mimo_local_edit"'))
    result = _run_helper(tmp_path)
    assert result.returncode != 0
    assert "does not apply" in result.stderr
