# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import re
from pathlib import Path

INSTALL_SCRIPT = Path(__file__).parents[1] / "docs" / "install-vllm-tt.sh"


def _assignment(script: str, name: str) -> str:
    match = re.search(rf"^{name}=([^\n]+)$", script, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_vllm_source_is_pinned_to_an_immutable_commit():
    script = INSTALL_SCRIPT.read_text()
    commit = _assignment(script, "VLLM_COMMIT")

    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert _assignment(script, "VLLM_VERSION") == "0.26.0"
    assert _assignment(script, "VLLM_REPOSITORY") == (
        "https://github.com/tenstorrent/vllm.git"
    )
    assert '"vllm @ git+${VLLM_REPOSITORY}@${VLLM_COMMIT}"' in script


def test_vllm_requirements_use_the_same_immutable_commit():
    script = INSTALL_SCRIPT.read_text()

    assert (
        '"https://raw.githubusercontent.com/tenstorrent/vllm/'
        '${VLLM_COMMIT}/requirements/common.txt"'
    ) in script
    assert "vllm-project/vllm/v0.26.0" not in script
