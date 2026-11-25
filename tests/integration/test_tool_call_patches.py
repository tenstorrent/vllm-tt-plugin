# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks that the TT platform hook fixes the API server's tool-call parsing."""

import pytest

from tests.support.tool_calls import Call, extract
from vllm_tt_plugin import platform as tt_platform


def test_platform_hook_installs_tool_call_patches(monkeypatch: pytest.MonkeyPatch):
    """``pre_register_and_update`` fixes typing and splitting for later parsers.

    The API server runs the hook from its CLI parser and again from
    ``create_engine_config``; the second run must not wrap the patches twice.
    The root fixture restores the patched vLLM attributes afterward.
    """
    for name in (
        "_pin_v1_model_runner",
        "_install_tt_harmony_truncation_patch",
        "_install_diffusion_gemma_architecture_patch",
    ):
        monkeypatch.setattr(tt_platform, name, lambda: None)
    monkeypatch.setattr(
        tt_platform, "register_tt_models", lambda register_test_models=False: None
    )
    tools = {
        "f": {
            "$defs": {"Count": {"type": "integer"}},
            "properties": {"n": {"$ref": "#/$defs/Count"}, "s": {"type": "string"}},
        }
    }
    output = (
        "<tool_call>\n<function=f>\n<parameter=n>\n2\n</parameter>\n"
        "<parameter=s>\na</parameter>b\n</function>\n</tool_call>"
    )

    tt_platform.TTPlatform.pre_register_and_update()
    tt_platform.TTPlatform.pre_register_and_update()

    assert extract(tools, output) == [Call("f", {"n": 2, "s": "a</parameter>b"})]
