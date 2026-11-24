# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks the general plugin entry point against vLLM's real parser registries."""

import pytest
import vllm.config  # noqa: F401  # finish plugin discovery before parser imports
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

from vllm_tt_plugin import entrypoints
from vllm_tt_plugin.parsers.registration import register_parsers


@pytest.fixture(autouse=True)
def parser_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uses copied registries with free Kolibri names and restores them afterward."""
    for manager, eager in (
        (ReasoningParserManager, "reasoning_parsers"),
        (ToolParserManager, "tool_parsers"),
    ):
        for attribute in (eager, "lazy_parsers"):
            entries = getattr(manager, attribute).copy()
            entries.pop("kolibri1", None)
            monkeypatch.setattr(manager, attribute, entries)


def test_general_entrypoint_registers_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    """General plugin activation connects parser setup without model construction."""
    from vllm_tt_plugin import model_registry

    def skip_models() -> None:
        """Leaves model registration outside this parser integration check."""

    monkeypatch.setattr(model_registry, "register_tt_models_from_plugin", skip_models)

    entrypoints.register()

    assert "kolibri1" in ReasoningParserManager.lazy_parsers
    assert "kolibri1" in ToolParserManager.lazy_parsers
    assert "kolibri1" not in ReasoningParserManager.reasoning_parsers
    assert "kolibri1" not in ToolParserManager.tool_parsers


def test_repeated_registration_resolves_the_same_parsers() -> None:
    """Both lazy and already-loaded plugin registrations are idempotent."""
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    from vllm_tt_plugin.parsers.kolibri1 import Kolibri1ReasoningParser

    for _ in range(2):
        register_parsers()
    for _ in range(2):
        assert (
            ReasoningParserManager.get_reasoning_parser("kolibri1")
            is Kolibri1ReasoningParser
        )
        assert ToolParserManager.get_tool_parser("kolibri1") is Hermes2ProToolParser
        register_parsers()


@pytest.mark.parametrize("resolve", [False, True], ids=["lazy", "loaded"])
def test_existing_parsers_win(resolve: bool) -> None:
    """Plugin defaults preserve existing parsers whether resolved or still lazy."""
    reasoning = ReasoningParserManager.get_reasoning_parser("qwen3")
    tool = ToolParserManager.get_tool_parser("llama3_json")
    ReasoningParserManager.register_lazy_module(
        "kolibri1", *ReasoningParserManager.lazy_parsers["qwen3"]
    )
    ToolParserManager.register_lazy_module(
        "kolibri1", *ToolParserManager.lazy_parsers["llama3_json"]
    )
    if resolve:
        ReasoningParserManager.get_reasoning_parser("kolibri1")
        ToolParserManager.get_tool_parser("kolibri1")
    reasoning_target = ReasoningParserManager.lazy_parsers["kolibri1"]
    tool_target = ToolParserManager.lazy_parsers["kolibri1"]

    register_parsers()

    assert ReasoningParserManager.lazy_parsers["kolibri1"] == reasoning_target
    assert ToolParserManager.lazy_parsers["kolibri1"] == tool_target
    assert ReasoningParserManager.get_reasoning_parser("kolibri1") is reasoning
    assert ToolParserManager.get_tool_parser("kolibri1") is tool
