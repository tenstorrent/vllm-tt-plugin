# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Text prompts checked against the pinned DeepSeek-V4.1 checkpoint encoder."""

from pathlib import Path

import pytest
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE = (
    Path(__file__).parents[1] / "src/vllm_tt_plugin/chat_templates/deepseek_v41.jinja"
)
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
SYS = "<｜System｜>"
USER = "<｜User｜>"
ASST = "<｜Assistant｜>"
RANGE = " (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
S = {"role": "system", "content": "You are terse."}
U1 = {"role": "user", "content": "First question?"}
U2 = {"role": "user", "content": "Second part."}
U3 = {"role": "user", "content": "Third question?"}
A1 = {
    "role": "assistant",
    "content": "First answer.",
    "reasoning_content": "Thought one.",
}
A2 = {"role": "assistant", "content": "Second answer."}
MS = {"role": "system", "content": "Now answer in French."}
LIST = {
    "role": "user",
    "content": [{"type": "text", "text": "Part A"}, {"type": "text", "text": "Part B"}],
}
CASES = {
    "few-shot user turns": (
        [S, U1, U2, U3],
        True,
        "max",
        [
            BOS,
            SYS,
            "Reasoning Effort: 100",
            RANGE,
            "You are terse.",
            USER,
            "First question?\n\nSecond part.\n\nThird question?",
            ASST,
            "<think>",
        ],
    ),
    "chat history": (
        [U1, A1, U3],
        False,
        None,
        [
            BOS,
            USER,
            "First question?",
            ASST,
            "</think>First answer.",
            EOS,
            USER,
            "Third question?",
            ASST,
            "</think>",
        ],
    ),
    "thinking history": (
        [S, U1, A1, U2, A2, U3],
        True,
        None,
        [
            BOS,
            SYS,
            "Reasoning Effort: 75",
            RANGE,
            "You are terse.",
            USER,
            "First question?",
            ASST,
            "</think>First answer.",
            EOS,
            USER,
            "Second part.",
            ASST,
            "</think>Second answer.",
            EOS,
            USER,
            "Third question?",
            ASST,
            "<think>",
        ],
    ),
    "mid-conversation system": (
        [U1, A1, MS, U3],
        True,
        37,
        [
            BOS,
            SYS,
            "Reasoning Effort: 37",
            RANGE,
            USER,
            "First question?",
            ASST,
            "</think>First answer.",
            EOS,
            SYS,
            "Now answer in French.",
            USER,
            "Third question?",
            ASST,
            "<think>",
        ],
    ),
    "list content": (
        [LIST, U3],
        False,
        None,
        [BOS, USER, "Part A\n\nPart B\n\nThird question?", ASST, "</think>"],
    ),
    "assistant continuation": (
        [U1, A1],
        True,
        "high",
        [
            BOS,
            SYS,
            "Reasoning Effort: 75",
            RANGE,
            USER,
            "First question?",
            ASST,
            "<think>Thought one.</think>First answer.",
            EOS,
        ],
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_deepseek_v41_template_matches_encoder(name):
    messages, thinking, effort, expected = CASES[name]
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    kwargs = {"thinking": thinking}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    actual = env.from_string(TEMPLATE.read_text()).render(
        messages=messages, add_generation_prompt=True, **kwargs
    )
    assert actual == "".join(expected)
