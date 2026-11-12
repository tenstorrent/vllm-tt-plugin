# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A non-streamed tool call cut off by max_tokens reports finish_reason
"length" (common, truncated tool arguments; finish-reason part of upstream
#59324, in v0.33). Streaming already reports "length" in 0.31 (#46303)."""

import asyncio
import json

import pytest

from tests.vllm_patches import requires_patched_vllm
from tests.vllm_patches.test_mimo_parsing import CALL, CONTENT, TOOLS, _tokenizer

pytestmark = requires_patched_vllm


def _full_response(text: str, engine_finish_reason: str):
    from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.outputs import CompletionOutput, RequestOutput
    from vllm.parser import ParserManager

    parser_cls = ParserManager.get_parser("mimo", "mimo", enable_auto_tools=True)
    # Only what the non-streaming generator reads; the full server needs a
    # model, a tokenizer and an engine.
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.parser_cls = parser_cls
    serving.enable_auto_tools = True
    serving.response_role = "assistant"
    serving.return_tokens_as_token_ids = False
    serving.enable_prompt_tokens_details = False
    serving.enable_force_include_usage = False
    serving.enable_log_outputs = False
    serving.request_logger = None
    serving.default_sampling_params = {}
    serving._include_reasoning_tokens_details = False
    serving.enable_per_request_metrics = False
    serving.system_fingerprint = None

    request = ChatCompletionRequest(
        model="mimo",
        messages=[{"role": "user", "content": "x"}],
        tools=TOOLS,
        tool_choice="auto",
    )
    parser = parser_cls(
        _tokenizer(), TOOLS, chat_template_kwargs={"enable_thinking": False}
    )
    output = RequestOutput(
        request_id="r",
        prompt="x",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[5] * len(text),
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=engine_finish_reason,
            )
        ],
        finished=True,
    )

    async def results():
        yield output

    response = asyncio.run(
        serving.chat_completion_full_generator(
            request,
            results(),
            "r",
            "mimo",
            [],
            _tokenizer(),
            RequestResponseMetadata(request_id="r", model_name="mimo"),
            parser=parser,
        )
    )
    return response.choices[0]


@pytest.mark.parametrize(
    ("text", "engine_finish_reason", "finish_reason", "args"),
    [
        (CALL, "stop", "tool_calls", {"path": "notes.txt", "content": CONTENT}),
        (
            CALL[: CALL.index("ünïcødé")],
            "length",
            "length",
            {"path": "notes.txt", "content": CONTENT[: CONTENT.index("ünïcødé")]},
        ),
    ],
    ids=["complete", "cut_by_max_tokens"],
)
def test_non_streamed_finish_reason(text, engine_finish_reason, finish_reason, args):
    choice = _full_response(text, engine_finish_reason)
    assert choice.finish_reason == finish_reason
    (call,) = choice.message.tool_calls
    parsed = json.loads(call.function.arguments)
    assert {k: parsed[k] for k in args} == args
