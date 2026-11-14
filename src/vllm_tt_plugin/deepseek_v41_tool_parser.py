# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""OpenAI tool-call transport for the checkpoint's V4.1 DSML grammar."""

import json
import uuid

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser

from vllm_tt_plugin.deepseek_v41_dsml import (
    END,
    START,
    parse_calls,
    safe_content_prefix,
)

logger = init_logger(__name__)


class DeepSeekV41ToolParser(ToolParser):
    # The model emits DSML, not guided-decoding JSON. TT's sampler does not
    # implement a structural grammar mask; do not silently request one.
    supports_required_and_named = False
    structural_tag_model = None

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._content_sent = 0
        self._calls_sent = False

    def adjust_request(self, request):
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(self, model_output, request):
        try:
            content, calls = parse_calls(model_output)
        except ValueError:
            logger.exception("Invalid DeepSeek-V4.1 DSML completion")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            content=content,
            tool_calls=[
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=name, arguments=json.dumps(arguments, ensure_ascii=False)
                    ),
                )
                for name, arguments in calls
            ],
        )

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        if not previous_text:
            self._content_sent, self._calls_sent = 0, False
        prefix = safe_content_prefix(current_text)
        content = prefix[self._content_sent :]
        self._content_sent = len(prefix)
        deltas = []
        if not self._calls_sent and START in current_text and END in current_text:
            _, calls = parse_calls(current_text)
            for index, (name, arguments) in enumerate(calls):
                deltas.append(
                    DeltaToolCall(
                        index=index,
                        id="call_" + uuid.uuid4().hex[:24],
                        type="function",
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )
            self._calls_sent = True
        if content or deltas:
            return DeltaMessage(content=content or None, tool_calls=deltas or None)
        return None
