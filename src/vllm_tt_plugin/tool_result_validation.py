# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Validation of tool results (``role: "tool"`` messages) of a chat request.

The DeepSeek-V4.1 prompt encoder turns every ``role: "tool"`` message into a
``<tool_result>`` block and silently accepts any ``tool_call_id``: an unknown id, an
id answered twice and a missing id all reach the model, which answers as if the
result were valid. The OpenAI contract (and the tool-calling contract of this
deployment) is that a result must be associable with exactly one call, so such a
request is refused with HTTP 400 before it is rendered (host only, the model and the
device path are not involved).

Rules, applied in message order:

* ``tool_call_id`` must be a non-empty string;
* it must match the ``id`` of a tool call of an EARLIER assistant message of the same
  request (any earlier turn, not only the last one);
* an id is answered at most once. A later assistant message that declares the same
  id again (clients that restart ``call_0`` every round) opens it again.

Results for only SOME of the calls of an assistant message, and results in any
order, are accepted. Disable with ``VLLM_TT_TOOL_RESULT_VALIDATE=0``.
"""

from __future__ import annotations

import functools
import os
from http import HTTPStatus
from typing import Any

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

_PATCHED_ATTR = "_tt_tool_result_wrapped"


def enabled() -> bool:
    return os.environ.get("VLLM_TT_TOOL_RESULT_VALIDATE", "1").strip().lower() not in ("0", "false", "off", "no")


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _call_id(call: Any) -> str | None:
    cid = _get(call, "id")
    if isinstance(cid, str) and cid:
        return cid
    return None


def find_tool_result_error(messages: Any) -> tuple[str, str] | None:
    """Return ``(message, param)`` for the first invalid tool result, ``None`` when all are valid."""
    if not isinstance(messages, (list, tuple)):
        return None
    answered: dict[str, bool] = {}  # declared tool call id -> already answered
    for idx, msg in enumerate(messages):
        role = _get(msg, "role")
        if role == "assistant":
            calls = _get(msg, "tool_calls")
            if calls:
                for call in calls:
                    cid = _call_id(call)
                    if cid is not None:
                        answered[cid] = False
        elif role == "tool":
            param = f"messages[{idx}].tool_call_id"
            cid = _get(msg, "tool_call_id")
            if not isinstance(cid, str) or not cid.strip():
                return (
                    f"messages[{idx}]: a tool message must carry a non-empty tool_call_id "
                    "(the id of the assistant tool call it answers)",
                    param,
                )
            if cid not in answered:
                return (
                    f"messages[{idx}]: tool_call_id {cid!r} does not match any tool call of an earlier assistant message",
                    param,
                )
            if answered[cid]:
                return (
                    f"messages[{idx}]: duplicate result for tool_call_id {cid!r}; each tool call can be answered once",
                    param,
                )
            answered[cid] = True
    return None


def wrap_endpoint(fn):
    """Wrap ``create_chat_completion`` (``async def (self, request, raw_request=None)``)."""
    if getattr(fn, _PATCHED_ATTR, False):
        return fn

    @functools.wraps(fn)
    async def wrapper(self, request, raw_request=None):
        if enabled():
            err = find_tool_result_error(getattr(request, "messages", None))
            if err is not None:
                message, param = err
                logger.info("refused chat request: %s", message)
                return self.create_error_response(
                    message,
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                    param=param,
                )
        return await fn(self, request, raw_request)

    setattr(wrapper, _PATCHED_ATTR, True)
    return wrapper


def install() -> None:
    """Patch the chat serving class once (the API-server process; elsewhere it is a no-op)."""
    module, cls, name = (
        "vllm.entrypoints.openai.chat_completion.serving",
        "OpenAIServingChat",
        "create_chat_completion",
    )
    try:
        mod = __import__(module, fromlist=[cls])
        klass = getattr(mod, cls)
        setattr(klass, name, wrap_endpoint(getattr(klass, name)))
    except Exception as exc:  # pragma: no cover - a vLLM layout without this class
        logger.debug("tool result validation not installed on %s.%s: %s", module, cls, exc)
