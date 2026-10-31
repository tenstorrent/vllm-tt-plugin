# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Pre-stream validation for the OpenAI chat and completion endpoints.

The TT platform refuses a request it cannot serve faithfully in
``TTPlatform.validate_request`` (``VLLMValidationError``). For a non-streaming
request vLLM turns that into an HTTP 400. For a streaming request the engine call
happens lazily inside the SSE generator, whose ``except Exception`` yields a single
``data: {"error": ...}`` event and ``[DONE]``: the client has already received a 200.

The wrapper installed here awaits the endpoint's own call and, for a streaming
result, reads the generator's FIRST event before handing it to the HTTP layer. The
engine call has been made by then, so a validation error shows up as that first
event; it is turned into the same ``ErrorResponse`` (HTTP status = the error's
code) the non-streaming path returns. A good stream pays nothing: its first event
(the role chunk) is only emitted after the first output anyway. Disable with
``VLLM_TT_PRESTREAM_VALIDATE=0``.
"""

from __future__ import annotations

import functools
import json
import os
from http import HTTPStatus
from typing import Any, AsyncGenerator

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

_PATCHED_ATTR = "_tt_prestream_wrapped"


def enabled() -> bool:
    return os.environ.get("VLLM_TT_PRESTREAM_VALIDATE", "1") != "0"


def parse_error_event(event: Any) -> dict | None:
    """The ``error`` object of a streamed ``data: {"error": ...}`` event with a 4xx code, else None."""
    if isinstance(event, bytes):
        try:
            event = event.decode()
        except UnicodeDecodeError:
            return None
    if not isinstance(event, str) or not event.startswith("data:"):
        return None
    payload = event[len("data:") :].strip()
    if not payload.startswith("{"):
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    err = obj.get("error") if isinstance(obj, dict) else None
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if not isinstance(code, int) or not 400 <= code < 500:
        return None
    return err


async def _replay(first: Any, rest: AsyncGenerator) -> AsyncGenerator:
    try:
        yield first
        async for item in rest:
            yield item
    finally:
        aclose = getattr(rest, "aclose", None)
        if aclose is not None:
            await aclose()


async def peek_stream(owner: Any, result: Any) -> Any:
    """``result`` unchanged unless it is a stream whose first event is a client error: then the ``ErrorResponse`` for it."""
    if not hasattr(result, "__aiter__"):
        return result
    it = result.__aiter__()
    try:
        first = await it.__anext__()
    except StopAsyncIteration:
        return _replay_empty()
    err = parse_error_event(first)
    if err is None:
        return _replay(first, it)
    aclose = getattr(it, "aclose", None)
    if aclose is not None:
        await aclose()
    code = int(err["code"])
    logger.info("pre-stream validation: HTTP %s for a streaming request: %s", code, err.get("message"))
    try:
        status = HTTPStatus(code)
    except ValueError:
        status = HTTPStatus.BAD_REQUEST
    return owner.create_error_response(
        message=str(err.get("message", "")),
        err_type=str(err.get("type") or "BadRequestError"),
        status_code=status,
        param=err.get("param"),
    )


async def _replay_empty() -> AsyncGenerator:
    return
    yield  # pragma: no cover


def wrap_endpoint(fn):
    """Wrap ``create_chat_completion`` / ``create_completion`` (``async def (self, request, raw_request=None)``)."""
    if getattr(fn, _PATCHED_ATTR, False):
        return fn

    @functools.wraps(fn)
    async def wrapper(self, request, raw_request=None):
        result = await fn(self, request, raw_request)
        if not enabled() or not getattr(request, "stream", False):
            return result
        return await peek_stream(self, result)

    setattr(wrapper, _PATCHED_ATTR, True)
    return wrapper


def install() -> None:
    """Patch the OpenAI serving classes once, where they import (the API-server process; elsewhere it is a no-op)."""
    for module, cls, name in (
        ("vllm.entrypoints.openai.chat_completion.serving", "OpenAIServingChat", "create_chat_completion"),
        ("vllm.entrypoints.openai.completion.serving", "OpenAIServingCompletion", "create_completion"),
    ):
        try:
            mod = __import__(module, fromlist=[cls])
            klass = getattr(mod, cls)
            setattr(klass, name, wrap_endpoint(getattr(klass, name)))
        except Exception as exc:  # pragma: no cover - a vLLM layout without these classes
            logger.debug("pre-stream validation not installed on %s.%s: %s", module, cls, exc)
