# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests of the pre-stream validation wrapper (vllm_tt_plugin.prestream)."""

import asyncio
import json
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import prestream


def _event(code, message="bad", typ="BadRequestError"):
    err = {"error": {"message": message, "type": typ, "param": None, "code": code}}
    return f"data: {json.dumps(err)}\n\n"


class _Owner:
    def create_error_response(self, message, err_type, status_code, param=None):
        return SimpleNamespace(message=message, err_type=err_type, status_code=status_code, param=param)


async def _stream(*events):
    for e in events:
        yield e


def _run(coro):
    return asyncio.run(coro)


async def _collect(gen):
    return [x async for x in gen]


def test_parse_error_event_only_accepts_client_errors():
    assert prestream.parse_error_event(_event(400))["message"] == "bad"
    assert prestream.parse_error_event(_event(422)) is not None
    assert prestream.parse_error_event(_event(500)) is None
    assert prestream.parse_error_event('data: {"id": "x", "choices": []}\n\n') is None
    assert prestream.parse_error_event("data: [DONE]\n\n") is None
    assert prestream.parse_error_event(b"not an event") is None
    assert prestream.parse_error_event(None) is None


def test_a_first_event_error_becomes_an_http_error():
    out = _run(prestream.peek_stream(_Owner(), _stream(_event(400, "structured_outputs refused"), "data: [DONE]\n\n")))
    assert out.status_code == HTTPStatus.BAD_REQUEST
    assert out.message == "structured_outputs refused"
    assert out.err_type == "BadRequestError"


def test_a_good_stream_is_replayed_unchanged():
    events = ['data: {"id": "1"}\n\n', 'data: {"id": "2"}\n\n', "data: [DONE]\n\n"]

    async def go():
        out = await prestream.peek_stream(_Owner(), _stream(*events))
        return await _collect(out)

    assert _run(go()) == events


def test_an_empty_stream_stays_empty_and_a_plain_result_is_untouched():
    async def go():
        out = await prestream.peek_stream(_Owner(), _stream())
        return await _collect(out)

    assert _run(go()) == []
    plain = object()
    assert _run(prestream.peek_stream(_Owner(), plain)) is plain


def test_server_errors_stay_in_the_stream():
    async def go():
        out = await prestream.peek_stream(_Owner(), _stream(_event(500), "data: [DONE]\n\n"))
        return await _collect(out)

    got = _run(go())
    assert len(got) == 2 and got[0] == _event(500)


def _endpoint(result):
    async def create(self, request, raw_request=None):
        return result

    return create


def test_wrapper_checks_streams_only_and_respects_the_switch(monkeypatch):
    stream_req, plain_req = SimpleNamespace(stream=True), SimpleNamespace(stream=False)
    wrapped = prestream.wrap_endpoint(_endpoint(_stream(_event(400))))
    out = _run(wrapped(_Owner(), stream_req))
    assert out.status_code == HTTPStatus.BAD_REQUEST
    # non-streaming results are the endpoint's own
    sentinel = object()
    assert _run(prestream.wrap_endpoint(_endpoint(sentinel))(_Owner(), plain_req)) is sentinel
    # switch off: the stream goes through untouched
    monkeypatch.setenv("VLLM_TT_PRESTREAM_VALIDATE", "0")
    gen = _stream(_event(400))
    assert _run(prestream.wrap_endpoint(_endpoint(gen))(_Owner(), stream_req)) is gen


def test_wrapping_twice_is_a_no_op():
    once = prestream.wrap_endpoint(_endpoint(None))
    assert prestream.wrap_endpoint(once) is once


def test_install_wraps_the_real_serving_classes_and_builds_a_real_error_response():
    chat = pytest.importorskip("vllm.entrypoints.openai.chat_completion.serving")
    original = chat.OpenAIServingChat.create_chat_completion
    try:
        prestream.install()
        assert getattr(chat.OpenAIServingChat.create_chat_completion, "_tt_prestream_wrapped", False)
        out = _run(prestream.peek_stream(chat.OpenAIServingChat, _stream(_event(400, "nope"))))
        assert type(out).__name__ == "ErrorResponse" and out.error.code == 400 and out.error.message == "nope"
    finally:
        chat.OpenAIServingChat.create_chat_completion = original
