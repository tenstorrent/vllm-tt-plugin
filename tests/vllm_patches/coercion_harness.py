# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Drive real vLLM tool parsers on synthetic model output, CPU only.

Standalone: imports only `vllm` (no vLLM test helpers), so the tests that use
it run against any installed, patched vLLM.

Each `Fmt` renders one tool call in a parser's native syntax.  A parameter
value is one of:

- `Raw(text)`:   text with no type encoding (Qwen3/MiMo `<parameter=x>5<...>`,
                 DeepSeek `string="false"`, a Gemma4 bare value).
- `Str(text)`:   a value the model explicitly encoded as a string (JSON
                 `"5"`, Gemma4 `<|"|>5<|"|>`, DeepSeek `string="true"`).
- `Typed(v)`:    an explicitly typed JSON value (JSON `5`, DeepSeek
                 `string="false"` with JSON text).

`non_streaming()` goes through the registered tool parser
(`ToolParserManager`), `streaming()` through the parser engine's
`parse_delta` at a given chunk size (in tokens).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager


@dataclass(frozen=True)
class Raw:
    text: str


@dataclass(frozen=True)
class Str:
    text: str


@dataclass(frozen=True)
class Typed:
    value: object


Value = Raw | Str | Typed
_CHAR_BASE = 1_000_000


def _tokenizer(special: dict[str, int]) -> MagicMock:
    inv = {v: k for k, v in special.items()}
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.get_vocab.return_value = dict(special)
    tok.decode.side_effect = lambda ids: "".join(
        inv[i] if i in inv else chr(i - _CHAR_BASE) for i in ids
    )
    tok.all_special_tokens = list(special)
    tok.all_special_ids = list(special.values())
    tok.eos_token_id = None
    tok.bos_token_id = None
    tok.pad_token_id = None
    return tok


def _special(*tokens: str) -> dict[str, int]:
    return {t: 100 + i for i, t in enumerate(tokens)}


@dataclass
class Fmt:
    name: str
    tool_parser: str
    engine: Callable[[], type]
    special: dict[str, int]
    render: Callable[[list[tuple[str, Value]]], str]
    supports: tuple[type, ...]
    # No arg converter: the engine streams the model's raw JSON arguments and
    # only the non-streaming result is schema-corrected (upstream behaviour).
    raw_json_stream: bool = False
    engine_kwargs: dict = field(
        default_factory=lambda: {
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}
        }
    )

    def tokenizer(self) -> MagicMock:
        return _tokenizer(self.special)

    def atoms(self, text: str) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        i = 0
        specials = sorted(self.special.items(), key=lambda kv: -len(kv[0]))
        while i < len(text):
            for tok, tid in specials:
                if text.startswith(tok, i):
                    out.append((tok, tid))
                    i += len(tok)
                    break
            else:
                out.append((text[i], _CHAR_BASE + ord(text[i])))
                i += 1
        return out

    def non_streaming(self, tools: list[dict], text: str) -> str:
        parser = ToolParserManager.get_tool_parser(self.tool_parser)(
            self.tokenizer(), tools
        )
        result = parser.extract_tool_calls(text, request(tools))
        assert result.tools_called, f"{self.name}: no tool call in {text!r}"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "f"
        return result.tool_calls[0].function.arguments

    def streaming_pieces(self, tools: list[dict], text: str, chunk: int) -> list[str]:
        parser = self.engine()(self.tokenizer(), tools, **self.engine_kwargs)
        req = request(tools)
        atoms = self.atoms(text)
        groups = [atoms[i : i + chunk] for i in range(0, len(atoms), chunk)]
        pieces: list[str] = []
        for n, group in enumerate(groups):
            delta = parser.parse_delta(
                "".join(t for t, _ in group),
                [i for _, i in group],
                req,
                prompt_token_ids=[] if n == 0 else None,
                finished=n == len(groups) - 1,
            )
            for tc in (delta.tool_calls if delta else None) or []:
                if tc.function and tc.function.arguments:
                    pieces.append(tc.function.arguments)
        return pieces

    def streaming(self, tools: list[dict], text: str, chunk: int) -> str:
        return "".join(self.streaming_pieces(tools, text, chunk))


def request(tools: list[dict]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice="auto",
    )


def tools_for(parameters: dict) -> list[dict]:
    return [{"type": "function", "function": {"name": "f", "parameters": parameters}}]


def canon(value: object) -> str:
    # json.dumps tells 1 / 1.0 / True / "1" apart, unlike ==.
    return json.dumps(value, sort_keys=True)


# ── Formats ──────────────────────────────────────────────────────────


def _qwen3(params: list[tuple[str, Value]]) -> str:
    body = "".join(f"<parameter={k}>\n{v.text}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function=f>\n{body}</function>\n</tool_call>"


def _mimo(params: list[tuple[str, Value]]) -> str:
    # MiMo's chat template writes calls compactly and values verbatim; the
    # MiMo parser (patches/vllm-0.31.0/until-0.32) keeps values verbatim.
    body = "".join(f"<parameter={k}>{v.text}</parameter>" for k, v in params)
    return f"<tool_call><function=f>{body}</function></tool_call>"


def _gemma4(params: list[tuple[str, Value]]) -> str:
    parts = []
    for k, v in params:
        if isinstance(v, Str):
            parts.append(f'{k}:<|"|>{v.text}<|"|>')
        elif isinstance(v, Raw):
            parts.append(f"{k}:{v.text}")
        else:
            parts.append(f"{k}:{json.dumps(v.value)}")
    return "<|tool_call>call:f{" + ",".join(parts) + "}<tool_call|>"


def _dsml(
    start: str, end: str, invoke: str, invoke_end: str, param: str, param_close: str
) -> Callable[[list[tuple[str, Value]]], str]:
    def render(params: list[tuple[str, Value]]) -> str:
        lines = []
        for k, v in params:
            if isinstance(v, Str):
                is_str, text = "true", v.text
            elif isinstance(v, Raw):
                is_str, text = "false", v.text
            else:
                is_str, text = "false", json.dumps(v.value, ensure_ascii=False)
            lines.append(f'{param} name="{k}" string="{is_str}">{text}{param_close}')
        body = "\n".join(lines)
        return f'{start}\n{invoke}f">\n{body}\n{invoke_end}\n{end}'

    return render


def _kimi(params: list[tuple[str, Value]]) -> str:
    args = {}
    for k, v in params:
        args[k] = v.text if isinstance(v, Str) else v.value
    return (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0"
        f"<|tool_call_argument_begin|>{json.dumps(args, ensure_ascii=False)}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    )


def _qwen3_engine():
    from vllm.parser.qwen3 import Qwen3Parser

    return Qwen3Parser


def _mimo_engine():
    from vllm.parser.mimo import MiMoParser

    return MiMoParser


def _gemma4_engine():
    from vllm.parser.gemma4 import Gemma4Parser

    return Gemma4Parser


def _v32_engine():
    from vllm.parser.deepseek_v32 import DeepSeekV32Parser

    return DeepSeekV32Parser


def _v4_engine():
    from vllm.parser.deepseek_v4 import DeepSeekV4Parser

    return DeepSeekV4Parser


def _v41_engine():
    from vllm.parser.deepseek_v41 import DeepSeekV41Parser

    return DeepSeekV41Parser


def _kimi_engine():
    from vllm.parser.kimi_k2 import KimiK2Parser

    return KimiK2Parser


_D = "｜DSML｜"
QWEN3_SPECIAL = _special("<tool_call>", "</tool_call>", "<think>", "</think>")

FORMATS: dict[str, Fmt] = {
    "qwen3_coder": Fmt(
        "qwen3_coder", "qwen3_coder", _qwen3_engine, QWEN3_SPECIAL, _qwen3, (Raw,)
    ),
    "mimo": Fmt("mimo", "mimo", _mimo_engine, QWEN3_SPECIAL, _mimo, (Raw,)),
    "gemma4": Fmt(
        "gemma4",
        "gemma4",
        _gemma4_engine,
        _special(
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            '<|"|>',
            "<|turn>",
            "<|tool_response>",
        ),
        _gemma4,
        (Str, Raw),
    ),
    "deepseek_v32": Fmt(
        "deepseek_v32",
        "deepseek_v32",
        _v32_engine,
        _special(f"<{_D}function_calls>", f"</{_D}function_calls>"),
        _dsml(
            f"<{_D}function_calls>",
            f"</{_D}function_calls>",
            f'<{_D}invoke name="',
            f"</{_D}invoke>",
            f"<{_D}parameter",
            f"</{_D}parameter>",
        ),
        (Str, Raw, Typed),
    ),
    "deepseek_v4": Fmt(
        "deepseek_v4",
        "deepseek_v4",
        _v4_engine,
        _special("<think>", "</think>", f"<{_D}tool_calls>", f"</{_D}tool_calls>"),
        _dsml(
            f"<{_D}tool_calls>",
            f"</{_D}tool_calls>",
            f'<{_D}invoke name="',
            f"</{_D}invoke>",
            f"<{_D}parameter",
            f"</{_D}parameter>",
        ),
        (Str, Raw, Typed),
    ),
    "deepseek_v41": Fmt(
        "deepseek_v41",
        "deepseek_v41",
        _v41_engine,
        _special("<think>", "</think>", f"<{_D} calls>", f"</{_D} calls>"),
        _dsml(
            f"<{_D} calls>",
            f"</{_D} calls>",
            f'<{_D} invoke name="',
            f"</{_D} invoke>",
            f"<{_D} parameter",
            f"</{_D} parameter>",
        ),
        (Str, Raw, Typed),
    ),
    "kimi_k2": Fmt(
        "kimi_k2",
        "kimi_k2",
        _kimi_engine,
        _special(
            "<think>",
            "</think>",
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_end|>",
            "<|tool_call_argument_begin|>",
        ),
        _kimi,
        (Str, Typed),
        raw_json_stream=True,
    ),
}
RAW_FORMATS = ["qwen3_coder", "mimo"]
EXPLICIT_FORMATS = ["gemma4", "deepseek_v32", "deepseek_v4", "deepseek_v41", "kimi_k2"]
TYPED_FORMATS = ["deepseek_v32", "deepseek_v4", "deepseek_v41", "kimi_k2"]
