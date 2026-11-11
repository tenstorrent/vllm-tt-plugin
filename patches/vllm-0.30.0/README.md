# vLLM 0.30.0 patches

`docs/install-vllm-tt.sh` applies every `*.patch` in this directory to the
installed vLLM, through `patches/apply-vllm-patches.sh`, right after it installs
vLLM. The helper skips a patch that is already applied, applies with no fuzz,
and fails the install on a conflict. It does nothing for a vLLM version that has
no `patches/vllm-<version>/` directory.

`tests/` holds the vLLM-side tests for these patches, as a patch against the
vLLM v0.30.0 source tree. It is not applied at install time. To run it, check
out vLLM `v0.30.0`, `git apply` both the patch here and the test patch, and run
`tests/parser` and `tests/tool_parsers`.

## 0001-mimo-tool-calling.patch

MiMo-V2.6-Flash tool calling (`--tool-call-parser mimo`). Four fixes:

| Fix | Files | Upstream |
|---|---|---|
| **Untyped schema.** Argument values whose property schema has no direct `type` (`$ref`/`$defs`, `not`, `if/then/else`, `const`, `{}`) are coerced to their natural JSON type instead of coming back as JSON strings. `$defs` is no longer popped from the request's tools. | `tool_parsers/utils.py`, `parser/engine/parser_engine.py` | Not fixed upstream. Our own fix, in the direction of open #50933 and #63385. |
| **Strict schema.** The `mimo` tool parser builds xgrammar's built-in `mimo` structural tag. Before, it used Qwen3-Coder's, whose trigger never matches MiMo output, so `strict: true` constrained nothing. Honours `parallel_tool_calls=false`. Adds `VLLM_TOOL_STRICT_LEVEL` (`auto`, `function`, `parameter`; default `auto`, the 0.30.0 behaviour). | `tool_parsers/mimo_tool_parser.py` (new), `tool_parsers/structural_tag_registry.py`, `tool_parsers/__init__.py`, `envs.py` | Strict half of #58019 (in v0.32.0); #56268 (`--tool-strict-level`, in v0.31.0). |
| **Streaming truncation.** A tool call cut off by `max_tokens` streams valid JSON identical to the non-streamed result, and reports `finish_reason=length`. A repeated parameter keeps its first value. | `parser/qwen3.py`, `parser/engine/parser_engine.py`, `entrypoints/openai/chat_completion/serving.py` | #46303 (in v0.31.0), #59324 (in v0.33.0), open #53739 and #65775. Keeping the partial value in non-streaming is our own. |
| **Newlines.** MiMo parameter values keep their leading and trailing newlines; Qwen3-Coder still trims them. | `parser/mimo.py` (new), `parser/engine/registered_adapters.py`, `tool_parsers/__init__.py` | Parser half of #58019 (in v0.32.0). |

PR numbers are `vllm-project/vllm`. The `mimo` structural tag needs xgrammar
0.2.8 or newer. vLLM 0.30.0 asks for `xgrammar>=0.2.1,<1`, which resolves to
a newer release; with an older one the parser falls back to `qwen_3_coder` and
logs a warning.

**Remove when** vLLM is bumped to a release that contains #53739, #65775 and
an untyped-schema fix (#50933 or #63385). Before that, a version bump needs this
patch rebased onto the new version: the parts already in that release drop out,
and the rest moves to `patches/vllm-<new version>/`. Without that, the helper
finds no directory for the new version and the fixes silently go away.

The patch adds two files (`vllm/parser/mimo.py`, `vllm/tool_parsers/mimo_tool_parser.py`)
that vLLM's install record does not list, so uninstalling vLLM leaves them behind.
