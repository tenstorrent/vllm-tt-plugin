# vLLM 0.31.0 patches

`docs/install-vllm-tt.sh` applies the patches listed in `series`, in order, to
the installed vLLM through `patches/apply-vllm-patches.sh`, right after it
installs vLLM. The helper skips what is already applied, applies with no fuzz,
and fails the install on a conflict. It does nothing for a vLLM version without
a `patches/vllm-<version>/series`.

The behaviour tests are plugin host tests in `tests/vllm_patches/`. They run in
CI against the installed, patched vLLM, and skip on a vLLM with no series.

There are two sets, applied `until-0.32/` first so that `common/` applies on
top of the same files 0.32 ships.

## `until-0.32/`: delete when the pin moves to vLLM 0.32

| Patch | What it does | Upstream (`vllm-project/vllm`) |
|---|---|---|
| `0001-mimo-parser-and-grammar.patch` | `mimo` tool and reasoning parsers go through a MiMo parser engine that keeps parameter values verbatim (Qwen3-Coder's engine trims a leading and trailing newline). The tool parser builds xgrammar's builtin `mimo` structural tag (0.31 uses Qwen3-Coder's, whose trigger never matches MiMo output, so `strict: true` constrained nothing) and honours `parallel_tool_calls=false`. It raises `ImportError` on xgrammar older than 0.2.8 instead of falling back to the Qwen3-Coder tag. | #58019, in v0.32.0. `vllm/parser/mimo.py` and the reasoning module are byte-identical to v0.32.0; 0.32 generalises the parallel-call limit. |

The `mimo` tag needs xgrammar 0.2.8 or newer, and vLLM 0.31.0 pins 0.2.7.
`docs/vllm-overrides.txt` overrides it to `xgrammar==0.2.9`, the version the MiMo
endpoint was measured on. vLLM 0.32.0 pins 0.2.8 itself.

`--tool-strict-level` (0.31, #56268) decides which tools get a schema; this
patch decides the grammar format. `auto` is 0.31's default.

**At 0.32:** delete `until-0.32/` and its line in `series`, and delete the
`xgrammar==` line in `docs/vllm-overrides.txt` if 0.32's pin (0.2.8) is enough.

## `common/`: keep after 0.32 (rebase onto the new version)

| Patch | What it does | Upstream |
|---|---|---|
| `0001-schema-coercion.patch` | Tool-argument values whose property schema has no direct `type` (`$ref`/`$defs`, `not`, `if/then/else`, `const`/`enum`, `{}`), and every value under a root-level `$ref`/`allOf`/`anyOf`/`oneOf` `parameters`, are coerced through the schema. Each reading of a value is validated with `jsonschema` against the full property schema, so `5` under `not: {type: integer}` stays `"5"`. The untyped JSON fallback runs only for raw-text formats (`ParserEngineConfig.raw_text_arg_values`: Qwen3-Coder/XML, MiMo, Seed-OSS, GLM-4.7, MiniMax-M2), so a value a parser encoded explicitly as a string (Gemma4 `<|"|>5<|"|>`, DeepSeek `string="true"`, JSON) stays a string under `{}`. `$defs` is no longer popped from the request's tools. | Not fixed upstream (v0.32.0, v0.33.0rc3 and main still default untyped to string); #50933 and #63385 are open. |
| `0002-truncated-tool-arguments.patch` | A tool call cut off by `max_tokens` streams valid JSON equal to the non-streamed arguments (the open trailing parameter is kept, and a flush that cannot extend the streamed prefix closes it). A repeated parameter keeps its first value. The non-streamed response reports `finish_reason="length"`. | Open #53739 and #65775; the finish reason is part of #59324 (v0.33). Streamed `length` is already in 0.31 (#46303), so it is not patched. |
| `0003-xgrammar-pattern-search.patch` | JSON Schema `pattern`/`patternProperties` keep search semantics under xgrammar, which compiles them as whole-string matches (`"^x_"` admitted only the key `x_`). Tool schemas in structural tags and JSON structured outputs. `VLLM_XGRAMMAR_PATTERN_SEARCH=0` turns it off. | xgrammar issue mlc-ai/xgrammar#838; fix PR #925 is open and unreleased. Not in vLLM 0.32. |

Against v0.32.0 (checked with `git apply --check`): `0002` applies as it is
(it touches `vllm/parser/mimo.py`, which `until-0.32/` adds and 0.32 ships
byte-identical). `0001` needs most of its `tool_parsers/utils.py` hunks rebased
(CustomTool support moved their context, #57322), and `0003` needs its context
rebased in `structural_tag_registry.py` and `backend_xgrammar.py`. No semantic
conflict is known.

**Remove a common patch** when the pinned vLLM contains its upstream fix:
`0001` with #50933/#63385 (or equivalent), `0002` with #53739, #65775 and
#59324, `0003` with an xgrammar release that has #925 and a vLLM pin on it.

The patches add files that vLLM's install record does not list, so
uninstalling vLLM leaves them behind.
