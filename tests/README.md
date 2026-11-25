# Tests

Tests follow the component and boundary they exercise. Logprobs, logging, and
model-output parsers use this layout; existing tests at the root and under
`spec/` remain in place until their owning area is reorganized.

| Location | Scope |
|---|---|
| [unit/test_logprobs.py](unit/test_logprobs.py) | Public packing helpers, deterministic tensor inputs, shapes, ranks, and ownership |
| [unit/test_logger.py](unit/test_logger.py) | Logger setup, warning deduplication, record context, and stdlib-only startup |
| [unit/parsers/test_kolibri1.py](unit/parsers/test_kolibri1.py) | Kolibri thinking switches, reasoning/content extraction, streaming, and input ownership |
| [unit/parsers/test_qwen3_arguments.py](unit/parsers/test_qwen3_arguments.py) | Qwen3 tool-call arguments: unclosed last parameter, literal tags in values, call boundaries, streamed versus complete output, and the streaming fast path |
| [unit/parsers/test_tool_schema.py](unit/parsers/test_tool_schema.py) | Tool-call argument types from `$ref` and `allOf` schemas and string unions |
| [integration/test_parser_registration.py](integration/test_parser_registration.py) | General entry point and vLLM registries: lazy resolution, aliases, repeated setup, and existing-name precedence |
| [integration/test_decode_logprobs.py](integration/test_decode_logprobs.py) | Decode controller and runner cooperating on host logprob payloads |
| [integration/test_tool_call_patches.py](integration/test_tool_call_patches.py) | TT platform hook installing the tool-call parser patches |
| [support/logprobs.py](support/logprobs.py) | Ranked-payload test data shared by those two modules |
| [support/tool_calls.py](support/tool_calls.py) | Character tokenizer and one-shot and streamed `qwen3_coder` parsing for the tool-call tests |
| [support/parser_patches.py](support/parser_patches.py) | Restore of the vLLM parser attributes that the tool-call patches replace |
| [tt/test_logprobs.py](tt/test_logprobs.py) | Logprobs returned by a live server on TT hardware |

Engineering and test-design rules live in
[AGENTS.md](../AGENTS.md#test-design-and-review-evidence). Environment preparation
is documented in [host and TT setup](../AGENTS.md#3-environment-and-installation).

## Run tests

Run from the repository root with the intended virtual environment active.
Use that environment's interpreter explicitly; no task runner is required.
The unit, integration, and doctest commands below use the import-only TTNN stub.
Each `PYTHONPATH` assignment applies only to its command.

Run the unit suite without loading ancestor conftests:

```bash
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/unit --confcutdir=tests/unit
```

Run all integration tests with the root host fixtures:

```bash
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/integration
```

The normal host-suite command discovers both groups and the existing host tests:

```bash
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/ --ignore=tests/tt
```

To run the same host suite against installed `ttnn`, use the prepared tt-metal
environment and ensure its `PYTHONPATH` does not contain `tests/host-stubs`.
Check the import origin before running:

```bash
"$VIRTUAL_ENV/bin/python" -c "import ttnn; print(ttnn.__file__)"
"$VIRTUAL_ENV/bin/python" -m pytest tests/ --ignore=tests/tt
```

Run the executable examples in the exception, logger, logprobs, and parser modules:

```bash
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest --doctest-modules \
  src/vllm_tt_plugin/exceptions.py src/vllm_tt_plugin/logger.py \
  src/vllm_tt_plugin/logprobs.py \
  src/vllm_tt_plugin/parsers
```

To narrow a run, replace the suite directory with a file or node ID, retaining
the command's environment and fixture options. For example:

```bash
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest \
  tests/unit/test_logger.py --confcutdir=tests/unit -q -k warning
```

For device tests, start a TT model server first, use the prepared tt-metal
environment without host stubs, and pass the server's URL and served model name:

```bash
"$VIRTUAL_ENV/bin/python" -m pytest tests/tt -v \
  --tt-server-url=http://localhost:8000 --tt-model-name=MODEL
```

Replace `MODEL` with the actual served name. See [server testing](../README.md#testing)
for model-specific options and [speculative regression tests](tt/spec/README.md)
for their launch configurations, evidence requirements, and driver.

## Fixture boundaries

[unit/conftest.py](unit/conftest.py) overrides the root
`reset_tt_platform_class_state` fixture so ordinary full-suite collection also
keeps unit tests independent of engine-reset setup. Unit tests still import the
software dependencies of the component under test, including vLLM output types.

Integration tests inherit [the root fixtures](conftest.py), which restore runtime
patches and platform state. Decode/logprob integration tests keep their model-input
fixture local and use the real `TTModelInput` and `TTSamplingParams` data classes.
The logprob unit and integration modules import their shared ranked payload from
`support/logprobs.py`.

Parser registration integration tests use the real vLLM managers with copied
eager and lazy registries, restored after each test. They replace model
registration when exercising the general entry point so the check stays focused
on parser activation. Kolibri unit tests use a deterministic tokenizer and real
chat request objects, with no downloaded model files.
