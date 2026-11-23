# Tests

Tests follow the component and boundary they exercise. The logprobs pilot uses
this layout; existing tests at the root and under `spec/` remain in place until
their owning area is reorganized.

| Location | Scope |
|---|---|
| [unit/test_logprobs.py](unit/test_logprobs.py) | Public packing helpers, deterministic tensor inputs, shapes, ranks, and ownership |
| [unit/test_logger.py](unit/test_logger.py) | Logger setup, warning deduplication, record context, and stdlib-only startup |
| [integration/test_decode_logprobs.py](integration/test_decode_logprobs.py) | Decode controller and runner cooperating on host logprob payloads |
| [support/logprobs.py](support/logprobs.py) | Ranked-payload test data shared by those two modules |
| [tt/test_logprobs.py](tt/test_logprobs.py) | Logprobs returned by a live server on TT hardware |

Engineering and test-design rules live in
[AGENTS.md](../AGENTS.md#test-design-and-review-evidence). Environment preparation
is documented in [host and TT setup](../AGENTS.md#3-environment-and-installation).

## Run tests

Run from the repository root with the intended virtual environment active.
The examples use the host stub; omit the `PYTHONPATH` assignment when testing
with real `ttnn`.

Run the unit suite without loading ancestor conftests:

```bash
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/unit --confcutdir=tests/unit
```

Run the decode integration tests with the root host fixtures:

```bash
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/integration/test_decode_logprobs.py
```

The normal host-suite command discovers both groups and the existing host tests:

```bash
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/ --ignore=tests/tt
```

See [server testing](../README.md#testing) for device prerequisites and commands.

## Fixture boundaries

[unit/conftest.py](unit/conftest.py) overrides the root
`reset_tt_platform_class_state` fixture so ordinary full-suite collection also
keeps unit tests independent of engine-reset setup. Unit tests still import the
software dependencies of the component under test, including vLLM output types.

Integration tests inherit [the root fixtures](conftest.py), which restore runtime
patches and platform state. Their model-input fixture is local and uses the real
`TTModelInput` and `TTSamplingParams` data classes. Both pilot modules import
their shared ranked payload from `support/logprobs.py`.
