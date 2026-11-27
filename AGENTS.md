# Agent Instructions for `vllm-tt-plugin`

These instructions apply to all AI-assisted contributions to
`tenstorrent/vllm-tt-plugin`. Read this file before changing anything in the
repository. Repository-owned agent instructions, skills, and prompts must
follow these rules too.

## Before making a change

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the task-specific references
   below. Follow their links as needed to understand the affected contract.
2. Inspect the implementation, callers, and existing tests. Establish the
   current behavior from code; do not treat nearby code as a design standard.
3. Identify the responsibility being changed, the behavior to preserve or
   correct, and the validation that will demonstrate it. Keep the change small
   enough for a human to follow from its entry point to its result.
4. Update the implementation, docstrings, examples, and affected reference
   documentation together. Report the checks actually run and any gaps.

| Task | Required reference |
|---|---|
| Model registration, construction, KV storage, execution, lifecycle | [Model interface](docs/MODEL_INTERFACE.md) and its relevant linked contracts |
| Capability declarations or consumers | [Model capabilities](docs/MODEL_CAPABILITIES.md) and section 6 below |
| Scheduling or execution modes | [Scheduling](docs/SCHEDULING.md) |
| Installation or dependency changes | Section 3 below, [install script](docs/install-vllm-tt.sh), and [dependency overrides](docs/vllm-overrides.txt) |
| vLLM compatibility assessment | [Upgrade assessment prompt](.github/prompts/vllm-upgrade-assessment.prompt.md) |
| Tests or test setup | Sections 3, 5, 7, and 8 below, plus the applicable `conftest.py` |

The engineering standards in section 7 apply to new and substantively changed
code, including internal helpers. A mechanical edit does not require rewriting
unrelated code. Existing debt is not permission to repeat it.

### Recommended skills for Claude and Codex

Installing [`tt-buddy`](https://github.com/tenstorrent/skills/tree/main/plugins/tt-buddy)
is **strongly recommended** for sustained plugin work, but is optional. Follow
the upstream installation instructions for
[Claude Code](https://github.com/tenstorrent/skills#claude-code) or
[Codex](https://github.com/tenstorrent/skills#codex).

When available, use its plugin recipes for setup, serving, tests, and benchmarks;
`learn` for source-based research and reuse of current notes; and `note` for
meaningful findings, decisions, attempts, and results across sessions. These
instructions and the current repository contracts take precedence over skills.

The upstream [host-test recipe](https://github.com/tenstorrent/skills/blob/fb21b80/plugins/tt-buddy/recipes/vllm-tt-plugin/test.md#L9)
references `ci/host-stubs`; use this repository's `tests/host-stubs` path and
the command in section 3.

## 1. What this repository is

`vllm-tt-plugin` integrates Tenstorrent hardware into vLLM through vLLM's
standard out-of-tree plugin mechanism. Two entry points, both declared in
`pyproject.toml`, activate platform detection and registration:

- `vllm.platform_plugins:tt` maps to `vllm_tt_plugin.entrypoints:platform_plugin`
- `vllm.general_plugins:tt_model_registry` maps to `vllm_tt_plugin.entrypoints:register`

Platform detection and model registration gate on `import ttnn`; without it,
the plugin does not select the TT platform or register TT models.

The plugin is self-contained by design. Model registration, platform detection,
config validation, scheduling, worker execution, model loading, async decode,
and all three (data-parallelism) execution modes live here.

### Repository boundaries

| Repository | Role today |
|---|---|
| `tenstorrent/vllm-tt-plugin` | The plugin. The only place plugin code lives. |
| `tenstorrent/tt-metal` | TTNN ops and the (legacy) model side. Generators, model classes, and the `model_capabilities` declarations the plugin reads. |
| `tenstorrent/tt-transformers` | Newly established package for model implementations. |
| `tenstorrent/vllm` | Deprecated fork. Do not mirror changes there or use it as a reference for current behavior. |
| `vllm-project/vllm` | Upstream vLLM. |

## 2. Hard rules

Violating any of these means the change is wrong, not merely imperfect.

1. **Do not land TT features in vLLM core.** Do not open TT-feature pull
   requests against `vllm-project/vllm` or `tenstorrent/vllm`. Extend this
   plugin: add a hook if one exists, otherwise add a runtime monkeypatch next
   to the `_install_*` functions in `src/vllm_tt_plugin/platform.py`. An
   upstream-general bug may still be fixed in vLLM core; see the TODO on
   `_install_tt_harmony_truncation_patch` for an example.
2. **Never use system `python3` or bare `pip`.** The plugin runs inside a
   tt-metal `python_env`. Use that environment's interpreter, or `uv` when
   creating a fresh one. Invoke tests as `$VIRTUAL_ENV/bin/python -m pytest`,
   not a `pytest` on `PATH`.
3. **Never gate new behaviour on a model type allowlist.** Read the model
   class's `model_capabilities` dictionary. Leftover identity gates still exist;
   do not copy them, and do not rip them out in an unrelated pull request. See
   section 6.
4. **Never use `assert` to enforce a new runtime invariant.** Python run with
   `-O` removes asserts. Raise instead, and include the offending values. `assert`
   in pytest tests is required. Existing production asserts in `src/` are debt;
   do not convert them in an unrelated pull request. See section 7.
5. **Run `pre-commit run` before every commit.** Agent shells often have no
   git hooks, even when a human has run `pre-commit install`. Do not skip the
   explicit run. See section 4.
6. **Do not commit review documents, plans, or analysis notes to `docs/`.**
   `docs/` holds runtime documentation for operators. Keep working notes
   untracked, or in a personal notes directory outside this repository.

## 3. Environment and installation

There are two environments, and most work needs only the first.

### Host-only, no Tenstorrent hardware

The host suite requires no Tenstorrent hardware. `tests/host-stubs/ttnn/`
supplies an import-only `ttnn` stand-in whose device entry points raise.
Work from the repository root with `uv` installed. vLLM requires Python 3.11
or newer; CI covers Python 3.12.

```bash
export VIRTUAL_ENV="$PWD/.venv"
export UV_TORCH_BACKEND=cpu
uv venv --python 3.12 "$VIRTUAL_ENV"
uv pip install torch                # stands in for what tt-metal owns
source docs/install-vllm-tt.sh      # note: sourced, not executed
uv pip install "pytest>=8,<9" pre-commit
PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/ --ignore=tests/tt
```

`docs/install-vllm-tt.sh` must be **sourced**, not executed. It uses `return`
on failure because it is designed to run in the caller's shell.

The `dev` extra pins `pytest>=8,<9` because pytest 9's `caplog`
double-captures records on non-propagating loggers (see
`tests/unit/test_logger.py`). This is a plugin test requirement, independent of
tt-metal's pytest version. CI installs the extra; do not install an unpinned
pytest over it.

See the [test commands](tests/README.md#run-tests) for isolated unit tests,
integration tests, both host-suite modes, doctests, and device tests.

### On a Tenstorrent host

Install tt-metal first, activate its environment, then run
`source docs/install-vllm-tt.sh` from the repository root. The relative paths
inside the script assume that working directory.

`tests/tt/` drives a live vLLM server over HTTP. Start a server, then:

```bash
"$VIRTUAL_ENV/bin/python" -m pytest tests/tt -v \
  --tt-server-url=http://localhost:8000 --tt-model-name=MODEL
```

Replace `MODEL` with the served model name. Use the real TTNN environment
without host stubs for device tests. See
`tests/tt/conftest.py` for the full option set, including
`--tt-max-num-seqs` and `--tt-chunked-prefill-budget`.

### The install script owns the dependency set

Read `docs/install-vllm-tt.sh` and `docs/vllm-overrides.txt` before changing
either. The script owns dependency resolution for vLLM.

Summary of the constraint: vLLM's PyPI metadata is generated on a CUDA machine,
so `uv` resolves `requirements/cuda.txt` regardless of `VLLM_TARGET_DEVICE`.
The script therefore fetches `requirements/common.txt` for the pinned version
explicitly, installs it, and then installs vLLM itself with `--no-deps` and
`--no-binary vllm`.

`docs/vllm-overrides.txt` forces `numpy<2` and `opencv-python-headless==4.11.0.86`,
and sets the whole `transformers` range of the install.
`numpy<2` wins because tt-metal fixes it and it cannot be moved from here.

**When bumping the pinned vLLM version, all of the following must change
together:**

- `docs/install-vllm-tt.sh`, the `common.txt` URL (the `v0.x` tag)
- `docs/install-vllm-tt.sh`, the `vllm==0.x` install line. Do **not** touch
  `torchvision==0.x` line as part of a vLLM version replacement: that pin
  tracks tt-metal's `requirements-dev.txt` and must match its torch pair.
- `README.md`: the stated version and the `launcher.py` package-layout note.
  Leave generic `compat/vllm-X.Y.Z` guidance generic.
- `docs/diffusion-gemma.md`: the pinned-build version and any affected
  reasoning/tool parser compatibility guidance.
- `docs/vllm-overrides.txt`: re-check every override against the new
  `common.txt` and the tt-metal environment, including version ceilings.

The pin is spread across a sourced shell script that installs with `--no-deps`,
so what resolved and what the docs claim can diverge silently. After
installing, confirm the version rather than trusting the pin:

```bash
"$VIRTUAL_ENV/bin/python" -c "import vllm; print(vllm.__version__)"
```

## 4. Linting, formatting and pre-commit

`.pre-commit-config.yaml` runs `trailing-whitespace`, `end-of-file-fixer`,
`check-yaml`, `check-toml`, `check-merge-conflict`, `ruff-check --fix`, and
`ruff-format`.

`ruff` is pinned to `0.14.0` in both `.pre-commit-config.yaml` and the `dev`
extra in `pyproject.toml`. `pytest>=8,<9` is pinned in that extra for the
reason in section 3. Keep those pins in step. `README.md` tells humans to
`pre-commit install` so hooks run on `git commit`; still run `pre-commit run`
explicitly, because agent environments often have no hooks.

Lint rules are aligned to upstream vLLM and are declared in `pyproject.toml`
under `[tool.ruff.lint]`.

```bash
pre-commit run              # staged files
pre-commit run --all-files
```

Record the command and result in the pull request's Validation section. Resolve
hook failures before committing.

Every Python source file carries an SPDX header. New ones need it. Markdown in
this repository does not carry SPDX headers.

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
```

## 5. Continuous integration

[Lint and host tests](.github/workflows/lint-and-host-tests.yaml) runs for
non-draft pull requests into `main` and `release/*`. Converting to draft
cancels an in-progress run. `required-checks` gates on `pre-commit` and all
`host-tests-stubbed` legs; require that aggregate check in branch rules.

The [stubbed host workflow](.github/workflows/host-tests-stubbed.yaml) tests
Python 3.12 through the documented install script. The release watcher
also uses it for compatibility checks. `host-tests-ttnn` tests real `ttnn`
Ubuntu 24.04 release images without hardware and is advisory. Its tag
resolution is independent of the required stubbed job. Both install pytest
through `.[dev]`.

CI resolves torch from tt-metal's `tt_metal/python_env/requirements-dev.txt`
using [.github/scripts/tt-metal-torch-pin.sh](.github/scripts/tt-metal-torch-pin.sh):
the stable tag for stubbed tests and each tested tag for real-`ttnn` tests.
Read the workflows for matrix and dispatch details.

**There is no hardware CI in this repository.** Plugin pull requests do not
dispatch tt-metal `vllm-model-tests`. That workflow is workflow_dispatch, and
its plugin ref defaults to `main`. Device-affecting plugin changes still need a
pasted host, model, and mesh result in Validation, or a named tt-metal job
with `vllm-tt-plugin-ref` set to this branch. The pull request template has a
`Validation` section for exactly this.

Inactivity automation is defined in
[close-stale.yaml](.github/workflows/close-stale.yaml) and
[remove-stale-branches.yaml](.github/workflows/remove-stale-branches.yaml).
Consult those files for timers, exemptions, and whether deletion is enabled.

## 6. Model interface and capabilities

Before changing model registration, construction, KV allocation, forward calls,
or lifecycle hooks, read [docs/MODEL_INTERFACE.md](docs/MODEL_INTERFACE.md)
and the relevant linked references. Update the affected interface pages in the
same PR when changing model-facing arguments, return values, capability
consumers, or state ownership. Keep API reference details in `docs/`; keep
agent contribution rules in `AGENTS.md`.

New behaviour is gated on capabilities a model class declares, not on a new
model-type allowlist. `TTPlatform.check_and_update_config` reads a
`model_capabilities` dictionary off the model class at config time, before an
instance exists. The authoritative key inventory, absent-value defaults,
validation rules, and `SpecPlan` fields are in
[docs/MODEL_CAPABILITIES.md](docs/MODEL_CAPABILITIES.md). Update that reference
when changing a capability consumer or a model declaration.

Downstream code reads rewritten scheduler and cache fields,
`store_tt_*` / `get_tt_*` on `VllmConfig`, or `TTPlatform` class attributes.
Some consumers also read `model_capabilities` directly: `TTWorker.init_device`
uses `fabric_config`, `TTModelRunner` checks sampling and hidden-state
declarations, and `TTAsyncDecodeController.plan_decode_reload` checks
`supports_async_decode`. Do not assume config-time resolution eliminates all
runtime reads. Keep class and instance declarations consistent.

Absent keys default via `.get`. That is the live contract. Do not add
fail-on-missing for a new key unless the matching tt-metal generators will
declare it in the same change. Raise when a *present* value contradicts
another resolved property (block-output plus prefix caching; DiffusionGemma
without `output_tokens_per_step > 1`) or when an operator request is
unsupported. Use the existing validation conventions and include the
contradictory values in the error.

Leftover identity gates still exist. Do not copy them for new work, and do
not remove them in an unrelated pull request:

- lane-DP folding: GPT-OSS identity check
- GPT-OSS top-K logprobs: `hf_config.model_type == "gpt_oss"`

Other contracts are methods, not dictionary keys: hybrid KV opt-in is
`get_kv_cache_spec`; block-output lifecycle uses `release_request`,
`release_persistent_capture`, and, for adaptive block output with multiple
requests, `note_state_slots_moved`; speculative feasibility is `spec_plan`, a
classmethod specified in [docs/SPEC_DECODE_CONTRACT.md](docs/SPEC_DECODE_CONTRACT.md).
`supports_narrow_decode` belongs to `SpecPlan`, not `model_capabilities`.
`SpecPlan` resource declarations do not yet enforce row or byte budgets.

The model classes and their capability declarations live in **tt-metal**, under
`models.ttt_compat.tt.generator_vllm` and the per-demo generators such as
`models.demos.llama3_70b_galaxy.tt.generator_vllm`. However, `ttt_compat` is a
legacy subpackage and accepts only bug fixes; the new **tenstorrent/tt-transformers**
repository re-implements model building blocks and we should strive to adapt to
it as much as possible. tt-metal's `models/tt_transformers` is deprecated.

Consequence: a plugin change that consumes a new capability needs a matching
tt-metal change that declares it. Those two land as a pair. Say so in both pull
request bodies and cross-link them. A plugin pull request that reads a
capability no tt-metal model declares is incomplete.

### Model capabilities and plugin configuration require human authorization

Model capabilities describe supported model behavior so shared code can choose
compatible execution paths. Environment variables read by the plugin and
options accepted through `--additional-config` also affect plugin behavior.
Each capability, environment variable, and `--additional-config` option adds
interface, branching, testing, and maintenance costs. Keep these interfaces
small and extend them only for demonstrated requirements.

These requirements apply to new model capabilities, environment variables that
the plugin starts reading, and new user-facing JSON keys accepted through
`--additional-config`, including keys nested under `tt`. An environment variable
already used outside the plugin still requires authorization before the plugin
starts reading that environment variable.

1. **Require a specific human request.** An agent must not introduce a model
   capability, start reading another environment variable in the plugin, or
   expose a new `--additional-config` option on its own initiative. A human must
   explicitly request or approve the specific model capability, environment
   variable, or `--additional-config` option before implementation. A general
   request to add a model, support a feature, or fix a failure does not authorize
   a new model capability, environment variable, or `--additional-config` option.

2. **Verify necessity before implementation.** The agent must inspect the
   existing model interface, capability declarations, environment variables read
   by the plugin, `--additional-config` options, and affected callers. The agent
   must explain the required behavior, why existing mechanisms cannot express
   the required behavior, and why a change inside the model or plugin
   implementation is insufficient. Human authorization does not replace this
   check.

3. **Do not hide bugs or missing features.** An agent must not introduce a model
   capability, environment-variable read, or `--additional-config` option to
   bypass a model or plugin bug, exempt a model from required interface behavior,
   suppress a failing test, or present an incomplete implementation as an
   intentional limitation. The agent must fix the model or plugin implementation
   or report the missing implementation directly. A model capability must
   describe a justified support boundary.

4. **Keep the addition minimal.** The agent must use existing mechanisms when
   existing mechanisms satisfy the requirement. The agent must not add
   speculative model capabilities, environment-variable reads, or
   `--additional-config` options for hypothetical future needs. These requirements
   also apply to new capability values and equivalent support checks implemented
   through model-name conditions, attributes, configuration options, or
   environment variables.

5. **Expect additional code-review scrutiny.** Every PR that introduces a
   model capability, starts reading another environment variable in the plugin,
   or exposes a new `--additional-config` option must identify the human request
   or approval and provide the necessity assessment, precise semantics, affected
   models and callers, default behavior, and relevant validation. Reviewers must
   challenge whether the model capability, environment variable, or
   `--additional-config` option belongs in the shared interface and whether the
   added complexity is justified. Missing authorization or an unsupported
   necessity claim blocks acceptance.

## 7. Engineering standards

### Design direction and reference examples

Favor clear module and class hierarchies, compact names that use their context,
small cohesive files, and implementations a reader can follow from entry point
to result. Apply the same direction to tests: focused scenarios, visible setup
and assertions, and purposeful fixtures and helpers. Keep code, abstractions,
comments, and test scaffolding proportional to the behavior they explain or
verify. Prefer direct control flow and remove duplication and obsolete code
while preserving explicit contracts and meaningful coverage.

Use these implementations and tests as worked examples of the standards below.
Adapt their patterns to the responsibility and actual callers of the code being
changed.

| Area | Implementation | Tests |
|---|---|---|
| Logprobs | [logprobs.py](src/vllm_tt_plugin/logprobs.py) | [Unit](tests/unit/test_logprobs.py), [decode integration](tests/integration/test_decode_logprobs.py) |
| Logging | [logger.py](src/vllm_tt_plugin/logger.py) | [Unit](tests/unit/test_logger.py) |
| Parsers | [Kolibri adapter](src/vllm_tt_plugin/parsers/kolibri1.py), [registration](src/vllm_tt_plugin/parsers/registration.py) | [Unit](tests/unit/parsers/test_kolibri1.py), [registration integration](tests/integration/test_parser_registration.py) |

### Responsibilities, state, and dependencies

- Give each module one explainable responsibility. Its module docstring must
  state its purpose, principal entry points, and relevant boundaries. Organize
  its contents so a reader can follow those entry points into their helpers.
  Keep files compact; split a growing file along responsibility boundaries
  when its size obscures that flow, keeping related behavior together.
- Order class members by their role: construction (`__new__`, `__init__`,
  `__post_init__`, and alternative-constructor classmethods), primary public
  attributes, properties, and methods, secondary public members, then protected
  and private helpers. Place helper staticmethods at the bottom. Keep field
  declarations together where required by Python or dataclasses.
- Use functions for operations and classes for cohesive state, identity, or
  lifecycle behavior. Introduce an abstraction only when actual callers need
  a responsibility that existing objects cannot express clearly. Do not add
  wrappers, base classes, or delegation layers just to follow a pattern.
- Give mutable state an explicit owner. Document who initializes, mutates, and
  releases it, its lifetime, and any ordering or concurrency constraints.
  Distinguish borrowed and owned buffers where that affects correctness.
- Keep dependencies directed from orchestration to the operations it uses.
  Extract behavior that can be understood through a clear interface; moving
  coupled code to several files without clarifying its contract is insufficient.
  Prefer an existing responsible module over a miscellaneous helpers module.
- Identify which interfaces callers may rely on and which helpers are internal.
  Check callers before changing either. State which observable behaviors a
  refactor preserves and explain intentional corrections, including changes to
  exceptions, warnings, or logging.
- Reuse existing paths, delete code made unreachable by the change, and keep
  unrelated cleanup separate. Every hunk must serve the stated change; explain
  changed constants and defaults explicitly.

### Types, docstrings, and examples

- Use concise, meaningful names for objects and arguments. Use the enclosing
  module, class, or function to supply context instead of repeating it in every
  name: prefer `from_top_k_logprobs(logprobs, ...)` over
  `from_top_k_logprobs(top_k_logprobs, ...)`. Retain distinctions needed to
  understand the value; avoid cryptic abbreviations.
- Annotate new and substantively changed function and method parameters,
  return values, and non-obvious stored state. Types must describe the runtime
  values callers actually pass. Avoid `Any` or unstructured containers when
  the contract is known; do not introduce a runtime wrapper merely to add types.
- Document every module, class, function or method, and nontrivial variable, including
  internal ones. Each docstring should begin with a one-liner and followed by optional
  details (see the table below). Use [**numpydoc**](https://numpydoc.readthedocs.io/en/latest/format.html): a concise purpose followed by the
  applicable sections below. Omit empty sections and information already clear from the
  signature.

  | Section | Required content when applicable |
  |---|---|
  | `Parameters` | Meaning, constraints, defaults with behavioral significance, and input ownership |
  | `Returns` / `Yields` | Result meaning, ordering, ownership, and sentinel values |
  | `Raises` | Deliberate exception types and the conditions that trigger them |
  | `Attributes` | Caller-visible state and its meaning |
  | `Notes` | Non-obvious algorithm, mutation, lifecycle, or concurrency rules |
  | `Examples` | Minimal use of the principal standalone interfaces |
  | `See Also` | References to owners of the object, related objects, etc. |

- Document tensor shapes, dimension meanings, dtypes, layouts, and units where
  they form part of the contract. Describe mutation and other side effects.
  Do not list every incidental exception a dependency might raise.
- Give principal standalone APIs small, deterministic examples with explicit
  inputs and observable results. Include imports and required setup; do not
  require unrelated engine scaffolding. State hardware prerequisites when
  needed. Run executable examples with doctest or a named test exercising the
  same example, and record the command and result.
- Put durable explanations in docstrings. Keep inline comments concise and
  local to a non-obvious operation, ordering constraint, or invariant. Bug-fix
  history belongs in commits, pull requests, or issues, not comment blocks
  above repaired code. Update inaccurate comments and docstrings with the code.

### Validation, exceptions, and diagnostics

- Validate at the boundary that owns the requirement. Distinguish invalid
  caller input from broken execution state. Raise useful exceptions with the
  relevant offending values; never use runtime `assert` or invent a fallback
  that continues after a broken invariant.
- Reuse an appropriate exception type. Prefer domain exceptions or a hierarchy,
  esp. when callers need that distinction. Preserve the cause when translating
  exceptions with `raise ... from exc`.
- Warn for recoverable conditions that need attention. Log meaningful events
  at the appropriate level with enough context to act. Make the logs concise,
  readable, traceable, and searchable. Avoid duplicate reports, routine per-step
  noise, and expensive formatting when logging is disabled.

### Test design and review evidence

- **Strictly follow test-driven-development best practices.** Do not invent
  post-hoc tests addressing only symptoms to fool yourself and others that your
  implementation is correct.
- Place new or reorganized module unit tests in `tests/unit/test_<module>.py`.
  Tests of collaborating plugin components belong in `tests/integration/`,
  including hardware-free integration tests; device tests belong in `tests/tt/`.
  Keep unaffected legacy tests in place. See the [testing map](tests/README.md)
  for commands and fixture boundaries.
- Test behavior and the mechanism that guarantees it. Private helpers may be
  tested when they own the relevant mechanism; avoid tests that merely repeat
  implementation steps or freeze an incidental call shape or full error string.
- Group related guarantees in focused test files and explain non-obvious intent.
  Keep setup compact and fixtures and helpers close to the cases they serve.
  Parameterize equivalent cases with meaningful IDs; keep distinct behaviors
  readable.
  Test classes are optional, not a required mirror of production classes, but
  preferred when have local fixtures and specific, focused testing scenarios or
  helpers. Give reused or non-obvious fixture and case structures typed, named
  fields so readers can identify each value without tracing tuple positions.
- Keep inputs deterministic unless randomness is under test. Restore mutable
  global state. Share builders through test support modules or fixtures, never
  imports from neighboring test files. Unit setup must avoid unrelated engine,
  platform, or device initialization; inspect inherited fixtures as well.
  Verify isolation both when running unit tests alone and through normal host
  collection. Keep engine-reset fixtures scoped to the integration tests that
  need them; see the [unit fixture override](tests/unit/conftest.py).
- A bug fix needs a regression test that exercises the failed mechanism and
  fails on the old code, or an explanation of how existing coverage already
  exercises it. A behavior-preserving refactor may use retained coverage;
  do not manufacture tests solely to validate a move or rename. Documentation
  changes need applicable link, example, and consistency checks.
- Review findings must identify a concrete violated requirement or reachable
  incorrect behavior, with evidence from the affected code or documentation.
  Separate optional suggestions from defects; a review with no findings is
  valid. Do not demand speculative guards, configuration, or abstractions.

## 8. State and fixture isolation

Resolved TT state lives in two places. Pick the bucket that matches the
lifetime of the value:

- **`VllmConfig.additional_config`** via `store_tt_*` / `get_tt_*` in
  `src/vllm_tt_plugin/config.py`. Use this for anything a scheduler or worker
  must read after copy or pickle (for example `output_tokens_per_step` and
  lane count).
- **`TTPlatform` class attributes.** Process-global. One test that calls
  `check_and_update_config` configures every later test. Current names in
  `tests/conftest.py` `_TT_PLATFORM_CONFIG_ATTRS`:
  `_standard_dp_visible_device_groups`, `_standard_dp_mesh_grids`,
  `sample_on_device_mode`, `always_compat_sampling`, `_tt_vllm_config`.
  `always_compat_sampling` is assigned during config validation, not declared
  on the class body.

If you add a new class-level attribute to `TTPlatform`, add its name to
`_TT_PLATFORM_CONFIG_ATTRS`, or you will leak state into unrelated tests.

The autouse `reset_tt_platform_class_state` fixture in
[tests/conftest.py](tests/conftest.py) also restores runtime patches, their
aliases, and original-function markers. When adding an `_install_*_patch`,
extend its restore set for every modified target and marker. Restoring class
attributes alone does not undo installed wrappers.

**`tests/tt/conftest.py` deliberately neutralizes that fixture by using the
same name.** Those tests drive a server over HTTP and never touch
`TTPlatform` class state. The *host* fixture's import lands while
`vllm.platforms` is still half-initialized by the plugin entry point, which
errors on `cannot import name 'current_platform'`. A new autouse fixture that
imports the plugin must be neutralized in `tests/tt/conftest.py` under the
same name. Host-only tests belong under `tests/`, not `tests/tt/` (plugin CI
runs `tests/ --ignore=tests/tt`).

**Importing the plugin from a conftest at module scope is circular.** Defer
plugin imports into the fixture body, as `tests/conftest.py` does.

## 9. Pull request conventions

- **Title prefix.** Use one of `[bug]`, `[feat]`, `[perf]`, `[ref]`, `[test]`,
  `[ci]`, `[doc]`, `[deps]`. Example: `[bug] Reject invalid device IDs for
  standard TT DP.`
- **Fill in the template.** `.github/pull_request_template.md` asks for TL;DR,
  Details, Related Artifacts, and Validation. The Validation section is where
  the commands and their results go, including hardware, model, and device-mesh
  configuration for device tests.
- **One logical change per pull request.** Split independently reviewable
  changes rather than bundling them.
- **Link the tt-metal or tt-transformers counterpart** when the change pairs with
  one, in both directions.
- **Branch naming.** `<user>/<label>.<short-description>`, dashes not underscores.
- **Commit messages.** Follow the [contribution guidelines](CONTRIBUTING.md#pull-request-guidelines)
  for message content and where validation evidence belongs.
- **Reviewers are assigned by** `.github/CODEOWNERS`.

### Before opening a pull request

Check that the work is not already being done:

```bash
gh discussion list --repo tenstorrent/vllm-tt-plugin --state all --search "<area keywords>"
gh issue view <issue_number> --repo tenstorrent/vllm-tt-plugin --comments
gh pr list --repo tenstorrent/vllm-tt-plugin --state open --search "<area keywords>"
```

If an open pull request already addresses the same thing, do not open another.
Explain a materially different approach to the human submitter so they can
coordinate on the existing pull request. Posting comments requires authorization.

## 10. AI-assisted contributions

- A human submitter must understand and defend the change end to end. Pure
  agent-authored pull requests are not acceptable.
- The submitter reviews every changed line and runs the relevant tests.
- State in the pull request body that AI assistance was used, and record the
  test commands and their results.
- Add accurate attribution trailers to commits. Use the actual assistant and
  human signer's identity; do not copy placeholder identities. Example:

```text
Co-authored-by: <assistant used>
Signed-off-by: Your Name <your.email@example.com>
```

### Fail-closed

If the requested work is a duplicate, is trivial busywork, or is a TT-feature
change against `vllm-project/vllm` or `tenstorrent/vllm`, do not proceed.
Return a short statement of what is blocking and why. A new runtime monkeypatch
in this plugin is in bounds; see section 1.

## 11. Editing these instructions

This file is the repository's agent source of truth. It takes precedence over
conflicting repository prompts, skills, review instructions, and
`CONTRIBUTING.md`. Human instructions for the task still apply.

- Keep `CLAUDE.md` as an adapter importing `@AGENTS.md`; do not put independent
  contribution rules there.
- Give each rule one canonical home. Link to shared rules from agent adapters,
  prompts, and skills instead of copying them. Keep operator instructions in
  `README.md`, runtime and model contracts in `docs/`, and shared contributor
  guidance in `CONTRIBUTING.md`. Before removing guidance from this file,
  ensure its destination contains the full procedure and link to it.
- Give each repository-owned prompt or skill a clear purpose and scope,
  required inputs, expected output, action boundaries, and applicable validation.
  Refer to the shared design, docstring, example, and test rules. Task-specific
  procedures may add necessary detail without weakening or duplicating them.
- Check that referenced files, commands, tools, and skills exist. State required
  setup and working directories. Do not make optional skills an undeclared
  prerequisite or invent commands, results, or tool availability.
- When changing a rule, check agent adapters, review instructions, prompts, and
  generated instructions for contradictory copies. Reconcile affected copies
  within the agreed scope; report remaining conflicts explicitly.
- Keep rules concise and tied to demonstrated failure modes. Remove obsolete
  rules and review anecdotes rather than accumulating them. Verify changed
  concrete claims against the current code and check documentation links.
