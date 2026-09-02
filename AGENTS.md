# Agent Instructions for `vllm-tt-plugin`

These instructions apply to all AI-assisted contributions to
`tenstorrent/vllm-tt-plugin`. Read this file before changing anything in the
repository.

## 1. What this repository is

`vllm-tt-plugin` integrates Tenstorrent hardware into vLLM through vLLM's
standard out-of-tree plugin mechanism. Two entry points, both declared in
`pyproject.toml`, are the whole hook surface:

- `vllm.platform_plugins:tt` maps to `vllm_tt_plugin.entrypoints:platform_plugin`
- `vllm.general_plugins:tt_model_registry` maps to `vllm_tt_plugin.entrypoints:register`

Both gate on `import ttnn`. When `ttnn` is not importable, the plugin declines
to activate and vLLM proceeds as if it were not installed.

The plugin is self-contained by design. Model registration, platform detection,
config validation, scheduling, worker execution, model loading, async decode,
and all three execution modes live here:

- single-process non-DP
- single-process lane-DP
- standard multi-process DP (per-rank mesh and `TT_VISIBLE_DEVICES` discovery)

See `docs/SCHEDULING.md` and `src/vllm_tt_plugin/utils/dp_discovery.py`.
`src/vllm_tt_plugin/launcher.py` is retained MPI / tt-run and is not hooked by
vLLM; it is not a fourth live mode.

| Term | Meaning |
|---|---|
| DP | Data parallelism |
| standard DP | Multi-process, one device mesh per rank |
| lane-DP | Single-process, multiple TT lanes in one engine |
| mesh | Device mesh (`MESH_DEVICE`), not a service mesh |
| `python_env` | tt-metal virtualenv at `$PYTHON_ENV_DIR` |

Do not open TT-feature pull requests against `vllm-project/vllm` or the
deprecated `tenstorrent/vllm` fork. When plugin entry points cannot express
the behaviour, runtime-monkeypatch vLLM from this repo next to the existing
`_install_*` helpers in `src/vllm_tt_plugin/platform.py`. An upstream-general
bug may still be fixed in vLLM core (see the TODO on
`_install_tt_harmony_truncation_patch`). Neighbors for a new patch:
`_install_diffusion_gemma_architecture_patch`,
`_install_tt_harmony_truncation_patch`, `_pin_v1_model_runner`, and the
`_install_block_output_*` functions.

### Repository boundaries, and what changed recently

Three repositories used to carry TT vLLM work. Their roles now:

| Repository | Role today |
|---|---|
| `tenstorrent/vllm-tt-plugin` | The plugin. The only place plugin code lives. |
| `tenstorrent/tt-metal` | The model side. Generators, model classes, and the `model_capabilities` declarations the plugin reads. |
| `tenstorrent/vllm` | **Deprecated as of 2026-08-24** (see `tenstorrent/vllm#476`). It formerly vendored a copy of this plugin under `plugins/vllm-tt-plugin/`. Do not mirror changes there. Do not cite it as a reference for current behaviour. |

Historical pull request titles in this repository carry markers such as
`(port of tenstorrent/vllm#466)`. Those record a mirroring practice that has
ended. Do not start new ones.

## 2. Hard rules

Violating any of these means the change is wrong, not merely imperfect.

1. **Do not land TT features in vLLM core.** Do not open TT-feature pull
   requests against `vllm-project/vllm` or `tenstorrent/vllm`. Extend this
   plugin: add a hook if one exists, otherwise add a runtime monkeypatch next
   to the `_install_*` functions in `platform.py`. An upstream-general bug
   (not TT-specific) may still be fixed in vLLM core. See section 1.
2. **Never use system `python3` or bare `pip`.** The plugin runs inside a
   tt-metal `python_env`. Use that environment's interpreter, or `uv` when
   creating a fresh one. Invoke tests as `$VIRTUAL_ENV/bin/python -m pytest`,
   not a `pytest` on `PATH`.
3. **Never gate new behaviour on a model type allowlist.** Read the model
   class's `model_capabilities` dictionary. Leftover identity gates still exist;
   do not copy them, and do not rip them out in an unrelated pull request. See
   section 6.
4. **Never use `assert` to enforce a new runtime invariant.** Python run with
   `-O` removes asserts. Raise instead, and print the offending values. `assert`
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

The whole unit suite runs on any machine. `ci/host-stubs/ttnn/` supplies an
import-only `ttnn` stand-in whose every device-reaching entry point raises. A
test that starts depending on real hardware fails loudly there rather than
passing against a fake device. Work from the repository root. `uv` must
already be installed. CI also runs this recipe on Python 3.10.

```bash
export VIRTUAL_ENV="$PWD/.venv"
export UV_TORCH_BACKEND=cpu
uv venv --python 3.12 "$VIRTUAL_ENV"
uv pip install torch                # stands in for what tt-metal owns
source docs/install-vllm-tt.sh      # note: sourced, not executed
uv pip install "pytest>=8,<9" pre-commit
PYTHONPATH=ci/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/ --ignore=tests/tt
```

`docs/install-vllm-tt.sh` must be **sourced**, not executed. It uses `return`
on failure because it is designed to run in the caller's shell.

`pytest` is pinned below 9 because pytest 9's `caplog` attaches its own capture
handler to non-propagating loggers and double-captures records (see
`tests/test_logger.py`). That pin is **not** "match tt-metal": tt-metal
`python_env` currently ships pytest 9.x. Plugin CI uses host stubs plus
`pytest<9`. tt-metal's plugin host job uses real `ttnn` plus `.[dev]`. The
`dev` extra pins `pytest>=8,<9` so `uv pip install -e ".[dev]"` cannot float
to 9. Do not install an unpinned pytest over that extra.

### On a Tenstorrent host

Install tt-metal first, activate its environment, then run
`source docs/install-vllm-tt.sh` from the repository root. The relative paths
inside the script assume that working directory.

`tests/tt/` drives a live vLLM server over HTTP. Start a server, then:

```bash
"$VIRTUAL_ENV/bin/python" -m pytest tests/tt -v \
  --tt-server-url http://localhost:8000 --tt-model-name <model>
```

See `tests/tt/conftest.py` for the full option set, including
`--tt-max-num-seqs` and `--tt-chunked-prefill-budget`.

### The install script owns the dependency set

This is the single most fragile part of the repository, and its rationale is
written out in the script and in `docs/vllm-overrides.txt`. Read both before
touching either.

Summary of the constraint: vLLM's PyPI metadata is generated on a CUDA machine,
so `uv` resolves `requirements/cuda.txt` regardless of `VLLM_TARGET_DEVICE`.
The script therefore fetches `requirements/common.txt` for the pinned version
explicitly, installs it, and then installs vLLM itself with `--no-deps` and
`--no-binary vllm`.

`docs/vllm-overrides.txt` forces `numpy<2` and `opencv-python-headless==4.11.0.86`.
`numpy<2` wins because tt-metal fixes it and it cannot be moved from here.

**When bumping the pinned vLLM version, all of the following must change
together:**

- `docs/install-vllm-tt.sh`, the `common.txt` URL (the `v0.x` tag)
- `docs/install-vllm-tt.sh`, the `vllm==0.x` install line. Do **not** touch
  `torchvision==0.x` a few lines above it: that pin tracks tt-metal's
  `requirements-dev.txt`, not vLLM, and the two currently carry the same
  version string by coincidence. A find-and-replace on the version breaks the
  torch pair.
- `README.md`, twice: the stated version, and the `launcher.py` line in the
  package layout. The `compat/vllm-X.Y.Z` guidance below them is generic and
  stays as it is.
- `docs/diffusion-gemma.md`, twice: the pinned-build line and the `gemma4`
  reasoning and tool parsers line.
- `docs/vllm-overrides.txt`, re-checked against the new `common.txt` (upstream
  may have relaxed the opencv floor)

The pin is spread across a sourced shell script that installs with `--no-deps`,
so what resolved and what the docs claim can diverge silently. After
installing, confirm the version rather than trusting the pin:

```bash
python -c "import vllm; print(vllm.__version__)"
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

**Record the result in the pull request body.** This repository has a clean CI
record and that is not an accident: pull requests here carry pre-push evidence,
and the habit of pushing follow-up `lint`, `lint2`, `lint3` fixup commits
stopped when work moved to this repository. Do not restart it.

Every Python source file carries an SPDX header. New ones need it. Markdown in
this repository does not carry SPDX headers.

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
```

## 5. Continuous integration

`.github/workflows/ci.yaml` runs two jobs on every pull request:

- `pre-commit` on `ubuntu-latest`
- `unit-tests` on Python 3.10 and 3.12, which installs through
  `docs/install-vllm-tt.sh`. That job therefore also fails when the documented
  install path rots, which is intentional.

**There is no hardware CI in this repository.** Plugin pull requests do not
dispatch tt-metal `vllm-model-tests`. That workflow is workflow_dispatch, and
its plugin ref defaults to `main`. Device-affecting plugin changes still need a
pasted host, model, and mesh result in Validation, or a named tt-metal job
with `vllm-tt-plugin-ref` set to this branch. The pull request template has a
`Validation` section for exactly this.

Known dead configuration: `ci.yaml` has a `push: branches: [dev]` trigger, but
the default branch is `main` and no `dev` branch exists on the remote. That
trigger never fires.

## 6. Model capabilities: the gating contract

New behaviour is gated on capabilities a model class declares, not on a new
model-type allowlist. `TTPlatform.check_and_update_config` reads a
`model_capabilities` dictionary off the model class at config time, before an
instance exists. Downstream code reads rewritten scheduler and cache fields,
`store_tt_*` / `get_tt_*` on `VllmConfig`, or `TTPlatform` class attributes.
It does not re-read the dict.

Keys currently consumed by `src/vllm_tt_plugin/platform.py`:

| Key | Default if absent | Effect |
|---|---|---|
| `supports_chunked_prefill` | `False` | Whether scheduler-driven chunked prefill is enabled |
| `supports_prefix_caching` | `False` | Whether the vLLM prefix cache may be used |
| `output_tokens_per_step` | `1` | Committed output width per step. `1` is token-at-a-time; `>1` is block-output width |
| `supports_sample_on_device` | `False` | Opt-in for on-device sampling. A requested `sample_on_device_mode` is rejected when `False` |
| `supports_async_decode` | `False` | Whether async scheduling may stay on. When `False`, the platform warns and clears `async_scheduling` |

Absent keys default via `.get`. That is the live contract. Do not add
fail-on-missing for a new key unless the matching tt-metal generators will
declare it in the same change. Raise when a *present* value contradicts
another resolved property (block-output plus prefix caching; DiffusionGemma
without `output_tokens_per_step > 1`) or when an operator request is
unsupported. Copy those existing `ValueError` messages.

Leftover identity gates still exist. Do not copy them for new work, and do
not remove them in an unrelated pull request:

- lane-DP folding: Galaxy generator version plus `model_class.__name__ == "GptOssForCausalLM"`
- GPT-OSS top-K logprobs: `hf_config.model_type == "gpt_oss"`
- hybrid KV: `get_kv_cache_spec` on the model class, not `model_capabilities`

Two more contracts are methods, not dict keys: hybrid KV opt-in is
`get_kv_cache_spec`; block-output lifecycle is `release_request` and
`release_persistent_capture`.

The model classes and their capability declarations live in **tt-metal**, under
`models.tt_transformers.tt.generator_vllm` and the per-demo generators such as
`models.demos.llama3_70b_galaxy.tt.generator_vllm`.

Consequence: a plugin change that consumes a new capability needs a matching
tt-metal change that declares it. Those two land as a pair. Say so in both pull
request bodies and cross-link them. A plugin pull request that reads a
capability no tt-metal model declares is incomplete.

## 7. What reviewers in this repository actually enforce

These are the recurring review themes. Check your own diff against them before
opening a pull request.

**Dead code and leftovers.** If the change makes something unreachable, delete
it in the same pull request. Recurring reviewer comments: "this is unused
right?", "is this used anywhere? not seeing it", "if X is never used anymore,
should delete the file", "this should never happen right? can we remove the
guardrail if unnecessary".

**Comments that state the wrong invariant.** A comment must describe the actual
mechanism, not the intended one. This is enforced precisely here. Example of the
standard being applied: "This comment is slightly misleading. Base scheduler not
adding `skipped_waiting` to a decode batch relies on us hiding `skipped_waiting`
when scheduling decodes, not on `skipped_waiting` implying prefill intent."
A comment that is approximately true is a defect.

**Scope creep.** Every hunk must be explained by the pull request title. This is
the most consistent complaint on large changes. A silently changed constant is
the classic case: a review caught `trace_region_size` moving inside a change
described as a config-mechanism rename. If a constant changes, say why in the
body.

**Duplication.** A new code path that mirrors an existing one draws a
de-duplication request. Prefer extending the existing path.

**Fail loudly, do not recover.** Where an invariant is broken, raise with a
message that prints the offending values. Do not add fallback logic that
continues in a degraded state. Using `assert` for a new runtime invariant is
rejected because `-O` removes asserts. pytest `assert` is required. Existing
production asserts are debt; do not convert them here.

**File size and layering.** `model_runner.py` and `platform.py` are already
thousands of lines. The predecessor repository accumulated a multi-thousand-line
file and drew "we really need to do something about this before it becomes
completely unmaintainable". Adding a new conditional branch to either file is a
prompt to split into an existing neighbor (`config.py`, `scheduler.py`,
`lane_scheduler.py`, `model_registry.py`) rather than grow the file.

**Tests: cover the mechanism, not the symptom.** Do not add a test that freezes
one error string or one call shape. Add a host test that exercises the
mechanism that failed and will keep exercising it. A change ships with a host
test that would fail on the old code, or names the existing test that already
exercises that mechanism. "Unsure" is not a skip.

## 8. Testing notes that will bite you

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
  `always_compat_sampling` is assigned at runtime inside
  `_apply_check_and_update_config`, not declared as a `ClassVar` on the class
  body.

If you add a new class-level attribute to `TTPlatform`, add its name to
`_TT_PLATFORM_CONFIG_ATTRS`, or you will leak state into unrelated tests.

The same autouse fixture, `reset_tt_platform_class_state` in
`tests/conftest.py`, also saves and restores the patches `platform.py`
installs: `InputProcessor.process_inputs`,
`AsyncLLM._add_streaming_input_request`, `EngineCore.reset_prefix_cache`,
`EngineCore.pause_scheduler`, `EngineCoreProc.pause_scheduler`, and the
`_tt_original_*` markers. A new `_install_*_patch` must extend that restore
set. A leaked wrapper raises in unrelated tests.

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
  standard TT DP.` The history contains a second style using a module prefix
  such as `platform:`. Follow the template's bracket form for new work.
- **Fill in the template.** `.github/pull_request_template.md` asks for TL;DR,
  Details, Related Artifacts, and Validation. The Validation section is where
  the commands and their results go, including hardware, model, and device-mesh
  configuration for device tests.
- **One logical change per pull request.** Small pull requests here merge fast
  and with little friction; large ones stall. Split rather than bundle.
- **Link the tt-metal counterpart** when the change pairs with one, in both
  directions.
- **Branch naming.** `<user>/<short-description>`, dashes not underscores.
- **Reviewers are assigned by** `.github/CODEOWNERS`.

### Before opening a pull request

Check that the work is not already being done:

```bash
gh pr list --repo tenstorrent/vllm-tt-plugin --state open --search "<area keywords>"
gh issue view <issue_number> --repo tenstorrent/vllm-tt-plugin --comments
```

If an open pull request already addresses the same thing, do not open another.
If your approach differs materially, say how in a comment on the existing one
rather than opening a competing pull request. Two undocumented pull requests
with the same goal read as low-effort automation and get treated that way.

## 10. AI-assisted contributions

- A human submitter must understand and defend the change end to end. Pure
  agent-authored pull requests are not acceptable.
- The submitter reviews every changed line and runs the relevant tests.
- State in the pull request body that AI assistance was used, and record the
  test commands and their results.
- Add attribution trailers to commits. Example:

```text
Co-authored-by: Claude
Signed-off-by: Your Name <your.email@example.com>
```

### Fail-closed

If the requested work is a duplicate, is trivial busywork, or is a TT-feature
change against `vllm-project/vllm` or `tenstorrent/vllm`, do not proceed.
Return a short statement of what is blocking and why. A new runtime monkeypatch
in this plugin is in bounds; see section 1.

## 11. Editing these instructions

This file is the agent source of truth. It wins over `CONTRIBUTING.md` where
they disagree (including `git rebase -i` / force-push guidance there).

`CLAUDE.md` exists only because Claude Code reads that name and not
`AGENTS.md`. It is a one-line `@AGENTS.md` import. Put instructions here, never
there, or the two toolchains drift.

- Keep it lean. Put operator docs in `docs/` and `README.md`; do not duplicate
  them here.
- Do not grow this file with review lore or one-off anecdotes.
- A new hard rule needs a failure mode an agent actually hits, not a preference.
- Verify every concrete claim against the code in the same change.
