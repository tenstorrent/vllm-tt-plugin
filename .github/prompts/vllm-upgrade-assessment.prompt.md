---
name: "vLLM Upgrade Compatibility Assessment"
description: "Assess (and optionally apply) the changes needed to move vllm-tt-plugin to a new upstream vLLM version. Diffs the plugin's vLLM coupling surface between the installed baseline and a target tag, classifies breakage by severity, and produces an issue-ready findings report plus a change plan. Use when a new vLLM release lands or when bumping the pinned version."
argument-hint: "Target vLLM tag, e.g. v0.28.0 (optional; defaults to the latest upstream release)"
agent: agent
---

# vLLM upgrade compatibility assessment

## Usage

Point an agent at this file with the target version — only TARGET is needed
(BASELINE = the currently installed vLLM):

```
Follow .github/prompts/vllm-upgrade-assessment.prompt.md; assess this plugin against vLLM v0.28.0.
```

In Copilot for VS Code this file is also slash-invocable as `/vllm-upgrade-assessment`.

Determine, and if asked apply, the plugin changes required to target a new upstream
vLLM version. The plugin binds vLLM's **internal v1 APIs**, so every minor bump can
break it — assess the actual source, never infer from release notes (those are a PR
index, not truth). Investigate read-only first; edit only once the plan is clear.

- **TARGET** — the vLLM tag to move to (e.g. `v0.28.0`); default = latest upstream release.
- **BASELINE** — the vLLM version the checkout targets: the pin in
  `docs/install-vllm-tt.sh`, not merely whatever happens to be installed.

If tt-buddy skills are available, use `tt:learn`/`tt:note` and subagents as accelerants;
otherwise the phases below stand alone.

## Phase 0 — Pin the environment
- Activate the tt-metal venv, then resolve the installed checkout (the only authority):
  `python -c "import importlib.util as u; print(u.find_spec('vllm_tt_plugin').origin)"`.
  Confirm it is the tree you intend to assess.
- Derive BASELINE from the repo, not the environment: the pin in `docs/install-vllm-tt.sh`
  — `sed -nE 's/.*vllm==([^[:space:]]+).*/\1/p' docs/install-vllm-tt.sh` (capture the full
  version token, including any `.postN`/`rcN` suffix). That is the version the checkout targets.
- Gate on installed == pin: `python -c "import vllm; print(vllm.__version__)"` (ignore any
  `+empty`/local suffix). On a **mismatch** the environment has drifted and the installed
  tree is the wrong baseline — never diff against it, and never auto-rebuild the venv (that
  mutates the environment and belongs to Phase 5 / explicit consent). Instead:
  - Report all three versions: installed, pin, TARGET.
  - **Interactive:** ask the user which is the intended baseline — the repo pin (the usual
    answer) or the installed version — and proceed only on their answer.
  - **Unattended** (no one to answer, e.g. an agent picking up the tracking issue):
    **hard-stop.** Do no assessment; post a blocking note stating the drift (installed vs
    pin vs TARGET) and that a human must reconcile it (rebuild the pin, or confirm intent)
    before the assessment can run.
- Confirm `import ttnn` works; note the installed source dir (`.../site-packages/vllm/`).

## Phase 1 — Map the coupling surface (the breakage surface)
Re-derive from the plugin source every time — it evolves. Use the greps below and the
**Known surface** snapshot as a starting checklist; durable, hard-to-re-derive nuances live
in **Gotchas** at the end.
- Imports: `grep -rnE "^\s*(from|import)\s+vllm" src/`
- vLLM subclasses: `grep -rnE "^class [A-Za-z_]+\(" src/` (note which extend vLLM bases).
- Constructed vLLM dataclasses: grep for `ModelRunnerOutput(`, `SchedulerOutput(`,
  `SamplingMetadata(`, `CachedRequestState(`, `LogprobsTensors(`, `EngineCoreOutputs(`,
  `KVCacheConfig`, `MultiGroupBlockTable(`, etc.
- Monkeypatch / override sites: read `platform.py` (patches live here), `entrypoints.py`,
  `scheduler.py`, `lane_scheduler.py`, `worker.py`, `model_runner.py`, `input_batch.py`,
  `launcher.py`.

Known surface (verify against current source):
- **Subclassed bases** — `TTPlatform(Platform)`, `TTModelLoader(BaseModelLoader)`,
  `TTScheduler(AsyncScheduler)`, `TTLaneCoordinator(SchedulerInterface)`,
  `TTWorker(WorkerBase)`, the plugin's own `InputBatch` + `TTLaneInputBatch`,
  `Deferred/AsyncTTModelRunnerOutput(AsyncModelRunnerOutput)`, and
  `TT*Launcher/Plan(CoreEngineLauncher/EngineLaunchPlan)`.
- **Constructed dataclasses** — `ModelRunnerOutput`, `SchedulerOutput`, `SamplingMetadata`,
  `CachedRequestState`, `LogprobsTensors`, `KVCacheConfig`/specs, `EngineCoreOutputs`,
  `GrammarOutput`, `SchedulerStats`, and the `MultiGroupBlockTable(...)` call.
- **Monkeypatch targets (in `platform.py`)** — `vllm.config.model.get_config`,
  `vllm.tokenizers.registry.cached_tokenizer_from_config`,
  `InputProcessor.process_inputs`, `AsyncLLM._add_streaming_input_request`,
  `EngineCore.reset_prefix_cache`, `EngineCore`/`EngineCoreProc.pause_scheduler`.
- **Helpers** — `vllm.utils.{torch_utils,system_utils,network_utils,math_utils,import_utils,argparse_utils}`
  and `vllm.utils.length_from_prompt_token_ids_or_embeds`.

The snapshot above is the surface *as last observed* — a concrete instance of a general
pattern, not a current guarantee. Follow the general pattern regardless: re-derive the four
categories (subclassed bases, constructed dataclasses, monkeypatch targets, imported
helpers) from source and treat that as authoritative. If a coupling was added or removed,
the greps catch it and this list will not — update the snapshot when they diverge.

## Phase 2 — Diff the coupled files (BASELINE vs TARGET)
- BASELINE side = the pinned baseline source (per the Phase 0 gate). When installed == pin,
  read the exact local tree `.../site-packages/vllm/<path>` — fastest and byte-exact.
  Otherwise read the pinned tag `refs/tags/v<pin>` the same way as TARGET below; never diff
  against a drifted installed version.
- TARGET side = upstream at the tag: fetch `vllm/<path>` at `refs/tags/<TARGET>` via your
  GitHub file tool, or `https://raw.githubusercontent.com/vllm-project/vllm/<TARGET>/<path>`.
  No shell `curl`/`gh`/`git` clone needed.
- Also diff `requirements/common.txt` (BASELINE tag vs TARGET tag) for dependency changes
  and any impact on `docs/vllm-overrides.txt`.
- When subagents are available, parallelize by subsystem — worker/platform,
  scheduler/lane, runner/outputs/sampling, config/loader/kv-cache,
  entrypoints/monkeypatch/utils — handing each the exact symbols + plugin `file:line` anchors.

## Phase 3 — Classify each delta
For every coupled symbol: **CHANGE** (what differs) → **PLUGIN IMPACT** (`file:line`) → **SEVERITY**:
- `BREAKING-IMPORT` — symbol moved / renamed / removed.
- `BREAKING-CONSTRUCT` — dataclass gained a required field, or a field was removed/reordered.
- `BREAKING-OVERRIDE` — a subclassed base's method signature changed, or a new abstract method appeared.
- `BREAKING-PATCH (silent)` — a monkeypatch target moved / renamed / changed signature.
- `BEHAVIORAL` — still runs, but semantics changed.
- `NONE`.

Prioritize: new required `__init__` params or abstract methods on subclassed bases;
dataclass field changes; moved/renamed/removed imported symbols; monkeypatch target
existence and signature. `MultiGroupBlockTable.__init__` and the scheduler's stale-token
handling have churned across bumps — check them every time.

## Phase 4 — Synthesize the change plan
- Required code edits (each with `file:line` + the concrete change).
- Version-pin bumps: `docs/install-vllm-tt.sh` (the `requirements/common.txt` URL and the
  `vllm==<ver>` pin), `README.md`, `docs/diffusion-gemma.md`.
- Low / conditional watch items (behavioral deltas, new optional fields, new default methods).

## Phase 5 — Validate (only when applying)
- Static: `uvx ruff@<pyproject pin> check` and `uvx ruff@<pyproject pin> format --check`
  on changed files (the commit/CI pre-commit job runs these plus the other hooks).
- Build TARGET: bump `docs/install-vllm-tt.sh`, then (tt-metal venv active)
  `source docs/install-vllm-tt.sh` — rebuilds vLLM's `empty` target and reinstalls the
  plugin. Re-verify the import origin and `vllm.__version__`.
- Host-only tests: `python -m pytest tests/ --ignore=tests/tt`. Add device/server tests
  (`tests/tt`, needs a running server) only when the change touches device behavior.

## Phase 6 — Record & ship
- Findings → the tracking issue: a concise summary + concise per-version diffs, **no
  proposed solutions**.
- Change → a PR per `.github/pull_request_template.md`, title prefixed `[deps]`.
- Branch / version model: `main` follows the latest supported vLLM; each still-supported
  line lives on `release/vX.Y`; cherry-pick version-agnostic TT fixes across lines; tag
  releases (e.g. `vX.Y.Z-tt.N`).

## Gotchas (learned)
- Release-notes highlights are GPU/model/kernel noise; only v1 internal API churn matters here.
- Confirm which plugin form you are assessing — the checked-out working tree may not be the
  branch that targets the version in question.
- The plugin reimplements its own `InputBatch` (it does not subclass vLLM's), so vLLM's
  `gpu_input_batch.InputBatch` churn is decoupled — but shared types it imports
  (`CachedRequestState`) and the `MultiGroupBlockTable` call are not.
- Monkeypatches live in `platform.py` (not `entrypoints.py`, which only wires the plugin
  entry points) — grep there for the exact patched `module.attr` targets and the signatures
  they assume.
- The tt-run / MPI launcher is dormant: upstream `CoreEngineLauncher` / `EngineLaunchPlan`
  are absent, so `launcher.py` runs its `ImportError` fallback and stays unhooked. Re-check
  each bump whether they appeared upstream.
