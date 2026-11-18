---
name: "vLLM Upgrade Compatibility Assessment"
description: "Assess (and optionally apply) the changes needed to move vllm-tt-plugin to a new upstream vLLM version. Diffs the plugin's vLLM coupling surface between the installed baseline and a target tag, classifies breakage by severity, and produces an issue-ready findings report plus a change plan. Use when a new vLLM release lands or when bumping the pinned version."
argument-hint: "Target vLLM tag, e.g. v0.28.0 (optional; defaults to the latest upstream release)"
agent: agent
---

# vLLM upgrade compatibility assessment

## Usage

Point an agent at this file with the target version — only TARGET is needed
(BASELINE = the currently installed vLLM), e.g.:

```
Follow .github/prompts/vllm-upgrade-assessment.prompt.md; assess this plugin against vLLM v0.28.0.
```

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
- Check for an in-flight bump first: search PRs (all states) and remote branches for TARGET
  or any intermediate version (e.g., `gh pr list --state all --search "vllm in:title"`,
  `git ls-remote --heads origin '*compat*'`). If exists, ask before working on the existing
  PR/branch or starting a parallel one. A tracking issue's **Generated** section lists them
  as of its creation — check again anyway.
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
- Run the greps over `tests/` too: fakes and stubs mimic upstream internals (e.g., a fake
  engine's `_finish_pause`) and silently stop matching when those change.
- Reverse coupling — upstream calling into or reading plugin objects. For each subclassed
  base, grep TARGET for what upstream calls or reads on that object (e.g., `self.scheduler.`
  under `vllm/v1/engine/`, worker calls under `vllm/v1/executor/` and `vllm/v1/worker/`) and
  confirm the TT class provides it with TT-correct behavior — a base default TT doesn't
  override now runs on TT. Also list dataclass fields the plugin only reads: a rename or
  semantic change there fails no import and may show only on models that exercise it.
- Overridden methods — for each method TT overrides, also diff the upstream implementations
  it replaces (the base plus the GPU `Worker`/`GPUModelRunner` counterpart); duties added
  there are silently skipped on TT.

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
- **Reverse coupling** — `WorkerBase.get_supported_kv_cache_layouts()` (TT returns `LBNHC`;
  the base default fails on TT), `EngineCore` reading `scheduler.ec_connector` (set to
  `None` on `TTLaneCoordinator`), and the read-only `KVCacheTensor.layers` (formerly
  `shared_by`).
- **Upstream duties TT carries** — `TTWorker.compile_or_warm_up_model` calls
  `set_torch_threads_for_runtime()`; `TTPlatform` forces `mm_device_do_normalize = False`.

The snapshot above is the surface *as last observed* — a concrete instance of a general
pattern, not a current guarantee. Follow the general pattern regardless: re-derive the five
categories (subclassed bases, constructed dataclasses, monkeypatch targets, imported
helpers, reverse coupling) from source and treat that as authoritative. If a coupling was
added or removed, the greps catch it and this list will not — update the snapshot when they
diverge.

## Phase 2 — Diff upstream release by release, then shadow-run the tests
- Start from the tracking issue's **Generated** section when it applies: per-release change
  summaries (full diffs: the `upstream-diffs` artifact, via the `gh run download` command
  given there) and host-test results on BASELINE and TARGET, with TARGET installed for real.
  They are inputs, not findings. Skip them and do the steps below when the issue's
  TARGET/BASELINE differ from yours, when `git diff --quiet <generated-at commit> origin/main
  -- src tests docs` fails (`main` moved since), or when the artifact has expired.
- Fetch source tarballs for BASELINE (the pin, per the Phase 0 gate), every intermediate
  minor (`vX.Y.0`), and TARGET:
  `mkdir -p /tmp/vllm-assess/<tag> && curl -sL https://github.com/vllm-project/vllm/archive/refs/tags/<tag>.tar.gz | tar xz --strip-components=1 -C /tmp/vllm-assess/<tag>`.
- Diff each consecutive pair over whole directories, not only the coupled files — moved
  symbols show up only there: `diff -ru /tmp/vllm-assess/{<A>,<B>}/vllm/v1`, likewise
  `vllm/config/`, `vllm/distributed/`, `vllm/envs.py`, and `requirements/common.txt`
  (dependency changes; check `docs/vllm-overrides.txt`). Consecutive pairs attribute each
  change to its release, which the tracking issue's per-version diffs need.
- Environment and startup preconditions: from the `envs.py` diff, note new or renamed
  variables and changed defaults; scan the engine/executor/worker startup diffs for new
  `raise`/`assert`/resource checks.
- Config derived from the upstream model class: TT runs its own model, so a flag that hands
  work from the processor to the upstream model skips that work on TT — force such flags
  off in `TTPlatform`.
- When subagents are available, parallelize by subsystem — worker/platform,
  scheduler/lane, runner/outputs/sampling, config/loader/kv-cache,
  entrypoints/monkeypatch/utils — handing each the exact symbols + plugin `file:line` anchors.
- Shadow-run the host tests on TARGET before editing anything — the fastest breakage list,
  and the venv stays untouched:
  - The tarball lacks the generated `vllm/_version.py`; add a stub with
    `__version__ = version = "X.Y.Z"`, `__version_tuple__ = version_tuple = (X, Y, Z)`,
    `__commit_id__ = commit_id = None`.
  - Confirm the shadow wins:
    `PYTHONPATH=/tmp/vllm-assess/<TARGET> python -c "import vllm; print(vllm.__file__, vllm.__version__)"`.
  - Run `python -m pytest tests/ --ignore=tests/tt -q` plain (BASELINE) and with that
    `PYTHONPATH` (TARGET); failures new on TARGET seed Phase 3. Third-party deps stay at
    BASELINE versions, so cross-check failures against the `requirements/common.txt` diff.

## Phase 3 — Classify each delta
For every coupled symbol: **CHANGE** (what differs) → **PLUGIN IMPACT** (`file:line`) → **SEVERITY**:
- `BREAKING-IMPORT` — symbol moved / renamed / removed.
- `BREAKING-CONSTRUCT` — dataclass gained a required field, or a field was removed/reordered.
- `BREAKING-OVERRIDE` — a subclassed base's method signature changed, a new abstract method
  appeared, or upstream now calls/reads something on a plugin object that TT lacks or gets wrong.
- `BREAKING-PATCH (silent)` — a monkeypatch target moved / renamed / changed signature.
- `BREAKING-ENV` — a new env var, changed default, or startup check the TT deployment or CI
  doesn't satisfy.
- `BEHAVIORAL` — still runs, but semantics changed.
- `REDUNDANT` — upstream now does what a plugin workaround does; the workaround can go.
- `NONE`.

Prioritize: new required `__init__` params or abstract methods on subclassed bases;
dataclass field changes; moved/renamed/removed imported symbols; monkeypatch target
existence and signature. `MultiGroupBlockTable.__init__` and the scheduler's stale-token
handling have churned across bumps — check them every time. Also walk the plugin's
workarounds (`platform.py` patches, scheduler/lane overrides) for `REDUNDANT` ones.

## Phase 4 — Synthesize the change plan
- Required code edits (each with `file:line` + the concrete change).
- Version-pin bumps: `docs/install-vllm-tt.sh` (the `requirements/common.txt` URL and the
  `vllm==<ver>` pin), `README.md`, `docs/diffusion-gemma.md`.
- Deployment / CI changes (`BREAKING-ENV`) and removable workarounds (`REDUNDANT`).
- Low / conditional watch items (behavioral deltas, new optional fields, new default methods).

## Phase 5 — Validate (only when applying)
- Static: `uvx ruff@<pyproject pin> check` and `uvx ruff@<pyproject pin> format --check`
  on changed files (the commit/CI pre-commit job runs these plus the other hooks).
- Build TARGET: bump `docs/install-vllm-tt.sh`, then (tt-metal venv active)
  `source docs/install-vllm-tt.sh` — rebuilds vLLM's `empty` target and reinstalls the
  plugin. Re-verify the import origin and `vllm.__version__`. The rebuild replaces BASELINE
  for everything else using that venv — ask first, and offer the Phase 2 shadow run instead.
- Host-only tests: `python -m pytest tests/ --ignore=tests/tt`. They miss device-only
  regressions; Phase 6 covers those.

## Phase 6 — Device CI (before merging)
This repo has no hardware CI. tt-metal's "vLLM Model Tests" (`vllm-model-tests.yaml`)
installs the plugin through `docs/install-vllm-tt.sh` and runs one job per model/SKU. A full
run holds shared hardware for hours — confirm with the user before dispatching.

```bash
REPO=tenstorrent/tt-metal
WORKFLOW=vllm-model-tests.yaml

# One TSV row per model job: name, result, first failing step, job id.
# Works on runs still in progress.
JOB_ROW='.jobs[]
  | select(.name | startswith("vllm-tests / "))
  | [
      (.name | ltrimstr("vllm-tests / ")),
      (.conclusion // .status),
      ([.steps[]? | select(.conclusion == "failure") | .name] | first // "-"),
      .id
    ]
  | @tsv'

run_jobs() {
  local run_id=$1
  gh api "repos/$REPO/actions/runs/$run_id/jobs?per_page=100" --paginate --jq "$JOB_ROW" \
    | LC_ALL=C sort -t $'\t' -k 1,1
}
```

- **Reference nightly.** Scheduled jobs resolve plugin `main` at checkout, not at run
  creation. List candidate runs and determine the branch's merge-base:
  ```bash
  git fetch origin main
  git merge-base HEAD origin/main
  gh run list --repo "$REPO" --workflow "$WORKFLOW" --event schedule --limit 5 \
      --json databaseId,createdAt,url
  ```
  Read the actual plugin SHA from each relevant model job's checkout log and choose
  reference results that tested the merge-base. Jobs in one run can test different
  commits; do not infer their SHAs from run creation time or commit timestamps.
  A job that fails the same way across several nightlies is pre-existing: list it, don't
  chase it.
- **Dispatch** the full matrix once the branch is stable, then only the affected model (and
  SKU) after each fix:
  ```bash
  gh workflow run "$WORKFLOW" --repo "$REPO" --ref main \
    -f vllm-tt-plugin-ref=<branch> \
    -f model=all -f sku=all -f tier=all

  # Your latest dispatched run
  gh run list --repo "$REPO" --workflow "$WORKFLOW" --event workflow_dispatch \
    --user "$(gh api user --jq .login)" --limit 1 --json databaseId,url
  ```
  `--ref` is the tt-metal ref: `main`, or a tt-metal branch carrying CI fixes the bump needs
  (`BREAKING-ENV`). `model` takes an HF id from the workflow's choice list. Check back later
  instead of polling.
- **Compare** with the nightly, side by side:
  ```bash
  BRANCH_RUN=<run-id>
  NIGHTLY_RUN=<run-id>

  # job | branch result | branch failing step | nightly result | nightly failing step | branch job id
  # "-" marks a job absent from one of the runs.
  LC_ALL=C join -t $'\t' -a 1 -a 2 -e - -o 0,1.2,1.3,2.2,2.3,1.4 \
      <(run_jobs "$BRANCH_RUN") <(run_jobs "$NIGHTLY_RUN") \
    | column -t -s $'\t'
  ```
- **Classify** each job:
  - Same result, same failing step, same error → unchanged. Compare the errors in the
    server logs, not just the conclusions.
  - Fails only on the branch, or at a different step → regression until the logs show
    otherwise; reproduce with a targeted dispatch.
  - Failed in a setup or checkout step → infra; re-dispatch that model.
  - Several jobs on one machine failing at server start → check the tt-smi snapshot's
    `"processes"` list in the job log for a foreign process holding the devices before
    blaming code.
  - Passes only on the branch → note it; not a blocker.
- **Green is not enough:**
  - Compare benchmark TPOT and throughput with the nightly's (`grep "Mean TPOT"` in the
    log), DP jobs especially — a large slowdown is a regression even when the job passes.
  - Multimodal jobs: read the answer. The check only greps for a keyword, so a wrong
    description can pass.
  - AI job and run summaries can invent root causes; confirm each one in the server log.
- **Logs** of a finished job are available while the run continues (`gh run view
  --log-failed` waits for the whole run). Save each log once and grep locally; the server
  log is in the "Show server log" step:
  ```bash
  mkdir -p /tmp/ci
  gh api "repos/$REPO/actions/jobs/<job-id>/logs" > "/tmp/ci/<job-id>.log"
  ```

## Phase 7 — Record & ship
- Findings → the tracking issue: fill its Summary and per-release sections (tag each finding
  with its Phase 3 class) and keep the Generated section — concise, **no proposed solutions**.
- Change → a PR per `.github/pull_request_template.md`, title prefixed `[deps]`. Keep its
  validation section to the final full Phase 6 run plus one confirmation run per device fix,
  each with its run link, plugin commit, and per-job result against the nightly; list
  pre-existing failures separately and drop superseded runs.
- Version model: `main` follows the latest supported vLLM; there are no release branches.
  Before merging, tag the last `main` commit that targets BASELINE as an annotated
  `compat/vllm-<BASELINE>` and push it — the README points older-version users at these
  tags, and nothing is maintained on top of them.

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
