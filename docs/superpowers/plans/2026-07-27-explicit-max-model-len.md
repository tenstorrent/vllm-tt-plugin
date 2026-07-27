# Explicit `max_model_len` Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve numeric and omitted `max_model_len` configurations while retaining vLLM's explicit `-1` auto-fit path.

**Architecture:** Let vLLM's parsed `original_max_model_len` remain authoritative instead of overwriting it in the TT platform hook. Rely on vLLM 0.24's override-aware KV capacity validation, while retaining the TT worker synchronization hook used after explicit auto-fit.

**Tech Stack:** Python 3.10+, pytest, vLLM 0.24 configuration and worker APIs, pre-commit, Ruff

## Global Constraints

- Positive numeric values must remain unchanged.
- An omitted value represented by `None` must remain unchanged.
- `-1` must remain the only value that requests auto-fit.
- Do not change TT KV cache sizing or scheduler limits.
- Do not disable tests or features.
- Run `pre-commit run` on staged files before every commit.

---

### Task 1: Enforce the explicit auto-fit contract

**Files:**
- Modify: `tests/test_dp_modes.py`
- Modify: `src/vllm_tt_plugin/platform.py:824-856`
- Modify: `src/vllm_tt_plugin/worker.py:17,138-161,411-434`

**Interfaces:**
- Consumes: `TTPlatform.check_and_update_config(vllm_config) -> None`
- Preserves: `model_config.original_max_model_len: int | None`
- Preserves: `TTWorker.update_max_model_len(max_model_len: int) -> None`
- Produces: Worker cache initialization that trusts upstream capacity validation

- [ ] **Step 1: Add policy and worker synchronization tests**

Add these methods to `TestDPModes` in `tests/test_dp_modes.py`:

```python
    @pytest.mark.parametrize("original_max_model_len", [None, 8192, -1])
    def test_check_and_update_config_preserves_original_max_model_len(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        original_max_model_len: int | None,
    ) -> None:
        vllm_config.model_config.original_max_model_len = original_max_model_len

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.model_config.original_max_model_len
            == original_max_model_len
        )

    def test_update_max_model_len_syncs_worker_model_config(self) -> None:
        worker_instance = TTWorker.__new__(TTWorker)
        worker_instance.model_config = SimpleNamespace(max_model_len=262_144)

        TTWorker.update_max_model_len(worker_instance, 131_072)

        assert worker_instance.model_config.max_model_len == 131_072
```

- [ ] **Step 2: Run the focused tests and verify the policy test fails**

Run:

```bash
pytest \
  tests/test_dp_modes.py::TestDPModes::test_check_and_update_config_preserves_original_max_model_len \
  tests/test_dp_modes.py::TestDPModes::test_update_max_model_len_syncs_worker_model_config \
  -v
```

Expected: the `None` and `8192` policy cases fail because
`TTPlatform.check_and_update_config` changes both values to `-1`. The `-1`
policy case and worker synchronization test pass.

- [ ] **Step 3: Stop TT from forcing auto-fit**

Delete the auto-fit comment and assignment from
`TTPlatform.check_and_update_config` in `src/vllm_tt_plugin/platform.py`:

```python
        model_config.original_max_model_len = -1
```

Leave the surrounding structured-output configuration and model registration
logic unchanged.

- [ ] **Step 4: Remove superseded TT capacity validation**

Remove this import from `src/vllm_tt_plugin/worker.py`:

```python
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
```

Delete the complete `_validate_tt_kv_cache_capacity` function. Change worker
cache initialization to delegate directly:

```python
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate TT KV cache and initialize persistent input batch.

        Every standard-DP rank owns its own TT mesh/KV cache, while
        single-process lane mode has only one rank.
        """
        self.model_runner.initialize_kv_cache(kv_cache_config)
```

Keep the worker synchronization hook, but replace its auto-fit comment with:

```python
    def update_max_model_len(self, max_model_len: int) -> None:
        # The engine calls this via collective_rpc after an explicit
        # max_model_len=-1 request auto-fits the value to available KV cache
        # capacity. WorkerBase has no such hook, so TTWorker must synchronize
        # the shared model config used by TTModelRunner.
        self.model_config.max_model_len = max_model_len
```

- [ ] **Step 5: Run the focused tests and verify they pass**

Run:

```bash
pytest \
  tests/test_dp_modes.py::TestDPModes::test_check_and_update_config_preserves_original_max_model_len \
  tests/test_dp_modes.py::TestDPModes::test_update_max_model_len_syncs_worker_model_config \
  -v
```

Expected: all four parametrized/test cases pass.

- [ ] **Step 6: Run the affected host test modules**

Run:

```bash
pytest tests/test_dp_modes.py tests/test_num_available_blocks.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Validate and commit the implementation**

Run:

```bash
git add \
  src/vllm_tt_plugin/platform.py \
  src/vllm_tt_plugin/worker.py \
  tests/test_dp_modes.py \
  docs/superpowers/plans/2026-07-27-explicit-max-model-len.md
pre-commit run
git commit -m "platform: preserve explicit max_model_len"
```

Expected: pre-commit passes and the implementation commit succeeds.

---

### Task 2: Verify the branch and prepare the pull request

**Files:**
- Verify: all files changed from `origin/main`

**Interfaces:**
- Consumes: implementation and design commits from this branch
- Produces: a reviewable branch with passing host checks

- [ ] **Step 1: Run the complete host test suite**

Run:

```bash
pytest tests --ignore=tests/tt -v
```

Expected: all host tests pass.

- [ ] **Step 2: Run repository-wide pre-commit checks**

Run:

```bash
pre-commit run --all-files
```

Expected: every hook passes without modifying files.

- [ ] **Step 3: Inspect the final branch diff**

Run:

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
```

Expected: the worktree is clean, the branch contains the design and
implementation commits, `git diff --check` reports no errors, and the diff is
limited to the approved specification, plan, platform, worker, and host test
files.
