---
description: 'Pull request review rules for vllm-tt-plugin'
applyTo: '**'
excludeAgent: "cloud-agent"
---

# Code review instructions

Be concise. For each finding, give a concrete failure scenario: the input or state, and the wrong result.

## 1. Describe the pull request

Start the review with a short description of the pull request:

- What the pull request changes. Name the changed functions, classes, and behavior.
- Why the pull request makes the change. Use the pull request description and the linked issues. If the reason is not clear, say so.

## 2. New model capabilities, environment variables, and `--additional-config` options

Apply the strictest review when the pull request does one of these:

- adds a key or a value to `model_capabilities`, or adds an equivalent support check (a model-name condition, an attribute check)
- makes the plugin read an environment variable that the plugin did not read before
- accepts a new `--additional-config` key, including keys under `tt`

For each addition, write a separate comment that states:

1. What the addition does and why the pull request needs it.
2. The category of the addition:
   - performance optimization: state the measured gain
   - true model necessity: state what makes this model different from all other models
   - workaround for a bug or a missing feature: mark the addition as blocking
3. Whether an existing mechanism can already do the same thing.
4. Whether the pull request documents the addition in all relevant locations: `docs/MODEL_CAPABILITIES.md` for model capabilities, the "Configuration" section of `README.md` for `--additional-config` options, and the code that reads the value.
5. Whether the pull request description identifies the human request or approval for the addition.

`AGENTS.md`, section "Model capabilities and plugin configuration require human authorization", gives the full rules.

## 3. Standard review

Then review the diff for:

- correctness bugs: wrong logic, unhandled edge cases, wrong state lifetime, race conditions
- regressions in the other execution modes: single-process non-DP, single-process lane-DP, multi-process DP
- broken invariants that the code does not detect: the code must raise with the offending values, not continue in a degraded state
- comments that describe a different mechanism than the code
- dead code, duplication, and hunks that the pull request title does not explain
- missing tests: a change needs a host test that fails on the old code

Do not comment on formatting that `pre-commit` checks.
