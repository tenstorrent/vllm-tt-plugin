---
description: 'Pull request review rules for vllm-tt-plugin'
applyTo: '**'
excludeAgent: "cloud-agent"
---

# Code review instructions

Read [AGENTS.md](../../AGENTS.md) and its applicable references before reviewing.
It defines the repository's engineering, validation, and authorization rules.
During a review, edit files or publish comments only when explicitly requested.

## 1. Establish intent

Read the pull request description, linked issues, and complete diff. Identify
the intended behavior, affected interfaces, and motivation. If the motivation or
required context is missing, say what remains unclear rather than guessing.

## 2. Trace impact and check evidence

- Inspect affected callers, state transitions, and tests alongside the diff.
  Follow the relevant paths in each affected
  [execution mode](../../docs/SCHEDULING.md).
- Apply the [engineering standards](../../AGENTS.md#7-engineering-standards)
  and, where relevant, [state and fixture isolation](../../AGENTS.md#8-state-and-fixture-isolation).
- For interface additions, apply the
  [authorization and necessity rules](../../AGENTS.md#model-capabilities-and-plugin-configuration-require-human-authorization).
  Trace the supporting evidence and required documentation. Check measurements
  when an addition claims a performance benefit. Comment on a demonstrated
  violation; a compliant addition does not need a separate comment.
- Assess validation under
  [test design and review evidence](../../AGENTS.md#test-design-and-review-evidence).
  Distinguish fixes, features, refactors, and documentation changes when judging
  coverage. Identify the material behavior left untested, if any.

## 3. Report the review

Start with a short description of what changed and why, naming the principal
functions, classes, or documentation involved.

For each finding, give the affected file and precise location, the evidence,
and the consequence:

- For a correctness defect, identify a reachable input or state and the wrong
  result introduced or exposed by the change.
- For a requirements violation, link the specific rule and show how the change
  violates it. A runtime failure is not necessary to establish a policy defect.

Be concise, group duplicate findings, and put blocking findings first. Label
optional suggestions separately. If no supported findings remain, say so.
Do not comment on formatting handled by `pre-commit`.

Finish with the validation reviewed, checks you ran, and any remaining gaps.
Distinguish the submitter's reported results from checks you performed yourself.
