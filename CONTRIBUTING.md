# Contributing to `vllm-tt-plugin`

Thank you for your interest in contributing to `vllm-tt-plugin`! Contributions include
documentation, bug fixes, tests, performance improvements, and model integrations.
Start with the problem, agree on the scope, and keep the result small enough for a human
to understand and validate end to end.

## How to Contribute

Search existing [discussions](https://github.com/tenstorrent/vllm-tt-plugin/discussions),
[issues](https://github.com/tenstorrent/vllm-tt-plugin/issues), and
[pull requests](https://github.com/tenstorrent/vllm-tt-plugin/pulls) before starting.
Coordinate on existing work instead of opening a competing pull request.

### Asking Questions

Use Discussions for questions, ideas, and uncertain proposals. Not every idea
needs an implementation; establish the requirement before starting a large change.

### Creating Issues

Open an issue for a reproducible bug or a planned enhancement. Use the applicable
[issue template](.github/ISSUE_TEMPLATE/); if none fits, describe the problem,
expected behavior, and evidence needed to reproduce or assess it. Include relevant
versions and runtime configuration. Add appropriate labels, plus assignees,
project information, and priority when applicable.

**Open the corresponding issue before submitting a pull request to `main`.**
The issue describes the problem or proposal; the pull request describes its
implementation and validation.

## Prepare the Change

Use the [README environment setup](README.md#environment-setup), or the
[host-only setup](AGENTS.md#3-environment-and-installation) when working without
Tenstorrent hardware.

The [engineering standards](AGENTS.md#7-engineering-standards) define design,
typing, numpydoc, examples, exceptions, and test expectations. Consult
[state and fixture isolation](AGENTS.md#8-state-and-fixture-isolation) when
changing shared state or test setup.

Start with the reference examples linked from those standards. The
[testing map](tests/README.md) explains the unit, integration, support, and
device-test layout and the commands for each group.

For model integrations, start with the
[model-plugin interface](docs/MODEL_INTERFACE.md) and its linked contracts.
Update affected interface pages with model-facing changes. Follow the
[repository boundaries](AGENTS.md#repository-boundaries) and cross-link paired
plugin and model pull requests.

### Package versions

Follow the versioning rule in [pyproject.toml](pyproject.toml). Update the literal
`__version__` in [the package initializer](src/vllm_tt_plugin/__init__.py) for each
plugin release; `pyproject.toml` reads it dynamically.
When moving to another vLLM series, also follow the
[upstream upgrade checklist](AGENTS.md#the-install-script-owns-the-dependency-set).

## Submitting Pull Requests

Pull requests are reviewed on a **weekly basis**.

### Pull Request Guidelines

- Fill in the [pull request template](.github/pull_request_template.md) and link
  the corresponding issue.
- Keep the pull request focused on one logical change.
- Follow the [engineering standards](AGENTS.md#7-engineering-standards) and
  [validation instructions](AGENTS.md#4-linting-formatting-and-pre-commit).
  Record commands, results, and validation gaps in the template's Validation
  section.
- Use the [pull request conventions](AGENTS.md#9-pull-request-conventions) for
  titles and branches.
- Write concise, imperative commit subjects with a bracketed label, for example
  `[feat] Enable on-device grammar sampling.` Add details when useful.
  Commit messages describe what changed and why. Keep validation commands,
  results, and pass/fail notes in the PR's Validation section or the task
  handoff, not in commit messages.
- Add appropriate labels; codeowners should specify priority.

**Keep the pull request in draft while implementation, local checks, or applicable**
**manually dispatched validation are incomplete.** Once those checks pass, mark it
ready to trigger automatic PR checks. The
[CI workflow](.github/workflows/lint-and-host-tests.yaml) skips draft PRs, so
its automatic checks cannot be a prerequisite for leaving draft. Confirm the
required checks and applicable manual validation pass before requesting final
review and merging.

### Responding to Change Requests

Address feedback with focused commits. If you rewrite commits with
`git rebase -i`, coordinate with collaborators and use `git push --force-with-lease`
when updating the remote branch. Re-run checks affected by the revisions and
update Validation before requesting another review.

AI-assisted contributions must also follow [AGENTS.md](AGENTS.md).

## Code of Conduct

This project adheres to the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating,
you are expected to uphold this code. Please report unacceptable behavior to
ospo@tenstorrent.com.
