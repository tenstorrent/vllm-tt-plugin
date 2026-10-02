# Contributing to vllm-tt-plugin

Thank you for your interest in contributing to vllm-tt-plugin! We welcome bug
reports, bug fixes, and enhancements from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub:

1. Navigate to the [Issues](https://github.com/tenstorrent/vllm-tt-plugin/issues) page
2. Click **New Issue**
3. Provide a clear description of the bug, including:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment details (hardware, OS, Python version, tt-metal version)
   - Any relevant logs or error messages

### Submitting Pull Requests

Pull requests are reviewed on a **weekly basis**.

#### Pull Request Guidelines

- Keep PRs focused on a single logical change to simplify review.
- Ensure existing tests pass before submitting.
- Add tests for new functionality where applicable.
- Follow the existing code style and run `pre-commit run` explicitly before
  every commit. Run `pre-commit run --all-files` to check the complete checkout.
  Record the commands and results in the pull request's Validation section.
- For model integrations, use the
  [model capability reference](docs/MODEL_CAPABILITIES.md) and
  [speculative decoding contract](docs/SPEC_DECODE_CONTRACT.md) to identify the
  plugin and tt-metal responsibilities. Keep plugin changes in this repository
  and model adapter changes in `tenstorrent/tt-metal`; cross-link paired pull
  requests.
- Write clear commit messages in the imperative mood (e.g., "Fix scheduler edge
  case" rather than "Fixed scheduler edge case").

#### Responding to Change Requests

Address review feedback with new commits or by amending existing commits via
`git rebase -i`. Force-push your updated branch when ready for re-review.

## Code of Conduct

This project adheres to the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating,
you are expected to uphold this code. Please report unacceptable behavior to
ospo@tenstorrent.com.

## Questions?

If you have questions about contributing, feel free to open a discussion or
issue on GitHub.
