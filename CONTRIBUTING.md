# Contributing to `vllm-tt-plugin`

Thank you for your interest in contributing to `vllm-tt-plugin`! There are many ways to contribute to the project: from improving the docs to bringing up new major features. Consider asking your questions in the [discussions](https://github.com/tenstorrent/vllm-tt-plugin/discussions) page, raising your concerns in [issues](https://github.com/tenstorrent/vllm-tt-plugin/issues), and/or proposing your solutions in [PRs](https://github.com/tenstorrent/vllm-tt-plugin/pulls).

## How to Contribute

Before contributing, please make sure that your ideas and concerns are not covered in our issues, PRs, and discussions pages. When made sure, please follow the instructions above.

### Asking Questions

[Discussions](https://github.com/tenstorrent/vllm-tt-plugin/discussions) are dedicated to Q&A, ideas, announcements, polls, and other types of general information. Don't hesitate to share your questions or concerns here first. Keep in mind that not every issue becomes a planned task, nor is every PR merged – so if not sure, begin your chat here.

### Creating Issues

If you find bugs or propose required / planned enhancements, please open an issue on [Issues](https://github.com/tenstorrent/vllm-tt-plugin/issues). Follow the project's [issue templates](./github/ISSUE_TEMPLATE/); if irrelevant, get inspirations from the templates, then make sure to briefly summarize your issue, explain details, and provide other relevant information for reproduction and/or comparison. Always mark labels. If applicable, mark assignees, add project tasks, and specify priority labels.

### Submitting Pull Requests

**_PRs to `main` must be followed by their corresponding issues first._** This way, we separate the problem / idea in Issues and the solution in PRs.

Pull requests are reviewed on a **weekly basis**.

#### Pull Request Guidelines

- Follow the [PR template](./github/pull_request_template.md) to create your PR.
- Keep PRs focused on a single logical change to simplify review.
- Add tests for new functionality where applicable.
- Follow the existing code style, incl. the `numpydoc` format, type hints, package structure, and documentation practices. Set up `pre-commit` to check the code every commit. Record the commands and results in the pull request's Validation section.
- Write clear and concise commit messages in the imperative mood. Your first message should be a short lable (e.g., "[feat]") and a one-liner (e.g., "Enable on-device grammar sampling.") followed by optional listed quick implementation details.
- If you're a codeowner, specify a priority label.
- **_Mark your PR ready for a review only after all validation steps pass._** This includes automatic and manual workflow runs as well. So by default, your PR should be a draft.


P.S. For model integrations, use the [model-plugin interface](docs/MODEL_INTERFACE.md) for construction, KV storage, execution, and lifecycle responsibilities, with links to the capability, reload, and speculative contracts. Update the affected interface pages when changing model-facing behavior. Keep plugin changes in this repository and model adapter changes in `tenstorrent/tt-metal`; cross-link paired pull requests.

#### Responding to Change Requests

Address review feedback with new commits or by amending existing commits via
`git rebase -i`. Force-push your updated branch when ready for re-review.

## Code of Conduct

This project adheres to the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating,
you are expected to uphold this code. Please report unacceptable behavior to
ospo@tenstorrent.com.
