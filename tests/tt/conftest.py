# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
import openai
import pytest


@pytest.fixture(autouse=True)
def reset_tt_platform_class_state():
    """Neutralize the host-suite fixture of the same name.

    These tests drive a server over HTTP and never touch ``TTPlatform`` class
    state, so there is nothing to save and restore. The host version imports the
    platform at setup time, and in a file that does not import vLLM at module
    scope that import lands while ``vllm.platforms`` is still half-initialized
    by the plugin entry point, so the first test of the file errors out on
    ``cannot import name 'current_platform'``.
    """
    yield


def pytest_addoption(parser):
    """Add TT-specific command line options."""
    parser.addoption(
        "--tt-server-url",
        action="store",
        required=True,
        help="URL of the running vLLM server (e.g., http://localhost:8000)",
    )
    parser.addoption(
        "--tt-model-name",
        action="store",
        required=True,
        help="Model name served by the server (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.addoption(
        "--tt-max-num-seqs",
        action="store",
        type=int,
        default=32,
        help="Max batch size for testing (default: 32)",
    )
    parser.addoption(
        "--tt-reasoning-token-budget",
        action="store",
        type=int,
        default=0,
        help=(
            "Additional generated tokens for reasoning before final-answer "
            "assertions in bad-word and mixed structured/plain chat tests. "
            "Use an allowance measured on a native reference (default: 0)."
        ),
    )
    parser.addoption(
        "--tt-recall-reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help=(
            "Use chat completions for the three chunked-prefill recall tests "
            "with this reasoning effort. Default: legacy completions."
        ),
    )
    parser.addoption(
        "--tt-recall-reasoning-token-budget",
        type=int,
        default=0,
        help=(
            "Extra tokens added to the original 24-token recall cap when "
            "--tt-recall-reasoning-effort is set. Other test caps are unchanged."
        ),
    )
    parser.addoption(
        "--tt-chunked-prefill-budget",
        action="store",
        type=int,
        default=0,
        help=(
            "The served max_num_batched_tokens, when the server runs with chunked "
            "prefill enabled. Tests that need a prompt long enough to be split "
            "across engine steps size it from this and skip when it is 0 (default)."
        ),
    )


@pytest.fixture(scope="session")
def tt_server_url(request):
    """Returns the server URL."""
    return request.config.getoption("--tt-server-url")


@pytest.fixture(scope="session")
def tt_model_name(request):
    """Returns the model name being tested."""
    return request.config.getoption("--tt-model-name")


@pytest.fixture(scope="session")
def max_batch_size(request):
    """Returns the max batch size for testing."""
    return request.config.getoption("--tt-max-num-seqs")


@pytest.fixture(scope="session")
def chunked_prefill_budget(request):
    """The served ``max_num_batched_tokens``, or 0 when chunked prefill is off."""
    return request.config.getoption("--tt-chunked-prefill-budget")


@pytest.fixture(scope="session")
def reasoning_token_budget(request):
    """Extra tokens added to the original total generation caps."""
    budget = request.config.getoption("--tt-reasoning-token-budget")
    if budget < 0:
        raise pytest.UsageError("--tt-reasoning-token-budget must be non-negative")
    return budget


@pytest.fixture(scope="session")
def recall_reasoning_effort(request):
    """An explicit chat-recall selection; never inferred from the model name."""
    return request.config.getoption("--tt-recall-reasoning-effort")


@pytest.fixture(scope="session")
def recall_reasoning_token_budget(request):
    budget = request.config.getoption("--tt-recall-reasoning-token-budget")
    if budget < 0:
        raise pytest.UsageError(
            "--tt-recall-reasoning-token-budget must be non-negative"
        )
    return budget


@pytest.fixture(scope="session")
def tt_server(tt_server_url):
    """
    Returns a simple object with get_async_client() method
    to match the interface expected by tests.
    """

    class ServerWrapper:
        def __init__(self, base_url: str):
            self.base_url = base_url.rstrip("/")

        def get_async_client(self):
            return openai.AsyncOpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="dummy",  # vLLM doesn't require a real key
            )

        def get_client(self):
            return openai.OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="dummy",
            )

    return ServerWrapper(tt_server_url)
