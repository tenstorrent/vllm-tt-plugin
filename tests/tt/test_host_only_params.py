# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""
Tests for sampling parameters which require host ("compat") sampling.
"""

import httpx

from tests.tt.utils import RequestConfig, run_concurrent_batch


def _bad_word_token_patterns(tokenize, bad_words):
    """Match upstream SamplingParams.update_from_tokenizer prefix selection."""
    patterns = []
    for word in dict.fromkeys(bad_words):
        bare = tokenize(word.lstrip())
        assert (
            isinstance(bare, list)
            and bare
            and all(type(token) is int and token >= 0 for token in bare)
        ), f"Invalid bare bad-word tokenization for {word!r}: {bare!r}"
        patterns.append(bare)
        spaced = tokenize(" " + word.lstrip())
        assert isinstance(spaced, list) and all(
            type(token) is int and token >= 0 for token in spaced
        ), f"Invalid space-prefixed tokenization for {word!r}: {spaced!r}"
        if spaced and len(spaced) == len(bare) and spaced[0] != bare[0]:
            patterns.append(spaced)
    assert patterns, "No forbidden token patterns were obtained"
    return patterns


def _served_bad_word_token_patterns(tt_server, model, bad_words):
    # Query the running server's tokenizer, including any configured revision.
    # Generated output is never re-tokenized: that could collapse a legal
    # alternate segmentation into a forbidden canonical sequence.
    with httpx.Client(timeout=30) as client:

        def tokenize(text):
            response = client.post(
                f"{tt_server.base_url}/tokenize",
                json={"model": model, "prompt": text, "add_special_tokens": False},
            )
            response.raise_for_status()
            return response.json()["tokens"]

        return _bad_word_token_patterns(tokenize, bad_words)


def _assert_no_bad_word_tokens(token_ids, patterns, response_index):
    assert (
        isinstance(token_ids, list)
        and token_ids
        and all(type(token) is int and token >= 0 for token in token_ids)
    ), (
        f"Response {response_index} omitted valid nonempty output token IDs: "
        f"{token_ids!r}"
    )
    for pattern in patterns:
        for start in range(len(token_ids) - len(pattern) + 1):
            assert token_ids[start : start + len(pattern)] != pattern, (
                f"Forbidden token sequence {pattern} at output offset {start} "
                f"in response {response_index}: {token_ids}"
            )


class TestHostOnlyParameters:
    def test_min_p(self, tt_server, tt_model_name, max_batch_size):
        """Test min_p parameter (smoke test - verifies it doesn't error)."""
        configs = [
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.1),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.2),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.3),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.4),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.5),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.6),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.7),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.8),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.9),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=1.0),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        # min_p affects sampling distribution - just verify we get output
        assert results[0] is not None and len(results[0]) > 0, (
            "should produce non-empty output"
        )

    def test_bad_words(
        self, tt_server, tt_model_name, max_batch_size, reasoning_token_budget
    ):
        """Test the exact forbidden token sequences used by bad_words masking."""
        bad_words = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]
        patterns = _served_bad_word_token_patterns(tt_server, tt_model_name, bad_words)
        configs = [
            # Run multiple times with high temperature to increase coverage
            RequestConfig(
                prompt="Say hello to me",
                max_tokens=100 + reasoning_token_budget,
                bad_words=bad_words,
                temperature=1.0,
                seed=i,
                return_token_ids=True,
            )
            for i in range(5)
        ]
        # bad_words is only available in chat completions API
        results = run_concurrent_batch(
            tt_server, tt_model_name, configs, use_chat=True, return_full_response=True
        )
        assert len(results) == len(configs)

        for i, response in enumerate(results):
            choice = response.choices[0]
            text = choice.message.content
            assert text is not None, f"Response {i} content is None"
            _assert_no_bad_word_tokens(getattr(choice, "token_ids", None), patterns, i)

    def test_logit_bias(self, tt_server, tt_model_name, max_batch_size):
        """Test logit_bias parameter (smoke test)."""
        configs = [
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={1: 0.1}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={4: 0.2}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={2: 0.3}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={16: 0.4}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10, logit_bias={7: 0.5}),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        assert results[0] is not None, "should produce output with logit_bias"

    def test_allowed_token_ids(self, tt_server, tt_model_name, max_batch_size):
        """Test allowed_token_ids parameter (smoke test)."""
        configs = [
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[1, 2, 3]
            ),
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[4, 5, 6]
            ),
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[7, 8, 9]
            ),
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[10, 11, 12]
            ),
            RequestConfig(
                prompt="Allowed: ", max_tokens=10, allowed_token_ids=[13, 14, 15]
            ),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        # With only 3 allowed tokens, output should be limited
        for i, result in enumerate(results):
            assert result is not None, f"should produce output for request {i}"
            assert len(result) > 0, f"should produce non-empty output for request {i}"

    def test_min_tokens(self, tt_server, tt_model_name, max_batch_size):
        """Test min_tokens parameter ensures minimum output length."""
        min_tokens = 5
        configs = [
            # Use a prompt that might naturally produce short output
            RequestConfig(prompt="Say OK.", max_tokens=100, min_tokens=min_tokens),
        ]
        results = run_concurrent_batch(
            tt_server, tt_model_name, configs, return_full_response=True
        )
        assert len(results) == len(configs)

        response = results[0]
        # Check usage stats for token count
        assert response.usage is not None, "usage stats should be returned"
        assert response.usage.completion_tokens >= min_tokens, (
            f"should produce at least {min_tokens} tokens, "
            f"got {response.usage.completion_tokens}"
        )
