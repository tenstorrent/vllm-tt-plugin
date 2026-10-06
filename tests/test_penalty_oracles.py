# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host regressions for live penalty test oracles, without a serving model."""

from types import SimpleNamespace

import pytest

from tests.tt.utils import (
    assert_frequency_penalty_reduces_token_repeats,
    render_penalty_prompts,
)


@pytest.mark.parametrize("template", [None, ""])
def test_penalty_prompts_keep_base_completion_format(template):
    prompts = ["She opened the door and", "a b c a b c"]

    def unexpected_render(*args, **kwargs):
        raise AssertionError("Base checkpoint must not get a chat wrapper")

    tokenizer = SimpleNamespace(
        chat_template=template, apply_chat_template=unexpected_render
    )
    assert render_penalty_prompts(prompts, tokenizer) == prompts


def test_penalty_prompts_follow_declared_chat_template_and_generation_prefix():
    prompts = ["She opened the door and", "a b c a b c"]
    seen = []

    def render(messages, *, tokenize, add_generation_prompt):
        seen.append((messages, tokenize, add_generation_prompt))
        return f"<user>{messages[0]['content']}<assistant>"

    tokenizer = SimpleNamespace(
        chat_template="declared-template", apply_chat_template=render
    )
    assert render_penalty_prompts(prompts, tokenizer) == [
        f"<user>{prompt}<assistant>" for prompt in prompts
    ]
    assert seen == [
        ([{"role": "user", "content": prompt}], False, True) for prompt in prompts
    ]


def test_frequency_oracle_counts_ids_despite_more_repeated_characters():
    # Actual failing vocabulary pieces: ' a', ' aa', 'aaaaaaaa', 'aaaa',
    # 'aaa', 'aa', 'bbbb'. Character count rises from 15 to 52 while the
    # dominant ID drops from 15 to 2 and total repeated occurrences 14 to 8.
    pieces = {
        941: " a",
        122047: " aa",
        55746: "aaaaaaaa",
        33741: "aaaa",
        73903: "aaa",
        33444: "aa",
        80573: "bbbb",
    }
    baseline = [941] * 15
    penalized = (
        [941] * 2
        + [122047] * 3
        + [55746] * 3
        + [33741] * 3
        + [73903] * 2
        + [33444, 80573]
    )
    assert "".join(pieces[token] for token in baseline).count("a") == 15
    assert "".join(pieces[token] for token in penalized).count("a") == 52
    assert_frequency_penalty_reduces_token_repeats(baseline, penalized)


@pytest.mark.parametrize("penalized", [[1] * 15, [2] * 15])
def test_frequency_oracle_rejects_unchanged_or_replaced_repetition(penalized):
    with pytest.raises(AssertionError):
        assert_frequency_penalty_reduces_token_repeats([1] * 15, penalized)


@pytest.mark.parametrize(
    "baseline, penalized", [("aaaa", "aa"), ([1, 2], [1]), ([], [1])]
)
def test_frequency_oracle_requires_token_ids_and_repetition_stimulus(
    baseline, penalized
):
    with pytest.raises(AssertionError):
        assert_frequency_penalty_reduces_token_repeats(baseline, penalized)
