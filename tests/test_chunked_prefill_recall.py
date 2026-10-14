# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only coverage for chunked-prefill passphrase matching."""

import pytest

from tests.tt.utils import recalled_passphrase


@pytest.mark.parametrize(
    "output",
    [
        "The secret passphrase is **cobalt-heron-256**.",
        "The secret passphrase is **cobalt\N{HYPHEN}heron\N{HYPHEN}256**.",
        (
            "The secret passphrase is "
            "**cobalt\N{NON-BREAKING HYPHEN}heron\N{NON-BREAKING HYPHEN}256**."
        ),
    ],
    ids=["hyphen-minus", "hyphen", "non-breaking-hyphen"],
)
def test_recalled_passphrase_accepts_equivalent_hyphens(output):
    assert recalled_passphrase(output, "cobalt-heron-256")


@pytest.mark.parametrize(
    "output",
    [
        None,
        "The secret passphrase is cobalt-egret-0-2.",
        "The secret passphrase is cobalt-heron-0-3.",
        (
            "The secret passphrase is "
            "cobalt\N{NON-BREAKING HYPHEN}heron\N{NON-BREAKING HYPHEN}2"
            "\N{NON-BREAKING HYPHEN}2."
        ),
    ],
    ids=["missing-output", "different-name", "different-suffix", "other-request"],
)
def test_recalled_passphrase_rejects_different_passphrases(output):
    assert not recalled_passphrase(output, "cobalt-heron-0-2")


_ORIGINAL_IDENTIFIERS = ["cobalt-heron-42", "cobalt-heron-256"] + [
    f"cobalt-heron-{rnd}-{i}" for rnd in range(3) for i in range(4)
]


@pytest.mark.parametrize("identifier", _ORIGINAL_IDENTIFIERS)
@pytest.mark.parametrize("hyphen", ["-", "\u2010", "\u2011"])
def test_accepts_every_complete_original_identifier(identifier, hyphen):
    output = f"The secret passphrase is **{identifier.replace('-', hyphen).upper()}**."
    assert recalled_passphrase(output, identifier)


@pytest.mark.parametrize(
    "expected,other",
    [(a, b) for a in _ORIGINAL_IDENTIFIERS for b in _ORIGINAL_IDENTIFIERS if a != b],
)
def test_rejects_every_other_original_requests_identifier(expected, other):
    assert not recalled_passphrase(other.replace("-", "\u2011"), expected)


@pytest.mark.parametrize(
    "output",
    [
        "cobalt-heron-420",
        "cobalt-heron-42-extra",
        "prefixcobalt-heron-42",
        "prefix-cobalt-heron-42",
        "cobalt-heron-42_suffix",
        "cobalt-heron-42\u0661",
        "cobalt-heron-4\u0662",
        "cobalt-heron-\uff14\uff12",
        "cobalt_heron_42",
        "cobalt heron 42",
        "cobalt\u2212heron\u221242",
        "cobalt\u2013heron\u201342",
    ],
)
def test_rejects_embedded_or_partial_identifiers(output):
    assert not recalled_passphrase(output, "cobalt-heron-42")


def test_rejects_longer_request_suffix():
    assert not recalled_passphrase(
        "cobalt\u2011heron\u20110\u201100", "cobalt-heron-0-0"
    )
