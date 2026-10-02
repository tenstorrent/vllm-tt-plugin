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
