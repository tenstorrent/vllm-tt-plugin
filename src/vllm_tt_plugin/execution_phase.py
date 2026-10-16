# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Optional host phase observation, without changing device completion boundaries."""


def finish_observed_phase(finish, phase, grammar_output):
    """Observe the existing synchronous sampler's completion."""
    result = finish(grammar_output)
    phase.complete()
    return result
