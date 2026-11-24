# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Defines shared exception categories for plugin-owned failures.

``VTTError`` is the shared base for deliberately raised plugin failures.
``VTTValueError`` and ``VTTRuntimeError`` also preserve their built-in categories
for existing callers. Dependency exceptions retain their original types unless
an owning boundary explicitly translates them. This module imports no runtime
components and does not depend on feature-specific exception hierarchies.

Examples
--------
>>> from vllm_tt_plugin.exceptions import VTTError, VTTValueError
>>> error = VTTValueError("invalid plugin argument")
>>> isinstance(error, VTTError), isinstance(error, ValueError)
(True, True)
"""

__all__ = ("VTTError", "VTTValueError", "VTTRuntimeError")


class VTTError(Exception):
    """Base class for plugin-owned failures."""


class VTTValueError(VTTError, ValueError):
    """Reports invalid argument values while preserving ValueError compatibility."""


class VTTRuntimeError(VTTError, RuntimeError):
    """Reports inconsistent execution state with RuntimeError compatibility."""
