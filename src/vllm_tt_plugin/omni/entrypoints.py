# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


def omni_platform_plugin() -> str | None:
    """Return the TT Omni platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT Omni platform is not available because: %s", exc)
        return None
    return "vllm_tt_plugin.omni.platform.TTOmniPlatform"
