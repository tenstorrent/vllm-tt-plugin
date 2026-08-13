# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from vllm_tt_plugin.logger import init_tt_logger


def test_tt_logger_provides_once_methods():
    """*_once call sites must work regardless of whether transformers (which
    patches warning_once onto logging.Logger) has been imported yet."""
    logger = init_tt_logger("vllm_tt_plugin.logger_probe")

    logger.warning_once("probe warning %s", "arg")
    logger.info_once("probe info")
    # Second call with identical arguments must be deduplicated silently.
    logger.warning_once("probe warning %s", "arg")
