# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import logging

import pytest

from vllm_tt_plugin.logger import _log_once, init_tt_logger


@pytest.fixture(autouse=True)
def clear_log_once_cache():
    _log_once.cache_clear()
    yield
    _log_once.cache_clear()


def test_warning_once_deduplicates_identical_calls(caplog):
    logger = init_tt_logger("vllm_tt_plugin.logger_probe_dedup")

    # vLLM configures its logger tree with propagate=False, so records never
    # reach the root logger where caplog listens; attach its handler directly.
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="vllm.tt.logger_probe_dedup"):
            logger.warning_once("dedup probe %s", "arg")
            logger.warning_once("dedup probe %s", "arg")
    finally:
        logger.removeHandler(caplog.handler)

    records = [r for r in caplog.records if "dedup probe" in r.getMessage()]
    assert len(records) == 1
    assert records[0].getMessage() == "dedup probe arg"
