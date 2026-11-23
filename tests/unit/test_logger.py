# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks TT logger setup and warning deduplication through emitted log records."""

import inspect
import logging
import subprocess
import sys
from collections.abc import Iterator

import pytest

from vllm_tt_plugin.logger import _log_once, init_tt_logger


@pytest.fixture
def logger(caplog: pytest.LogCaptureFixture) -> Iterator[logging.Logger]:
    """Captures one TT logger while restoring its attributes and shared once-cache."""
    logger = logging.getLogger("vllm.tt.unit_logger_probe")
    saved = logger.__dict__.copy()
    logger.handlers = [caplog.handler]
    logger.filters = []
    logger.propagate = False
    logger.disabled = False
    logger.setLevel(logging.WARNING)
    _log_once.cache_clear()
    try:
        yield init_tt_logger("vllm_tt_plugin.unit_logger_probe")
    finally:
        _log_once.cache_clear()
        logger.__dict__.clear()
        logger.__dict__.update(saved)
        logger.setLevel(saved["level"])


class TestInitTTLogger:
    """Checks namespace selection, shared identity, and TT filename tagging."""

    def test_import_needs_only_standard_library(self) -> None:
        """Plugin discovery can import logging before Torch, vLLM, or TTNN exist."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules.update(torch=None, vllm=None, ttnn=None); "
                "from vllm_tt_plugin.logger import init_tt_logger; "
                "print(init_tt_logger('bootstrap_probe').name)",
            ],
            text=True,
            capture_output=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "vllm.tt.bootstrap_probe"

    def test_repeated_setup_keeps_one_filter(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Repeated setup reuses the logger and prefixes emitted filenames once."""
        again = init_tt_logger("vllm_tt_plugin.unit_logger_probe")

        logger.warning("tag probe")

        assert again is logger
        assert logger.name == "vllm.tt.unit_logger_probe"
        assert len(caplog.records) == 1
        assert caplog.records[0].filename == "tt/test_logger.py"

    def test_setup_preserves_logging_configuration(
        self, logger: logging.Logger
    ) -> None:
        """Initialization leaves handlers, level, propagation, and disabling alone."""
        handlers = logger.handlers.copy()
        logger.setLevel(logging.ERROR)
        logger.disabled = True

        init_tt_logger("vllm_tt_plugin.unit_logger_probe")

        assert logger.handlers == handlers
        assert logger.level == logging.ERROR
        assert logger.disabled is True
        assert logger.propagate is False

    def test_setup_shadows_an_existing_warning_once(
        self,
        logger: logging.Logger,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An inherited extension cannot replace the instance's TT deduplication."""

        def inherited_warning_once(*args: object, **kwargs: object) -> None:
            """Fails if initialization leaves the class-level extension in use."""
            pytest.fail("Inherited warning_once was not replaced")

        monkeypatch.delattr(logger, "warning_once")
        monkeypatch.setattr(
            logging.Logger, "warning_once", inherited_warning_once, raising=False
        )

        init_tt_logger("vllm_tt_plugin.unit_logger_probe").warning_once("probe")

        assert [record.getMessage() for record in caplog.records] == ["probe"]


class TestWarningOnce:
    """Checks rendered-message keys, cache lifetime, and standard logging options."""

    @pytest.mark.parametrize(
        "args",
        [
            pytest.param(("all",), id="string"),
            pytest.param(({"mode": "all"},), id="unhashable-dict"),
            pytest.param(([1, 2],), id="unhashable-list"),
        ],
    )
    def test_repeated_arguments_emit_one_warning(
        self,
        logger: logging.Logger,
        caplog: pytest.LogCaptureFixture,
        args: tuple[object, ...],
    ) -> None:
        """Deduplication accepts ordinary and unhashable formatting arguments."""
        logger.warning_once("value=%s", *args)
        logger.warning_once("value=%s", *args)

        assert len(caplog.records) == 1
        assert caplog.records[0].getMessage() == f"value={args[0]}"

    def test_deduplicates_rendered_text(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Different format strings with the same rendered text share an entry."""
        logger.warning_once("value=%s", 3)
        logger.warning_once("value=3")
        logger.warning_once("value=%s", 4)

        assert [record.getMessage() for record in caplog.records] == [
            "value=3",
            "value=4",
        ]

    def test_leaves_malformed_formatting_for_the_handler(
        self,
        logger: logging.Logger,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Key generation preserves malformed messages for a non-formatting handler."""
        records: list[logging.LogRecord] = []
        monkeypatch.setattr(caplog.handler, "emit", records.append)

        logger.warning_once("%s %s", "one argument")
        logger.warning_once("%s %s", "one argument")

        (record,) = records
        assert record.msg == "%s %s"
        assert record.args == ("one argument",)

    def test_does_not_format_or_remember_disabled_messages(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A suppressed warning neither formats arguments nor consumes its entry."""

        class RecordingArg:
            """Counts string conversions requested by logging."""

            def __init__(self) -> None:
                """Starts with no rendering attempts."""
                self.render_count = 0

            def __str__(self) -> str:
                """Records one formatting operation."""
                self.render_count += 1
                return "probe"

        argument = RecordingArg()
        logger.setLevel(logging.ERROR)
        logger.warning_once("value=%s", argument)
        assert argument.render_count == 0
        assert caplog.records == []

        logger.setLevel(logging.WARNING)
        logger.warning_once("value=%s", argument)
        assert [record.getMessage() for record in caplog.records] == ["value=probe"]

    def test_cache_eviction_allows_a_warning_again(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The bounded 128-entry cache does not promise suppression forever."""
        logger.warning_once("oldest")
        for index in range(128):
            logger.warning_once("entry %s", index)
        logger.warning_once("oldest")

        assert len(caplog.records) == 130
        assert caplog.records[0].getMessage() == "oldest"
        assert caplog.records[-1].getMessage() == "oldest"

    def test_forwards_exception_and_context(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception information and extra context reach the first emitted record."""
        error = ValueError("bad row")
        try:
            raise error
        except ValueError:
            logger.warning_once("failed", exc_info=True, extra={"request_id": "r1"})

        (record,) = caplog.records
        assert record.exc_info is not None
        assert record.exc_info[1] is error
        assert record.request_id == "r1"

    @pytest.mark.parametrize("stacklevel", [1, 2], ids=["caller", "callers-caller"])
    def test_attributes_record_to_requested_caller(
        self,
        logger: logging.Logger,
        caplog: pytest.LogCaptureFixture,
        stacklevel: int,
    ) -> None:
        """The warning wrapper preserves logging's caller-relative stacklevel."""

        def emit() -> int:
            """Returns the line that emits the warning."""
            line = inspect.currentframe().f_lineno + 1
            logger.warning_once("location", stacklevel=stacklevel, stack_info=True)
            return line

        outer_line = inspect.currentframe().f_lineno + 1
        inner_line = emit()

        (record,) = caplog.records
        assert record.lineno == (inner_line if stacklevel == 1 else outer_line)
        assert record.stack_info is not None

    def test_context_does_not_change_the_deduplication_key(
        self, logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Repeated text keeps the first record's context instead of emitting again."""
        logger.warning_once("same", extra={"request_id": "first"})
        logger.warning_once("same", extra={"request_id": "second"})

        (record,) = caplog.records
        assert record.request_id == "first"
