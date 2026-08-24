import logging
import uuid

from concurrent_log_handler import ConcurrentRotatingFileHandler

from shin_ai.utils.logger_config import (
    ThirdPartyNoiseFilter,
    bind_log_context,
    setup_logger,
)


def test_file_logs_include_event_context_and_source(tmp_path) -> None:
    log_path = tmp_path / "app.log"
    test_logger = setup_logger(
        f"test.{uuid.uuid4().hex}",
        log_file=log_path,
        debug=True,
        max_bytes=100_000,
        backup_count=2,
    )
    try:
        with bind_log_context(
            interaction_id="abc123",
            platform="telegram",
            chat_id="chat",
            message_id="msg",
            user_id="user",
        ):
            test_logger.info("sent reply", extra={"event_name": "response.sent"})
        for handler in test_logger.handlers:
            handler.flush()

        contents = log_path.read_text(encoding="utf-8")
        assert "response.sent" in contents
        assert "rid=abc123 platform=telegram chat=chat msg=msg user=user" in contents
        assert "test_logger_config:test_file_logs_include_event_context_and_source" in contents
        assert "sent reply" in contents
    finally:
        for handler in test_logger.handlers:
            handler.close()
        test_logger.handlers.clear()


def test_file_handler_rotates_at_configured_size(tmp_path) -> None:
    log_path = tmp_path / "rotating.log"
    test_logger = setup_logger(
        f"test.{uuid.uuid4().hex}",
        log_file=log_path,
        debug=False,
        max_bytes=200,
        backup_count=2,
    )
    try:
        for index in range(10):
            test_logger.info("line=%d %s", index, "x" * 80)
        for handler in test_logger.handlers:
            handler.flush()

        assert any(
            isinstance(handler, ConcurrentRotatingFileHandler)
            for handler in test_logger.handlers
        )
        assert log_path.exists()
        assert (tmp_path / "rotating.log.1").exists()
    finally:
        for handler in test_logger.handlers:
            handler.close()
        test_logger.handlers.clear()


def test_console_warnings_include_navigable_source(capsys) -> None:
    test_logger = setup_logger(
        f"test.{uuid.uuid4().hex}",
        log_file=None,
        debug=False,
    )
    try:
        test_logger.warning("something failed")
        output = capsys.readouterr().out

        assert "something failed" in output
        assert "source=test_logger_config:test_console_warnings_include_navigable_source:" in output
    finally:
        test_logger.handlers.clear()


def test_info_console_uses_compact_context_and_fallback_event(capsys) -> None:
    test_logger = setup_logger(
        f"test.{uuid.uuid4().hex}",
        log_file=None,
        debug=False,
    )
    try:
        with bind_log_context(
            interaction_id="abc123",
            platform="whatsapp",
            chat_id="very-long-chat-id",
            message_id="very-long-message-id",
            user_id="very-long-user-id",
        ):
            test_logger.info("processed")
        output = capsys.readouterr().out

        assert "test_logger_config.info" in output
        assert "rid=abc123 platform=whatsapp" in output
        assert "very-long-chat-id" not in output
        assert "very-long-message-id" not in output
        assert "very-long-user-id" not in output
    finally:
        test_logger.handlers.clear()


def test_debug_console_keeps_full_context(capsys) -> None:
    test_logger = setup_logger(
        f"test.{uuid.uuid4().hex}",
        log_file=None,
        debug=True,
    )
    try:
        with bind_log_context(
            interaction_id="abc123",
            platform="whatsapp",
            chat_id="chat-id",
            message_id="message-id",
            user_id="user-id",
        ):
            test_logger.debug("processed")
        output = capsys.readouterr().out

        assert "rid=abc123 platform=whatsapp chat=chat-id msg=message-id user=user-id" in output
    finally:
        test_logger.handlers.clear()


def test_repeated_sdk_advisories_are_suppressed() -> None:
    noise_filter = ThirdPartyNoiseFilter()
    record = logging.LogRecord(
        "huggingface_hub.utils._http",
        30,
        __file__,
        1,
        "You are sending unauthenticated requests to the HF Hub.",
        (),
        None,
    )

    assert not noise_filter.filter(record)


def test_errors_capture_active_exception_traceback(tmp_path) -> None:
    log_path = tmp_path / "errors.log"
    test_logger = setup_logger(
        f"test.{uuid.uuid4().hex}",
        log_file=log_path,
        debug=False,
    )
    try:
        try:
            raise ValueError("bad value")
        except ValueError as error:
            test_logger.error("operation failed: %s", error)
        for handler in test_logger.handlers:
            handler.flush()

        contents = log_path.read_text(encoding="utf-8")
        assert "Traceback (most recent call last)" in contents
        assert "ValueError: bad value" in contents
    finally:
        for handler in test_logger.handlers:
            handler.close()
        test_logger.handlers.clear()
