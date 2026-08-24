import uuid

from shin_ai.utils.logger_config import bind_log_context, setup_logger


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
