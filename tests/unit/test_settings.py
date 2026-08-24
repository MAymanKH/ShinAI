from shin_ai import settings


def test_suite_always_uses_the_example_configuration() -> None:
    expected_path = settings.PROJECT_ROOT / "config.yaml.example"

    assert settings.DEFAULT_CONFIG_PATH == expected_path
    parsed = settings.load_settings(expected_path)
    assert parsed.coordination.backend == "sqlite"
    assert parsed.coordination.reply_state_ttl_seconds == 86_400
    assert parsed.runtime.typing_action_timeout_seconds == 2
    assert parsed.runtime.context_message_chars == 4_000
    assert parsed.runtime.media_max_items == 5
    assert parsed.runtime.media_max_total_bytes == 20_000_000
    assert parsed.chroma.path == settings.PROJECT_ROOT / "chroma_db"
    assert parsed.ai.primary in parsed.ai.providers
