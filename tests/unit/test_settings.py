from shin_ai import settings


def test_suite_always_uses_the_example_configuration() -> None:
    expected_path = settings.PROJECT_ROOT / "config.yaml.example"

    assert settings.DEFAULT_CONFIG_PATH == expected_path
    parsed = settings.load_settings(expected_path)
    assert parsed.coordination.backend == "sqlite"
    assert parsed.coordination.reply_state_ttl_seconds == 86_400
    assert parsed.ai.primary in parsed.ai.providers
