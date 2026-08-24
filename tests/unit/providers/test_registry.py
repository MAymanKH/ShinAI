import pytest

from shin_ai.providers.registry import _parse_config


def _minimal_config() -> dict:
    return {
        "platform": {
            "telegram": {"enabled": False},
            "discord": {"enabled": False},
            "whatsapp": {"enabled": False},
        },
        "admin_user_id": 1,
        "debug": False,
        "ai": {
            "providers": [
                {
                    "name": "gemini",
                    "type": "gemini",
                    "models": ["gemini-test"],
                }
            ],
            "primary": "gemini",
            "fallbacks": [],
            "rotation": "failover",
        },
    }


def test_parse_config_applies_stable_defaults() -> None:
    config = _parse_config(_minimal_config())

    assert config.ai.timeout_seconds == 60
    assert config.ai.max_retries == 3
    assert config.ai.global_timeout_seconds == 180
    assert config.random_trigger_probability == 0.05
    assert config.whisper.model == "large-v3-turbo"
    assert config.whisper.timeout_seconds == 180
    assert config.whisper.max_file_bytes == 25_000_000
    assert config.ai.providers["gemini"].models == ("gemini-test",)
    assert config.runtime.max_concurrent_interactions == 24
    assert config.runtime.platform_message_cache_size == 500
    assert config.coordination.backend == "sqlite"
    assert config.coordination.reply_state_ttl_seconds == 86_400
    assert config.chroma.path.name == "chroma_db"


def test_parse_config_rejects_unknown_fallback() -> None:
    raw = _minimal_config()
    raw["ai"]["fallbacks"] = ["missing"]

    with pytest.raises(ValueError, match="Fallback provider 'missing'"):
        _parse_config(raw)


def test_parse_config_rejects_duplicate_provider_names() -> None:
    raw = _minimal_config()
    raw["ai"]["providers"].append({"name": "gemini", "type": "gemini", "models": ["other-model"]})

    with pytest.raises(ValueError, match="Duplicate provider name"):
        _parse_config(raw)


def test_parse_config_reads_runtime_coordination_and_logging(tmp_path) -> None:
    raw = _minimal_config()
    raw.update(
        {
            "logging": {"debug": True, "file": None, "content_preview_chars": 80},
            "runtime": {
                "max_concurrent_interactions": 12,
                "context": {"max_chats": 250, "ttl_seconds": 60},
            },
            "coordination": {
                "backend": "sqlite",
                "namespace": "shared-bot",
                "database_path": "state/coordination.sqlite3",
            },
            "chroma": {"path": "state/chroma"},
        }
    )

    config = _parse_config(raw, project_root=tmp_path)

    assert config.debug is True
    assert config.logging.file is None
    assert config.runtime.max_concurrent_interactions == 12
    assert config.runtime.context_max_chats == 250
    assert config.coordination.namespace == "shared-bot"
    assert config.coordination.database_path == tmp_path / "state/coordination.sqlite3"
    assert config.chroma.path == tmp_path / "state/chroma"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("runtime", "max_concurrent_interactions"), 0, "must be greater than zero"),
        (("coordination", "backend"), "redis", "must be 'sqlite' or 'memory'"),
        (("response", "random_trigger_probability"), 1.5, "must be between 0 and 1"),
    ],
)
def test_parse_config_rejects_invalid_operational_limits(path, value, message) -> None:
    raw = _minimal_config()
    section, key = path
    raw.setdefault(section, {})[key] = value

    with pytest.raises(ValueError, match=message):
        _parse_config(raw)
