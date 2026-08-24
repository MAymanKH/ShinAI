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
    assert config.ai.providers["gemini"].models == ["gemini-test"]


def test_parse_config_rejects_unknown_fallback() -> None:
    raw = _minimal_config()
    raw["ai"]["fallbacks"] = ["missing"]

    with pytest.raises(ValueError, match="Fallback provider 'missing'"):
        _parse_config(raw)


def test_parse_config_rejects_duplicate_provider_names() -> None:
    raw = _minimal_config()
    raw["ai"]["providers"].append(
        {"name": "gemini", "type": "gemini", "models": ["other-model"]}
    )

    with pytest.raises(ValueError, match="Duplicate provider name"):
        _parse_config(raw)

