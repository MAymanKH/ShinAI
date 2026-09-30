from dataclasses import replace

import pytest

from shin_ai.providers import registry
from shin_ai.settings import ProviderSettings, get_settings, parse_settings


def _config(chats):
    return {
        "ai": {
            "primary": "dedicated",
            "providers": [
                {
                    "name": "dedicated",
                    "type": "gemini",
                    "models": ["model"],
                    "api_keys": {"key": "secret"},
                    "chats": chats,
                }
            ],
        }
    }


def test_chat_assignments_normalize_ids_and_match_platforms():
    provider = parse_settings(
        _config(
            {
                "telegram": [-100123, "-100456"],
                "discord": [123],
                "whatsapp": ["123:12@s.whatsapp.net"],
            }
        )
    ).ai.providers["dedicated"]

    assert provider.is_available_for("telegram", "-100123")
    assert provider.is_available_for("telegram", -100456)
    assert provider.is_available_for("discord", "123")
    assert provider.is_available_for("whatsapp", "123:7@s.whatsapp.net")
    assert not provider.is_available_for("telegram", 123)
    assert not provider.is_available_for("discord", -100123)
    assert not provider.is_available_for(None, None)


@pytest.mark.parametrize(
    "chats",
    [
        None,
        [],
        {"slack": [1]},
        {"telegram": []},
        {"telegram": "123"},
        {"telegram": [None]},
        {"telegram": [True]},
        {"telegram": [""]},
    ],
)
def test_invalid_chat_assignments_are_rejected(chats):
    with pytest.raises(ValueError, match="chats"):
        parse_settings(_config(chats))


@pytest.fixture
def configured_providers(monkeypatch):
    providers = {
        "shared": ProviderSettings(name="shared", type="openai"),
        "dedicated": ProviderSettings(name="dedicated", type="gemini", chats={"telegram": ("10", "20")}),
        "second": ProviderSettings(
            name="second", type="gemini", chats={"telegram": ("10",), "discord": ("30",)}
        ),
        "fallback": ProviderSettings(name="fallback", type="gemini"),
    }
    base = get_settings()
    settings = replace(
        base,
        ai=replace(
            base.ai,
            providers=providers,
            primary="shared",
            fallbacks=("dedicated", "fallback"),
            rotation="failover",
        ),
    )
    monkeypatch.setattr(registry, "get_settings", lambda: settings)
    return settings


@pytest.mark.parametrize(
    ("platform", "chat_id", "expected"),
    [
        ("telegram", 10, ["dedicated", "second", "shared", "fallback"]),
        ("telegram", 20, ["dedicated", "shared", "fallback"]),
        ("discord", 30, ["second", "shared", "fallback"]),
        ("discord", 10, ["shared", "fallback"]),
        ("telegram", 99, ["shared", "fallback"]),
        (None, None, ["shared", "fallback"]),
    ],
)
def test_assigned_providers_take_priority_only_in_their_chats(
    configured_providers, platform, chat_id, expected
):
    assert [p.name for p in registry.get_provider_chain(platform, chat_id)] == expected


def test_round_robin_keeps_assigned_providers_first(configured_providers, monkeypatch):
    settings = replace(configured_providers, ai=replace(configured_providers.ai, rotation="round_robin"))
    monkeypatch.setattr(registry, "get_settings", lambda: settings)
    chains = [registry.get_provider_chain("telegram", 10) for _ in range(2)]
    assert all([p.name for p in chain[:2]] == ["dedicated", "second"] for chain in chains)
    assert {chain[2].name for chain in chains} == {"shared", "fallback"}


def test_scoped_primary_is_unavailable_outside_its_chats(configured_providers, monkeypatch):
    settings = replace(
        configured_providers, ai=replace(configured_providers.ai, primary="dedicated", fallbacks=())
    )
    monkeypatch.setattr(registry, "get_settings", lambda: settings)
    assert registry.get_provider_chain("telegram", 99) == []
    assert [p.name for p in registry.get_provider_chain("telegram", 20)] == ["dedicated"]


def test_media_gemini_selection_respects_chat_assignments(configured_providers):
    assert registry.get_first_gemini_provider("telegram", 10).name == "dedicated"
    assert registry.get_first_gemini_provider("discord", 30).name == "second"
    assert registry.get_first_gemini_provider("discord", 10).name == "fallback"
    assert registry.get_first_gemini_provider().name == "fallback"
