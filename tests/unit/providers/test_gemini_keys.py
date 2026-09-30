import json
from dataclasses import replace

import pytest

from shin_ai.providers import gemini_keys
from shin_ai.settings import ProviderSettings, get_settings


@pytest.fixture
def key_file(monkeypatch, tmp_path):
    path = tmp_path / "data" / "gemini_keys.json"
    monkeypatch.setattr(gemini_keys, "GEMINI_KEYS_FILE", path)
    monkeypatch.setattr(gemini_keys, "DATA_DIR", path.parent)
    monkeypatch.setattr(gemini_keys, "_api_keys", None)
    return path


def test_startup_replaces_stale_keys_and_refreshes_cached_keys(key_file) -> None:
    key_file.parent.mkdir()
    key_file.write_text('{"removed": "old-secret"}')
    assert gemini_keys.get_api_keys() == {"removed": "old-secret"}
    providers = {
        "first": ProviderSettings(name="first", type="gemini", api_keys={"one": "secret-one"}),
        "second": ProviderSettings(name="second", type="gemini", api_keys={"two": "secret-two"}),
        "other": ProviderSettings(name="other", type="openai", api_key="openai-secret"),
    }
    settings = replace(get_settings().ai, providers=providers)

    gemini_keys.initialize_keys(settings)

    expected = {"one": "secret-one", "two": "secret-two"}
    assert json.loads(key_file.read_text()) == expected
    assert gemini_keys.get_api_keys() == expected
    gemini_keys.initialize_keys(settings)
    assert gemini_keys.load_keys() == expected


def test_startup_creates_key_file_and_clears_keys_without_gemini_providers(key_file) -> None:
    gemini_keys.initialize_keys(get_settings().ai)
    assert gemini_keys.get_api_keys()

    gemini_keys.initialize_keys(replace(get_settings().ai, providers={}))

    assert json.loads(key_file.read_text()) == {}
    assert gemini_keys.get_api_keys() == {}
