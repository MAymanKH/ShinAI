"""Hermetic test configuration bootstrap."""

import dataclasses

import pytest

from shin_ai import settings as settings_module

settings_module.DEFAULT_CONFIG_PATH = settings_module.PROJECT_ROOT / "config.yaml.example"


@pytest.fixture
def override_settings(monkeypatch):
    """Patch a module's get_settings with a modified copy of the real settings.

    Application modules read configuration through get_settings() at the point
    of use, so a test changes behaviour by swapping that function rather than
    by reaching in and rebinding module constants.

        override_settings(media, "runtime", media_max_items=2)
    """

    def _override(module, section: str | None = None, **values):
        base = settings_module.get_settings()
        if section is not None:
            replaced = dataclasses.replace(getattr(base, section), **values)
            base = dataclasses.replace(base, **{section: replaced})
        elif values:
            base = dataclasses.replace(base, **values)
        monkeypatch.setattr(module, "get_settings", lambda: base)
        return base

    return _override
