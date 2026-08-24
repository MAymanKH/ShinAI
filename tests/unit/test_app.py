"""The composition root decides which platforms exist and survives their failures."""

import asyncio

import pytest

from shin_ai import app as app_module
from shin_ai.app import Application
from shin_ai.settings import get_settings


class _Platform:
    def __init__(self, name, *, fail_start=False):
        self.name = name
        self.fail_start = fail_start
        self.started = False
        self.stopped = False

    async def start(self):
        if self.fail_start:
            raise RuntimeError(f"{self.name} refused to start")
        self.started = True

    async def stop(self):
        self.stopped = True


@pytest.fixture
def wire(monkeypatch):
    """Stub every handler module's register() and the Telegram client factory."""

    def _wire(telegram=None, discord=None, whatsapp=None, *, discord_raises=False):
        from shin_ai.core import client as client_module
        from shin_ai.handlers import analytics, discord_chat, stats, telegram_chat, whatsapp_chat

        registered = []
        monkeypatch.setattr(client_module, "create_telegram_client", lambda _p: "tg-client")
        monkeypatch.setattr(
            telegram_chat, "register", lambda client: (registered.append(client), telegram)[1]
        )
        monkeypatch.setattr(stats, "register", lambda client: registered.append(("stats", client)))
        monkeypatch.setattr(analytics, "register", lambda client: registered.append(("analytics", client)))

        def discord_register():
            if discord_raises:
                raise RuntimeError("discord blew up during registration")
            return discord

        monkeypatch.setattr(discord_chat, "register", discord_register)
        monkeypatch.setattr(whatsapp_chat, "register", lambda: whatsapp)
        return registered

    return _wire


class TestRegistration:
    def test_registers_only_the_platforms_that_return_an_adapter(self, wire) -> None:
        wire(telegram=_Platform("tg"), discord=None, whatsapp=_Platform("wa"))
        application = Application.build(get_settings())
        assert [label for label, _ in application.platforms] == ["Telegram", "WhatsApp"]

    def test_admin_commands_attach_only_when_telegram_registers(self, wire) -> None:
        registered = wire(telegram=_Platform("tg"))
        Application.build(get_settings())
        assert ("stats", "tg-client") in registered
        assert ("analytics", "tg-client") in registered

    def test_admin_commands_are_skipped_without_telegram(self, wire) -> None:
        registered = wire(telegram=None, discord=_Platform("dc"))
        Application.build(get_settings())
        assert not any(entry[0] == "stats" for entry in registered if isinstance(entry, tuple))

    def test_one_platform_failing_to_register_does_not_stop_the_others(self, wire) -> None:
        wire(telegram=_Platform("tg"), discord_raises=True, whatsapp=_Platform("wa"))
        application = Application.build(get_settings())
        assert [label for label, _ in application.platforms] == ["Telegram", "WhatsApp"]

    def test_no_platforms_configured_is_not_an_error(self, wire) -> None:
        wire()
        application = Application.build(get_settings())
        assert application.platforms == []


class TestStart:
    def test_returns_only_platforms_that_started(self, wire) -> None:
        good, bad = _Platform("tg"), _Platform("wa", fail_start=True)
        wire(telegram=good, whatsapp=bad)
        application = Application.build(get_settings())

        started = asyncio.run(application.start())

        assert [label for label, _ in started] == ["Telegram"]
        assert good.started is True
        assert bad.started is False

    def test_warns_when_nothing_is_active(self, wire, monkeypatch) -> None:
        warnings = []
        monkeypatch.setattr(
            app_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg % a if a else msg)
        )
        wire(telegram=_Platform("tg", fail_start=True))
        application = Application.build(get_settings())

        asyncio.run(application.start())

        assert any("No chat platforms are active" in w for w in warnings)

    def test_shutdown_only_stops_what_started(self, wire) -> None:
        good, bad = _Platform("tg"), _Platform("wa", fail_start=True)
        wire(telegram=good, whatsapp=bad)
        application = Application.build(get_settings())

        asyncio.run(application.start())
        asyncio.run(application.shutdown())

        assert good.stopped is True
        assert bad.stopped is False
