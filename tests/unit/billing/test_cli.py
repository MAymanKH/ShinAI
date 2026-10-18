from types import SimpleNamespace

import pytest

from shin_ai.billing import __main__ as cli
from shin_ai.billing.store import CreditStore, Identity


@pytest.mark.asyncio
async def test_grant_replay_and_balance(tmp_path, monkeypatch, capsys):
    database = tmp_path / "credits.sqlite3"
    settings = SimpleNamespace(billing=SimpleNamespace(enabled=True, database_path=database))
    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)
    args = cli.parser().parse_args(
        [
            "grant",
            "--platform",
            "whatsapp",
            "--user",
            "123:4@s.whatsapp.net",
            "--credits",
            "20",
            "--reference",
            "manual-001",
        ]
    )
    await cli.run(args)
    await cli.run(args)
    assert (await CreditStore(database).balance(Identity("whatsapp", "123@s.whatsapp.net"))).available == 20
    assert "Available credits: 20" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_disabled_cli_creates_no_database(tmp_path, monkeypatch):
    database = tmp_path / "credits.sqlite3"
    settings = SimpleNamespace(billing=SimpleNamespace(enabled=False, database_path=database))
    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)
    args = cli.parser().parse_args(["balance", "--platform", "discord", "--user", "1"])
    with pytest.raises(ValueError, match="Enable billing"):
        await cli.run(args)
    assert not database.exists()


@pytest.mark.asyncio
async def test_cli_rejects_negative_grant(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        billing=SimpleNamespace(enabled=True, database_path=tmp_path / "credits.sqlite3")
    )
    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)
    args = cli.parser().parse_args(
        [
            "grant",
            "--platform",
            "telegram",
            "--user",
            "1",
            "--credits",
            "-1",
            "--reference",
            "bad",
        ]
    )
    with pytest.raises(ValueError):
        await cli.run(args)


def test_grant_requires_reference():
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["grant", "--platform", "telegram", "--user", "1", "--credits", "5"])
