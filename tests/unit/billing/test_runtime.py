import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from shin_ai.billing import runtime
from shin_ai.billing.store import Balance, CreditStore, Identity, Scope

OWNER = Identity("discord", "payer")
CHAT = Scope("discord", "chat", "channel")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def close_runtime_between_tests():
    asyncio.run(runtime.close_billing())
    yield
    asyncio.run(runtime.close_billing())


def configure_runtime(monkeypatch, database, *, enabled=True):
    settings = SimpleNamespace(
        billing=SimpleNamespace(enabled=enabled, database_path=database, web=SimpleNamespace(enabled=False))
    )
    monkeypatch.setattr(runtime, "get_settings", lambda: settings)


def subprocess_python(code, database):
    return subprocess.run(
        [sys.executable, "-c", code, str(database)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )


def test_disabled_runtime_creates_no_directories(tmp_path, monkeypatch):
    directory = tmp_path / "unused" / "billing"
    configure_runtime(monkeypatch, directory / "credits.sqlite3", enabled=False)
    assert runtime.get_credit_store() is None
    asyncio.run(runtime.initialize_billing())
    assert not directory.exists()


def test_second_process_cannot_own_database_but_cli_can_grant(tmp_path, monkeypatch):
    database = tmp_path / "billing" / "credits.sqlite3"
    configure_runtime(monkeypatch, database)
    store = runtime.get_credit_store()
    assert store is runtime.get_credit_store()

    locked = subprocess_python(
        """
import sys
from pathlib import Path
from types import SimpleNamespace
from shin_ai.billing import runtime
runtime.get_settings = lambda: SimpleNamespace(
    billing=SimpleNamespace(enabled=True, database_path=Path(sys.argv[1]))
)
try:
    runtime.get_credit_store()
except BlockingIOError:
    print("ownership blocked")
else:
    raise AssertionError("A second process acquired billing ownership")
""",
        database,
    )
    assert locked.stdout.strip() == "ownership blocked"

    granted = subprocess_python(
        """
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from shin_ai.billing import __main__ as cli
cli.load_settings = lambda _path: SimpleNamespace(
    billing=SimpleNamespace(enabled=True, database_path=Path(sys.argv[1]))
)
args = cli.parser().parse_args([
    "grant", "--platform", "discord", "--user", "payer", "--amount-usd", "0.000005",
    "--reference", "while-bot-running",
])
asyncio.run(cli.run(args))
""",
        database,
    )
    assert "Available balance: $0.000005" in granted.stdout
    assert asyncio.run(store.balance(OWNER)) == Balance(5, 0)


def test_initialization_recovers_only_after_runtime_owns_database(tmp_path, monkeypatch):
    async def scenario():
        database = tmp_path / "credits.sqlite3"
        store = CreditStore(database)
        await store.grant(OWNER, 2, "grant")
        await store.configure(OWNER, CHAT, "only_me")
        reservation = await store.reserve("interrupted", OWNER, (CHAT,), 1)
        assert reservation is not None
        configure_runtime(monkeypatch, database)

        reopened = runtime.get_credit_store()
        assert await reopened.balance(OWNER) == Balance(1, 1)
        assert runtime._ownership_file is not None
        assert not runtime._ownership_file.closed
        await runtime.initialize_billing()
        assert await reopened.balance(OWNER) == Balance(2, 0)
        active = await reopened.reserve("live-after-startup", OWNER, (CHAT,), 1)
        assert active is not None
        await runtime.initialize_billing()
        assert await reopened.balance(OWNER) == Balance(1, 1)

    asyncio.run(scenario())


def test_closing_runtime_releases_process_ownership(tmp_path, monkeypatch):
    database = tmp_path / "credits.sqlite3"
    configure_runtime(monkeypatch, database)
    original = runtime.get_credit_store()
    ownership_file = runtime._ownership_file
    asyncio.run(runtime.close_billing())
    assert ownership_file.closed

    acquired = subprocess_python(
        """
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from shin_ai.billing import runtime
runtime.get_settings = lambda: SimpleNamespace(
    billing=SimpleNamespace(enabled=True, database_path=Path(sys.argv[1]))
)
assert runtime.get_credit_store() is not None
asyncio.run(runtime.close_billing())
print("ownership acquired")
""",
        database,
    )
    assert acquired.stdout.strip() == "ownership acquired"
    assert runtime.get_credit_store() is not original


@pytest.mark.asyncio
async def test_failed_web_start_releases_ownership_and_allows_retry(tmp_path, monkeypatch):
    from shin_ai.billing import web

    settings = SimpleNamespace(
        billing=SimpleNamespace(
            enabled=True, database_path=tmp_path / "billing.sqlite3", web=SimpleNamespace(enabled=True)
        )
    )
    monkeypatch.setattr(runtime, "get_settings", lambda: settings)

    async def fail(*_args):
        raise OSError("port already bound")

    monkeypatch.setattr(web, "start_web", fail)
    with pytest.raises(OSError, match="port already bound"):
        await runtime.initialize_billing()
    assert runtime._ownership_file is None
    assert not runtime._initialized

    class Runner:
        async def cleanup(self):
            pass

    runner = Runner()

    async def succeed(*_args):
        return runner

    monkeypatch.setattr(web, "start_web", succeed)
    await runtime.initialize_billing()
    assert runtime._initialized and runtime._web_runner is runner
