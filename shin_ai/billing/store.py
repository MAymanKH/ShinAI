"""Durable prepaid credits with atomic reservations and an append-only ledger."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class Identity:
    platform: str
    user_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", str(self.user_id))
        if not self.platform or not self.user_id:
            raise ValueError("An identity requires a platform and user ID")


@dataclass(frozen=True, slots=True)
class Scope:
    platform: str
    kind: Literal["chat", "guild"]
    id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        if not self.platform or not self.id or self.kind not in ("chat", "guild"):
            raise ValueError("Invalid usage scope")


@dataclass(frozen=True, slots=True)
class Balance:
    available: int
    reserved: int


@dataclass(frozen=True, slots=True)
class Reservation:
    reference: str
    owner: Identity
    credits: int


class DuplicateRequest(ValueError):
    """The interaction already has a reservation, including a terminal one."""


class SponsorConflict(ValueError):
    """Another account sponsors this scope."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    platform TEXT NOT NULL, user_id TEXT NOT NULL,
    available INTEGER NOT NULL DEFAULT 0 CHECK(available >= 0 AND typeof(available) = 'integer'),
    reserved INTEGER NOT NULL DEFAULT 0 CHECK(reserved >= 0 AND typeof(reserved) = 'integer'),
    PRIMARY KEY(platform, user_id)
);
CREATE TABLE IF NOT EXISTS permissions (
    owner_platform TEXT NOT NULL, owner_id TEXT NOT NULL,
    platform TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('chat','guild')), scope_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('only_me','everyone')),
    PRIMARY KEY(owner_platform, owner_id, platform, kind, scope_id),
    FOREIGN KEY(owner_platform, owner_id) REFERENCES wallets(platform, user_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_sponsor_per_scope
    ON permissions(platform,kind,scope_id) WHERE mode = 'everyone';
CREATE TABLE IF NOT EXISTS grants (
    reference TEXT PRIMARY KEY, owner_platform TEXT NOT NULL, owner_id TEXT NOT NULL,
    credits INTEGER NOT NULL CHECK(credits > 0),
    FOREIGN KEY(owner_platform, owner_id) REFERENCES wallets(platform, user_id)
);
CREATE TABLE IF NOT EXISTS reservations (
    reference TEXT PRIMARY KEY, owner_platform TEXT NOT NULL, owner_id TEXT NOT NULL,
    credits INTEGER NOT NULL CHECK(credits > 0),
    status TEXT NOT NULL CHECK(status IN ('held','settled','released')),
    FOREIGN KEY(owner_platform, owner_id) REFERENCES wallets(platform, user_id)
);
CREATE TABLE IF NOT EXISTS ledger (
    event_reference TEXT PRIMARY KEY, owner_platform TEXT NOT NULL, owner_id TEXT NOT NULL,
    available_delta INTEGER NOT NULL, reserved_delta INTEGER NOT NULL,
    kind TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(owner_platform, owner_id) REFERENCES wallets(platform, user_id)
);
CREATE TRIGGER IF NOT EXISTS immutable_ledger_update BEFORE UPDATE ON ledger
    BEGIN SELECT RAISE(ABORT, 'ledger is immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_ledger_delete BEFORE DELETE ON ledger
    BEGIN SELECT RAISE(ABORT, 'ledger is immutable'); END;
"""


def _validate_credits(credits: int) -> None:
    if type(credits) is not int or not 0 < credits <= 2**63 - 1:
        raise ValueError("Credits must be a positive 64-bit integer")


def _validate_reference(reference: str) -> None:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("A nonempty reference is required")


class CreditStore:
    """Each operation owns a connection; independent instances share SQLite locks.

    Construction does not recover reservations. The application must hold its
    exclusive billing lock before explicitly calling recover().
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def _transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout=15000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def _run[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        def transact() -> T:
            with self._transaction() as connection:
                return operation(connection)

        # SQLite work cannot be cancelled once its worker thread starts.
        task = asyncio.create_task(asyncio.to_thread(transact))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            result = await task
            if isinstance(result, Reservation):
                await self.release(result)
            raise

    @staticmethod
    def _wallet(connection: sqlite3.Connection, owner: Identity) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO wallets(platform,user_id) VALUES (?,?)",
            (owner.platform, owner.user_id),
        )

    @staticmethod
    def _balance(connection: sqlite3.Connection, owner: Identity) -> Balance:
        row = connection.execute(
            "SELECT available,reserved FROM wallets WHERE platform=? AND user_id=?",
            (owner.platform, owner.user_id),
        ).fetchone()
        return Balance(*row) if row else Balance(0, 0)

    @staticmethod
    def _record(connection, reference, owner, available, reserved, kind) -> None:
        connection.execute(
            "INSERT INTO ledger(event_reference,owner_platform,owner_id,available_delta,reserved_delta,kind) "
            "VALUES (?,?,?,?,?,?)",
            (reference, owner.platform, owner.user_id, available, reserved, kind),
        )

    async def balance(self, owner: Identity) -> Balance:
        return await self._run(lambda connection: self._balance(connection, owner))

    async def grant(self, owner: Identity, credits: int, reference: str) -> int:
        _validate_credits(credits)
        _validate_reference(reference)

        def grant(connection):
            row = connection.execute(
                "SELECT owner_platform,owner_id,credits FROM grants WHERE reference=?", (reference,)
            ).fetchone()
            if row:
                if row != (owner.platform, owner.user_id, credits):
                    raise ValueError("Grant reference already identifies a different grant")
                return self._balance(connection, owner).available
            self._wallet(connection, owner)
            connection.execute(
                "INSERT INTO grants VALUES (?,?,?,?)", (reference, owner.platform, owner.user_id, credits)
            )
            connection.execute(
                "UPDATE wallets SET available=available+? WHERE platform=? AND user_id=?",
                (credits, owner.platform, owner.user_id),
            )
            self._record(connection, f"grant:{reference}", owner, credits, 0, "grant")
            return self._balance(connection, owner).available

        return await self._run(grant)

    async def configure(
        self, owner: Identity, scope: Scope, mode: Literal["off", "only_me", "everyone"]
    ) -> None:
        if mode not in ("off", "only_me", "everyone"):
            raise ValueError("Invalid spending mode")
        if owner.platform != scope.platform:
            raise ValueError("Owner and scope must belong to the same platform")

        def configure(connection):
            self._wallet(connection, owner)
            key = (owner.platform, owner.user_id, scope.platform, scope.kind, scope.id)
            if mode == "off":
                connection.execute(
                    "DELETE FROM permissions WHERE owner_platform=? AND owner_id=? "
                    "AND platform=? AND kind=? AND scope_id=?",
                    key,
                )
                return
            try:
                connection.execute(
                    "INSERT INTO permissions VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(owner_platform,owner_id,platform,kind,scope_id) "
                    "DO UPDATE SET mode=excluded.mode",
                    (*key, mode),
                )
            except sqlite3.IntegrityError as error:
                raise SponsorConflict("Another account already sponsors this scope") from error

        await self._run(configure)

    async def reserve(
        self, reference: str, requester: Identity, scopes: tuple[Scope, ...], credits: int
    ) -> Reservation | None:
        _validate_reference(reference)
        _validate_credits(credits)
        if any(scope.platform != requester.platform for scope in scopes):
            raise ValueError("Requester and scopes must belong to the same platform")
        ordered_scopes = sorted(dict.fromkeys(scopes), key=lambda scope: scope.kind != "chat")

        def reserve(connection):
            if connection.execute("SELECT 1 FROM reservations WHERE reference=?", (reference,)).fetchone():
                raise DuplicateRequest(reference)
            own: list[Identity] = []
            shared: list[Identity] = []
            for scope in ordered_scopes:
                rows = connection.execute(
                    "SELECT owner_platform,owner_id,mode FROM permissions "
                    "WHERE platform=? AND kind=? AND scope_id=? ORDER BY owner_platform,owner_id",
                    (scope.platform, scope.kind, scope.id),
                ).fetchall()
                for owner_platform, owner_id, mode in rows:
                    owner = Identity(owner_platform, owner_id)
                    if owner == requester:
                        own.append(owner)
                    elif mode == "everyone":
                        shared.append(owner)
            for owner in dict.fromkeys([*own, *shared]):
                updated = connection.execute(
                    "UPDATE wallets SET available=available-?,reserved=reserved+? "
                    "WHERE platform=? AND user_id=? AND available>=?",
                    (credits, credits, owner.platform, owner.user_id, credits),
                )
                if updated.rowcount:
                    connection.execute(
                        "INSERT INTO reservations VALUES (?,?,?,?,'held')",
                        (reference, owner.platform, owner.user_id, credits),
                    )
                    self._record(connection, f"reserve:{reference}", owner, -credits, credits, "reserve")
                    return Reservation(reference, owner, credits)
            return None

        return await self._run(reserve)

    def _finish(self, connection, reservation: Reservation, status: str, *, recovery=False) -> None:
        row = connection.execute(
            "SELECT owner_platform,owner_id,credits,status FROM reservations WHERE reference=?",
            (reservation.reference,),
        ).fetchone()
        if row is None:
            raise ValueError("Unknown reservation")
        if row[:3] != (reservation.owner.platform, reservation.owner.user_id, reservation.credits):
            raise ValueError("Reservation details do not match")
        if row[3] != "held":
            return
        refunded = reservation.credits if status == "released" else 0
        connection.execute(
            "UPDATE wallets SET available=available+?,reserved=reserved-? WHERE platform=? AND user_id=?",
            (refunded, reservation.credits, reservation.owner.platform, reservation.owner.user_id),
        )
        connection.execute(
            "UPDATE reservations SET status=? WHERE reference=?", (status, reservation.reference)
        )
        kind = "recover" if recovery else status
        self._record(
            connection,
            f"finish:{reservation.reference}",
            reservation.owner,
            refunded,
            -reservation.credits,
            kind,
        )

    async def settle(self, reservation: Reservation) -> None:
        await self._run(lambda connection: self._finish(connection, reservation, "settled"))

    async def release(self, reservation: Reservation) -> None:
        await self._run(lambda connection: self._finish(connection, reservation, "released"))

    async def recover(self) -> int:
        """Refund held reservations while the caller holds the exclusive runtime lock."""

        def recover(connection):
            rows = connection.execute(
                "SELECT reference,owner_platform,owner_id,credits FROM reservations WHERE status='held'"
            ).fetchall()
            for reference, platform, user_id, credits in rows:
                self._finish(
                    connection,
                    Reservation(reference, Identity(platform, user_id), credits),
                    "released",
                    recovery=True,
                )
            return len(rows)

        return await self._run(recover)
