"""Transactional USD balances, request accounting, sessions, and payment receipts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import secrets
import sqlite3
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shin_ai.billing.types import Checkout, PaymentState, UsageCall

_UNIT = "usd_micros_v1"
_MAX = 2**63 - 1


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
    debt: int = 0


@dataclass(frozen=True, slots=True)
class Reservation:
    reference: str
    owner: Identity
    credits: int


@dataclass(frozen=True, slots=True)
class Session:
    owner: Identity
    csrf: str
    expires_at: float
    login_platform: str | None = None

    def __post_init__(self) -> None:
        if self.login_platform is None:
            object.__setattr__(self, "login_platform", self.owner.platform)


@dataclass(frozen=True, slots=True)
class Purchase:
    reference: str
    owner: Identity
    amount_cents: int
    currency: str
    credited_units: int
    quote: dict
    status: str
    order_id: str | None = None
    intention_id: str | None = None
    checkout_url: str | None = None
    transaction_id: str | None = None
    refunded_cents: int = 0
    frozen: bool = False


class DuplicateRequest(ValueError):
    """A request already has a durable reservation, including terminal records."""


class SponsorConflict(ValueError):
    """Another account sponsors this scope."""


class LegacyBalanceError(ValueError):
    """Old response credits require an explicit, audited currency conversion."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS billing_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS wallets (
    platform TEXT NOT NULL,user_id TEXT NOT NULL,
    available INTEGER NOT NULL DEFAULT 0 CHECK(available >= 0 AND typeof(available)='integer'),
    reserved INTEGER NOT NULL DEFAULT 0 CHECK(reserved >= 0 AND typeof(reserved)='integer'),
    debt INTEGER NOT NULL DEFAULT 0 CHECK(debt >= 0 AND typeof(debt)='integer'),
    PRIMARY KEY(platform,user_id)
);
CREATE TABLE IF NOT EXISTS identity_aliases (
    platform TEXT NOT NULL,user_id TEXT NOT NULL,owner_platform TEXT NOT NULL,owner_id TEXT NOT NULL,
    PRIMARY KEY(platform,user_id),
    FOREIGN KEY(owner_platform,owner_id) REFERENCES wallets(platform,user_id)
);
CREATE TABLE IF NOT EXISTS permissions (
    owner_platform TEXT NOT NULL,owner_id TEXT NOT NULL,
    platform TEXT NOT NULL,kind TEXT NOT NULL CHECK(kind IN ('chat','guild')),scope_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('only_me','everyone')),
    PRIMARY KEY(owner_platform,owner_id,platform,kind,scope_id),
    FOREIGN KEY(owner_platform,owner_id) REFERENCES wallets(platform,user_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_sponsor_per_scope
    ON permissions(platform,kind,scope_id) WHERE mode='everyone';
CREATE TABLE IF NOT EXISTS grants (
    reference TEXT PRIMARY KEY,owner_platform TEXT NOT NULL,owner_id TEXT NOT NULL,
    credits INTEGER NOT NULL CHECK(credits > 0),
    FOREIGN KEY(owner_platform,owner_id) REFERENCES wallets(platform,user_id)
);
CREATE TABLE IF NOT EXISTS reservations (
    reference TEXT PRIMARY KEY,owner_platform TEXT NOT NULL,owner_id TEXT NOT NULL,
    credits INTEGER NOT NULL CHECK(credits > 0),
    status TEXT NOT NULL CHECK(status IN ('held','settled','released')),
    markup_bps INTEGER NOT NULL DEFAULT 0,
    delivered INTEGER NOT NULL DEFAULT 0,charged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(owner_platform,owner_id) REFERENCES wallets(platform,user_id)
);
CREATE TABLE IF NOT EXISTS ledger (
    event_reference TEXT PRIMARY KEY,owner_platform TEXT NOT NULL,owner_id TEXT NOT NULL,
    available_delta INTEGER NOT NULL,reserved_delta INTEGER NOT NULL,kind TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,debt_delta INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(owner_platform,owner_id) REFERENCES wallets(platform,user_id)
);
CREATE TRIGGER IF NOT EXISTS money_ledger_update BEFORE UPDATE ON ledger
    BEGIN SELECT RAISE(ABORT,'ledger is immutable'); END;
CREATE TRIGGER IF NOT EXISTS money_ledger_delete BEFORE DELETE ON ledger
    BEGIN SELECT RAISE(ABORT,'ledger is immutable'); END;
CREATE TABLE IF NOT EXISTS call_usage (
    reference TEXT PRIMARY KEY,request_reference TEXT NOT NULL,attempt TEXT NOT NULL,
    provider TEXT NOT NULL,model TEXT NOT NULL,cost_nano_usd INTEGER,
    usage TEXT NOT NULL,price TEXT NOT NULL,billable INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(request_reference) REFERENCES reservations(reference),
    CHECK(cost_nano_usd IS NULL OR (cost_nano_usd >= 0 AND typeof(cost_nano_usd)='integer'))
);
CREATE TABLE IF NOT EXISTS login_tokens (
    digest TEXT PRIMARY KEY,owner_platform TEXT NOT NULL,owner_id TEXT NOT NULL,expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    digest TEXT PRIMARY KEY,owner_platform TEXT NOT NULL,owner_id TEXT NOT NULL,
    csrf TEXT NOT NULL,expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS purchases (
    reference TEXT PRIMARY KEY,owner_platform TEXT NOT NULL,owner_id TEXT NOT NULL,
    amount_cents INTEGER NOT NULL CHECK(amount_cents > 0),currency TEXT NOT NULL,
    credited_units INTEGER NOT NULL CHECK(credited_units > 0),quote TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',order_id TEXT UNIQUE,intention_id TEXT,checkout_url TEXT,
    transaction_id TEXT UNIQUE,refunded_cents INTEGER NOT NULL DEFAULT 0,
    funded INTEGER NOT NULL DEFAULT 0,refund_debited_units INTEGER NOT NULL DEFAULT 0,
    frozen INTEGER NOT NULL DEFAULT 0,refund_target_cents INTEGER,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(owner_platform,owner_id) REFERENCES wallets(platform,user_id)
);
"""


def _execute_schema(connection: sqlite3.Connection) -> None:
    statement = ""
    for line in _SCHEMA.splitlines():
        statement += line + "\n"
        if sqlite3.complete_statement(statement):
            connection.execute(statement)
            statement = ""


def _positive(value: int, label="Amount") -> None:
    if type(value) is not int or not 0 < value <= _MAX:
        raise ValueError(f"{label} must be a positive 64-bit integer")


def _nonnegative(value: int, label="Amount") -> None:
    if type(value) is not int or not 0 <= value <= _MAX:
        raise ValueError(f"{label} must be a nonnegative 64-bit integer")


def _reference(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty reference is required")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _ttl(value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Token lifetime must be positive and finite")


class CreditStore:
    """Money is stored in microUSD; every financial mutation is one SQLite transaction."""

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock

    @staticmethod
    def _tables(connection) -> set[str]:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def _schema(self, connection) -> None:
        tables = self._tables(connection)
        if "billing_meta" in tables:
            unit = connection.execute("SELECT value FROM billing_meta WHERE key='unit'").fetchone()
            if unit and unit[0] != _UNIT:
                raise LegacyBalanceError("Unsupported billing currency schema")
            if unit:
                version = connection.execute("SELECT value FROM billing_meta WHERE key='version'").fetchone()
                if version is None or version[0] != "1":
                    raise LegacyBalanceError("Unsupported billing schema version")
                return
        if "wallets" in tables:
            nonzero = connection.execute(
                "SELECT 1 FROM wallets WHERE available != 0 OR reserved != 0 LIMIT 1"
            ).fetchone()
            historical = any(
                name in tables and connection.execute(f"SELECT 1 FROM {name} LIMIT 1").fetchone()
                for name in ("grants", "ledger", "reservations")
            )
            if nonzero or historical:
                raise LegacyBalanceError(
                    "Legacy response credits need explicit conversion: run migrate-legacy with a microUSD-per-credit rate"
                )
        _execute_schema(connection)
        additions = {
            "wallets": {"debt": "INTEGER NOT NULL DEFAULT 0"},
            "reservations": {
                "markup_bps": "INTEGER NOT NULL DEFAULT 0",
                "delivered": "INTEGER NOT NULL DEFAULT 0",
                "charged": "INTEGER NOT NULL DEFAULT 0",
                "created_at": "TEXT",
            },
            "ledger": {"debt_delta": "INTEGER NOT NULL DEFAULT 0"},
        }
        for table, columns in additions.items():
            present = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns.items():
                if name not in present:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
        connection.executemany(
            "INSERT OR REPLACE INTO billing_meta VALUES (?,?)", (("unit", _UNIT), ("version", "1"))
        )

    @contextmanager
    def _transaction(self, *, legacy=False):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=15000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            if not legacy:
                self._schema(connection)
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def _run[T](self, operation: Callable[[sqlite3.Connection], T], *, legacy=False) -> T:
        def transact() -> T:
            with self._transaction(legacy=legacy) as connection:
                return operation(connection)

        task = asyncio.create_task(asyncio.to_thread(transact))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            result = await task
            if isinstance(result, Reservation):
                await self.release(result)
            raise

    @staticmethod
    def _canonical(connection, owner: Identity) -> Identity:
        row = connection.execute(
            "SELECT owner_platform,owner_id FROM identity_aliases WHERE platform=? AND user_id=?",
            (owner.platform, owner.user_id),
        ).fetchone()
        return Identity(row[0], row[1]) if row else owner

    @staticmethod
    def _owner_clause(connection, owner: Identity, prefix: str = "") -> tuple[str, tuple]:
        primary = CreditStore._canonical(connection, owner)
        aliases = connection.execute(
            "SELECT platform,user_id FROM identity_aliases WHERE owner_platform=? AND owner_id=?",
            (primary.platform, primary.user_id),
        ).fetchall()
        identities = list(dict.fromkeys([primary, *(Identity(row[0], row[1]) for row in aliases)]))
        clause = (
            "(" + " OR ".join(f"({prefix}owner_platform=? AND {prefix}owner_id=?)" for _ in identities) + ")"
        )
        values = tuple(value for identity in identities for value in (identity.platform, identity.user_id))
        return clause, values

    @staticmethod
    def _wallet(connection, owner: Identity) -> None:
        owner = CreditStore._canonical(connection, owner)
        connection.execute(
            "INSERT OR IGNORE INTO wallets(platform,user_id) VALUES (?,?)", (owner.platform, owner.user_id)
        )

    @staticmethod
    def _balance(connection, owner: Identity) -> Balance:
        owner = CreditStore._canonical(connection, owner)
        row = connection.execute(
            "SELECT available,reserved,debt FROM wallets WHERE platform=? AND user_id=?",
            (owner.platform, owner.user_id),
        ).fetchone()
        return Balance(*row) if row else Balance(0, 0)

    @staticmethod
    def _record(connection, reference, owner, available, reserved, kind, debt=0) -> None:
        connection.execute(
            "INSERT INTO ledger(event_reference,owner_platform,owner_id,available_delta,reserved_delta,kind,debt_delta) "
            "VALUES (?,?,?,?,?,?,?)",
            (reference, owner.platform, owner.user_id, available, reserved, kind, debt),
        )

    def _credit(self, connection, owner, amount, reference, kind) -> None:
        owner = self._canonical(connection, owner)
        self._wallet(connection, owner)
        old = self._balance(connection, owner)
        repaid = min(old.debt, amount)
        connection.execute(
            "UPDATE wallets SET available=available+?,debt=debt-? WHERE platform=? AND user_id=?",
            (amount - repaid, repaid, owner.platform, owner.user_id),
        )
        self._record(connection, reference, owner, amount - repaid, 0, kind, -repaid)

    def _debit(self, connection, owner, amount, reference, kind) -> None:
        owner = self._canonical(connection, owner)
        old = self._balance(connection, owner)
        used = min(old.available, amount)
        connection.execute(
            "UPDATE wallets SET available=available-?,debt=debt+? WHERE platform=? AND user_id=?",
            (used, amount - used, owner.platform, owner.user_id),
        )
        self._record(connection, reference, owner, -used, 0, kind, amount - used)

    async def balance(self, owner: Identity) -> Balance:
        return await self._run(lambda c: self._balance(c, owner))

    async def grant(self, owner: Identity, credits: int, reference: str) -> int:
        _positive(credits)
        _reference(reference)

        def grant(c):
            primary = self._canonical(c, owner)
            row = c.execute(
                "SELECT owner_platform,owner_id,credits FROM grants WHERE reference=?", (reference,)
            ).fetchone()
            if row:
                if self._canonical(c, Identity(row[0], row[1])) != primary or row[2] != credits:
                    raise ValueError("Grant reference already identifies a different grant")
                return self._balance(c, owner).available
            self._wallet(c, owner)
            c.execute(
                "INSERT INTO grants VALUES (?,?,?,?)", (reference, primary.platform, primary.user_id, credits)
            )
            self._credit(c, owner, credits, f"grant:{reference}", "grant")
            return self._balance(c, owner).available

        return await self._run(grant)

    async def configure(
        self, owner: Identity, scope: Scope, mode: Literal["off", "only_me", "everyone"]
    ) -> None:
        if mode not in ("off", "only_me", "everyone"):
            raise ValueError("Invalid spending mode")
        if owner.platform != scope.platform:
            raise ValueError("Owner and scope must belong to the same platform")

        def configure(c):
            self._wallet(c, owner)
            primary = self._canonical(c, owner)
            key = (primary.platform, primary.user_id, scope.platform, scope.kind, scope.id)
            if mode == "off":
                c.execute(
                    "DELETE FROM permissions WHERE owner_platform=? AND owner_id=? AND platform=? AND kind=? AND scope_id=?",
                    key,
                )
                return
            try:
                c.execute(
                    "INSERT INTO permissions VALUES (?,?,?,?,?,?) ON CONFLICT(owner_platform,owner_id,platform,kind,scope_id) DO UPDATE SET mode=excluded.mode",
                    (*key, mode),
                )
            except sqlite3.IntegrityError as error:
                raise SponsorConflict("Another account already sponsors this scope") from error

        await self._run(configure)

    async def reserve(
        self, reference: str, requester: Identity, scopes: tuple[Scope, ...], credits: int
    ) -> Reservation | None:
        _reference(reference)
        _positive(credits)
        if any(scope.platform != requester.platform for scope in scopes):
            raise ValueError("Requester and scopes must belong to the same platform")
        ordered = sorted(dict.fromkeys(scopes), key=lambda scope: scope.kind != "chat")

        def reserve(c):
            if c.execute("SELECT 1 FROM reservations WHERE reference=?", (reference,)).fetchone():
                raise DuplicateRequest(reference)
            primary_requester = self._canonical(c, requester)
            own, shared = [], []
            for scope in ordered:
                for row in c.execute(
                    "SELECT owner_platform,owner_id,mode FROM permissions WHERE platform=? AND kind=? AND scope_id=? ORDER BY owner_platform,owner_id",
                    (scope.platform, scope.kind, scope.id),
                ):
                    owner = self._canonical(c, Identity(row[0], row[1]))
                    if owner == primary_requester:
                        own.append(owner)
                    elif row[2] == "everyone":
                        shared.append(owner)
            for owner in dict.fromkeys([*own, *shared]):
                owner_clause, owner_values = self._owner_clause(c, owner)
                blocked = c.execute(
                    f"SELECT 1 FROM purchases WHERE {owner_clause} AND frozen=1 LIMIT 1",
                    owner_values,
                ).fetchone()
                if blocked:
                    continue
                changed = c.execute(
                    "UPDATE wallets SET available=available-?,reserved=reserved+? WHERE platform=? AND user_id=? AND available>=? AND debt=0",
                    (credits, credits, owner.platform, owner.user_id, credits),
                )
                if changed.rowcount:
                    c.execute(
                        "INSERT INTO reservations(reference,owner_platform,owner_id,credits,status,created_at) VALUES (?,?,?,?,'held',CURRENT_TIMESTAMP)",
                        (reference, owner.platform, owner.user_id, credits),
                    )
                    self._record(c, f"reserve:{reference}", owner, -credits, credits, "reserve")
                    return Reservation(reference, owner, credits)
            return None

        return await self._run(reserve)

    @staticmethod
    def _reservation(c, reservation: Reservation):
        row = c.execute("SELECT * FROM reservations WHERE reference=?", (reservation.reference,)).fetchone()
        if row is None:
            raise ValueError("Unknown reservation")
        if (row["owner_platform"], row["owner_id"], row["credits"]) != (
            reservation.owner.platform,
            reservation.owner.user_id,
            reservation.credits,
        ):
            raise ValueError("Reservation details do not match")
        return row

    async def configure_reservation(self, reservation: Reservation, markup_bps: int) -> None:
        _nonnegative(markup_bps, "Markup")

        def configure(c):
            row = self._reservation(c, reservation)
            if row["status"] != "held":
                raise ValueError("Reservation is already complete")
            if c.execute(
                "SELECT 1 FROM call_usage WHERE request_reference=? LIMIT 1", (reservation.reference,)
            ).fetchone():
                if row["markup_bps"] != markup_bps:
                    raise ValueError("Reservation pricing cannot change after usage")
                return
            c.execute(
                "UPDATE reservations SET markup_bps=? WHERE reference=?", (markup_bps, reservation.reference)
            )

        await self._run(configure)

    async def mark_delivered(self, reservation: Reservation) -> None:
        def mark(c):
            self._reservation(c, reservation)
            c.execute(
                "UPDATE reservations SET delivered=1 WHERE reference=? AND status='held'",
                (reservation.reference,),
            )

        await self._run(mark)

    async def record_call(self, reservation: Reservation, call: UsageCall) -> None:
        for value in (call.reference, call.attempt, call.provider, call.model):
            _reference(value)
        if call.cost_nano_usd is not None:
            _nonnegative(call.cost_nano_usd, "Call cost")

        def record(c):
            row = self._reservation(c, reservation)
            if row["status"] != "held":
                raise ValueError("Cannot record calls after settlement")
            existing = c.execute("SELECT * FROM call_usage WHERE reference=?", (call.reference,)).fetchone()
            expected = (reservation.reference, call.attempt, call.provider, call.model, call.price)
            if existing:
                actual = tuple(
                    existing[name] for name in ("request_reference", "attempt", "provider", "model", "price")
                )
                if actual != expected:
                    raise ValueError("Conflicting usage reference")
                if existing["cost_nano_usd"] is not None:
                    if (existing["cost_nano_usd"], existing["usage"]) != (call.cost_nano_usd, call.usage):
                        raise ValueError("Conflicting usage replay")
                    return
                c.execute(
                    "UPDATE call_usage SET cost_nano_usd=?,usage=? WHERE reference=?",
                    (call.cost_nano_usd, call.usage, call.reference),
                )
                return
            c.execute(
                "INSERT INTO call_usage(reference,request_reference,attempt,provider,model,cost_nano_usd,usage,price) VALUES (?,?,?,?,?,?,?,?)",
                (call.reference, *expected[:4], call.cost_nano_usd, call.usage, call.price),
            )

        await self._run(record)

    async def accept_attempt(self, reservation: Reservation, attempt: str) -> None:
        def accept(c):
            row = self._reservation(c, reservation)
            if row["status"] != "held":
                raise ValueError("Cannot accept usage after settlement")
            if not c.execute(
                "SELECT 1 FROM call_usage WHERE request_reference=? AND attempt=? LIMIT 1",
                (reservation.reference, attempt),
            ).fetchone():
                raise ValueError("Cannot accept an attempt without recorded calls")
            unknown = c.execute(
                "SELECT 1 FROM call_usage WHERE request_reference=? AND attempt=? AND cost_nano_usd IS NULL LIMIT 1",
                (reservation.reference, attempt),
            ).fetchone()
            if unknown:
                raise ValueError("Cannot bill incomplete usage")
            c.execute(
                "UPDATE call_usage SET billable=1 WHERE request_reference=? AND attempt=?",
                (reservation.reference, attempt),
            )

        await self._run(accept)

    @staticmethod
    def _cost(c, reference: str, markup_bps: int, cap: int) -> int:
        costs = c.execute(
            "SELECT cost_nano_usd FROM call_usage WHERE request_reference=? AND billable=1", (reference,)
        )
        nano = sum(row[0] for row in costs if row[0] is not None)
        scaled = nano * (10_000 + markup_bps)
        return min(cap, (scaled + 9_999_999) // 10_000_000)

    async def final_cost(self, reservation: Reservation, markup_bps: int) -> int:
        _nonnegative(markup_bps, "Markup")

        def cost(c):
            row = self._reservation(c, reservation)
            if row["status"] != "held":
                return row["charged"]
            if markup_bps != row["markup_bps"]:
                raise ValueError("Markup does not match reservation pricing snapshot")
            return (
                self._cost(c, reservation.reference, markup_bps, reservation.credits)
                if row["delivered"]
                else 0
            )

        return await self._run(cost)

    def _finish(self, c, reservation: Reservation, status: str, amount: int, *, recovery=False) -> None:
        row = self._reservation(c, reservation)
        if row["status"] != "held":
            return
        refunded = reservation.credits - amount
        balance = self._balance(c, reservation.owner)
        repaid = min(balance.debt, refunded)
        c.execute(
            "UPDATE wallets SET available=available+?,reserved=reserved-?,debt=debt-? WHERE platform=? AND user_id=?",
            (
                refunded - repaid,
                reservation.credits,
                repaid,
                reservation.owner.platform,
                reservation.owner.user_id,
            ),
        )
        c.execute(
            "UPDATE reservations SET status=?,charged=? WHERE reference=?",
            (status, amount, reservation.reference),
        )
        kind = "recover" if recovery else status
        self._record(
            c,
            f"finish:{reservation.reference}",
            reservation.owner,
            refunded - repaid,
            -reservation.credits,
            kind,
            -repaid,
        )

    async def settle(self, reservation: Reservation, amount: int) -> None:
        _nonnegative(amount)
        if amount > reservation.credits:
            raise ValueError("Charge exceeds reserved amount")
        await self._run(lambda c: self._finish(c, reservation, "settled", amount))

    async def release(self, reservation: Reservation) -> None:
        await self._run(lambda c: self._finish(c, reservation, "released", 0))

    async def recover(self) -> int:
        def recover(c):
            rows = c.execute("SELECT * FROM reservations WHERE status='held'").fetchall()
            for row in rows:
                reservation = Reservation(
                    row["reference"], Identity(row["owner_platform"], row["owner_id"]), row["credits"]
                )
                cost = (
                    self._cost(c, reservation.reference, row["markup_bps"], reservation.credits)
                    if row["delivered"]
                    else 0
                )
                self._finish(
                    c, reservation, "settled" if row["delivered"] else "released", cost, recovery=True
                )
            return len(rows)

        return await self._run(recover)

    async def migrate_legacy(self, micro_usd_per_credit: int) -> int:
        """Explicit conversion while the administrator holds the runtime ownership lock."""
        _positive(micro_usd_per_credit, "Conversion factor")

        def migrate(c):
            tables = self._tables(c)
            if "billing_meta" in tables:
                unit = c.execute("SELECT value FROM billing_meta WHERE key='unit'").fetchone()
                if unit:
                    old_factor = c.execute(
                        "SELECT value FROM billing_meta WHERE key='legacy_factor'"
                    ).fetchone()
                    if old_factor and old_factor[0] == str(micro_usd_per_credit):
                        return 0
                    raise LegacyBalanceError("Database already uses monetary balances")
            if "wallets" not in tables:
                self._schema(c)
                return 0
            if c.execute("SELECT 1 FROM wallets WHERE reserved!=0 LIMIT 1").fetchone() or (
                "reservations" in tables
                and c.execute("SELECT 1 FROM reservations WHERE status='held' LIMIT 1").fetchone()
            ):
                raise LegacyBalanceError("Release legacy reservations before converting balances")
            wallets = c.execute("SELECT platform,user_id,available FROM wallets").fetchall()
            permissions = c.execute("SELECT * FROM permissions").fetchall() if "permissions" in tables else []
            for row in wallets:
                _nonnegative(row[2] * micro_usd_per_credit, "Converted balance")
            for name in ("permissions", "grants", "reservations", "ledger", "wallets"):
                if name in tables:
                    c.execute(f"ALTER TABLE {name} RENAME TO legacy_{name}")
            # The archived index would otherwise keep its globally unique name.
            c.execute("DROP INDEX IF EXISTS one_sponsor_per_scope")
            self._schema(c)
            for platform, user_id, available in wallets:
                owner = Identity(platform, user_id)
                self._wallet(c, owner)
                amount = available * micro_usd_per_credit
                if amount:
                    reference = f"legacy:{platform}:{user_id}"
                    c.execute("INSERT INTO grants VALUES (?,?,?,?)", (reference, platform, user_id, amount))
                    self._credit(c, owner, amount, f"grant:{reference}", "legacy_conversion")
            for row in permissions:
                c.execute("INSERT INTO permissions VALUES (?,?,?,?,?,?)", tuple(row))
            c.execute("INSERT INTO billing_meta VALUES ('legacy_factor',?)", (str(micro_usd_per_credit),))
            return len(wallets)

        return await self._run(migrate, legacy=True)

    async def issue_login(self, owner: Identity, ttl_seconds: float = 600) -> str:
        _ttl(ttl_seconds)
        token = secrets.token_urlsafe(32)

        def issue(c):
            now = self.clock()
            c.execute(
                "DELETE FROM login_tokens WHERE expires_at<=? OR (owner_platform=? AND owner_id=?)",
                (now, owner.platform, owner.user_id),
            )
            c.execute(
                "INSERT INTO login_tokens VALUES (?,?,?,?)",
                (_hash(token), owner.platform, owner.user_id, now + ttl_seconds),
            )
            return token

        return await self._run(issue)

    async def redeem_login(self, token: str, ttl_seconds: float = 86400) -> tuple[str, Session] | None:
        _ttl(ttl_seconds)
        if not isinstance(token, str) or not token:
            return None
        session_token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)

        def redeem(c):
            now = self.clock()
            c.execute("DELETE FROM sessions WHERE expires_at<=?", (now,))
            row = c.execute(
                "DELETE FROM login_tokens WHERE digest=? AND expires_at>? RETURNING owner_platform,owner_id",
                (_hash(token), now),
            ).fetchone()
            if row is None:
                return None
            session = Session(self._canonical(c, Identity(row[0], row[1])), csrf, now + ttl_seconds, row[0])
            c.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?)",
                (_hash(session_token), row[0], row[1], csrf, session.expires_at),
            )
            return session_token, session

        return await self._run(redeem)

    async def get_session(self, token: str) -> Session | None:
        if not isinstance(token, str) or not token:
            return None

        def get(c):
            row = c.execute(
                "SELECT owner_platform,owner_id,csrf,expires_at FROM sessions WHERE digest=? AND expires_at>?",
                (_hash(token), self.clock()),
            ).fetchone()
            return (
                Session(self._canonical(c, Identity(row[0], row[1])), row[2], row[3], row[0]) if row else None
            )

        return await self._run(get)

    async def revoke_session(self, token: str) -> None:
        await self._run(lambda c: c.execute("DELETE FROM sessions WHERE digest=?", (_hash(token),)))

    async def link_identity(self, session_owner: Identity, login_token: str) -> Identity:
        if not isinstance(login_token, str) or not login_token:
            raise ValueError("A valid sign-in token is required")

        def link(c):
            proof = c.execute(
                "SELECT owner_platform,owner_id FROM login_tokens WHERE digest=? AND expires_at>?",
                (_hash(login_token), self.clock()),
            ).fetchone()
            if proof is None:
                raise ValueError("Sign-in token is invalid or expired")
            destination = self._canonical(c, session_owner)
            source = self._canonical(c, Identity(proof[0], proof[1]))
            if source == destination:
                c.execute("DELETE FROM login_tokens WHERE digest=?", (_hash(login_token),))
                return destination
            self._wallet(c, destination)
            self._wallet(c, source)
            for owner in (destination, source):
                clause, values = self._owner_clause(c, owner)
                if (
                    self._balance(c, owner).reserved
                    or c.execute(
                        f"SELECT 1 FROM reservations WHERE {clause} AND status='held' LIMIT 1",
                        values,
                    ).fetchone()
                ):
                    raise ValueError("Wait for active requests to finish before linking accounts")
                if c.execute(
                    f"SELECT 1 FROM purchases WHERE {clause} AND frozen=1 LIMIT 1",
                    values,
                ).fetchone():
                    raise ValueError("Resolve pending refunds before linking accounts")
            old_source = self._balance(c, source)
            old_destination = self._balance(c, destination)
            available = old_source.available + old_destination.available
            debt = old_source.debt + old_destination.debt
            paid_debt = min(available, debt)
            available -= paid_debt
            debt -= paid_debt
            _nonnegative(available)
            _nonnegative(debt)
            c.execute(
                "UPDATE wallets SET available=0,debt=0 WHERE platform=? AND user_id=?",
                (source.platform, source.user_id),
            )
            c.execute(
                "UPDATE wallets SET available=?,debt=? WHERE platform=? AND user_id=?",
                (available, debt, destination.platform, destination.user_id),
            )
            prefix = f"link:{_hash(login_token)}"
            self._record(
                c, f"{prefix}:out", source, -old_source.available, 0, "identity_transfer", -old_source.debt
            )
            self._record(
                c,
                f"{prefix}:in",
                destination,
                available - old_destination.available,
                0,
                "identity_transfer",
                debt - old_destination.debt,
            )
            rows = c.execute(
                "SELECT * FROM permissions WHERE owner_platform=? AND owner_id=?",
                (source.platform, source.user_id),
            ).fetchall()
            c.execute(
                "DELETE FROM permissions WHERE owner_platform=? AND owner_id=?",
                (source.platform, source.user_id),
            )
            for row in rows:
                c.execute(
                    "INSERT INTO permissions VALUES (?,?,?,?,?,?) ON CONFLICT(owner_platform,owner_id,platform,kind,scope_id) "
                    "DO UPDATE SET mode=CASE WHEN permissions.mode='everyone' OR excluded.mode='everyone' THEN 'everyone' ELSE 'only_me' END",
                    (
                        destination.platform,
                        destination.user_id,
                        row["platform"],
                        row["kind"],
                        row["scope_id"],
                        row["mode"],
                    ),
                )
            c.execute(
                "UPDATE identity_aliases SET owner_platform=?,owner_id=? WHERE owner_platform=? AND owner_id=?",
                (destination.platform, destination.user_id, source.platform, source.user_id),
            )
            for identity in (source, destination):
                c.execute(
                    "INSERT INTO identity_aliases VALUES (?,?,?,?) ON CONFLICT(platform,user_id) DO UPDATE SET owner_platform=excluded.owner_platform,owner_id=excluded.owner_id",
                    (identity.platform, identity.user_id, destination.platform, destination.user_id),
                )
            c.execute("DELETE FROM login_tokens WHERE digest=?", (_hash(login_token),))
            return destination

        return await self._run(link)

    @staticmethod
    def _purchase(row) -> Purchase | None:
        if row is None:
            return None
        return Purchase(
            row["reference"],
            Identity(row["owner_platform"], row["owner_id"]),
            row["amount_cents"],
            row["currency"],
            row["credited_units"],
            json.loads(row["quote"]),
            row["status"],
            row["order_id"],
            row["intention_id"],
            row["checkout_url"],
            row["transaction_id"],
            row["refunded_cents"],
            bool(row["frozen"]),
        )

    async def create_purchase(
        self,
        owner: Identity,
        reference: str,
        amount_cents: int,
        currency: str,
        credited_units: int,
        quote: dict,
    ) -> Purchase:
        _reference(reference)
        _positive(amount_cents)
        _positive(credited_units)
        if not isinstance(currency, str) or len(currency) != 3 or not currency.isalpha():
            raise ValueError("Purchase currency must be a three-letter code")
        currency = currency.upper()
        encoded = json.dumps(quote, sort_keys=True, separators=(",", ":"), allow_nan=False)

        def create(c):
            primary = self._canonical(c, owner)
            row = c.execute("SELECT * FROM purchases WHERE reference=?", (reference,)).fetchone()
            if row:
                if (
                    self._canonical(c, Identity(row["owner_platform"], row["owner_id"])),
                    row["amount_cents"],
                    row["currency"],
                    row["credited_units"],
                    row["quote"],
                ) != (primary, amount_cents, currency, credited_units, encoded):
                    raise ValueError("Purchase reference already identifies a different purchase")
                return self._purchase(row)
            self._wallet(c, owner)
            c.execute(
                "INSERT INTO purchases(reference,owner_platform,owner_id,amount_cents,currency,credited_units,quote) VALUES (?,?,?,?,?,?,?)",
                (
                    reference,
                    primary.platform,
                    primary.user_id,
                    amount_cents,
                    currency,
                    credited_units,
                    encoded,
                ),
            )
            return self._purchase(
                c.execute("SELECT * FROM purchases WHERE reference=?", (reference,)).fetchone()
            )

        return await self._run(create)

    async def claim_checkout(self, reference: str) -> bool:
        return await self._run(
            lambda c: bool(
                c.execute(
                    "UPDATE purchases SET status='creating' WHERE reference=? AND status='created'",
                    (reference,),
                ).rowcount
            )
        )

    async def bind_checkout(self, reference: str, checkout: Checkout) -> None:
        for value in (checkout.order_id, checkout.intention_id, checkout.url):
            _reference(value)

        def bind(c):
            row = c.execute("SELECT * FROM purchases WHERE reference=?", (reference,)).fetchone()
            if row is None:
                raise ValueError("Unknown purchase")
            existing = (row["order_id"], row["intention_id"], row["checkout_url"])
            if row["order_id"] is not None:
                if (
                    row["order_id"] != checkout.order_id
                    or (row["intention_id"] is not None and row["intention_id"] != checkout.intention_id)
                    or (row["checkout_url"] is not None and row["checkout_url"] != checkout.url)
                ):
                    raise ValueError("Purchase checkout cannot change")
                if existing == (checkout.order_id, checkout.intention_id, checkout.url):
                    return
            c.execute(
                "UPDATE purchases SET order_id=?,intention_id=?,checkout_url=?, "
                "status=CASE WHEN status IN ('created','creating') THEN 'pending' ELSE status END WHERE reference=?",
                (checkout.order_id, checkout.intention_id, checkout.url, reference),
            )

        await self._run(bind)

    async def restore_order(self, reference: str, payment: PaymentState) -> None:
        def restore(c):
            row = c.execute("SELECT * FROM purchases WHERE reference=?", (reference,)).fetchone()
            if row is None or payment.special_reference != reference:
                raise ValueError("Canonical payment reference does not match purchase")
            quote = json.loads(row["quote"])
            if (payment.amount_cents, payment.currency) != (row["amount_cents"], row["currency"]):
                raise ValueError("Payment amount or currency does not match purchase")
            if (
                payment.integration_id not in quote.get("integration_ids", [])
                or payment.merchant_id != quote.get("merchant_id")
                or payment.is_live != quote.get("is_live")
            ):
                raise ValueError("Payment merchant, integration, or mode does not match purchase")
            if row["order_id"] is not None and row["order_id"] != str(payment.order_id):
                raise ValueError("Purchase already has a different gateway order")
            c.execute(
                "UPDATE purchases SET order_id=?,status=CASE WHEN status IN ('created','creating') THEN 'pending' ELSE status END WHERE reference=?",
                (str(payment.order_id), reference),
            )

        await self._run(restore)

    async def get_purchase(self, reference: str, owner: Identity | None = None) -> Purchase | None:
        def get(c):
            row = c.execute("SELECT * FROM purchases WHERE reference=?", (reference,)).fetchone()
            if (
                row
                and owner is not None
                and self._canonical(c, Identity(row["owner_platform"], row["owner_id"]))
                != self._canonical(c, owner)
            ):
                return None
            return self._purchase(row)

        return await self._run(get)

    async def get_purchase_by_order(self, order_id: str) -> Purchase | None:
        return await self._run(
            lambda c: self._purchase(
                c.execute("SELECT * FROM purchases WHERE order_id=?", (str(order_id),)).fetchone()
            )
        )

    async def freeze_purchase(self, reference: str, target_refunded_cents: int) -> None:
        _positive(target_refunded_cents, "Refund target")

        def freeze(c):
            row = c.execute("SELECT * FROM purchases WHERE reference=?", (reference,)).fetchone()
            if row is None or not row["funded"] or row["transaction_id"] is None:
                raise ValueError("Only a paid purchase can be frozen for refund")
            if row["frozen"]:
                raise ValueError("A refund is already pending; reconcile before retrying")
            if target_refunded_cents > row["amount_cents"] or target_refunded_cents <= row["refunded_cents"]:
                raise ValueError("Refund target must increase cumulative refund within payment amount")
            c.execute(
                "UPDATE purchases SET frozen=1,refund_target_cents=? WHERE reference=?",
                (target_refunded_cents, reference),
            )

        await self._run(freeze)

    async def apply_payment(self, payment: PaymentState) -> Purchase:
        _positive(payment.amount_cents)
        _nonnegative(payment.refunded_cents, "Refund")
        if payment.refunded_cents > payment.amount_cents:
            raise ValueError("Refund exceeds payment amount")

        def apply(c):
            row = c.execute("SELECT * FROM purchases WHERE order_id=?", (str(payment.order_id),)).fetchone()
            if row is None:
                raise ValueError("Unknown gateway order; reconcile checkout binding before retry")
            quote = json.loads(row["quote"])
            if (row["amount_cents"], row["currency"]) != (payment.amount_cents, payment.currency):
                raise ValueError("Payment amount or currency does not match purchase")
            if (
                payment.integration_id not in quote.get("integration_ids", [])
                or payment.merchant_id != quote.get("merchant_id")
                or payment.is_live != quote.get("is_live")
            ):
                raise ValueError("Payment merchant, integration, or mode does not match purchase")
            if payment.special_reference is not None and payment.special_reference != row["reference"]:
                raise ValueError("Payment reference does not match purchase")
            if row["transaction_id"] is not None and row["transaction_id"] != str(payment.transaction_id):
                raise ValueError("Payment transaction does not match purchase")
            if payment.pending:
                return self._purchase(row)
            owner = Identity(row["owner_platform"], row["owner_id"])
            refunded = max(
                row["refunded_cents"], payment.amount_cents if payment.voided else payment.refunded_cents
            )
            funded = bool(row["funded"])
            transaction_id = row["transaction_id"]
            if payment.success or refunded:
                transaction_id = str(payment.transaction_id)
            if payment.success and not funded:
                self._credit(c, owner, row["credited_units"], f"payment:{row['reference']}", "payment")
                funded = True
            debited = row["refund_debited_units"]
            if funded:
                cumulative = (row["credited_units"] * refunded + row["amount_cents"] - 1) // row[
                    "amount_cents"
                ]
                if cumulative > debited:
                    self._debit(
                        c, owner, cumulative - debited, f"refund:{row['reference']}:{refunded}", "refund"
                    )
                    debited = cumulative
            if funded:
                status = (
                    "refunded"
                    if refunded == row["amount_cents"]
                    else "partially_refunded"
                    if refunded
                    else "paid"
                )
            else:
                status = "refunded" if refunded else "failed"
            c.execute(
                "UPDATE purchases SET status=?,transaction_id=?,refunded_cents=?,funded=?,refund_debited_units=?,frozen=? WHERE reference=?",
                (
                    status,
                    transaction_id,
                    refunded,
                    int(funded),
                    debited,
                    int(
                        bool(
                            row["frozen"]
                            and row["refund_target_cents"] is not None
                            and refunded < row["refund_target_cents"]
                        )
                    ),
                    row["reference"],
                ),
            )
            return self._purchase(
                c.execute("SELECT * FROM purchases WHERE reference=?", (row["reference"],)).fetchone()
            )

        return await self._run(apply)

    async def history(self, owner: Identity, limit: int = 30) -> dict[str, list[dict]]:
        _positive(limit, "History limit")
        limit = min(limit, 100)

        def history(c):
            owner_clause, owner_values = self._owner_clause(c, owner)
            reservation_clause, reservation_values = self._owner_clause(c, owner, "r.")
            purchases = [
                dict(row)
                for row in c.execute(
                    "SELECT reference,amount_cents,currency,credited_units,status,refunded_cents,created_at "
                    f"FROM purchases WHERE {owner_clause} ORDER BY rowid DESC LIMIT ?",
                    (*owner_values, limit),
                )
            ]
            ledger = [
                dict(row)
                for row in c.execute(
                    "SELECT event_reference,available_delta,reserved_delta,debt_delta,kind,created_at "
                    f"FROM ledger WHERE {owner_clause} ORDER BY rowid DESC LIMIT ?",
                    (*owner_values, limit),
                )
            ]
            usage = [
                dict(row)
                for row in c.execute(
                    "SELECT r.reference,r.credits,r.charged AS customer_charge,r.status,r.delivered,r.created_at, "
                    "COALESCE(SUM(u.cost_nano_usd),0) AS cost_nano_usd, "
                    "GROUP_CONCAT(DISTINCT u.provider) AS provider,GROUP_CONCAT(DISTINCT u.model) AS model "
                    "FROM reservations r LEFT JOIN call_usage u ON u.request_reference=r.reference AND u.billable=1 "
                    f"WHERE {reservation_clause} GROUP BY r.reference ORDER BY r.rowid DESC LIMIT ?",
                    (*reservation_values, limit),
                )
            ]
            return {"purchases": purchases, "usage": usage, "ledger": ledger}

        return await self._run(history)

    async def list_purchases(self, status: str | None = None, limit: int = 100) -> list[Purchase]:
        _positive(limit, "Purchase limit")

        def purchases(c):
            if status is None:
                rows = c.execute("SELECT * FROM purchases ORDER BY rowid DESC LIMIT ?", (min(limit, 1000),))
            else:
                rows = c.execute(
                    "SELECT * FROM purchases WHERE status=? ORDER BY rowid DESC LIMIT ?",
                    (status, min(limit, 1000)),
                )
            return [self._purchase(row) for row in rows]

        return await self._run(purchases)
