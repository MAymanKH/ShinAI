"""Fast, coordinated Gemini key/model pair scheduling."""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any

from shin_ai.coordination.store import CoordinationStore
from shin_ai.providers.gemini_errors import GeminiFailure, GeminiFailureKind


@dataclass(frozen=True, slots=True)
class PairHealth:
    status: str = "unknown"
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    last_error: str | None = None
    last_success_at: float | None = None

    @classmethod
    def decode(cls, raw: str | None) -> "PairHealth":
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
            return cls(
                status=str(data.get("status", "unknown")),
                cooldown_until=float(data.get("cooldown_until", 0.0)),
                consecutive_failures=int(data.get("consecutive_failures", 0)),
                last_error=data.get("last_error"),
                last_success_at=data.get("last_success_at"),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return cls()

    def encode(self) -> str:
        return json.dumps(
            {
                "status": self.status,
                "cooldown_until": self.cooldown_until,
                "consecutive_failures": self.consecutive_failures,
                "last_error": self.last_error,
                "last_success_at": self.last_success_at,
            },
            separators=(",", ":"),
        )


@dataclass(slots=True)
class GeminiReservation:
    scheduler: "GeminiScheduler"
    key_name: str
    api_key: str
    model: str
    token: str
    _released: bool = False

    async def succeeded(self) -> None:
        if self._released:
            return
        await self.scheduler.record_success(self)
        self._released = True

    async def failed(self, failure: GeminiFailure) -> None:
        if self._released:
            return
        await self.scheduler.record_failure(self, failure)
        self._released = True

    async def release(self) -> None:
        if self._released:
            return
        await self.scheduler.release(self)
        self._released = True


class GeminiScheduler:
    """Reserve healthy pairs fairly across threads and local bot processes."""

    def __init__(
        self,
        keys: dict[str, str],
        models: list[str] | tuple[str, ...],
        store: CoordinationStore,
        *,
        clock=time.time,
        owner_id: str | None = None,
        reservation_seconds: float = 75.0,
    ) -> None:
        self.keys = {name: value for name, value in keys.items() if value}
        self.models = tuple(models)
        self.store = store
        self.clock = clock
        self.owner_id = owner_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.reservation_seconds = reservation_seconds

    @staticmethod
    def _safe_component(value: str) -> str:
        return value.replace(":", "_").replace("/", "_")

    def _pair_key(self, prefix: str, model: str, key_name: str) -> str:
        return (
            f"gemini:{prefix}:{self._safe_component(model)}:"
            f"{self._safe_component(key_name)}"
        )

    def _key_health_key(self, key_name: str) -> str:
        return f"gemini:key-health:{self._safe_component(key_name)}"

    async def _is_key_disabled(self, key_name: str) -> bool:
        return await self.store.get(self._key_health_key(key_name)) is not None

    async def pair_health(self, model: str, key_name: str) -> PairHealth:
        return PairHealth.decode(await self.store.get(self._pair_key("health", model, key_name)))

    async def reserve(
        self,
        model: str,
        *,
        excluded_keys: set[str] | None = None,
    ) -> GeminiReservation | None:
        if model not in self.models or not self.keys:
            return None
        excluded = excluded_keys or set()
        key_names = list(self.keys)
        cursor = await self.store.increment(f"gemini:cursor:{self._safe_component(model)}")
        start = (cursor - 1) % len(key_names)
        ordered = key_names[start:] + key_names[:start]
        now = self.clock()

        for key_name in ordered:
            if key_name in excluded or await self._is_key_disabled(key_name):
                continue
            health = await self.pair_health(model, key_name)
            if health.cooldown_until > now:
                continue

            token = f"{self.owner_id}:{uuid.uuid4().hex}"
            claimed = await self.store.claim(
                self._pair_key("lease", model, key_name),
                token,
                ttl_seconds=self.reservation_seconds,
            )
            if claimed:
                return GeminiReservation(
                    scheduler=self,
                    key_name=key_name,
                    api_key=self.keys[key_name],
                    model=model,
                    token=token,
                )
        return None

    def _cooldown_seconds(self, failure: GeminiFailure, failures: int) -> float:
        if failure.retry_after_seconds is not None:
            return max(1.0, failure.retry_after_seconds)
        if failure.kind == GeminiFailureKind.RATE_LIMIT:
            return 900.0
        if failure.kind == GeminiFailureKind.AUTHENTICATION:
            return 3_600.0
        if failure.kind == GeminiFailureKind.INVALID_REQUEST:
            return 300.0
        if failure.kind == GeminiFailureKind.TIMEOUT:
            return min(120.0, 10.0 * 2 ** min(failures - 1, 3))
        if failure.kind == GeminiFailureKind.TRANSIENT:
            return min(120.0, 5.0 * 2 ** min(failures - 1, 4))
        return min(300.0, 15.0 * 2 ** min(failures - 1, 4))

    async def record_success(self, reservation: GeminiReservation) -> None:
        now = self.clock()
        health = PairHealth(
            status="healthy",
            cooldown_until=0.0,
            consecutive_failures=0,
            last_error=None,
            last_success_at=now,
        )
        await self.store.set(
            self._pair_key("health", reservation.model, reservation.key_name),
            health.encode(),
            ttl_seconds=604_800.0,
        )
        await self.store.delete(self._key_health_key(reservation.key_name))
        await self.release(reservation)

    async def record_failure(
        self,
        reservation: GeminiReservation,
        failure: GeminiFailure,
    ) -> None:
        previous = await self.pair_health(reservation.model, reservation.key_name)
        failures = previous.consecutive_failures + 1
        cooldown = self._cooldown_seconds(failure, failures)
        health = PairHealth(
            status=failure.kind.value,
            cooldown_until=self.clock() + cooldown,
            consecutive_failures=failures,
            last_error=failure.message[:200],
            last_success_at=previous.last_success_at,
        )
        await self.store.set(
            self._pair_key("health", reservation.model, reservation.key_name),
            health.encode(),
            ttl_seconds=max(86_400.0, cooldown * 2),
        )
        if failure.kind == GeminiFailureKind.AUTHENTICATION:
            await self.store.set(
                self._key_health_key(reservation.key_name),
                failure.message[:200],
                ttl_seconds=cooldown,
            )
        await self.release(reservation)

    async def release(self, reservation: GeminiReservation) -> None:
        await self.store.delete(
            self._pair_key("lease", reservation.model, reservation.key_name),
            expected_value=reservation.token,
        )

    async def health_snapshot(self) -> dict[str, Any]:
        now = self.clock()
        models: dict[str, Any] = {}
        for model in self.models:
            pairs = []
            available = 0
            for key_name in self.keys:
                disabled = await self._is_key_disabled(key_name)
                health = await self.pair_health(model, key_name)
                eligible = not disabled and health.cooldown_until <= now
                if eligible:
                    available += 1
                pairs.append(
                    {
                        "key": key_name,
                        "status": "authentication" if disabled else health.status,
                        "eligible": eligible,
                        "cooldown_until": health.cooldown_until,
                        "last_error": health.last_error,
                    }
                )
            models[model] = {
                "available": available > 0,
                "eligible_keys": available,
                "total_keys": len(self.keys),
                "pairs": pairs,
            }
        return {"models": models, "total_keys": len(self.keys)}
