"""Request-scoped state shared by provider tools."""

import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PaidRoute:
    providers: tuple[str, ...]
    deadline: float
    attempt_timeout_seconds: float
    max_retries: int


paid_route: contextvars.ContextVar[PaidRoute | None] = contextvars.ContextVar("paid_route", default=None)
delivery_callback: contextvars.ContextVar[Callable[[], Awaitable[None]] | None] = contextvars.ContextVar(
    "delivery_callback",
    default=None,
)


async def record_delivery() -> None:
    callback = delivery_callback.get()
    if callback is not None:
        await callback()


web_search_count = contextvars.ContextVar("web_search_count", default=0)
web_search_start_time = contextvars.ContextVar("web_search_start_time", default=0.0)
web_search_exhausted = contextvars.ContextVar("web_search_exhausted", default=False)


def reset_request_context() -> None:
    web_search_count.set(0)
    web_search_start_time.set(0.0)
    web_search_exhausted.set(False)
