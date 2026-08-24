"""Cancellation-safe concurrency limits for non-cancellable native work."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

Result = TypeVar("Result")
Commit = Callable[[], None]
WorkFactory = Callable[[Commit], Awaitable[Result]]


@dataclass(slots=True)
class _WorkState(Generic[Result]):
    task: asyncio.Task[Result] | None = None
    committed: bool = False
    on_cancel: Callable[[], None] | None = None


class NativeWorkLimiter:
    """Keep slots reserved until native work really finishes.

    Awaiting a thread-backed coroutine through ``asyncio.shield`` prevents task
    cancellation from pretending that the native call stopped. Work factories
    call ``commit`` immediately before entering non-cancellable native code;
    queued or preparatory work can still be cancelled normally before then.
    """

    def __init__(self, max_concurrency: int, *, task_name: str) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._task_name = task_name
        self._states: dict[asyncio.Task, _WorkState] = {}
        self._closing = False

    @property
    def active_count(self) -> int:
        return sum(state.committed for state in self._states.values())

    @property
    def pending_count(self) -> int:
        return len(self._states)

    def _finish(self, task: asyncio.Task) -> None:
        self._states.pop(task, None)
        if task.cancelled():
            return
        # Consume failures when the original caller was cancelled and therefore
        # can no longer retrieve the background task's result.
        task.exception()

    async def run(
        self,
        factory: WorkFactory[Result],
        *,
        on_cancel: Callable[[], None] | None = None,
    ) -> Result:
        if self._closing:
            raise RuntimeError("native work limiter is closed")

        state: _WorkState[Result] = _WorkState(on_cancel=on_cancel)

        async def run_work() -> Result:
            async with self._semaphore:
                def commit() -> None:
                    state.committed = True

                return await factory(commit)

        task = asyncio.create_task(run_work(), name=self._task_name)
        state.task = task
        self._states[task] = state
        task.add_done_callback(self._finish)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if state.committed:
                if state.on_cancel is not None:
                    state.on_cancel()
            else:
                task.cancel()
            raise

    async def close(self) -> None:
        self._closing = True
        states = tuple(self._states.values())
        for state in states:
            if state.task is None:
                continue
            if state.committed:
                if state.on_cancel is not None:
                    state.on_cancel()
            else:
                state.task.cancel()
        tasks = [state.task for state in states if state.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
