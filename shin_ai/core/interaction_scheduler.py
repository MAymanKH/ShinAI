"""Bounded, per-chat ordered scheduling for interaction work."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections import deque
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

Payload = TypeVar("Payload")
Handler = Callable[[Payload], Awaitable[None]]
ErrorHandler = Callable[[Payload, BaseException], None]
DropHandler = Callable[[Payload, str], None]


@dataclass(frozen=True, slots=True)
class SubmissionResult(Generic[Payload]):
    accepted: bool
    reason: str | None = None
    dropped: Payload | None = None
    delay_applied: float = 0.0


@dataclass(slots=True)
class _Job(Generic[Payload]):
    sequence: int
    chat_key: Hashable
    payload: Payload
    ready_at: float
    expires_at: float


@dataclass(slots=True)
class _ChatQueue(Generic[Payload]):
    jobs: deque[_Job[Payload]] = field(default_factory=deque)
    active: bool = False


class InteractionScheduler(Generic[Payload]):
    """Runs at most ``max_concurrent`` jobs while preserving chat order."""

    def __init__(
        self,
        handler: Handler[Payload],
        *,
        max_concurrent: int,
        max_pending: int,
        per_chat_limit: int,
        job_ttl_seconds: float,
        on_error: ErrorHandler[Payload] | None = None,
        on_drop: DropHandler[Payload] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min(max_concurrent, max_pending, per_chat_limit) <= 0:
            raise ValueError("scheduler limits must be greater than zero")
        self._handler = handler
        self.max_concurrent = max_concurrent
        self.max_pending = max_pending
        self.per_chat_limit = per_chat_limit
        self.job_ttl_seconds = job_ttl_seconds
        self._on_error = on_error
        self._on_drop = on_drop
        self._clock = clock

        self._condition = asyncio.Condition()
        self._queues: dict[Hashable, _ChatQueue[Payload]] = {}
        self._ready_heap: list[tuple[float, int, Hashable]] = []
        self._sequence = itertools.count()
        self._pending = 0
        self._active_tasks: set[asyncio.Task] = set()
        self._dispatcher: asyncio.Task | None = None
        self._accepting = True
        self._running = False
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def pending_count(self) -> int:
        return self._pending

    @property
    def active_count(self) -> int:
        return len(self._active_tasks)

    @property
    def chat_count(self) -> int:
        return len(self._queues)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._accepting = True
        self._dispatcher = asyncio.create_task(
            self._dispatch_loop(),
            name="shinai-interaction-dispatcher",
        )

    def _schedule_front(self, queue: _ChatQueue[Payload]) -> None:
        if not queue.active and queue.jobs:
            job = queue.jobs[0]
            heapq.heappush(self._ready_heap, (job.ready_at, job.sequence, job.chat_key))

    async def submit(
        self,
        chat_key: Hashable,
        payload: Payload,
        *,
        delay_seconds: float = 0.0,
    ) -> SubmissionResult[Payload]:
        if not self._running:
            await self.start()
        async with self._condition:
            if not self._accepting:
                return SubmissionResult(False, "scheduler_closed")
            if self._pending >= self.max_pending:
                return SubmissionResult(False, "global_queue_full")

            queue = self._queues.setdefault(chat_key, _ChatQueue())
            dropped_payload = None
            if len(queue.jobs) >= self.per_chat_limit:
                dropped = queue.jobs.popleft()
                dropped_payload = dropped.payload
                self._pending -= 1
                if self._on_drop:
                    self._on_drop(dropped.payload, "per_chat_queue_full")

            now = self._clock()
            apply_delay = not queue.active and not queue.jobs
            ready_at = now + max(0.0, delay_seconds) if apply_delay else now
            job = _Job(
                sequence=next(self._sequence),
                chat_key=chat_key,
                payload=payload,
                # Delay the first message in a chat burst, then drain that
                # chat in order, matching the bot's existing behavior.
                ready_at=ready_at,
                # Intentional human-like delay is not time spent waiting in
                # the work queue and must not consume the queue TTL.
                expires_at=ready_at + self.job_ttl_seconds,
            )
            was_empty = not queue.jobs
            queue.jobs.append(job)
            self._pending += 1
            self._idle.clear()
            if was_empty and not queue.active:
                self._schedule_front(queue)
            elif dropped_payload is not None and not queue.active:
                # The previous heap entry is now stale; add the new front.
                self._schedule_front(queue)
            self._condition.notify_all()
            return SubmissionResult(
                True,
                dropped=dropped_payload,
                delay_applied=max(0.0, delay_seconds) if apply_delay else 0.0,
            )

    def _pop_due_job(self, now: float) -> _Job[Payload] | None:
        while self._ready_heap:
            ready_at, sequence, chat_key = self._ready_heap[0]
            queue = self._queues.get(chat_key)
            if queue is None or queue.active or not queue.jobs or queue.jobs[0].sequence != sequence:
                heapq.heappop(self._ready_heap)
                continue
            if ready_at > now:
                return None
            heapq.heappop(self._ready_heap)
            queue.active = True
            self._pending -= 1
            return queue.jobs.popleft()
        return None

    def _next_wait(self, now: float) -> float | None:
        while self._ready_heap:
            ready_at, sequence, chat_key = self._ready_heap[0]
            queue = self._queues.get(chat_key)
            if queue is None or queue.active or not queue.jobs or queue.jobs[0].sequence != sequence:
                heapq.heappop(self._ready_heap)
                continue
            return max(0.0, ready_at - now)
        return None

    async def _dispatch_loop(self) -> None:
        while self._running:
            async with self._condition:
                job = None
                while self._running and job is None:
                    if len(self._active_tasks) < self.max_concurrent:
                        job = self._pop_due_job(self._clock())
                        if job is not None:
                            break
                    timeout = self._next_wait(self._clock())
                    try:
                        if timeout is None or len(self._active_tasks) >= self.max_concurrent:
                            await self._condition.wait()
                        else:
                            await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                    except TimeoutError:
                        pass
                if not self._running or job is None:
                    continue

                task = asyncio.create_task(
                    self._run_job(job),
                    name=f"shinai-interaction-{job.sequence}",
                )
                self._active_tasks.add(task)

    async def _run_job(self, job: _Job[Payload]) -> None:
        try:
            if self._clock() > job.expires_at:
                if self._on_drop:
                    self._on_drop(job.payload, "interaction_expired")
                return
            await self._handler(job.payload)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if self._on_error:
                self._on_error(job.payload, error)
        finally:
            async with self._condition:
                self._active_tasks.discard(asyncio.current_task())
                queue = self._queues.get(job.chat_key)
                if queue is not None:
                    queue.active = False
                    if queue.jobs:
                        self._schedule_front(queue)
                    else:
                        del self._queues[job.chat_key]
                if self._pending == 0 and not self._active_tasks:
                    self._idle.set()
                self._condition.notify_all()

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def close(self, *, grace_seconds: float = 30.0) -> None:
        self._accepting = False
        try:
            await asyncio.wait_for(self.wait_idle(), timeout=grace_seconds)
        except TimeoutError:
            for task in tuple(self._active_tasks):
                task.cancel()
            if self._active_tasks:
                await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._running = False
        async with self._condition:
            self._condition.notify_all()
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
        self._queues.clear()
        self._ready_heap.clear()
        self._pending = 0
        self._idle.set()
