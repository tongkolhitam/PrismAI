"""Task queue with priority support."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

from prismai.models.schemas import Priority, TaskStatus
from prismai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(order=True)
class Task:
    """A prioritized analysis task."""
    priority: int = field(compare=True)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    payload: Any = field(default=None, compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    created_at: datetime = field(default_factory=datetime.utcnow, compare=False)
    started_at: datetime | None = field(default=None, compare=False)
    completed_at: datetime | None = field(default=None, compare=False)
    result: Any = field(default=None, compare=False)
    error: str | None = field(default=None, compare=False)


class QueueManager:
    """Async task queue with priority ordering and status tracking."""

    def __init__(self, max_concurrent: int = 10) -> None:
        self._queue: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()
        self._tasks: dict[str, Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._processor: Callable[..., Coroutine[Any, Any, Any]] | None = None

    def set_processor(self, fn: Callable[..., Coroutine[Any, Any, Any]]) -> None:
        """Set the async function that processes tasks."""
        self._processor = fn

    async def enqueue(
        self,
        payload: Any,
        priority: Priority = Priority.NORMAL,
    ) -> str:
        """Add a task to the queue. Returns task ID."""
        task = Task(priority=priority.value, payload=payload)
        self._tasks[task.task_id] = task
        await self._queue.put(task)
        logger.info("task_enqueued", task_id=task.task_id, priority=priority.name)
        return task.task_id

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    async def start(self) -> None:
        """Start processing the queue."""
        if self._running:
            return
        self._running = True
        logger.info("queue_started")
        asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop processing the queue."""
        self._running = False
        logger.info("queue_stopped")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            async with self._semaphore:
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.utcnow()

                try:
                    if self._processor:
                        task.result = await self._processor(task.payload)
                    task.status = TaskStatus.COMPLETED
                except Exception as exc:
                    task.status = TaskStatus.FAILED
                    task.error = str(exc)
                    logger.error("task_failed", task_id=task.task_id, error=str(exc))
                finally:
                    task.completed_at = datetime.utcnow()
                    self._queue.task_done()

    @property
    def stats(self) -> dict[str, Any]:
        statuses = {}
        for task in self._tasks.values():
            s = task.status.value
            statuses[s] = statuses.get(s, 0) + 1
        return {
            "total_tasks": len(self._tasks),
            "queue_size": self._queue.qsize(),
            "running": self._running,
            "status_breakdown": statuses,
        }
