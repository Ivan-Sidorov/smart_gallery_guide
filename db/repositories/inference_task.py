"""Inference task repository."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import InferenceTask, TaskStatus, TaskType


class InferenceTaskRepository:
    """CRUD and lookup helpers for `inference_tasks` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, task_id: uuid.UUID) -> InferenceTask | None:
        """Get a task by ID."""
        return await self.session.get(InferenceTask, task_id)

    async def create(
        self,
        *,
        type: TaskType,
        request: dict[str, Any],
        user_id: int | None = None,
        session_id: uuid.UUID | None = None,
        model: str | None = None,
    ) -> InferenceTask:
        """Create a new task."""
        task = InferenceTask(
            type=type,
            status=TaskStatus.PENDING,
            user_id=user_id,
            session_id=session_id,
            request=request,
            model=model,
        )
        self.session.add(task)
        await self.session.flush()
        return task

    async def mark_running(
        self, task_id: uuid.UUID, worker: str | None = None
    ) -> InferenceTask | None:
        """Mark a task as running."""
        task = await self.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        if worker is not None:
            task.worker = worker
        await self.session.flush()
        return task

    async def mark_done(
        self,
        task_id: uuid.UUID,
        result: dict[str, Any],
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_usd: float | None = None,
    ) -> InferenceTask | None:
        """Mark a task as done."""
        task = await self.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus.DONE
        task.result = result
        task.finished_at = datetime.now(timezone.utc)
        task.tokens_in = tokens_in
        task.tokens_out = tokens_out
        task.cost_usd = cost_usd
        await self.session.flush()
        return task

    async def mark_error(self, task_id: uuid.UUID, error: str) -> InferenceTask | None:
        """Mark a task as errored."""
        task = await self.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus.ERROR
        task.error = error
        task.finished_at = datetime.now(timezone.utc)
        await self.session.flush()
        return task

    async def list_pending(self, limit: int = 100) -> list[InferenceTask]:
        """List pending tasks."""
        stmt = (
            select(InferenceTask)
            .where(InferenceTask.status == TaskStatus.PENDING)
            .order_by(InferenceTask.queued_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
