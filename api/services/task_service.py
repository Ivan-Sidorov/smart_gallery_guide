"""Inference task service."""

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.tasks import TaskDTO
from db.models import InferenceTask, TaskType
from db.repositories import InferenceTaskRepository


def _task_to_dto(task: InferenceTask) -> TaskDTO:
    return TaskDTO(
        id=task.id,
        type=task.type.value if hasattr(task.type, "value") else str(task.type),
        status=task.status.value if hasattr(task.status, "value") else str(task.status),
        queued_at=task.queued_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        request=dict(task.request or {}),
        result=dict(task.result) if task.result is not None else None,
        error=task.error,
    )


class TaskService:
    """Create/read inference tasks."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, task_id: uuid.UUID) -> TaskDTO | None:
        """Fetch a task row by id."""
        repo = InferenceTaskRepository(self._session)
        task = await repo.get(task_id)
        return _task_to_dto(task) if task is not None else None

    async def enqueue(
        self,
        *,
        type: TaskType,
        request: dict[str, Any],
        user_id: int | None = None,
        session_id: uuid.UUID | None = None,
        model: str | None = None,
    ) -> TaskDTO:
        """Persist a new pending task and return it."""
        repo = InferenceTaskRepository(self._session)
        task = await repo.create(
            type=type,
            request=request,
            user_id=user_id,
            session_id=session_id,
            model=model,
        )
        await self._session.commit()
        await self._session.refresh(task)
        return _task_to_dto(task)
