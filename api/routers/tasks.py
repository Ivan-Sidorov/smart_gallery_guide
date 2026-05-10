"""Task status endpoint."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_task_service
from api.schemas.tasks import TaskDTO
from api.services import TaskService

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskDTO)
async def get_task(
    task_id: uuid.UUID,
    service: TaskService = Depends(get_task_service),
) -> TaskDTO:
    """Fetch the status/result of a queued inference task."""
    dto = await service.get(task_id)
    if dto is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Task not found")
    return dto
