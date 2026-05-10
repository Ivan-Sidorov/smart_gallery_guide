"""Inference task DTO."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TaskDTO(BaseModel):
    """Async inference task as exposed over HTTP."""

    id: uuid.UUID
    type: str
    status: str
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
