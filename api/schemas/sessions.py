"""Session start/update request/response DTOs."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    """Open or reuse an active session for a Telegram user."""

    user_id: int = Field(description="Telegram user id.")
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    locale: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class SessionContextUpdate(BaseModel):
    """Partial context update for an existing session."""

    context: dict[str, Any]


class SessionDTO(BaseModel):
    """Session row exposed over HTTP."""

    id: uuid.UUID
    user_id: int
    started_at: datetime
    ended_at: datetime | None = None
    context: dict[str, Any] = Field(default_factory=dict)
