"""Message create/response DTOs."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ExhibitEvent = Literal["select", "question"]


class MessageCreateRequest(BaseModel):
    """Log a user–exhibit interaction event."""

    session_id: uuid.UUID
    user_id: int = Field(description="Telegram user id.")
    exhibit_id: str = Field(min_length=1, max_length=128)
    event: ExhibitEvent
    content: str | None = Field(
        default=None,
        max_length=2000,
        description="Question text when event is 'question'.",
    )


class MessageDTO(BaseModel):
    """Persisted message row."""

    id: int
    session_id: uuid.UUID
    user_id: int
    direction: str
    type: str
    content: str | None
    attachments: dict[str, Any]
    created_at: datetime
