"""Feedback create/response DTOs."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FeedbackRating = Literal[1, -1]


class FeedbackCreateRequest(BaseModel):
    """Submit or update like/dislike for a bot reply."""

    message_id: int = Field(gt=0)
    user_id: int = Field(description="Telegram user id for authorization.")
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackDTO(BaseModel):
    """Persisted feedback row."""

    id: uuid.UUID
    message_id: int
    rating: int
    comment: str | None
    created_at: datetime
