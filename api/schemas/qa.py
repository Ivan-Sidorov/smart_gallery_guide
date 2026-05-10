"""Q&A request/response DTOs."""

import uuid

from pydantic import BaseModel, Field


class QAImageRequest(BaseModel):
    """Request for VLM Q&A over a user image."""

    question: str = Field(min_length=1, max_length=2000)
    exhibit_id: str | None = None
    user_id: int | None = None
    session_id: uuid.UUID | None = None


class QAExhibitRequest(BaseModel):
    """Request for FAQ-first Q&A about an exhibit already known to the system."""

    exhibit_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2000)
    user_id: int | None = None
    session_id: uuid.UUID | None = None


class QAResponse(BaseModel):
    """Either an immediate answer (from FAQ) or a task handle for VLM."""

    mode: str = Field(description="'faq' for immediate answer, 'task' for VLM.")
    answer: str | None = None
    task_id: uuid.UUID | None = None
