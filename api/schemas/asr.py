"""ASR API schemas."""

from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    """Speech-to-text result."""

    text: str = Field(description="Recognised utterance as plain text.")
