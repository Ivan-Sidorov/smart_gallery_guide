"""FAQ search request/response DTOs."""

from pydantic import BaseModel, Field


class FAQSearchRequest(BaseModel):
    """Search FAQ items inside a single exhibit by question text."""

    exhibit_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class FAQSearchResultDTO(BaseModel):
    """Single FAQ hit."""

    exhibit_id: str
    question: str
    answer: str
    similarity_score: float
