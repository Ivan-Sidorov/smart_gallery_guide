"""Exhibit search request/response DTOs."""

from typing import Any

from pydantic import BaseModel, Field


class ExhibitDTO(BaseModel):
    """Exhibit metadata as exposed via HTTP."""

    exhibit_id: str
    title: str
    author: str | None = None
    year: str | None = None
    description: str | None = None
    image_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ExhibitSearchRequest(BaseModel):
    """Request for hybrid text search over exhibits."""

    query: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of results to return (falls back to API default).",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine score (falls back to settings.exhibit_match_threshold).",
    )


class ExhibitSearchResultDTO(BaseModel):
    """Single hit of a hybrid text search."""

    exhibit_id: str
    title: str
    similarity_score: float
    metadata: dict[str, Any]
