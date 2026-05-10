"""Pydantic DTOs for the FastAPI."""

from api.schemas.common import ErrorResponse, HealthResponse
from api.schemas.exhibits import (
    ExhibitDTO,
    ExhibitSearchRequest,
    ExhibitSearchResultDTO,
)
from api.schemas.faq import FAQSearchRequest, FAQSearchResultDTO
from api.schemas.qa import QAExhibitRequest, QAImageRequest, QAResponse
from api.schemas.sessions import (
    SessionContextUpdate,
    SessionDTO,
    SessionStartRequest,
)
from api.schemas.tasks import TaskDTO

__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "ExhibitDTO",
    "ExhibitSearchRequest",
    "ExhibitSearchResultDTO",
    "FAQSearchRequest",
    "FAQSearchResultDTO",
    "QAImageRequest",
    "QAExhibitRequest",
    "QAResponse",
    "SessionStartRequest",
    "SessionContextUpdate",
    "SessionDTO",
    "TaskDTO",
]
