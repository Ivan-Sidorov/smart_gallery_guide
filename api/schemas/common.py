"""Common response shapes."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str = Field(description="'ok' if the check passed.")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Per-component status: 'ok' / 'down' / 'disabled'.",
    )


class ErrorResponse(BaseModel):
    """Uniform error response."""

    detail: str
    request_id: str | None = None
