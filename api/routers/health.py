"""Liveness/readiness endpoints."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db_session
from api.schemas.common import HealthResponse
from core.settings import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Always 200 if the process is running."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=HealthResponse)
async def readiness(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Verify Postgres and ML components (optional)."""
    components: dict[str, str] = {}

    try:
        await session.execute(text("SELECT 1"))
        components["postgres"] = "ok"
    except Exception:
        components["postgres"] = "down"

    if settings.api_load_ml:
        components["text_encoder"] = (
            "ok" if getattr(request.app.state, "text_encoder", None) else "down"
        )
        components["vector_db"] = (
            "ok" if getattr(request.app.state, "vector_db", None) else "down"
        )
    else:
        components["text_encoder"] = "disabled"
        components["vector_db"] = "disabled"

    overall = (
        "ok" if all(v in {"ok", "disabled"} for v in components.values()) else "down"
    )
    return HealthResponse(status=overall, components=components)
