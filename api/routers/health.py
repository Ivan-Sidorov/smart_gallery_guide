"""Liveness/readiness endpoints."""

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db_session
from api.schemas.common import HealthResponse
from core.settings import Settings, get_settings

router = APIRouter(tags=["health"])


def _vllm_models_url(base_url: str) -> str:
    """Resolve an OpenAI-compatible `/v1/models` URL from a configured base."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


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
        components["vision_encoder"] = (
            "ok" if getattr(request.app.state, "vision_encoder", None) else "down"
        )
        components["asr_encoder"] = (
            "ok" if getattr(request.app.state, "asr_encoder", None) else "down"
        )
        components["chroma"] = (
            "ok" if getattr(request.app.state, "vector_db", None) else "down"
        )
    else:
        components["text_encoder"] = "disabled"
        components["vision_encoder"] = "disabled"
        components["asr_encoder"] = "disabled"
        components["chroma"] = "disabled"

    vllm_url = _vllm_models_url(settings.vllm_api_base_url)
    headers = (
        {"Authorization": f"Bearer {settings.vllm_api_key}"}
        if settings.vllm_api_key
        else {}
    )
    try:
        async with httpx.AsyncClient(timeout=2.5, trust_env=False) as client:
            response = await client.get(vllm_url, headers=headers)
        components["vllm"] = "ok" if response.is_success else "down"
    except Exception:
        components["vllm"] = "down"

    overall = (
        "ok" if all(v in {"ok", "disabled"} for v in components.values()) else "down"
    )
    return HealthResponse(status=overall, components=components)
