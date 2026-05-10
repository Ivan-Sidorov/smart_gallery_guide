"""Session start/update endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_session_service
from api.schemas.sessions import (
    SessionContextUpdate,
    SessionDTO,
    SessionStartRequest,
)
from api.services import SessionService

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", response_model=SessionDTO, status_code=status.HTTP_200_OK)
async def start_or_resume_session(
    payload: SessionStartRequest,
    service: SessionService = Depends(get_session_service),
) -> SessionDTO:
    """Open/resume a session for a Telegram user."""
    return await service.start_or_resume(
        user_id=payload.user_id,
        username=payload.username,
        first_name=payload.first_name,
        last_name=payload.last_name,
        locale=payload.locale,
        context=payload.context or None,
    )


@router.get("/{session_id}", response_model=SessionDTO)
async def get_session(
    session_id: uuid.UUID,
    service: SessionService = Depends(get_session_service),
) -> SessionDTO:
    """Fetch session by id."""
    dto = await service.get(session_id)
    if dto is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
    return dto


@router.patch("/{session_id}/context", response_model=SessionDTO)
async def update_session_context(
    session_id: uuid.UUID,
    payload: SessionContextUpdate,
    service: SessionService = Depends(get_session_service),
) -> SessionDTO:
    """Replace the session's context."""
    dto = await service.update_context(session_id, payload.context)
    if dto is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
    return dto
