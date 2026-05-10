"""Q&A endpoints."""

import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.deps import get_qa_service
from api.schemas.qa import QAExhibitRequest, QAResponse
from api.services import QAService

router = APIRouter(prefix="/v1/qa", tags=["qa"])


@router.post("/exhibit", response_model=QAResponse)
async def qa_exhibit(
    payload: QAExhibitRequest,
    service: QAService = Depends(get_qa_service),
) -> QAResponse:
    """FAQ-first Q&A about a known exhibit (falls back to VLM)."""
    return await service.answer_about_exhibit(
        exhibit_id=payload.exhibit_id,
        question=payload.question,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )


@router.post("/image", response_model=QAResponse)
async def qa_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    exhibit_id: str | None = Form(default=None),
    user_id: int | None = Form(default=None),
    session_id: uuid.UUID | None = Form(default=None),
    service: QAService = Depends(get_qa_service),
) -> QAResponse:
    """VLM Q&A over a user image."""
    image_bytes = await image.read()
    return await service.answer_about_image(
        image_bytes=image_bytes,
        question=question,
        exhibit_id=exhibit_id,
        user_id=user_id,
        session_id=session_id,
    )
