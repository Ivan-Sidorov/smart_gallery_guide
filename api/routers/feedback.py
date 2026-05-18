"""Feedback endpoints."""

from fastapi import APIRouter, Depends, status

from api.deps import get_feedback_service
from api.schemas.feedback import FeedbackCreateRequest, FeedbackDTO
from api.services import FeedbackService

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackDTO, status_code=status.HTTP_201_CREATED)
async def create_feedback(
    payload: FeedbackCreateRequest,
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackDTO:
    """Submit like/dislike for a bot reply message."""
    return await service.submit(payload)
