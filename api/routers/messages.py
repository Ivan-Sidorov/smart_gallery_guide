"""Message logging endpoints."""

from fastapi import APIRouter, Depends, status

from api.deps import get_message_service
from api.schemas.messages import MessageCreateRequest, MessageDTO
from api.services import MessageService

router = APIRouter(prefix="/v1/messages", tags=["messages"])


@router.post("", response_model=MessageDTO, status_code=status.HTTP_201_CREATED)
async def create_message(
    payload: MessageCreateRequest,
    service: MessageService = Depends(get_message_service),
) -> MessageDTO:
    """Log a user–exhibit interaction (select or question)."""
    return await service.create_exhibit_event(payload)
