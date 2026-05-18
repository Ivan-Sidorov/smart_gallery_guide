"""Feedback service: rate bot replies."""

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.feedback import FeedbackCreateRequest, FeedbackDTO
from db.models import MessageDirection
from db.repositories import FeedbackRepository, MessageRepository


def _to_dto(feedback) -> FeedbackDTO:
    return FeedbackDTO(
        id=feedback.id,
        message_id=feedback.message_id,
        rating=feedback.rating,
        comment=feedback.comment,
        created_at=feedback.created_at,
    )


class FeedbackService:
    """Create or update user feedback on outbound messages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def submit(self, payload: FeedbackCreateRequest) -> FeedbackDTO:
        """Record like/dislike, updates an existing row for the same message."""
        message_repo = MessageRepository(self._session)
        message = await message_repo.get_by_id(payload.message_id)
        if message is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Message not found")
        if message.user_id != payload.user_id:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail="Message doesn't belong to this user",
            )
        if message.direction != MessageDirection.OUT:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Feedback is only allowed on bot replies",
            )

        feedback_repo = FeedbackRepository(self._session)
        existing = await feedback_repo.get_for_message(payload.message_id)
        if existing is not None:
            existing.rating = payload.rating
            if payload.comment is not None:
                existing.comment = payload.comment.strip() or None
            await self._session.commit()
            await self._session.refresh(existing)
            return _to_dto(existing)

        feedback = await feedback_repo.add(
            message_id=payload.message_id,
            rating=payload.rating,
            comment=payload.comment.strip() if payload.comment else None,
        )
        await self._session.commit()
        await self._session.refresh(feedback)
        return _to_dto(feedback)
