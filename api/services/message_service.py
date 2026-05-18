"""Message service: persist user–exhibit interaction events."""

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.messages import (
    BotReplyCreateRequest,
    MessageCreateRequest,
    MessageDTO,
)
from db.models import MessageDirection, MessageType
from db.repositories import MessageRepository, SessionRepository


def _to_dto(message) -> MessageDTO:
    return MessageDTO(
        id=message.id,
        session_id=message.session_id,
        user_id=message.user_id,
        direction=message.direction.value,
        type=message.type.value,
        content=message.content,
        attachments=dict(message.attachments or {}),
        created_at=message.created_at,
    )


class MessageService:
    """Create interaction messages tied to a user session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_exhibit_event(self, payload: MessageCreateRequest) -> MessageDTO:
        """Record select/question for an exhibit under the given session."""
        session_repo = SessionRepository(self._session)
        row = await session_repo.get(payload.session_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
        if row.user_id != payload.user_id:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail="Session does not belong to this user",
            )

        if payload.event == "question" and not (payload.content or "").strip():
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="content is required for question events",
            )

        msg_type = (
            MessageType.CALLBACK if payload.event == "select" else MessageType.TEXT
        )
        message = await MessageRepository(self._session).add(
            session_id=payload.session_id,
            user_id=payload.user_id,
            direction=MessageDirection.IN,
            type=msg_type,
            content=payload.content.strip() if payload.content else None,
            attachments={
                "exhibit_id": payload.exhibit_id,
                "event": payload.event,
            },
        )
        await self._session.commit()
        await self._session.refresh(message)
        return _to_dto(message)

    async def create_bot_reply(self, payload: BotReplyCreateRequest) -> MessageDTO:
        """Record an outbound bot answer that can receive feedback."""
        session_repo = SessionRepository(self._session)
        row = await session_repo.get(payload.session_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
        if row.user_id != payload.user_id:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail="Session does not belong to this user",
            )

        attachments: dict = {}
        if payload.exhibit_id:
            attachments["exhibit_id"] = payload.exhibit_id

        message = await MessageRepository(self._session).add(
            session_id=payload.session_id,
            user_id=payload.user_id,
            direction=MessageDirection.OUT,
            type=MessageType.TEXT,
            content=payload.content.strip(),
            attachments=attachments,
            api_task_id=payload.api_task_id,
        )
        await self._session.commit()
        await self._session.refresh(message)
        return _to_dto(message)
