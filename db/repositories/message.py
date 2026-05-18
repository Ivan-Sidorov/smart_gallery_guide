"""Message repository."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Message, MessageDirection, MessageType


class MessageRepository:
    """CRUD and lookup helpers for `messages` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        session_id: uuid.UUID,
        user_id: int,
        direction: MessageDirection,
        type: MessageType,
        content: str | None = None,
        attachments: dict[str, Any] | None = None,
        api_task_id: uuid.UUID | None = None,
        latency_ms: int | None = None,
    ) -> Message:
        """Add a message."""
        message = Message(
            session_id=session_id,
            user_id=user_id,
            direction=direction,
            type=type,
            content=content,
            attachments=attachments or {},
            api_task_id=api_task_id,
            latency_ms=latency_ms,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def get_by_id(self, message_id: int) -> Message | None:
        """Fetch a message by primary key."""
        stmt = select(Message).where(Message.id == message_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_session(
        self, session_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> list[Message]:
        """List messages for a session."""
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
