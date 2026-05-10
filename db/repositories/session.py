"""Session repository."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Session


class SessionRepository:
    """CRUD and lookup helpers for `sessions` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, session_id: uuid.UUID) -> Session | None:
        """Get a session by ID."""
        return await self.session.get(Session, session_id)

    async def get_active_for_user(self, user_id: int) -> Session | None:
        """Get the active session for a user."""
        stmt = (
            select(Session)
            .where(Session.user_id == user_id, Session.ended_at.is_(None))
            .order_by(Session.started_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self, user_id: int, context: dict[str, Any] | None = None
    ) -> Session:
        """Create a new session."""
        session = Session(user_id=user_id, context=context or {})
        self.session.add(session)
        await self.session.flush()
        return session

    async def update_context(
        self, session_id: uuid.UUID, context: dict[str, Any]
    ) -> Session | None:
        """Update the context of a session."""
        session = await self.get(session_id)
        if session is None:
            return None
        session.context = context
        await self.session.flush()
        return session

    async def end(self, session_id: uuid.UUID) -> None:
        """End a session."""
        session = await self.get(session_id)
        if session is None:
            return
        session.ended_at = datetime.now(timezone.utc)
        await self.session.flush()
