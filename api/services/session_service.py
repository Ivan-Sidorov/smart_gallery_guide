"""Session/User service: maps Telegram users to dialogue sessions in Postgres."""

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.sessions import SessionDTO
from db.models import Session
from db.repositories import SessionRepository, UserRepository


def _session_to_dto(session: Session) -> SessionDTO:
    return SessionDTO(
        id=session.id,
        user_id=session.user_id,
        started_at=session.started_at,
        ended_at=session.ended_at,
        context=dict(session.context or {}),
    )


class SessionService:
    """Open/resume/fetch/update dialogue sessions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start_or_resume(
        self,
        *,
        user_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        locale: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> SessionDTO:
        """Upsert the user and return their active session."""
        user_repo = UserRepository(self._session)
        await user_repo.upsert(
            user_id=user_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            locale=locale,
        )

        session_repo = SessionRepository(self._session)
        active = await session_repo.get_active_for_user(user_id)
        if active is None:
            active = await session_repo.create(user_id=user_id, context=context)
        elif context:
            merged = {**(active.context or {}), **context}
            await session_repo.update_context(active.id, merged)
            active = await session_repo.get(active.id)
            if active is None:
                raise RuntimeError(
                    "Failed to reload active session after context update"
                )
        await self._session.commit()
        return _session_to_dto(active)

    async def get(self, session_id: uuid.UUID) -> SessionDTO | None:
        """Fetch a session by id."""
        repo = SessionRepository(self._session)
        row = await repo.get(session_id)
        return _session_to_dto(row) if row is not None else None

    async def update_context(
        self, session_id: uuid.UUID, context: dict[str, Any]
    ) -> SessionDTO | None:
        """Replace a session's context."""
        repo = SessionRepository(self._session)
        row = await repo.update_context(session_id, context)
        if row is None:
            return None
        await self._session.commit()
        return _session_to_dto(row)
