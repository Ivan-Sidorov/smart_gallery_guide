"""User repository."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User


class UserRepository:
    """CRUD and lookup helpers for `users` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, user_id: int) -> User | None:
        """Get a user by ID."""
        return await self.session.get(User, user_id)

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[User]:
        """List all users."""
        stmt = select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def upsert(
        self,
        user_id: int,
        *,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        locale: str | None = None,
    ) -> User:
        """Insert or update a user."""
        user = await self.session.get(User, user_id)
        now = datetime.now(timezone.utc)
        if user is None:
            user = User(
                id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                locale=locale,
                last_seen_at=now,
            )
            self.session.add(user)
        else:
            if username is not None:
                user.username = username
            if first_name is not None:
                user.first_name = first_name
            if last_name is not None:
                user.last_name = last_name
            if locale is not None:
                user.locale = locale
            user.last_seen_at = now
        await self.session.flush()
        return user
