"""Feedback repository."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Feedback


class FeedbackRepository:
    """CRUD and lookup helpers for `feedback` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self, *, message_id: int, rating: int, comment: str | None = None
    ) -> Feedback:
        """Add feedback for a message."""
        feedback = Feedback(message_id=message_id, rating=rating, comment=comment)
        self.session.add(feedback)
        await self.session.flush()
        return feedback

    async def get_for_message(self, message_id: int) -> Feedback | None:
        """Return the latest feedback row for a message, if any."""
        stmt = (
            select(Feedback)
            .where(Feedback.message_id == message_id)
            .order_by(Feedback.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_message(self, message_id: int) -> list[Feedback]:
        """List feedback for a message."""
        stmt = select(Feedback).where(Feedback.message_id == message_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
