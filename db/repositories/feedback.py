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

    async def list_for_message(self, message_id: int) -> list[Feedback]:
        """List feedback for a message."""
        stmt = select(Feedback).where(Feedback.message_id == message_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
