"""FAQ item repository."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import FAQItem


class FAQItemRepository:
    """CRUD and lookup helpers for `faq_items` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, faq_id: uuid.UUID) -> FAQItem | None:
        return await self.session.get(FAQItem, faq_id)

    async def list_for_exhibit(self, exhibit_id: str) -> list[FAQItem]:
        """List FAQ items for an exhibit."""
        stmt = (
            select(FAQItem)
            .where(FAQItem.exhibit_id == exhibit_id)
            .order_by(FAQItem.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def add(
        self,
        *,
        exhibit_id: str,
        question: str,
        answer: str,
        source: str | None = None,
    ) -> FAQItem:
        """Add a FAQ item."""
        item = FAQItem(
            exhibit_id=exhibit_id,
            question=question,
            answer=answer,
            source=source,
        )
        self.session.add(item)
        await self.session.flush()
        return item
