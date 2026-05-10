"""Exhibit repository."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Exhibit


class ExhibitRepository:
    """CRUD and lookup helpers for `exhibits` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, exhibit_id: str) -> Exhibit | None:
        return await self.session.get(Exhibit, exhibit_id)

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Exhibit]:
        """List all exhibits."""
        stmt = (
            select(Exhibit)
            .order_by(Exhibit.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_pending_reindex(self, limit: int = 100) -> list[Exhibit]:
        """List exhibits flagged for reindex."""
        stmt = (
            select(Exhibit)
            .where(Exhibit.needs_reindex.is_(True))
            .order_by(Exhibit.updated_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def upsert(self, **fields: object) -> Exhibit:
        """Insert or update an exhibit by ``id``."""
        exhibit_id = fields.get("id")
        if not isinstance(exhibit_id, str):
            raise ValueError("Exhibit upsert requires 'id' (str).")
        exhibit = await self.session.get(Exhibit, exhibit_id)
        if exhibit is None:
            exhibit = Exhibit(**fields)
            self.session.add(exhibit)
        else:
            for key, value in fields.items():
                if key == "id":
                    continue
                setattr(exhibit, key, value)
        await self.session.flush()
        return exhibit

    async def mark_for_reindex(self, exhibit_id: str) -> None:
        """Flag an exhibit for reindex."""
        exhibit = await self.get(exhibit_id)
        if exhibit is None:
            return
        exhibit.needs_reindex = True
        await self.session.flush()
