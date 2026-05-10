"""Postgres –> Chroma metadata sync."""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.schemas import ExhibitMetadata
from core.vector_db import VectorDatabase
from db.models import Exhibit
from db.repositories import ExhibitRepository

logger = logging.getLogger(__name__)


def exhibit_to_metadata(exhibit: Exhibit) -> ExhibitMetadata:
    """Build an ``ExhibitMetadata`` from an ORM ``Exhibit`` row."""
    extra: dict[str, Any] = exhibit.extra or {}

    known_keys = {
        "material",
        "dimensions",
        "school",
        "department",
        "inventory_number",
        "description_perplexity",
        "antic_art_description",
    }
    additional_info = {k: v for k, v in extra.items() if k not in known_keys}

    return ExhibitMetadata(
        exhibit_id=exhibit.id,
        title=exhibit.title,
        artist=exhibit.author,
        year=exhibit.year,
        material=extra.get("material"),
        dimensions=extra.get("dimensions"),
        school=extra.get("school"),
        department=extra.get("department"),
        inventory_number=extra.get("inventory_number"),
        image_path=exhibit.image_path or "",
        description=exhibit.description,
        description_perplexity=extra.get("description_perplexity"),
        antic_art_description=extra.get("antic_art_description"),
        additional_info=additional_info,
    )


async def sync_exhibit_to_chroma(
    session: AsyncSession,
    exhibit_id: str,
    vector_db: VectorDatabase,
) -> bool:
    """Refresh Chroma metadata for a single exhibit from Postgres."""
    repo = ExhibitRepository(session)
    exhibit = await repo.get(exhibit_id)
    if exhibit is None:
        logger.warning("[sync] Exhibit %s not found in Postgres", exhibit_id)
        return False

    metadata = exhibit_to_metadata(exhibit)
    updated = vector_db.update_exhibit_metadata(exhibit_id, metadata)
    if not updated:
        logger.info(
            "[sync] Exhibit %s has no embedding in Chroma yet, needs full ingest",
            exhibit_id,
        )
        return False

    exhibit.indexed_at = datetime.now(timezone.utc)
    exhibit.needs_reindex = False
    await session.flush()
    return True


async def sync_pending_reindex(
    session: AsyncSession,
    vector_db: VectorDatabase,
    limit: int = 100,
) -> int:
    """Sync all exhibits flagged with ``needs_reindex=True`` (metadata only)."""
    repo = ExhibitRepository(session)
    pending = await repo.list_pending_reindex(limit=limit)
    if not pending:
        return 0

    now = datetime.now(timezone.utc)
    updated_count = 0
    for exhibit in pending:
        metadata = exhibit_to_metadata(exhibit)
        if not vector_db.update_exhibit_metadata(exhibit.id, metadata):
            logger.info(
                "[sync] Exhibit %s flagged for reindex but missing in Chroma, "
                "skipping (full ingest required)",
                exhibit.id,
            )
            continue
        exhibit.indexed_at = now
        exhibit.needs_reindex = False
        updated_count += 1

    await session.flush()
    logger.info(
        "[sync] Refreshed Chroma metadata for %d/%d pending exhibits",
        updated_count,
        len(pending),
    )
    return updated_count
