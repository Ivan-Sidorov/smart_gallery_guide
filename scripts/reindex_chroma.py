"""Sync Chroma indexes with Postgres."""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.encoders.text import TextEncoder  # noqa: E402
from core.encoders.vision import VisionEncoder  # noqa: E402
from core.schemas import ExhibitMetadata  # noqa: E402
from core.sync import exhibit_to_metadata  # noqa: E402
from core.vector_db import VectorDatabase  # noqa: E402
from db.models import Exhibit, FAQItem  # noqa: E402
from db.session import get_engine, session_scope  # noqa: E402

DESC_EXTRA_KEYS = (
    "anotation",
    "glossary",
    "text",
    "epoque",
    "category",
    "place",
    "techniq",
)


@dataclass
class ReindexStats:
    exhibit_targets: int = 0
    exhibits_indexed: int = 0
    exhibits_missing_image: int = 0
    exhibits_failed: int = 0
    faq_targets: int = 0
    faq_indexed_items: int = 0
    faq_skipped_empty: int = 0
    faq_skipped_duplicates: int = 0
    faq_failed_items: int = 0
    faq_failed_exhibits: int = 0


def _resolve_image_path(image_path: str | None) -> Path | None:
    """Resolve image path from metadata."""
    if not image_path:
        return None
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _build_title_text(metadata: ExhibitMetadata) -> str:
    """Build title text from metadata."""
    return " ".join(
        filter(
            None,
            [
                metadata.artist or "",
                metadata.title,
                metadata.school or "",
                metadata.year or "",
            ],
        )
    ).strip()


def _build_desc_text(metadata: ExhibitMetadata) -> str:
    """Build description text from metadata."""
    parts: list[str] = []
    display = metadata.display_description.strip()
    if display:
        parts.append(display)

    additional = metadata.additional_info or {}
    for key in DESC_EXTRA_KEYS:
        value = additional.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                parts.append(value)

    if metadata.material and metadata.material.strip():
        parts.append(metadata.material.strip())
    return " ".join(parts).strip()


async def _collect_exhibit_targets(*, full: bool, limit: int) -> list[str]:
    """Collect exhibit IDs from Postgres."""
    async with session_scope() as session:
        stmt = select(Exhibit.id)
        if full:
            stmt = stmt.order_by(Exhibit.updated_at.asc())
        else:
            stmt = stmt.where(Exhibit.needs_reindex.is_(True)).order_by(
                Exhibit.updated_at.asc()
            )
        if limit > 0:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def _collect_faq_targets(*, full: bool, limit: int) -> list[str]:
    """Collect FAQ exhibit IDs from Postgres."""
    async with session_scope() as session:
        stmt = select(FAQItem.exhibit_id).distinct().order_by(FAQItem.exhibit_id.asc())
        if not full:
            stmt = stmt.where(FAQItem.indexed_at.is_(None))
        if limit > 0:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


def _delete_exhibit_from_chroma(vector_db: VectorDatabase, exhibit_id: str) -> None:
    """Delete exhibit from Chroma."""
    vector_db.exhibits_collection.delete(ids=[exhibit_id])
    vector_db.title_collection.delete(ids=[exhibit_id])
    vector_db.desc_collection.delete(ids=[exhibit_id])


async def _reindex_exhibit(
    *,
    exhibit_id: str,
    vector_db: VectorDatabase,
    vision_encoder: VisionEncoder,
    text_encoder: TextEncoder,
    stats: ReindexStats,
) -> bool:
    """Reindex an exhibit."""
    async with session_scope() as session:
        exhibit = await session.get(Exhibit, exhibit_id)
        if exhibit is None:
            print(f"[reindex_chroma] {exhibit_id}: not found in Postgres")
            stats.exhibits_failed += 1
            return False

        image_path = _resolve_image_path(exhibit.image_path)
        if image_path is None or not image_path.exists():
            print(f"[reindex_chroma] {exhibit_id}: image not found at {image_path}")
            stats.exhibits_missing_image += 1
            return False

        metadata = exhibit_to_metadata(exhibit)

        try:
            with Image.open(image_path) as image:
                image_rgb = image.convert("RGB")
            image_embedding = vision_encoder.encode_image(image_rgb).tolist()
        except Exception as exc:
            print(f"[reindex_chroma] {exhibit_id}: can't encode image ({exc})")
            stats.exhibits_failed += 1
            return False

        title_text = _build_title_text(metadata)
        desc_text = _build_desc_text(metadata)

        title_embedding: list[float] | None = None
        desc_embedding: list[float] | None = None

        if title_text:
            try:
                title_embedding = text_encoder.encode_text(title_text).tolist()
            except Exception as exc:
                print(
                    f"[reindex_chroma] {exhibit_id}: "
                    f"title embedding failed, continue without it ({exc})"
                )

        if desc_text:
            try:
                desc_embedding = text_encoder.encode_text(desc_text).tolist()
            except Exception as exc:
                print(
                    f"[reindex_chroma] {exhibit_id}: "
                    f"description embedding failed, continue without it ({exc})"
                )

        try:
            _delete_exhibit_from_chroma(vector_db, exhibit_id)
            vector_db.add_exhibit(
                exhibit_id=exhibit_id,
                embedding=image_embedding,
                metadata=metadata,
                title_embedding=title_embedding,
                title_text=title_text or None,
                desc_embedding=desc_embedding,
                desc_text=desc_text or None,
            )
        except Exception as exc:
            print(f"[reindex_chroma] {exhibit_id}: failed to write to Chroma ({exc})")
            stats.exhibits_failed += 1
            return False

        exhibit.needs_reindex = False
        exhibit.indexed_at = datetime.now(timezone.utc)
        stats.exhibits_indexed += 1
        print(f"[exhibit:ok] {exhibit_id}")
        return True


async def _reindex_faq_for_exhibit(
    *,
    exhibit_id: str,
    vector_db: VectorDatabase,
    text_encoder: TextEncoder,
    stats: ReindexStats,
) -> bool:
    """Reindex FAQ for an exhibit."""
    async with session_scope() as session:
        stmt = (
            select(FAQItem)
            .where(FAQItem.exhibit_id == exhibit_id)
            .order_by(FAQItem.created_at.asc(), FAQItem.id.asc())
        )
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        now = datetime.now(timezone.utc)

        try:
            vector_db.faq_collection.delete(where={"exhibit_id": exhibit_id})
        except Exception as exc:
            print(
                f"[reindex_chroma] {exhibit_id}: failed to clear old Chroma docs ({exc})"
            )
            stats.faq_failed_exhibits += 1
            return False

        seen_questions: set[str] = set()
        indexed_for_exhibit = 0

        for row in rows:
            question = row.question.strip()
            answer = row.answer.strip()

            if not question or not answer:
                row.indexed_at = now
                stats.faq_skipped_empty += 1
                continue

            dedupe_key = question.casefold()
            if dedupe_key in seen_questions:
                row.indexed_at = now
                stats.faq_skipped_duplicates += 1
                continue
            seen_questions.add(dedupe_key)

            try:
                embedding = text_encoder.encode_text(question).tolist()
                vector_db.add_faq(
                    question=question,
                    answer=answer,
                    exhibit_id=exhibit_id,
                    embedding=embedding,
                )
                row.indexed_at = now
                indexed_for_exhibit += 1
            except Exception as exc:
                # Keep indexed_at=NULL to retry on next run.
                print(
                    f"[reindex_chroma] {exhibit_id}: failed to index question ({exc})"
                )
                stats.faq_failed_items += 1

        stats.faq_indexed_items += indexed_for_exhibit
        print(
            f"[reindex_chroma] {exhibit_id}: indexed={indexed_for_exhibit}, "
            f"rows_in_pg={len(rows)}"
        )
        return True


def _print_stats(stats: ReindexStats) -> None:
    """Print reindex statistics."""
    print("\nReindex completed.")
    print(
        "exhibit_targets={exhibit_targets}, exhibits_indexed={exhibits_indexed}, "
        "exhibits_missing_image={exhibits_missing_image}, exhibits_failed={exhibits_failed}".format(
            **stats.__dict__
        )
    )
    print(
        "faq_targets={faq_targets}, faq_indexed_items={faq_indexed_items}, "
        "faq_skipped_empty={faq_skipped_empty}, "
        "faq_skipped_duplicates={faq_skipped_duplicates}, "
        "faq_failed_items={faq_failed_items}, faq_failed_exhibits={faq_failed_exhibits}".format(
            **stats.__dict__
        )
    )


async def run_reindex(args: argparse.Namespace) -> None:
    """Run reindex."""
    stats = ReindexStats()

    exhibit_targets: list[str] = []
    if not args.only_faq:
        exhibit_targets = await _collect_exhibit_targets(
            full=args.full_exhibits,
            limit=args.limit,
        )
    stats.exhibit_targets = len(exhibit_targets)

    faq_targets: list[str] = []
    if not args.skip_faq:
        faq_targets = await _collect_faq_targets(
            full=args.full_faq,
            limit=args.faq_limit,
        )
        # If an exhibit is reindexed, refresh its FAQ view in Chroma too
        faq_targets = sorted(set(faq_targets) | set(exhibit_targets))
    stats.faq_targets = len(faq_targets)

    if not exhibit_targets and (args.skip_faq or not faq_targets):
        print("[reindex_chroma] No pending records for reindex.")
        return

    vector_db: VectorDatabase | None = None
    vision_encoder: VisionEncoder | None = None
    text_encoder: TextEncoder | None = None
    chroma_updated = False

    try:
        vector_db = VectorDatabase()

        needs_text_encoder = bool(exhibit_targets) or (
            not args.skip_faq and bool(faq_targets)
        )
        if needs_text_encoder:
            print("[reindex_chroma] Loading TextEncoder...")
            text_encoder = TextEncoder()

        if exhibit_targets:
            print("[reindex_chroma] Loading VisionEncoder...")
            vision_encoder = VisionEncoder()

        if exhibit_targets:
            if vision_encoder is None or text_encoder is None:
                raise RuntimeError("Encoders are not initialised for exhibit reindex")
            for exhibit_id in exhibit_targets:
                ok = await _reindex_exhibit(
                    exhibit_id=exhibit_id,
                    vector_db=vector_db,
                    vision_encoder=vision_encoder,
                    text_encoder=text_encoder,
                    stats=stats,
                )
                if ok:
                    chroma_updated = True

        if not args.skip_faq and faq_targets:
            if text_encoder is None:
                raise RuntimeError("TextEncoder is not initialised for FAQ reindex")
            for exhibit_id in faq_targets:
                ok = await _reindex_faq_for_exhibit(
                    exhibit_id=exhibit_id,
                    vector_db=vector_db,
                    text_encoder=text_encoder,
                    stats=stats,
                )
                if ok:
                    chroma_updated = True

        if chroma_updated and not args.skip_bm25:
            print("[reindex_chroma] Rebuilding BM25 indexes...")
            vector_db.build_bm25_indexes()
            print("[reindex_chroma] BM25 indexes ready.")

    finally:
        if vision_encoder is not None:
            vision_encoder.close()
        if text_encoder is not None:
            text_encoder.close()
        if vector_db is not None:
            vector_db.close()
        await get_engine().dispose()

    _print_stats(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Chroma indexes with Postgres.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max exhibits to reindex (0 = all selected).",
    )
    parser.add_argument(
        "--faq-limit",
        type=int,
        default=0,
        help="Max FAQ exhibit groups to sync before union with exhibit targets (0 = all selected).",
    )
    parser.add_argument(
        "--full-exhibits",
        action="store_true",
        help="Reindex all exhibits, not only needs_reindex=true.",
    )
    parser.add_argument(
        "--full-faq",
        action="store_true",
        help="Reindex FAQ for all exhibits that have rows in faq_items.",
    )
    parser.add_argument(
        "--only-faq",
        action="store_true",
        help="Skip exhibit reindex and only sync FAQ.",
    )
    parser.add_argument(
        "--skip-faq",
        action="store_true",
        help="Skip FAQ sync phase.",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 rebuild after writes to Chroma.",
    )
    args = parser.parse_args()
    asyncio.run(run_reindex(args))


if __name__ == "__main__":
    main()
