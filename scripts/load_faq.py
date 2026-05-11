"""Load FAQ JSON files into Postgres."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import FAQDocument  # noqa: E402
from core.settings import get_settings  # noqa: E402
from db.repositories import ExhibitRepository, FAQItemRepository  # noqa: E402
from db.session import get_engine, session_scope  # noqa: E402

settings = get_settings()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


async def load_faq_from_directory(
    faq_dir: Path | None = None,
    *,
    source: str = "local_faq_files",
) -> None:
    """Load/replace FAQ items in Postgres from local JSON files."""
    faq_dir = faq_dir or settings.faq_dir

    faq_files = sorted(faq_dir.glob("*.json"))
    if not faq_files:
        print(f"[load_faq] No FAQ files found in {faq_dir}")
        return

    print(f"[load_faq] Found {len(faq_files)} FAQ files")

    stats = {
        "processed_files": 0,
        "skipped_files": 0,
        "missing_exhibits": 0,
        "deleted_old_items": 0,
        "inserted_items": 0,
        "invalid_items": 0,
        "failed": 0,
    }

    async with session_scope() as session:
        exhibit_repo = ExhibitRepository(session)
        faq_repo = FAQItemRepository(session)

        for faq_file in faq_files:
            data = _read_json(faq_file)
            if data is None:
                print(f"[load_faq] can't decode JSON: {faq_file.name}")
                stats["skipped_files"] += 1
                continue

            try:
                doc = FAQDocument(**data)
            except Exception as exc:
                print(f"[load_faq] invalid FAQ schema in {faq_file.name}: {exc}")
                stats["skipped_files"] += 1
                continue

            exhibit = await exhibit_repo.get(doc.exhibit_id)
            if exhibit is None:
                print(
                    f"[load_faq] exhibit {doc.exhibit_id} not found in Postgres "
                    f"(file: {faq_file.name})"
                )
                stats["missing_exhibits"] += 1
                continue

            item_source = (
                doc.source_model.strip()
                if isinstance(doc.source_model, str) and doc.source_model.strip()
                else source
            )

            prepared_items: list[tuple[str, str]] = []
            for qa in doc.questions:
                question = qa.question.strip()
                answer = qa.answer.strip()
                if not question or not answer:
                    stats["invalid_items"] += 1
                    continue
                prepared_items.append((question, answer))

            if not prepared_items:
                print(
                    f"[load_faq] no valid FAQ items in {faq_file.name}; "
                    f"existing rows are left unchanged"
                )
                stats["skipped_files"] += 1
                continue

            try:
                deleted = await faq_repo.delete_for_exhibit(doc.exhibit_id)
                stats["deleted_old_items"] += deleted
            except Exception as exc:
                print(f"[load_faq] failed to clear old FAQ for {doc.exhibit_id}: {exc}")
                stats["failed"] += 1
                continue

            inserted_for_file = 0
            for question, answer in prepared_items:
                try:
                    await faq_repo.add(
                        exhibit_id=doc.exhibit_id,
                        question=question,
                        answer=answer,
                        source=item_source,
                    )
                except Exception as exc:
                    print(
                        f"[load_faq] failed to insert FAQ item for {doc.exhibit_id}: {exc}"
                    )
                    stats["failed"] += 1
                    continue

                inserted_for_file += 1

            stats["processed_files"] += 1
            stats["inserted_items"] += inserted_for_file
            print(
                f"[load_faq] exhibit={doc.exhibit_id} "
                f"questions={inserted_for_file} file={faq_file.name}"
            )

    print("\n[load_faq] FAQ loading into Postgres completed.")
    print(
        "processed_files={processed_files}, skipped_files={skipped_files}, "
        "missing_exhibits={missing_exhibits}, deleted_old_items={deleted_old_items}, "
        "inserted_items={inserted_items}, invalid_items={invalid_items}, "
        "failed={failed}".format(**stats)
    )
    print(
        "[load_faq] FAQ rows are now stored in Postgres and ready for downstream indexing."
    )


async def _async_main(args: argparse.Namespace) -> None:
    """Load FAQ JSON files into Postgres."""
    try:
        await load_faq_from_directory(
            faq_dir=Path(args.faq_dir) if args.faq_dir else None,
            source=args.source,
        )
    finally:
        await get_engine().dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load FAQ JSON files into Postgres.")
    parser.add_argument(
        "--faq-dir",
        type=str,
        default="",
        help="Path to FAQ directory (defaults to settings.faq_dir).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="local_faq_files",
        help="Fallback value written into faq_items.source.",
    )
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
