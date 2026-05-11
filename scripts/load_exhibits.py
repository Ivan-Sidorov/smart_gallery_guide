"""Load exhibit metadata from files into Postgres."""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import ExhibitMetadata  # noqa: E402
from core.settings import get_settings  # noqa: E402
from db.repositories import ExhibitRepository  # noqa: E402
from db.session import get_engine, session_scope  # noqa: E402

settings = get_settings()

METADATA_ENCODINGS = ("utf-8", "cp1251", "latin-1")
IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png")

EXHIBIT_SCHEMA_KEYS = {
    "exhibit_id",
    "title",
    "artist",
    "year",
    "material",
    "dimensions",
    "school",
    "department",
    "inventory_number",
    "image_path",
    "description",
    "description_perplexity",
    "antic_art_description",
    "additional_info",
}


def _is_non_empty(value: Any) -> bool:
    """Check if a value is non-empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _read_json_with_fallback(path: Path) -> tuple[dict[str, Any], bytes] | None:
    """Read JSON with fallback encodings."""
    raw_bytes = path.read_bytes()
    for enc in METADATA_ENCODINGS:
        try:
            payload = json.loads(raw_bytes.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload, raw_bytes
        return None
    return None


def _storage_path(path: Path) -> str:
    """Get storage path relative to project root."""
    try:
        return str(path.relative_to(settings.project_root))
    except ValueError:
        return str(path)


def _build_checksum(metadata_raw: bytes, image_file: Path) -> str:
    """Build checksum for metadata and image."""
    digest = hashlib.sha256()
    digest.update(metadata_raw)
    digest.update(b"\n")
    with image_file.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_extra(
    metadata: ExhibitMetadata, raw_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Build extra metadata for exhibit."""
    extra: dict[str, Any] = dict(metadata.additional_info or {})

    for key, value in (
        ("material", metadata.material),
        ("dimensions", metadata.dimensions),
        ("school", metadata.school),
        ("department", metadata.department),
        ("inventory_number", metadata.inventory_number),
        ("description_perplexity", metadata.description_perplexity),
        ("antic_art_description", metadata.antic_art_description),
    ):
        if _is_non_empty(value):
            extra[key] = value

    for key, value in raw_metadata.items():
        if key in EXHIBIT_SCHEMA_KEYS:
            continue
        if _is_non_empty(value):
            extra[key] = value

    return extra


def _normalize_for_schema(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize metadata for schema."""
    normalized = dict(raw_metadata)
    for key in (
        "artist",
        "year",
        "material",
        "dimensions",
        "school",
        "department",
        "inventory_number",
        "description",
        "description_perplexity",
        "antic_art_description",
    ):
        value = normalized.get(key)
        if isinstance(value, str):
            value = value.strip()
            normalized[key] = value or None

    if not isinstance(normalized.get("additional_info"), dict):
        normalized["additional_info"] = {}
    return normalized


async def load_exhibits_from_directory(
    exhibits_dir: Path | None = None,
    metadata_dir: Path | None = None,
    *,
    source: str = "local_metadata_files",
    force_reindex: bool = False,
) -> None:
    """Load/refresh exhibits in Postgres from local image+metadata files."""
    exhibits_dir = exhibits_dir or settings.exhibits_dir
    metadata_dir = metadata_dir or settings.metadata_dir

    image_files = sorted(
        {image for pattern in IMAGE_GLOBS for image in exhibits_dir.glob(pattern)}
    )
    if not image_files:
        print(f"[load_exhibits] No image files found in {exhibits_dir}")
        return

    print(f"[load_exhibits] Found {len(image_files)} exhibit images")

    stats = {
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
        "missing_metadata": 0,
        "invalid_metadata": 0,
        "failed": 0,
    }

    async with session_scope() as session:
        repo = ExhibitRepository(session)

        for image_file in image_files:
            exhibit_id = image_file.stem
            metadata_file = metadata_dir / f"{exhibit_id}.json"
            if not metadata_file.exists():
                print(f"[load_exhibits] metadata not found for {exhibit_id}")
                stats["missing_metadata"] += 1
                continue

            decoded = _read_json_with_fallback(metadata_file)
            if decoded is None:
                print(f"[load_exhibits] can't decode metadata for {exhibit_id}")
                stats["invalid_metadata"] += 1
                continue
            raw_metadata, raw_bytes = decoded

            storage_image_path = _storage_path(image_file)
            normalized = _normalize_for_schema(raw_metadata)
            normalized["exhibit_id"] = exhibit_id
            normalized["image_path"] = storage_image_path

            try:
                metadata = ExhibitMetadata(**normalized)
            except Exception as exc:
                print(f"[load_exhibits] invalid metadata for {exhibit_id}: {exc}")
                stats["invalid_metadata"] += 1
                continue

            try:
                checksum = _build_checksum(raw_bytes, image_file)
            except Exception as exc:
                print(f"[load_exhibits] can't hash payload for {exhibit_id}: {exc}")
                stats["failed"] += 1
                continue

            existing = await repo.get(exhibit_id)
            if (
                not force_reindex
                and existing is not None
                and existing.checksum == checksum
                and not existing.needs_reindex
            ):
                stats["unchanged"] += 1
                continue

            extra = _build_extra(metadata, normalized)
            description = metadata.display_description.strip() or None

            try:
                await repo.upsert(
                    id=exhibit_id,
                    title=metadata.title,
                    author=metadata.artist,
                    year=metadata.year,
                    description=description,
                    image_path=storage_image_path,
                    extra=extra,
                    source=source,
                    checksum=checksum,
                    needs_reindex=True,
                    indexed_at=None,
                )
            except Exception as exc:
                print(f"[load_exhibits] failed to upsert exhibit {exhibit_id}: {exc}")
                stats["failed"] += 1
                continue

            if existing is None:
                stats["inserted"] += 1
                print(f"[load_exhibits] {metadata.title} ({exhibit_id})")
            else:
                stats["updated"] += 1
                print(f"[load_exhibits] {metadata.title} ({exhibit_id})")

    print("\n[load_exhibits] Exhibit loading into Postgres completed.")
    print(
        "inserted={inserted}, updated={updated}, unchanged={unchanged}, "
        "missing_metadata={missing_metadata}, invalid_metadata={invalid_metadata}, "
        "failed={failed}".format(**stats)
    )
    print(
        "[load_exhibits] Rows changed are flagged with needs_reindex=true for downstream indexing."
    )


async def _async_main(args: argparse.Namespace) -> None:
    """Load exhibits from local files into Postgres."""
    try:
        await load_exhibits_from_directory(
            exhibits_dir=Path(args.exhibits_dir) if args.exhibits_dir else None,
            metadata_dir=Path(args.metadata_dir) if args.metadata_dir else None,
            source=args.source,
            force_reindex=args.force_reindex,
        )
    finally:
        await get_engine().dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load exhibits from local files into Postgres."
    )
    parser.add_argument(
        "--exhibits-dir",
        type=str,
        default="",
        help="Path to image directory (defaults to settings.exhibits_dir).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="",
        help="Path to metadata directory (defaults to settings.metadata_dir).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="local_metadata_files",
        help="Value written into exhibits.source.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Flag all matched rows with needs_reindex=true even if checksum is unchanged.",
    )
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
