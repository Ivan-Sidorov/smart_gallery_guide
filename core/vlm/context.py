"""Build exhibit context strings for VLM prompts."""

from typing import Any

from core.schemas import ExhibitMetadata

_EXTRA_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("material", "Материал"),
    ("school", "Школа/страна"),
    ("techniq", "Техника"),
    ("place", "Место"),
    ("epoque", "Эпоха"),
)


def build_exhibit_context(
    *,
    title: str,
    author: str | None = None,
    year: str | None = None,
    description: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render exhibit fields into the context block passed to the VLM."""
    parts = [f"Название: {title}"]
    if author:
        parts.append(f"Автор: {author}")
    if year:
        parts.append(f"Год: {year}")

    for key, label in _EXTRA_FIELD_LABELS:
        value = (extra or {}).get(key)
        if value:
            parts.append(f"{label}: {value}")

    if description:
        parts.append(f"Описание: {description}")
    if anotation := (extra or {}).get("anotation"):
        parts.append(f"Экспертный комментарий: {anotation}")
    return "\n".join(parts)


def build_exhibit_context_from_exhibit(exhibit: Any) -> str:
    """Build context from a DB `Exhibit` ORM row."""
    return build_exhibit_context(
        title=exhibit.title,
        author=exhibit.author,
        year=exhibit.year,
        description=exhibit.description,
        extra=dict(exhibit.extra or {}),
    )


def _metadata_extra_payload(
    metadata: ExhibitMetadata, raw: dict[str, Any]
) -> dict[str, Any]:
    """Mirror `load_exhibits._build_extra` for local JSON files."""
    extra: dict[str, Any] = dict(metadata.additional_info or {})
    for key in (
        "material",
        "dimensions",
        "school",
        "department",
        "inventory_number",
        "description_perplexity",
        "antic_art_description",
    ):
        value = getattr(metadata, key, None)
        if value is not None and str(value).strip():
            extra[key] = value
    skip = {
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
        "use_in_knowledge_base",
    }
    for key, value in raw.items():
        if key in skip:
            continue
        if value is not None and str(value).strip():
            extra[key] = value
    return extra


def build_exhibit_context_from_metadata(raw: dict[str, Any]) -> str:
    """Build context from `data/metadata/<exhibit_id>.json`."""
    exhibit_id = str(raw.get("exhibit_id") or "").strip() or "unknown"
    metadata = ExhibitMetadata(
        exhibit_id=exhibit_id,
        title=str(raw.get("title") or ""),
        artist=raw.get("artist"),
        year=raw.get("year"),
        material=raw.get("material"),
        dimensions=raw.get("dimensions"),
        school=raw.get("school"),
        department=raw.get("department"),
        inventory_number=raw.get("inventory_number"),
        image_path=str(raw.get("image_path") or ""),
        description=raw.get("description"),
        description_perplexity=raw.get("description_perplexity"),
        antic_art_description=raw.get("antic_art_description"),
        additional_info=dict(raw.get("additional_info") or {}),
    )
    description = metadata.display_description.strip() or None
    return build_exhibit_context(
        title=metadata.title,
        author=metadata.artist,
        year=metadata.year,
        description=description,
        extra=_metadata_extra_payload(metadata, raw),
    )
