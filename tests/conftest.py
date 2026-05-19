"""Shared pytest fixtures and helpers for smart gallery guide tests."""

from typing import Any

from core.schemas import ExhibitMetadata


def build_metadata(exhibit_id: str = "ex-1", **overrides: Any) -> ExhibitMetadata:
    """Construct a valid ExhibitMetadata with sensible defaults for tests."""
    defaults: dict[str, Any] = dict(
        exhibit_id=exhibit_id,
        title="Звёздная ночь",
        artist="Винсент Ван Гог",
        year="1889",
        image_path=f"data/exhibits/{exhibit_id}.jpg",
    )
    defaults.update(overrides)
    return ExhibitMetadata(**defaults)
