"""ML core of the Smart Gallery Guide service.

Heavy ML imports are loaded lazily so that Telegram adapter and scripts that only need
`Settings` do not pull in heavy dependencies.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # noqa: SIM108
    from core.agent import GuideAgent
    from core.encoders.text import TextEncoder
    from core.encoders.vision import VisionEncoder
    from core.schemas import (
        ExhibitMetadata,
        ExhibitSearchResult,
        FAQDocument,
        FAQItem,
        FAQSearchResult,
    )
    from core.search.web import WebSearchResult, WebSearchService
    from core.settings import Settings, get_settings
    from core.sync import (
        exhibit_to_metadata,
        sync_exhibit_to_chroma,
        sync_pending_reindex,
    )
    from core.vector_db import VectorDatabase
    from core.vlm.client import VLM, SearchEvaluation


__all__ = [
    "GuideAgent",
    "TextEncoder",
    "VisionEncoder",
    "VectorDatabase",
    "VLM",
    "SearchEvaluation",
    "WebSearchService",
    "WebSearchResult",
    "ExhibitMetadata",
    "ExhibitSearchResult",
    "FAQDocument",
    "FAQItem",
    "FAQSearchResult",
    "Settings",
    "get_settings",
    "exhibit_to_metadata",
    "sync_exhibit_to_chroma",
    "sync_pending_reindex",
]


# Maps public attribute name -> (submodule, attribute name in submodule).
_LAZY: dict[str, tuple[str, str]] = {
    "GuideAgent": ("core.agent", "GuideAgent"),
    "TextEncoder": ("core.encoders.text", "TextEncoder"),
    "VisionEncoder": ("core.encoders.vision", "VisionEncoder"),
    "VectorDatabase": ("core.vector_db", "VectorDatabase"),
    "VLM": ("core.vlm.client", "VLM"),
    "SearchEvaluation": ("core.vlm.client", "SearchEvaluation"),
    "WebSearchService": ("core.search.web", "WebSearchService"),
    "WebSearchResult": ("core.search.web", "WebSearchResult"),
    "ExhibitMetadata": ("core.schemas", "ExhibitMetadata"),
    "ExhibitSearchResult": ("core.schemas", "ExhibitSearchResult"),
    "FAQDocument": ("core.schemas", "FAQDocument"),
    "FAQItem": ("core.schemas", "FAQItem"),
    "FAQSearchResult": ("core.schemas", "FAQSearchResult"),
    "Settings": ("core.settings", "Settings"),
    "get_settings": ("core.settings", "get_settings"),
    "exhibit_to_metadata": ("core.sync", "exhibit_to_metadata"),
    "sync_exhibit_to_chroma": ("core.sync", "sync_exhibit_to_chroma"),
    "sync_pending_reindex": ("core.sync", "sync_pending_reindex"),
}


def __getattr__(name: str) -> Any:  # noqa: D401
    """Lazily resolve heavy ML symbols only when actually accessed."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'core' has no attribute {name!r}")
    module_name, attr_name = target
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
