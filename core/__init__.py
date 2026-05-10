"""ML core of the Smart Gallery Guide service."""

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
