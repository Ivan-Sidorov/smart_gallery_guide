from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ExhibitMetadata(BaseModel):
    """Schema for exhibit metadata."""

    exhibit_id: str
    title: str
    artist: Optional[str] = None
    year: Optional[str] = None
    description: str
    interesting_facts: List[str] = []
    image_path: str
    additional_info: Dict[str, Any] = {}


class FAQItem(BaseModel):
    """Schema for FAQ item."""

    question: str
    answer: str
    exhibit_id: str
    category: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ExhibitSearchResult(BaseModel):
    """Schema for exhibit search result."""

    exhibit_id: str
    title: str
    similarity_score: float
    metadata: ExhibitMetadata


class FAQSearchResult(BaseModel):
    """Schema for FAQ search result."""

    question: str
    answer: str
    exhibit_id: str
    similarity_score: float
