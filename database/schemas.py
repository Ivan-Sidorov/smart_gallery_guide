from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ExhibitMetadata(BaseModel):
    """Schema for exhibit metadata."""

    exhibit_id: str
    title: str
    artist: Optional[str] = None
    year: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[str] = None
    school: Optional[str] = None
    department: Optional[str] = None
    inventory_number: Optional[str] = None
    image_path: str
    description: Optional[str] = None
    description_perplexity: Optional[str] = None
    antic_art_description: Optional[str] = None
    additional_info: Dict[str, Any] = {}

    @property
    def display_description(self) -> str:
        """Best available description text with fallback chain."""
        if self.description:
            return self.description
        if self.description_perplexity:
            return self.description_perplexity
        if self.antic_art_description:
            return self.antic_art_description
        return self._build_fallback_description()

    def _build_fallback_description(self) -> str:
        parts: List[str] = []

        intro = f"Произведение «{self.title}»"
        details: List[str] = []
        if self.artist:
            details.append(f"автор — {self.artist}")
        if self.year:
            details.append(f"создано {self.year}")
        if details:
            intro += ", " + ", ".join(details)
        intro += "."
        parts.append(intro)

        if self.material:
            parts.append(f"Материал: {self.material}.")

        techniq = self.additional_info.get("techniq")
        if techniq:
            parts.append(f"Техника: {techniq}.")

        place = self.additional_info.get("place")
        if place:
            parts.append(f"Место создания: {place}.")

        epoque = self.additional_info.get("epoque")
        if epoque:
            parts.append(f"Эпоха: {epoque}.")

        return " ".join(parts)


class FAQItem(BaseModel):
    """Schema for a single FAQ question-answer pair."""

    question: str
    answer: str


class FAQDocument(BaseModel):
    """Schema for a FAQ file (one per exhibit)."""

    exhibit_id: str
    title: str
    artist: str = ""
    questions: List[FAQItem]
    question_count: int = 0
    source_model: str = ""
    source_file: str = ""
    generated_at_unix: int = 0


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
