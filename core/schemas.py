"""Domain schemas for the Smart Gallery Guide service."""

from typing import Any

from pydantic import BaseModel


class ExhibitMetadata(BaseModel):
    """Metadata for a single museum exhibit (painting, sculpture, etc.)."""

    exhibit_id: str
    title: str
    artist: str | None = None
    year: str | None = None
    material: str | None = None
    dimensions: str | None = None
    school: str | None = None
    department: str | None = None
    inventory_number: str | None = None
    image_path: str
    description: str | None = None
    description_perplexity: str | None = None
    antic_art_description: str | None = None
    additional_info: dict[str, Any] = {}

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
        parts: list[str] = []

        intro = f"Произведение «{self.title}»"
        details: list[str] = []
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
    """A single FAQ question-answer pair."""

    question: str
    answer: str


class FAQDocument(BaseModel):
    """A FAQ file (one per exhibit)."""

    exhibit_id: str
    title: str
    artist: str = ""
    questions: list[FAQItem]
    question_count: int = 0
    source_model: str = ""
    source_file: str = ""
    generated_at_unix: int = 0


class ExhibitSearchResult(BaseModel):
    """Result of an exhibit search (vector / hybrid)."""

    exhibit_id: str
    title: str
    similarity_score: float
    metadata: ExhibitMetadata


class FAQSearchResult(BaseModel):
    """Result of an FAQ search bound to a specific exhibit."""

    question: str
    answer: str
    exhibit_id: str
    similarity_score: float
