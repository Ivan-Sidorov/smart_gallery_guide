"""FAQ search service."""

import asyncio
import logging

from api.schemas.faq import FAQSearchResultDTO
from core.encoders.text import TextEncoder
from core.settings import Settings
from core.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class FAQService:
    """Searches FAQ entries for a single exhibit."""

    def __init__(
        self,
        vector_db: VectorDatabase | None,
        text_encoder: TextEncoder | None,
        settings: Settings,
    ) -> None:
        self._vector_db = vector_db
        self._text_encoder = text_encoder
        self._settings = settings

    async def search(
        self,
        exhibit_id: str,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[FAQSearchResultDTO]:
        """Return FAQ hits ordered by score."""
        if self._vector_db is None or self._text_encoder is None:
            logger.warning(
                "[FAQService] search called but ML components are not loaded"
            )
            return []

        settings = self._settings
        threshold = (
            score_threshold
            if score_threshold is not None
            else settings.faq_relevance_threshold
        )
        limit = top_k or settings.api_default_top_k
        limit = min(limit, settings.api_max_top_k)

        emb = await asyncio.to_thread(self._text_encoder.encode_text, question)
        results = await asyncio.to_thread(
            self._vector_db.search_faq,
            question_embedding=emb.tolist(),
            exhibit_id=exhibit_id,
            limit=limit,
            score_threshold=threshold,
            display_threshold=settings.display_score_threshold,
            query_text=question,
        )
        return [
            FAQSearchResultDTO(
                exhibit_id=r.exhibit_id,
                question=r.question,
                answer=r.answer,
                similarity_score=float(r.similarity_score),
            )
            for r in results
        ]
