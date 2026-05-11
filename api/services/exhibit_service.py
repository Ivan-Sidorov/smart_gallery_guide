"""Exhibit retrieval and metadata service."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.exhibits import ExhibitDTO, ExhibitSearchResultDTO
from core.encoders.text import TextEncoder
from core.encoders.vision import VisionEncoder
from core.settings import Settings
from core.vector_db import VectorDatabase
from db.models import Exhibit
from db.repositories import ExhibitRepository

logger = logging.getLogger(__name__)


def _exhibit_to_dto(exhibit: Exhibit) -> ExhibitDTO:
    return ExhibitDTO(
        exhibit_id=exhibit.id,
        title=exhibit.title,
        author=exhibit.author,
        year=exhibit.year,
        description=exhibit.description,
        image_path=exhibit.image_path,
        extra=dict(exhibit.extra or {}),
    )


class ExhibitService:
    """Read-only operations on exhibits + hybrid text search via Chroma."""

    def __init__(
        self,
        session: AsyncSession,
        vector_db: VectorDatabase | None,
        text_encoder: TextEncoder | None,
        vision_encoder: VisionEncoder | None,
        settings: Settings,
    ) -> None:
        self._session = session
        self._vector_db = vector_db
        self._text_encoder = text_encoder
        self._vision_encoder = vision_encoder
        self._settings = settings

    async def get(self, exhibit_id: str) -> ExhibitDTO | None:
        """Fetch a single exhibit row from Postgres."""
        repo = ExhibitRepository(self._session)
        row = await repo.get(exhibit_id)
        return _exhibit_to_dto(row) if row is not None else None

    async def list(self, limit: int = 100, offset: int = 0) -> list[ExhibitDTO]:
        """List exhibits ordered by ``created_at`` desc."""
        repo = ExhibitRepository(self._session)
        rows = await repo.list_all(limit=limit, offset=offset)
        return [_exhibit_to_dto(row) for row in rows]

    async def search_by_text(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[ExhibitSearchResultDTO]:
        """Cascading text search: title –> description –> image-text.

        Returns an empty list when the API was started without ML components.
        """
        if self._vector_db is None or self._text_encoder is None:
            logger.warning(
                "[exhibit_service] search_by_text called but ML components are not loaded"
            )
            return []

        settings = self._settings
        threshold = (
            score_threshold
            if score_threshold is not None
            else settings.exhibit_match_threshold
        )
        display_threshold = settings.display_score_threshold
        limit = top_k or settings.api_default_top_k
        limit = min(limit, settings.api_max_top_k)

        text_emb = await asyncio.to_thread(self._text_encoder.encode_text, query)
        text_emb_list: list[float] = text_emb.tolist()

        title_results = await asyncio.to_thread(
            self._vector_db.search_text,
            query_embedding=text_emb_list,
            variant="title",
            limit=limit,
            score_threshold=display_threshold,
            display_threshold=display_threshold,
            query_text=query,
        )
        if title_results and title_results[0].similarity_score >= threshold:
            return [_to_dto(r) for r in title_results]

        desc_results = await asyncio.to_thread(
            self._vector_db.search_text,
            query_embedding=text_emb_list,
            variant="desc",
            limit=limit,
            score_threshold=display_threshold,
            display_threshold=display_threshold,
            query_text=query,
        )
        return [_to_dto(r) for r in desc_results]

    async def recognize_by_image(
        self,
        image_bytes: bytes,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[ExhibitSearchResultDTO]:
        """Recognise an exhibit from a user photo."""
        if self._vector_db is None or self._vision_encoder is None:
            logger.warning(
                "[exhibit_service] recognize_by_image called but ML components are not loaded"
            )
            return []

        settings = self._settings
        threshold = (
            score_threshold
            if score_threshold is not None
            else settings.exhibit_match_threshold
        )
        display_threshold = settings.display_score_threshold
        limit = top_k or settings.api_default_top_k
        limit = min(limit, settings.api_max_top_k)

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            logger.warning("[exhibit_service] can't decode image: %s", exc)
            return []

        embedding = await asyncio.to_thread(self._vision_encoder.encode_image, image)
        results = await asyncio.to_thread(
            self._vector_db.search_exhibit,
            image_embedding=embedding.tolist(),
            limit=limit,
            score_threshold=threshold,
            display_threshold=display_threshold,
        )
        return [_to_dto(r) for r in results]


def _to_dto(result: Any) -> ExhibitSearchResultDTO:
    return ExhibitSearchResultDTO(
        exhibit_id=result.exhibit_id,
        title=result.title,
        similarity_score=float(result.similarity_score),
        metadata=result.metadata.model_dump(),
    )
