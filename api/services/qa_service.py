"""Q&A service: FAQ with VLM fallback."""

import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.qa import QAResponse
from api.services.faq_service import FAQService
from api.services.task_service import TaskService
from core.settings import Settings
from db.models import TaskType

logger = logging.getLogger(__name__)


class QAService:
    """Decides between FAQ hit and VLM enqueue."""

    def __init__(
        self,
        session: AsyncSession,
        faq_service: FAQService,
        task_service: TaskService,
        settings: Settings,
    ) -> None:
        self._session = session
        self._faq = faq_service
        self._tasks = task_service
        self._settings = settings

    async def answer_about_exhibit(
        self,
        *,
        exhibit_id: str,
        question: str,
        user_id: int | None = None,
        session_id: uuid.UUID | None = None,
    ) -> QAResponse:
        """FAQ lookup with VLM fallback."""
        hits = await self._faq.search(
            exhibit_id=exhibit_id,
            question=question,
            top_k=1,
            score_threshold=self._settings.faq_relevance_threshold,
        )
        if hits:
            best = hits[0]
            logger.info(
                "[QAService] FAQ hit (exhibit_id=%s, score=%.3f)",
                exhibit_id,
                best.similarity_score,
            )
            return QAResponse(mode="faq", answer=best.answer)

        request: dict[str, Any] = {
            "exhibit_id": exhibit_id,
            "question": question,
        }
        task = await self._tasks.enqueue(
            type=TaskType.VLM_QA,
            request=request,
            user_id=user_id,
            session_id=session_id,
            model=self._settings.vllm_vlm_model,
        )
        return QAResponse(mode="task", task_id=task.id)

    async def answer_about_image(
        self,
        *,
        question: str,
        image_bytes: bytes,
        exhibit_id: str | None = None,
        user_id: int | None = None,
        session_id: uuid.UUID | None = None,
    ) -> QAResponse:
        """VLM Q&A over a user image."""
        import base64

        request: dict[str, Any] = {
            "question": question,
            "exhibit_id": exhibit_id,
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "image_size_bytes": len(image_bytes),
        }
        task = await self._tasks.enqueue(
            type=TaskType.VLM_QA,
            request=request,
            user_id=user_id,
            session_id=session_id,
            model=self._settings.vllm_vlm_model,
        )
        return QAResponse(mode="task", task_id=task.id)
