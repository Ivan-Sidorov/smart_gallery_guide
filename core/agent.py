"""GuideAgent: orchestrates retrieval + VLM + web search."""

import logging
from pathlib import Path

from PIL import Image

from core.encoders.text import TextEncoder
from core.encoders.vision import VisionEncoder
from core.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult
from core.search.web import WebSearchService
from core.settings import get_settings
from core.vector_db import VectorDatabase
from core.vlm.client import VLM
from core.vlm.prompts import enriched_system_prompt

logger = logging.getLogger(__name__)


class GuideAgent:
    """Orchestrates ML components to recognise exhibits and answer questions."""

    def __init__(
        self,
        vector_db: VectorDatabase | None = None,
        vision_encoder: VisionEncoder | None = None,
        text_encoder: TextEncoder | None = None,
        web_search: WebSearchService | None = None,
    ):
        settings = get_settings()
        self.vector_db = vector_db or VectorDatabase()
        self.vision_encoder = vision_encoder or VisionEncoder()
        self.text_encoder = text_encoder or TextEncoder()

        if web_search is not None:
            self.web_search: WebSearchService | None = web_search
        elif settings.web_search_enabled:
            self.web_search = WebSearchService()
        else:
            self.web_search = None

        logger.info(
            "GuideAgent initialized (web_search=%s)", self.web_search is not None
        )

    async def recognize_exhibit(
        self,
        image: Image.Image,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[ExhibitSearchResult]:
        """Recognize exhibit from image using vector search.

        Args:
            image: Image to recognize the exhibit from.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score to return.

        Returns:
            List of exhibit search results.
        """
        try:
            settings = get_settings()
            image_embedding = self.vision_encoder.encode_image(image)
            image_embedding_list = image_embedding.tolist()

            threshold = (
                score_threshold
                if score_threshold is not None
                else settings.exhibit_match_threshold
            )
            results = self.vector_db.search_exhibit(
                image_embedding=image_embedding_list,
                limit=limit,
                score_threshold=threshold,
                display_threshold=settings.display_score_threshold,
            )

            logger.info("Recognized %d exhibits from image", len(results))
            return results

        except Exception as e:
            logger.error(f"Error recognizing exhibit: {e}", exc_info=True)
            return []

    async def answer_question_about_image(
        self,
        image: Image.Image,
        question: str,
        exhibit_id: str,
    ) -> str:
        """Answer a question about an image using VLM.

        Uses a two-step process when web search is enabled:
        1. VLM evaluates whether it can answer from local context alone.
        2. If not, VLM formulates a search query, fetches results, and
           answers again with enriched context.

        Args:
            image: Image to answer the question about.
            question: Question to answer.
            exhibit_id: ID of the exhibit to answer the question about.

        Returns:
            Answer to the question.
        """
        try:
            metadata = self.vector_db.get_exhibit_metadata(exhibit_id)
            if not metadata:
                logger.warning(f"Exhibit {exhibit_id} not found for question")
                return "Экспонат не найден. Пожалуйста, сначала распознайте экспонат."

            context = self._build_exhibit_context(metadata)
            answer = await self._answer_with_optional_search(
                image=image, question=question, context=context, exhibit_id=exhibit_id
            )

            logger.info(f"Answered question about image (exhibit_id: {exhibit_id})")
            return answer

        except Exception as e:
            logger.error(f"Error answering question about image: {e}", exc_info=True)
            return f"Произошла ошибка при обработке вопроса: {str(e)}"

    async def answer_question_about_exhibit(
        self, question: str, exhibit_id: str
    ) -> str:
        """Answer a text question about a specific exhibit.

        First tries to find an answer in FAQ. If nothing relevant is
        found, falls back to VLM using the exhibit image and metadata.

        Args:
            question: Question to answer.
            exhibit_id: ID of the exhibit to answer the question about.

        Returns:
            Answer to the question.
        """
        try:
            settings = get_settings()
            metadata = self.vector_db.get_exhibit_metadata(exhibit_id)
            if not metadata:
                logger.warning(f"Exhibit {exhibit_id} not found for text question")
                return "Экспонат не найден. Пожалуйста, выберите экспонат сначала."

            faq_results = await self.search_faq(
                question=question, exhibit_id=exhibit_id, limit=3
            )
            if faq_results:
                best_match = faq_results[0]
                if best_match.similarity_score >= settings.faq_relevance_threshold:
                    logger.info(
                        "Answered question from FAQ (exhibit_id: %s, score: %.3f)",
                        exhibit_id,
                        best_match.similarity_score,
                    )
                    return best_match.answer

            image_path = Path(metadata.image_path)
            if not image_path.is_absolute():
                image_path = settings.project_root / image_path

            if not image_path.exists():
                logger.warning(
                    "Image for exhibit %s not found at %s", exhibit_id, image_path
                )
                return (
                    "Не удалось найти изображение экспоната. "
                    "Попробуйте отправить фото экспоната с вопросом."
                )

            with Image.open(image_path) as img:
                exhibit_image = img.convert("RGB")

            context = self._build_exhibit_context(metadata)
            answer = await self._answer_with_optional_search(
                image=exhibit_image,
                question=question,
                context=context,
                exhibit_id=exhibit_id,
            )

            logger.info(
                "Answered question with VLM (exhibit_id: %s) after FAQ fallback",
                exhibit_id,
            )
            return answer

        except Exception as e:
            logger.error(
                "Error answering text question about exhibit %s: %s",
                exhibit_id,
                e,
                exc_info=True,
            )
            return f"Произошла ошибка при обработке вопроса: {str(e)}"

    async def search_faq(
        self,
        question: str,
        exhibit_id: str,
        limit: int = 3,
        score_threshold: float | None = None,
    ) -> list[FAQSearchResult]:
        """Search the FAQ collection for a question within a specific exhibit.

        Args:
            question: Question to search for.
            exhibit_id: ID of the exhibit to search the FAQ for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score to return.

        Returns:
            List of FAQ search results.
        """
        try:
            settings = get_settings()
            question_embedding = self.text_encoder.encode_text(question)
            question_embedding_list = question_embedding.tolist()

            threshold = (
                score_threshold
                if score_threshold is not None
                else settings.faq_relevance_threshold
            )
            results = self.vector_db.search_faq(
                question_embedding=question_embedding_list,
                exhibit_id=exhibit_id,
                limit=limit,
                score_threshold=threshold,
                display_threshold=settings.display_score_threshold,
                query_text=question,
            )

            logger.info(
                f"Found {len(results)} FAQ items for question (exhibit_id: {exhibit_id})"
            )
            return results

        except Exception as e:
            logger.error(f"Error searching FAQ: {e}", exc_info=True)
            return []

    def get_exhibit_info(self, exhibit_id: str) -> ExhibitMetadata | None:
        """Get information about a specific exhibit.

        Args:
            exhibit_id: ID of the exhibit to get information about.

        Returns:
            Metadata of the exhibit.
        """
        try:
            metadata = self.vector_db.get_exhibit_metadata(exhibit_id)
            if metadata:
                logger.info(f"Retrieved exhibit info for {exhibit_id}")
            else:
                logger.warning(f"Exhibit {exhibit_id} not found")
            return metadata

        except Exception as e:
            logger.error(f"Error getting exhibit info: {e}", exc_info=True)
            return None

    async def search_exhibits_by_text(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[ExhibitSearchResult]:
        """Search for exhibits by text query.

        Uses a cascading approach:
        1. Searches the title/artist index.
        2. If no results, searches the description index.
        3. If no results, searches the image index.

        Args:
            query: Text to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score to return.

        Returns:
            List of exhibit search results.
        """
        try:
            settings = get_settings()
            threshold = (
                score_threshold
                if score_threshold is not None
                else settings.exhibit_match_threshold
            )
            display_threshold = settings.display_score_threshold

            text_emb = self.text_encoder.encode_text(query).tolist()

            title_results = self.vector_db.search_text(
                query_embedding=text_emb,
                variant="title",
                limit=limit,
                score_threshold=display_threshold,
                display_threshold=display_threshold,
                query_text=query,
            )
            if title_results and title_results[0].similarity_score >= threshold:
                logger.info(
                    f"Found {len(title_results)} exhibits by title/artist for query: {query}"
                )
                return title_results

            desc_results = self.vector_db.search_text(
                query_embedding=text_emb,
                variant="desc",
                limit=limit,
                score_threshold=display_threshold,
                display_threshold=display_threshold,
                query_text=query,
            )
            if desc_results and desc_results[0].similarity_score >= threshold:
                logger.info(
                    f"Found {len(desc_results)} exhibits by description/facts for query: {query}"
                )
                return desc_results

            query_embedding = self.vision_encoder.encode_text(query)
            query_embedding_list = query_embedding.tolist()

            image_results = self.vector_db.search_exhibit(
                image_embedding=query_embedding_list,
                limit=limit,
                score_threshold=threshold,
                display_threshold=display_threshold,
            )

            logger.info(
                f"Found {len(image_results)} exhibits by image for query: {query}"
            )
            return image_results

        except Exception as e:
            logger.error(f"Error searching exhibits by text: {e}", exc_info=True)
            return []

    async def _answer_with_optional_search(
        self,
        image: Image.Image,
        question: str,
        context: str,
        exhibit_id: str,
    ) -> str:
        """Two-step VLM flow with optional web search.

        1. VLM decides whether it can answer from local context alone.
        2. If not, VLM formulates a search query, results are fetched, and
           VLM is asked again with enriched context.

        Args:
            image: Image to answer the question about.
            question: Question to answer.
            context: Context to use for the question.
            exhibit_id: ID of the exhibit to answer the question about.

        Returns:
            Answer to the question.
        """
        async with VLM() as vlm:
            if not self.web_search:
                return await vlm.answer_question(
                    image=image, question=question, context=context
                )

            evaluation = await vlm.evaluate_search_need(
                image=image, question=question, context=context
            )

            if not evaluation.needs_search:
                logger.info(
                    "VLM answered directly without web search (exhibit_id: %s)",
                    exhibit_id,
                )
                return evaluation.answer

            logger.info(
                "VLM requested web search (exhibit_id: %s, query: %r)",
                exhibit_id,
                evaluation.search_query,
            )
            search_results = await self.web_search.search(evaluation.search_query)
            web_context = WebSearchService.format_results(search_results)

            if web_context:
                enriched_context = f"{context}\n\nДополнительная информация из интернета:\n{web_context}"
            else:
                enriched_context = context

            return await vlm.answer_question(
                image=image,
                question=question,
                context=enriched_context,
                system_prompt=enriched_system_prompt(),
            )

    @staticmethod
    def _build_exhibit_context(metadata: ExhibitMetadata) -> str:
        """Build a context string for a specific exhibit.

        Args:
            metadata: Metadata of the exhibit.

        Returns:
            Context string for the exhibit.
        """
        context_parts = [f"Название: {metadata.title}"]

        if metadata.artist:
            context_parts.append(f"Автор: {metadata.artist}")
        if metadata.year:
            context_parts.append(f"Год: {metadata.year}")
        if metadata.material:
            context_parts.append(f"Материал: {metadata.material}")
        if metadata.school:
            context_parts.append(f"Школа/страна: {metadata.school}")

        ai = metadata.additional_info
        if techniq := ai.get("techniq"):
            context_parts.append(f"Техника: {techniq}")
        if place := ai.get("place"):
            context_parts.append(f"Место: {place}")
        if epoque := ai.get("epoque"):
            context_parts.append(f"Эпоха: {epoque}")

        if description := metadata.display_description:
            context_parts.append(f"Описание: {description}")
        if anotation := ai.get("anotation"):
            context_parts.append(f"Экспертный комментарий: {anotation}")

        return "\n".join(context_parts)

    async def close(self) -> None:
        """Release resources held by the agent and its components."""
        for attr in ("web_search", "vector_db", "vision_encoder", "text_encoder"):
            component = getattr(self, attr, None)
            close_fn = (
                getattr(component, "close", None) if component is not None else None
            )
            if close_fn is None:
                continue
            try:
                result = close_fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.warning("Error while closing %s: %s", attr, exc)

        logger.info("GuideAgent closed")

    async def __aenter__(self) -> "GuideAgent":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
