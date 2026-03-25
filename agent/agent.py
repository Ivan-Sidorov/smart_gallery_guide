import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

from config.config import (
    DISPLAY_SCORE_THRESHOLD,
    EXHIBIT_MATCH_THRESHOLD,
    FAQ_RELEVANCE_THRESHOLD,
    PROJECT_ROOT,
    VLLM_ENRICHED_SYSTEM_PROMPT,
    WEB_SEARCH_ENABLED,
)
from database.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult
from database.vector_db import VectorDatabase
from models.text_encoder import TextEncoder
from models.vision_encoder import VisionEncoder
from models.vlm import VLM
from services.web_search import WebSearchService

logger = logging.getLogger(__name__)


class GuideAgent:
    """Agent for processing user requests and orchestrating ML components."""

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        vision_encoder: Optional[VisionEncoder] = None,
        text_encoder: Optional[TextEncoder] = None,
    ):
        """
        Initialize GuideAgent.

        Args:
            vector_db (Optional[VectorDatabase]): VectorDatabase instance. Defaults to None (creates a new one).
            vision_encoder (Optional[VisionEncoder]): VisionEncoder instance. Defaults to None (creates a new one).
            text_encoder (Optional[TextEncoder]): TextEncoder instance. Defaults to None (creates a new one).
        """
        self.vector_db = vector_db or VectorDatabase()
        self.vision_encoder = vision_encoder or VisionEncoder()
        self.text_encoder = text_encoder or TextEncoder()
        self.web_search = WebSearchService() if WEB_SEARCH_ENABLED else None

        logger.info("GuideAgent initialized (web_search=%s)", WEB_SEARCH_ENABLED)

    async def recognize_exhibit(
        self,
        image: Image.Image,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[ExhibitSearchResult]:
        """
        Recognize exhibit from image using vector search.

        Args:
            image (Image.Image): PIL Image to recognize
            limit (int): Maximum number of results. Defaults to 5.
            score_threshold (Optional[float]): Minimum similarity score. Defaults to None (uses config default).

        Returns:
            List[ExhibitSearchResult]: List of ExhibitSearchResult sorted by similarity score
        """
        try:
            image_embedding = self.vision_encoder.encode_image(image)
            image_embedding_list = image_embedding.tolist()

            threshold = (
                score_threshold
                if score_threshold is not None
                else EXHIBIT_MATCH_THRESHOLD
            )
            results = self.vector_db.search_exhibit(
                image_embedding=image_embedding_list,
                limit=limit,
                score_threshold=threshold,
                display_threshold=DISPLAY_SCORE_THRESHOLD,
            )

            logger.info(f"Recognized {len(results)} exhibits from image")
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
        """
        Answer a question about an image using VLM.

        Uses a two-step flow when web search is enabled:
        1. VLM evaluates whether it can answer from local context alone.
        2. If not – VLM formulates a search query, results are fetched, and
           VLM answers again with enriched context.

        Args:
            image (Image.Image): PIL Image to analyze
            question (str): Question about the image
            exhibit_id (str): Exhibit ID (must be recognized first via recognize_exhibit)

        Returns:
            str: Answer string from VLM, or error message if exhibit not found
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
        """
        Answer a text question about a specific exhibit.

        The method first tries to find an answer in FAQ. If nothing relevant is
        found, it falls back to VLM using the exhibit image and metadata.

        Args:
            question (str): User question
            exhibit_id (str): Exhibit ID

        Returns:
            str: Answer string
        """
        try:
            metadata = self.vector_db.get_exhibit_metadata(exhibit_id)
            if not metadata:
                logger.warning(f"Exhibit {exhibit_id} not found for text question")
                return "Экспонат не найден. Пожалуйста, выберите экспонат сначала."

            faq_results = await self.search_faq(
                question=question, exhibit_id=exhibit_id, limit=3
            )
            if faq_results:
                best_match = faq_results[0]
                if best_match.similarity_score >= FAQ_RELEVANCE_THRESHOLD:
                    logger.info(
                        "Answered question from FAQ (exhibit_id: %s, score: %.3f)",
                        exhibit_id,
                        best_match.similarity_score,
                    )
                    return best_match.answer

            image_path = Path(metadata.image_path)
            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path

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
        score_threshold: Optional[float] = None,
    ) -> List[FAQSearchResult]:
        """
        Search for FAQ items by question for a specific exhibit.

        Args:
            question (str): Question text
            exhibit_id (str): Exhibit ID (required - FAQ search is always tied to an exhibit)
            limit (int): Maximum number of results. Defaults to 3.
            score_threshold (Optional[float]): Minimum similarity score. Defaults to None (uses config default).

        Returns:
            List[FAQSearchResult]: List of FAQSearchResult sorted by similarity score
        """
        try:
            question_embedding = self.text_encoder.encode_text(question)
            question_embedding_list = question_embedding.tolist()

            threshold = (
                score_threshold
                if score_threshold is not None
                else FAQ_RELEVANCE_THRESHOLD
            )
            results = self.vector_db.search_faq(
                question_embedding=question_embedding_list,
                exhibit_id=exhibit_id,
                limit=limit,
                score_threshold=threshold,
                display_threshold=DISPLAY_SCORE_THRESHOLD,
                query_text=question,
            )

            logger.info(
                f"Found {len(results)} FAQ items for question (exhibit_id: {exhibit_id})"
            )
            return results

        except Exception as e:
            logger.error(f"Error searching FAQ: {e}", exc_info=True)
            return []

    def get_exhibit_info(self, exhibit_id: str) -> Optional[ExhibitMetadata]:
        """
        Get information about a specific exhibit.

        Args:
            exhibit_id (str): Exhibit ID

        Returns:
            Optional[ExhibitMetadata]: ExhibitMetadata or None if not found
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
        score_threshold: Optional[float] = None,
    ) -> List[ExhibitSearchResult]:
        """
        Search for exhibits by text query.

        Args:
            query (str): Text query
            limit (int): Maximum number of results
            score_threshold (Optional[float]): Minimum similarity score. Defaults to None (uses config default).

        Returns:
            List[ExhibitSearchResult]: List of ExhibitSearchResult sorted by similarity score
        """
        try:
            threshold = (
                score_threshold
                if score_threshold is not None
                else EXHIBIT_MATCH_THRESHOLD
            )
            display_threshold = DISPLAY_SCORE_THRESHOLD

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
        """
        Two-step VLM flow:
        1. VLM decides if it can answer or needs web search.
        2. If search is needed – fetch snippets, enrich context, answer again.
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
                enriched_context = (
                    f"{context}\n\n"
                    f"Дополнительная информация из интернета:\n{web_context}"
                )
            else:
                enriched_context = context

            return await vlm.answer_question(
                image=image,
                question=question,
                context=enriched_context,
                system_prompt=VLLM_ENRICHED_SYSTEM_PROMPT,
            )

    def _build_exhibit_context(self, metadata: ExhibitMetadata) -> str:
        """
        Build context string from exhibit metadata for VLM.

        Args:
            metadata (ExhibitMetadata): ExhibitMetadata

        Returns:
            str: Context string for VLM
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
        techniq = ai.get("techniq")
        if techniq:
            context_parts.append(f"Техника: {techniq}")

        place = ai.get("place")
        if place:
            context_parts.append(f"Место: {place}")

        epoque = ai.get("epoque")
        if epoque:
            context_parts.append(f"Эпоха: {epoque}")

        description = metadata.display_description
        if description:
            context_parts.append(f"Описание: {description}")

        anotation = ai.get("anotation")
        if anotation:
            context_parts.append(f"Экспертный комментарий: {anotation}")

        return "\n".join(context_parts)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
