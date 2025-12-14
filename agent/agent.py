import logging
from typing import List, Optional

from PIL import Image

from config.config import EXHIBIT_MATCH_THRESHOLD, FAQ_RELEVANCE_THRESHOLD
from database.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult
from database.vector_db import VectorDatabase
from models.text_encoder import TextEncoder
from models.vision_encoder import VisionEncoder
from models.vlm import VLM

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

        logger.info("GuideAgent initialized")

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

        Requires exhibit_id to be provided (exhibit must be recognized first).

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

            async with VLM() as vlm:
                answer = await vlm.answer_question(
                    image=image,
                    question=question,
                    context=context,
                )

            logger.info(f"Answered question about image (exhibit_id: {exhibit_id})")
            return answer

        except Exception as e:
            logger.error(f"Error answering question about image: {e}", exc_info=True)
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
            query_embedding = self.vision_encoder.encode_text(query)
            query_embedding_list = query_embedding.tolist()

            threshold = (
                score_threshold
                if score_threshold is not None
                else EXHIBIT_MATCH_THRESHOLD
            )
            results = self.vector_db.search_exhibit(
                image_embedding=query_embedding_list,
                limit=limit,
                score_threshold=threshold,
            )

            logger.info(f"Found {len(results)} exhibits for text query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error searching exhibits by text: {e}", exc_info=True)
            return []

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
            context_parts.append(f"Художник: {metadata.artist}")

        if metadata.year:
            context_parts.append(f"Год: {metadata.year}")

        context_parts.append(f"Описание: {metadata.description}")

        if metadata.interesting_facts:
            facts = ", ".join(metadata.interesting_facts)
            context_parts.append(f"Интересные факты: {facts}")

        return "\n".join(context_parts)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
