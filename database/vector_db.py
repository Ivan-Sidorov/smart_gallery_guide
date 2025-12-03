import json
import logging
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from config.config import (
    CHROMA_COLLECTION_EXHIBITS,
    CHROMA_COLLECTION_FAQ,
    CHROMA_PERSIST_DIR,
)
from database.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Vector DB for exhibits and FAQ."""

    def __init__(self, persist_directory: Optional[str] = None):
        persist_dir = persist_directory or CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=str(persist_dir))

        self.exhibits_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_EXHIBITS,
            metadata={"description": "Exhibits with image embeddings"},
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 400,
                    "max_neighbors": 32,
                    "ef_search": 200,
                }
            },
        )

        self.faq_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_FAQ,
            metadata={"description": "FAQ items with question embeddings"},
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 400,
                    "max_neighbors": 32,
                    "ef_search": 200,
                }
            },
        )

    @staticmethod
    def _restore_metadata_types(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore metadata types from strings to original types.

        Args:
            metadata_dict (Dict[str, Any]): Metadata dict from ChromaDB

        Returns:
            Dict[str, Any]: Metadata dict with restored types
        """
        processed_dict = metadata_dict.copy()

        if "interesting_facts" in processed_dict:
            interesting_facts_str = processed_dict["interesting_facts"]
            if isinstance(interesting_facts_str, str) and interesting_facts_str:
                processed_dict["interesting_facts"] = [
                    fact.strip() for fact in interesting_facts_str.split(",")
                ]
            else:
                processed_dict["interesting_facts"] = []

        if "additional_info" in processed_dict:
            additional_info_str = processed_dict["additional_info"]
            if isinstance(additional_info_str, str) and additional_info_str:
                try:
                    processed_dict["additional_info"] = json.loads(additional_info_str)
                except (json.JSONDecodeError, TypeError):
                    processed_dict["additional_info"] = {}
            else:
                processed_dict["additional_info"] = {}

        return processed_dict

    @staticmethod
    def _normalize_embedding(embedding: List[float]) -> List[float]:
        """
        Normalize embedding vector to unit length.

        Args:
            embedding (List[float]): Embedding vector

        Returns:
            List[float]: Normalized embedding vector
        """
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        else:
            logger.warning("Zero norm embedding detected!")
        return arr.tolist()

    def add_exhibit(
        self, exhibit_id: str, image_embedding: List[float], metadata: ExhibitMetadata
    ) -> bool:
        """
        Add an exhibit to the DB.

        Args:
            exhibit_id (str): Unique exhibit ID
            image_embedding (List[float]): Image embedding
            metadata (ExhibitMetadata): Exhibit metadata

        Returns:
            bool: True if successful
        """
        metadata_dict = metadata.model_dump()

        # Convert types to strings for ChromaDB compatibility
        processed_metadata = {}
        for key, value in metadata_dict.items():
            if isinstance(value, list):
                processed_metadata[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                processed_metadata[key] = None
            else:
                processed_metadata[key] = value

        normalized_embedding = self._normalize_embedding(image_embedding)

        self.exhibits_collection.add(
            embeddings=[normalized_embedding],
            ids=[exhibit_id],
            metadatas=[processed_metadata],
        )
        return True

    def search_exhibit(
        self, image_embedding: List[float], limit: int = 5, score_threshold: float = 0.7
    ) -> List[ExhibitSearchResult]:
        """
        Search for similar exhibits.

        Args:
            image_embedding (List[float]): Image embedding
            limit (int): Maximum number of results
            score_threshold (float): Minimum similarity score

        Returns:
            List[ExhibitSearchResult]: List of search results
        """
        normalized_embedding = self._normalize_embedding(image_embedding)

        results = self.exhibits_collection.query(
            query_embeddings=[normalized_embedding], n_results=limit
        )

        search_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, exhibit_id in enumerate(results["ids"][0]):
                metadata_dict = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1.0 - distance

                if similarity >= score_threshold:
                    try:
                        processed_dict = self._restore_metadata_types(metadata_dict)
                        metadata = ExhibitMetadata(**processed_dict)
                        search_results.append(
                            ExhibitSearchResult(
                                exhibit_id=exhibit_id,
                                title=metadata.title,
                                similarity_score=similarity,
                                metadata=metadata,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error restoring metadata for {exhibit_id}: {e}",
                            exc_info=True,
                        )
                        continue

        logger.info(f"Found {len(search_results)} exhibits")
        return search_results

    def add_faq(
        self,
        question: str,
        question_embedding: List[float],
        answer: str,
        exhibit_id: str,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a FAQ item to the database.

        Args:
            question (str): FAQ question
            question_embedding (List[float]): Question embedding vector
            answer (str): FAQ answer
            exhibit_id (str): Associated exhibit ID
            category (Optional[str]): Optional category
            metadata (Optional[Dict[str, Any]]): Optional additional metadata

        Returns:
            bool: True if successful
        """
        faq_id = f"{exhibit_id}_{abs(hash(question))}"

        payload = {
            "question": question,
            "answer": answer,
            "exhibit_id": exhibit_id,
        }

        if category:
            payload["category"] = category
        if metadata:
            payload["metadata"] = metadata

        normalized_embedding = self._normalize_embedding(question_embedding)

        self.faq_collection.add(
            embeddings=[normalized_embedding], ids=[faq_id], metadatas=[payload]
        )
        return True

    def search_faq(
        self,
        question_embedding: List[float],
        exhibit_id: str,
        limit: int = 3,
        score_threshold: float = 0.6,
    ) -> List[FAQSearchResult]:
        """
        Search for similar FAQ items.

        Args:
            question_embedding (List[float]): Question embedding vector
            exhibit_id (str): Exhibit id
            limit (int): Maximum number of results
            score_threshold (float): Minimum similarity score

        Returns:
            List[FAQSearchResult]: List of FAQ search results
        """

        normalized_embedding = self._normalize_embedding(question_embedding)

        results = self.faq_collection.query(
            query_embeddings=[normalized_embedding],
            n_results=limit,
            where={"exhibit_id": exhibit_id},
        )

        search_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, faq_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1.0 - distance

                if similarity >= score_threshold:
                    search_results.append(
                        FAQSearchResult(
                            question=metadata.get("question", ""),
                            answer=metadata.get("answer", ""),
                            exhibit_id=metadata.get("exhibit_id", ""),
                            similarity_score=similarity,
                        )
                    )

        return search_results

    def get_exhibit_metadata(self, exhibit_id: str) -> Optional[ExhibitMetadata]:
        """
        Get metadata for a specific exhibit.

        Args:
            exhibit_id (str): Exhibit ID

        Returns:
            Optional[ExhibitMetadata]: Exhibit metadata or None if not found
        """
        results = self.exhibits_collection.get(ids=[exhibit_id])

        if results["ids"] and len(results["ids"]) > 0:
            try:
                metadata_dict = results["metadatas"][0]
                processed_dict = self._restore_metadata_types(metadata_dict)
                return ExhibitMetadata(**processed_dict)
            except Exception:
                return None

        return None

    def delete_exhibit(self, exhibit_id: str) -> bool:
        """
        Delete an exhibit from the database.

        Args:
            exhibit_id (str): Exhibit ID to delete

        Returns:
            bool: True if successful
        """
        try:
            self.exhibits_collection.delete(ids=[exhibit_id])
            return True
        except Exception:
            return False
