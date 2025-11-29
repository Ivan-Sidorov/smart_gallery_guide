from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config.config import CHROMA_COLLECTION_EXHIBITS, CHROMA_COLLECTION_FAQ, CHROMA_PERSIST_DIR
from database.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult


class VectorDatabase:
    """Vector DB for exhibits and FAQ."""

    def __init__(self, persist_directory: Optional[str] = None):
        persist_dir = persist_directory or CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(chroma_db_impl="duckdb+parquet", anonymized_telemetry=False),
        )

        self.exhibits_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_EXHIBITS,
            metadata={"description": "Exhibits with image embeddings"},
        )

        self.faq_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_FAQ,
            metadata={"description": "FAQ items with question embeddings"},
        )

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

        self.exhibits_collection.add(
            embeddings=[image_embedding], ids=[exhibit_id], metadatas=[metadata_dict]
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
        results = self.exhibits_collection.query(
            query_embeddings=[image_embedding], n_results=limit
        )

        search_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, exhibit_id in enumerate(results["ids"][0]):
                metadata_dict = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1 - distance

                if similarity >= score_threshold:
                    try:
                        metadata = ExhibitMetadata(**metadata_dict)
                        search_results.append(
                            ExhibitSearchResult(
                                exhibit_id=exhibit_id,
                                title=metadata.title,
                                similarity_score=similarity,
                                metadata=metadata,
                            )
                        )
                    except Exception:
                        continue

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

        self.faq_collection.add(embeddings=[question_embedding], ids=[faq_id], metadatas=[payload])
        return True

    def search_faq(
        self,
        question_embedding: List[float],
        exhibit_id: Optional[str] = None,
        limit: int = 3,
        score_threshold: float = 0.6,
    ) -> List[FAQSearchResult]:
        """
        Search for similar FAQ items.

        Args:
            question_embedding (List[float]): Question embedding vector
            exhibit_id (Optional[str]): Optional filter by exhibit ID
            limit (int): Maximum number of results
            score_threshold (float): Minimum similarity score

        Returns:
            List[FAQSearchResult]: List of FAQ search results
        """
        where = None
        if exhibit_id:
            where = {"exhibit_id": exhibit_id}

        results = self.faq_collection.query(
            query_embeddings=[question_embedding], n_results=limit, where=where
        )

        search_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, faq_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1 - distance

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
                return ExhibitMetadata(**metadata_dict)
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
