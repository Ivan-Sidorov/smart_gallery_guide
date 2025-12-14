import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import faiss
import numpy as np

from config.config import (
    CHROMA_COLLECTION_EXHIBITS,
    CHROMA_COLLECTION_FAQ,
    CHROMA_PERSIST_DIR,
    FAISS_STORAGE_DIR,
    VECTOR_BACKEND,
)
from database.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult

logger = logging.getLogger(__name__)


class FaissLocalIndex:
    """Lightweight FAISS-backed storage with on-disk persistence."""

    def __init__(self, storage_dir: Path, name: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_dir / f"{name}.index"
        self.meta_path = self.storage_dir / f"{name}_meta.json"
        self.embeddings_path = self.storage_dir / f"{name}_embeddings.npy"

        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension: Optional[int] = None
        self.ids: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

        self._load()

    def _load(self) -> None:
        """Load index, embeddings and metadata from disk if present."""
        if (
            self.index_path.exists()
            and self.meta_path.exists()
            and self.embeddings_path.exists()
        ):
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                self.ids = payload.get("ids", [])
                self.metadatas = payload.get("metadatas", [])
                self.dimension = payload.get("dimension")
                self.embeddings = np.load(self.embeddings_path)

                if self.dimension is None and self.embeddings.size > 0:
                    self.dimension = int(self.embeddings.shape[1])
                logger.info(
                    "Loaded FAISS index %s with %d items",
                    self.index_path,
                    len(self.ids),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load FAISS index %s: %s. Starting fresh.",
                    self.index_path,
                    exc,
                )
                self._reset()
        else:
            self._reset()

    def _reset(self) -> None:
        """Reset in-memory state for empty index."""
        self.index = None
        self.dimension = None
        self.ids = []
        self.metadatas = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)

    def _ensure_index(self, dimension: int) -> None:
        """Create an empty IndexFlatIP if it does not exist."""
        if self.index is None:
            self.dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)
            self.embeddings = np.empty((0, dimension), dtype=np.float32)

    def _persist(self) -> None:
        """Persist index, embeddings and metadata to disk."""
        if self.index is None or self.dimension is None:
            return

        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ids": self.ids,
                    "metadatas": self.metadatas,
                    "dimension": self.dimension,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        np.save(self.embeddings_path, self.embeddings)

    def add(
        self, item_id: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> None:
        """Add new item to the index (overwrites existing id)."""
        embedding_arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if embedding_arr.ndim != 2 or embedding_arr.shape[0] != 1:
            raise ValueError("Embedding must be a 1D vector")

        if item_id in self.ids:
            self.delete(item_id)

        self._ensure_index(embedding_arr.shape[1])
        if embedding_arr.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding_arr.shape[1]} "
                f"does not match index dimension {self.dimension}"
            )

        self.ids.append(item_id)
        self.metadatas.append(metadata)
        if self.embeddings.size == 0:
            self.embeddings = embedding_arr
        else:
            self.embeddings = np.vstack([self.embeddings, embedding_arr])

        self.index.add(embedding_arr)
        self._persist()

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for nearest neighbors with optional metadata filtering."""
        if self.index is None or self.index.ntotal == 0 or self.dimension is None:
            return []

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} "
                f"does not match index dimension {self.dimension}"
            )

        k = self.index.ntotal if where else min(limit, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results: List[Tuple[str, Dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            metadata = self.metadatas[idx]
            if where and any(
                metadata.get(key) != value for key, value in where.items()
            ):
                continue

            results.append((self.ids[idx], metadata, float(score)))
            if len(results) >= limit:
                break

        return results

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for item."""
        if item_id in self.ids:
            idx = self.ids.index(item_id)
            return self.metadatas[idx]
        return None

    def delete(self, item_id: str) -> bool:
        """Delete item by id, rebuilding the index."""
        if item_id not in self.ids or self.dimension is None:
            return False

        idx = self.ids.index(item_id)
        self.ids.pop(idx)
        self.metadatas.pop(idx)

        if self.embeddings.size > 0:
            self.embeddings = np.delete(self.embeddings, idx, axis=0)

        self.index = faiss.IndexFlatIP(self.dimension)
        if self.embeddings.size > 0:
            self.index.add(self.embeddings.astype(np.float32))

        self._persist()
        return True


class VectorDatabase:
    """Vector DB for exhibits and FAQ."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        backend: Optional[str] = None,
        faiss_storage_dir: Optional[str] = None,
    ):
        self.backend = (backend or VECTOR_BACKEND or "chroma").lower()
        self._use_faiss = self.backend == "faiss"

        if self._use_faiss:
            storage_dir = Path(faiss_storage_dir or FAISS_STORAGE_DIR)
            self.exhibits_index = FaissLocalIndex(storage_dir, "exhibits")
            self.faq_index = FaissLocalIndex(storage_dir, "faq")
            logger.info(
                "VectorDatabase initialized with FAISS backend at %s", storage_dir
            )
            return

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
            if isinstance(interesting_facts_str, list):
                processed_dict["interesting_facts"] = interesting_facts_str
            elif isinstance(interesting_facts_str, str) and interesting_facts_str:
                processed_dict["interesting_facts"] = [
                    fact.strip() for fact in interesting_facts_str.split(",")
                ]
            else:
                processed_dict["interesting_facts"] = []

        if "additional_info" in processed_dict:
            additional_info_str = processed_dict["additional_info"]
            if isinstance(additional_info_str, dict):
                processed_dict["additional_info"] = additional_info_str
            elif isinstance(additional_info_str, str) and additional_info_str:
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

        normalized_embedding = self._normalize_embedding(image_embedding)

        if self._use_faiss:
            self.exhibits_index.add(exhibit_id, normalized_embedding, metadata_dict)
            return True

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

        search_results = []
        if self._use_faiss:
            results = self.exhibits_index.search(
                query_embedding=normalized_embedding, limit=limit
            )
            for exhibit_id, metadata_dict, similarity in results:
                if similarity < score_threshold:
                    continue
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
            logger.info(f"Found {len(search_results)} exhibits")
            return search_results

        results = self.exhibits_collection.query(
            query_embeddings=[normalized_embedding], n_results=limit
        )

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

        if self._use_faiss:
            self.faq_index.add(faq_id, normalized_embedding, payload)
            return True

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

        search_results = []
        if self._use_faiss:
            results = self.faq_index.search(
                query_embedding=normalized_embedding,
                limit=limit,
                where={"exhibit_id": exhibit_id},
            )
            for _, metadata, similarity in results:
                if similarity < score_threshold:
                    continue
                search_results.append(
                    FAQSearchResult(
                        question=metadata.get("question", ""),
                        answer=metadata.get("answer", ""),
                        exhibit_id=metadata.get("exhibit_id", ""),
                        similarity_score=similarity,
                    )
                )
            return search_results

        results = self.faq_collection.query(
            query_embeddings=[normalized_embedding],
            n_results=limit,
            where={"exhibit_id": exhibit_id},
        )

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
        if self._use_faiss:
            metadata_dict = self.exhibits_index.get(exhibit_id)
            if not metadata_dict:
                return None
            try:
                processed_dict = self._restore_metadata_types(metadata_dict)
                return ExhibitMetadata(**processed_dict)
            except Exception:
                return None

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
            if self._use_faiss:
                return self.exhibits_index.delete(exhibit_id)

            self.exhibits_collection.delete(ids=[exhibit_id])
            return True
        except Exception:
            return False
