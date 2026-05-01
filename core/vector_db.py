"""Vector storage and hybrid retrieval over ChromaDB."""

import json
import logging
import math
import numbers
import pickle
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from core.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult
from core.search.bm25 import RRF_K, BM25Index
from core.settings import get_settings

logger = logging.getLogger(__name__)

HNSW_CONFIG = {
    "hnsw": {
        "space": "cosine",
        "ef_construction": 400,
        "max_neighbors": 32,
        "ef_search": 200,
    }
}


class VectorDatabase:
    """Vector database backed by ChromaDB with hybrid (vector + BM25) search."""

    def __init__(self, persist_directory: str | None = None):
        settings = get_settings()
        persist_dir = persist_directory or settings.chroma_persist_dir
        self._persist_path = Path(str(persist_dir))
        self._bm25_dir = self._persist_path / "bm25"

        self.client = chromadb.PersistentClient(path=str(self._persist_path))

        self.exhibits_collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_exhibits, configuration=HNSW_CONFIG
        )
        self.title_collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_title, configuration=HNSW_CONFIG
        )
        self.desc_collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_desc, configuration=HNSW_CONFIG
        )
        self.faq_collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_faq, configuration=HNSW_CONFIG
        )

        self._bm25_cache: dict[str, BM25Index | None] = {}
        self._load_bm25_indexes()

    # ----------------------------------------------------------------- ingest

    def add_exhibit(
        self,
        exhibit_id: str,
        embedding: list[float],
        metadata: ExhibitMetadata,
        title_embedding: list[float] | None = None,
        title_text: str | None = None,
        desc_embedding: list[float] | None = None,
        desc_text: str | None = None,
    ) -> None:
        """Add an exhibit to the vector database.

        Args:
            exhibit_id: ID of the exhibit.
            embedding: Embedding of the exhibit image.
            metadata: Metadata of the exhibit.
            title_embedding: Embedding of the title text.
            title_text: Text of the title.
            desc_embedding: Embedding of the description text.
            desc_text: Text of the description.
        """
        existing = self.exhibits_collection.get(ids=[exhibit_id])
        if existing["ids"]:
            logger.warning(
                "Exhibit %s already exists in the index, skipping.", exhibit_id
            )
            return

        serialized = self._serialize_metadata(metadata)
        normalized = self._normalize_embedding(embedding)

        self.exhibits_collection.add(
            embeddings=[normalized],
            ids=[exhibit_id],
            metadatas=[serialized],
        )

        if title_embedding is not None:
            add_kw: dict[str, Any] = {
                "embeddings": [self._normalize_embedding(title_embedding)],
                "ids": [exhibit_id],
                "metadatas": [{"exhibit_id": exhibit_id}],
            }
            if title_text:
                add_kw["documents"] = [title_text]
            self.title_collection.add(**add_kw)
            self._invalidate_bm25(self.title_collection)

        if desc_embedding is not None:
            add_kw = {
                "embeddings": [self._normalize_embedding(desc_embedding)],
                "ids": [exhibit_id],
                "metadatas": [{"exhibit_id": exhibit_id}],
            }
            if desc_text:
                add_kw["documents"] = [desc_text]
            self.desc_collection.add(**add_kw)
            self._invalidate_bm25(self.desc_collection)

    def add_faq(
        self,
        question: str,
        answer: str,
        exhibit_id: str,
        embedding: list[float],
    ) -> None:
        """Add a FAQ item to the vector database.

        Args:
            question: Question text.
            answer: Answer text.
            exhibit_id: ID of the exhibit.
            embedding: Embedding of the FAQ question.
        """
        faq_id = f"{exhibit_id}_{abs(hash(question))}"
        normalized = self._normalize_embedding(embedding)

        self.faq_collection.add(
            embeddings=[normalized],
            ids=[faq_id],
            documents=[question],
            metadatas=[
                {
                    "question": question,
                    "answer": answer,
                    "exhibit_id": exhibit_id,
                }
            ],
        )
        self._invalidate_bm25(self.faq_collection)

    # ----------------------------------------------------------------- search

    def search_exhibit(
        self,
        image_embedding: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        display_threshold: float | None = None,
    ) -> list[ExhibitSearchResult]:
        """Search for exhibits by image embedding.

        Args:
            image_embedding: Embedding of the image.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score to return.
            display_threshold: Minimum similarity score to display.

        Returns:
            List of exhibit search results.
        """
        if self.exhibits_collection.count() == 0:
            return []

        normalized = self._normalize_embedding(image_embedding)
        n = min(self.exhibits_collection.count(), limit)

        results = self.exhibits_collection.query(
            query_embeddings=[normalized],
            n_results=n,
        )

        cutoff = display_threshold if display_threshold is not None else score_threshold
        return self._build_exhibit_results(results, cutoff, limit)

    def search_text(
        self,
        query_embedding: list[float],
        variant: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        display_threshold: float | None = None,
        query_text: str | None = None,
    ) -> list[ExhibitSearchResult]:
        """Search for exhibits by text.

        Args:
            query_embedding: Embedding of the query text.
            variant: Variant of the text to search.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score to return.
            display_threshold: Minimum similarity score to display.
            query_text: Text to search for.

        Returns:
            List of exhibit search results.
        """
        if variant == "title":
            collection = self.title_collection
        elif variant == "desc":
            collection = self.desc_collection
        else:
            raise ValueError("variant must be 'title' or 'desc'")

        if collection.count() == 0:
            return []

        cutoff = display_threshold if display_threshold is not None else score_threshold

        if query_text:
            ranked = self._hybrid_search(
                collection=collection,
                query_embedding=query_embedding,
                query_text=query_text,
                n_results=limit,
            )
        else:
            ranked = self._vector_search(
                collection=collection,
                query_embedding=query_embedding,
                n_results=limit,
            )

        if not ranked:
            return []

        exhibit_ids = [doc_id for doc_id, _ in ranked]
        meta_by_id = self._fetch_exhibit_metadata(exhibit_ids)

        search_results: list[ExhibitSearchResult] = []
        for doc_id, similarity in ranked:
            if similarity < cutoff:
                continue

            raw_meta = meta_by_id.get(doc_id)
            if not raw_meta:
                continue

            try:
                metadata = self._deserialize_metadata(raw_meta)
                search_results.append(
                    ExhibitSearchResult(
                        exhibit_id=doc_id,
                        title=metadata.title,
                        similarity_score=similarity,
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.warning("Error restoring metadata for %s: %s", doc_id, e)

            if len(search_results) >= limit:
                break

        return search_results

    def search_faq(
        self,
        question_embedding: list[float],
        exhibit_id: str,
        limit: int = 3,
        score_threshold: float = 0.0,
        display_threshold: float | None = None,
        query_text: str | None = None,
    ) -> list[FAQSearchResult]:
        """Search for FAQ items by question embedding.

        Args:
            question_embedding: Embedding of the question.
            exhibit_id: ID of the exhibit.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score to return.
            display_threshold: Minimum similarity score to display.
            query_text: Text to search for.

        Returns:
            List of FAQ search results.
        """
        if self.faq_collection.count() == 0:
            return []

        cutoff = display_threshold if display_threshold is not None else score_threshold
        where = {"exhibit_id": exhibit_id}

        if query_text:
            ranked_raw = self._hybrid_search(
                collection=self.faq_collection,
                query_embedding=question_embedding,
                query_text=query_text,
                n_results=limit,
                where=where,
                exhibit_id_filter=exhibit_id,
            )
        else:
            ranked_raw = self._vector_search(
                collection=self.faq_collection,
                query_embedding=question_embedding,
                n_results=limit,
                where=where,
            )

        if not ranked_raw:
            return []

        faq_ids = [faq_id for faq_id, _ in ranked_raw]
        faq_lookup = self.faq_collection.get(ids=faq_ids)
        meta_by_id: dict[str, dict] = {}
        for i, fid in enumerate(faq_lookup["ids"]):
            meta_by_id[fid] = faq_lookup["metadatas"][i]

        search_results: list[FAQSearchResult] = []
        for faq_id, similarity in ranked_raw:
            if similarity < cutoff:
                continue

            meta = meta_by_id.get(faq_id)
            if not meta:
                continue

            search_results.append(
                FAQSearchResult(
                    question=meta.get("question", ""),
                    answer=meta.get("answer", ""),
                    exhibit_id=meta.get("exhibit_id", ""),
                    similarity_score=similarity,
                )
            )

            if len(search_results) >= limit:
                break

        return search_results

    def get_exhibit_metadata(self, exhibit_id: str) -> ExhibitMetadata | None:
        """Get metadata for an exhibit.

        Args:
            exhibit_id: ID of the exhibit.

        Returns:
            Metadata of the exhibit.
        """
        results = self.exhibits_collection.get(ids=[exhibit_id])
        if not results["ids"]:
            return None
        try:
            return self._deserialize_metadata(results["metadatas"][0])
        except Exception:
            return None

    # ----------------------------------------------------------------- hybrid

    def _hybrid_search(
        self,
        collection: chromadb.Collection,
        query_embedding: list[float],
        query_text: str,
        n_results: int,
        where: dict | None = None,
        exhibit_id_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Hybrid search: semantic + BM25 fused via RRF.

        Args:
            collection: Collection to search.
            query_embedding: Embedding of the query.
            query_text: Text to search for.
            n_results: Maximum number of results to return.
            where: Where clause to filter the search.
            exhibit_id_filter: Exhibit ID to filter the search.

        Returns:
            List of tuples of ``(doc_id, similarity_score)`` sorted by RRF score descending.
        """
        normalized = self._normalize_embedding(query_embedding)
        total = collection.count()
        if total == 0:
            return []

        fetch_n = min(total, n_results * 3)

        query_kw: dict[str, Any] = {
            "query_embeddings": [normalized],
            "n_results": fetch_n,
        }
        if where:
            query_kw["where"] = where
        vec_results = collection.query(**query_kw)

        rrf_scores: dict[str, float] = {}
        similarities: dict[str, float] = {}

        if vec_results["ids"] and vec_results["ids"][0]:
            for rank, (doc_id, dist) in enumerate(
                zip(vec_results["ids"][0], vec_results["distances"][0], strict=False)
            ):
                rrf_scores[doc_id] = 1.0 / (RRF_K + rank + 1)
                similarities[doc_id] = 1.0 - dist

        bm25_ranked = self._bm25_rank(
            collection, query_text, fetch_n, exhibit_id_filter
        )
        for rank, (doc_id, _bm25_score) in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        bm25_only = [did for did, _ in bm25_ranked if did not in similarities]
        if bm25_only:
            similarities.update(
                self._compute_similarities(collection, normalized, bm25_only)
            )

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (doc_id, similarities.get(doc_id, 0.0))
            for doc_id, _ in ranked[: n_results * 2]
        ]

    def _vector_search(
        self,
        collection: chromadb.Collection,
        query_embedding: list[float],
        n_results: int,
        where: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Pure vector search.

        Args:
            collection: Collection to search.
            query_embedding: Embedding of the query.
            n_results: Maximum number of results to return.
            where: Where condition to filter the search.

        Returns:
            List of tuples of ``(doc_id, similarity_score)`` sorted by similarity score descending.
        """
        normalized = self._normalize_embedding(query_embedding)
        total = collection.count()
        if total == 0:
            return []

        fetch_n = min(total, n_results)
        query_kw: dict[str, Any] = {
            "query_embeddings": [normalized],
            "n_results": fetch_n,
        }
        if where:
            query_kw["where"] = where
        results = collection.query(**query_kw)

        pairs: list[tuple[str, float]] = []
        if results["ids"] and results["ids"][0]:
            for doc_id, dist in zip(
                results["ids"][0], results["distances"][0], strict=False
            ):
                pairs.append((doc_id, 1.0 - dist))
        return pairs

    # ------------------------------------------------------------------ BM25

    def build_bm25_indexes(self) -> None:
        """Build BM25 indexes from current ChromaDB data and persist to disk."""
        self._bm25_dir.mkdir(parents=True, exist_ok=True)

        for collection in (
            self.title_collection,
            self.desc_collection,
            self.faq_collection,
        ):
            self._build_single_bm25(collection)

    def _build_single_bm25(self, collection: chromadb.Collection) -> None:
        name = collection.name

        if collection.count() == 0:
            logger.info("Collection '%s' is empty, skipping BM25 build.", name)
            return

        data = collection.get(include=["documents", "metadatas"])
        if not data["ids"] or not data.get("documents"):
            return

        doc_ids: list[str] = data["ids"]
        documents: list[str] = data["documents"]

        meta_map: dict[str, dict] = {}
        if data.get("metadatas"):
            for i, did in enumerate(doc_ids):
                meta_map[did] = data["metadatas"][i] or {}

        index = BM25Index.build(doc_ids=doc_ids, documents=documents, meta_map=meta_map)
        if index is None:
            return

        self._bm25_cache[name] = index

        path = self._bm25_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": index.bm25,
                    "doc_ids": index.doc_ids,
                    "meta_map": index.meta_map,
                },
                f,
            )
        logger.info("Built BM25 index for '%s': %d documents.", name, len(doc_ids))

    def _load_bm25_indexes(self) -> None:
        """Load pre-built BM25 indexes from disk into the in-memory cache."""
        if not self._bm25_dir.exists():
            return

        for collection in (
            self.title_collection,
            self.desc_collection,
            self.faq_collection,
        ):
            path = self._bm25_dir / f"{collection.name}.pkl"
            if not path.exists():
                continue
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                self._bm25_cache[collection.name] = BM25Index(
                    bm25=payload["bm25"],
                    doc_ids=payload["doc_ids"],
                    meta_map=payload["meta_map"],
                )
                logger.info(
                    "Loaded BM25 index for '%s': %d documents.",
                    collection.name,
                    len(payload["doc_ids"]),
                )
            except Exception:
                logger.exception("Failed to load BM25 index for '%s'.", collection.name)

    def _invalidate_bm25(self, collection: chromadb.Collection) -> None:
        """Drop in-memory BM25 cache for a collection.

        Args:
            collection: Collection to invalidate.
        """
        self._bm25_cache.pop(collection.name, None)

    def _bm25_rank(
        self,
        collection: chromadb.Collection,
        query_text: str,
        top_k: int,
        exhibit_id_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Rank documents in ``collection`` by BM25 relevance to ``query_text``."""
        index = self._bm25_cache.get(collection.name)
        if index is None:
            return []
        return index.rank(
            query_text=query_text, top_k=top_k, exhibit_id_filter=exhibit_id_filter
        )

    # ------------------------------------------------------------ utilities

    @staticmethod
    def _compute_similarities(
        collection: chromadb.Collection,
        query_embedding: list[float],
        doc_ids: list[str],
    ) -> dict[str, float]:
        """Compute cosine similarity for specific documents against a query."""
        if not doc_ids:
            return {}

        data = collection.get(ids=doc_ids, include=["embeddings"])
        query_arr = np.asarray(query_embedding, dtype=np.float32)

        result: dict[str, float] = {}
        for i, doc_id in enumerate(data["ids"]):
            doc_emb = np.asarray(data["embeddings"][i], dtype=np.float32)
            result[doc_id] = float(np.dot(query_arr, doc_emb))
        return result

    def _fetch_exhibit_metadata(self, exhibit_ids: list[str]) -> dict[str, dict]:
        """Fetch metadata from the exhibits collection."""
        if not exhibit_ids:
            return {}
        lookup = self.exhibits_collection.get(ids=exhibit_ids)
        meta_by_id: dict[str, dict] = {}
        for i, eid in enumerate(lookup["ids"]):
            meta_by_id[eid] = lookup["metadatas"][i]
        return meta_by_id

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        arr = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    @staticmethod
    def _serialize_metadata(metadata: ExhibitMetadata) -> dict[str, Any]:
        """Convert ``ExhibitMetadata`` to a flat dict accepted by ChromaDB metadata."""
        raw = metadata.model_dump(mode="json")
        processed: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                processed[key] = json.dumps(value, ensure_ascii=False, default=str)
            else:
                processed[key] = VectorDatabase._chroma_metadata_scalar(value)
        return processed

    @staticmethod
    def _chroma_metadata_scalar(value: Any) -> Any:
        """Convert a single value to a ChromaDB-compatible metadata scalar."""
        if value is None:
            raise ValueError("unexpected None in _chroma_metadata_scalar")
        if isinstance(value, np.generic):
            return VectorDatabase._chroma_metadata_scalar(value.item())

        if isinstance(value, bool):
            return value
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            x = float(value)
            if math.isnan(x) or math.isinf(x):
                return str(x)
            return x
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _deserialize_metadata(meta_dict: dict[str, Any]) -> ExhibitMetadata:
        """Restore ``ExhibitMetadata`` from a metadata dict."""
        restored = meta_dict.copy()

        if "additional_info" in restored and isinstance(
            restored["additional_info"], str
        ):
            try:
                restored["additional_info"] = json.loads(restored["additional_info"])
            except (json.JSONDecodeError, TypeError):
                restored["additional_info"] = {}

        return ExhibitMetadata(**restored)

    def _build_exhibit_results(
        self,
        results: dict,
        cutoff: float,
        limit: int,
    ) -> list[ExhibitSearchResult]:
        """Convert pure-vector Chroma response into typed ``ExhibitSearchResult``."""
        search_results: list[ExhibitSearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return search_results

        for i, exhibit_id in enumerate(results["ids"][0]):
            similarity = 1.0 - results["distances"][0][i]
            if similarity < cutoff:
                continue

            raw_meta = results["metadatas"][0][i]
            try:
                metadata = self._deserialize_metadata(raw_meta)
                search_results.append(
                    ExhibitSearchResult(
                        exhibit_id=exhibit_id,
                        title=metadata.title,
                        similarity_score=similarity,
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.warning("Error restoring metadata for %s: %s", exhibit_id, e)

            if len(search_results) >= limit:
                break

        return search_results

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Release resources held by the underlying Chroma client."""
        self._bm25_cache.clear()
        client = getattr(self, "client", None)
        if client is None:
            return
        for attr in ("reset", "close", "stop"):
            fn = getattr(client, attr, None)
            if fn is None:
                continue
            try:
                fn()
                break
            except Exception:
                continue
        self.client = None
