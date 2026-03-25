import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

from config.config import (
    CHROMA_COLLECTION_DESC,
    CHROMA_COLLECTION_EXHIBITS,
    CHROMA_COLLECTION_FAQ,
    CHROMA_COLLECTION_TITLE,
    CHROMA_PERSIST_DIR,
)
from database.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult

logger = logging.getLogger(__name__)

HNSW_CONFIG = {
    "hnsw": {
        "space": "cosine",
        "ef_construction": 400,
        "max_neighbors": 32,
        "ef_search": 200,
    }
}

_RRF_K = 60
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_stemmer = SnowballStemmer("russian")


class VectorDatabase:
    """Vector database backed by ChromaDB with hybrid (vector + BM25) search."""

    def __init__(self, persist_directory: Optional[str] = None):
        persist_dir = persist_directory or CHROMA_PERSIST_DIR
        self._persist_path = Path(str(persist_dir))
        self._bm25_dir = self._persist_path / "bm25"

        self.client = chromadb.PersistentClient(path=str(self._persist_path))

        self.exhibits_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_EXHIBITS,
            configuration=HNSW_CONFIG,
        )
        self.title_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_TITLE,
            configuration=HNSW_CONFIG,
        )
        self.desc_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_DESC,
            configuration=HNSW_CONFIG,
        )
        self.faq_collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_FAQ,
            configuration=HNSW_CONFIG,
        )

        self._bm25_cache: Dict[
            str, Optional[Tuple[BM25Okapi, List[str], Dict[str, Dict]]]
        ] = {}
        self._load_bm25_indexes()

    def add_exhibit(
        self,
        exhibit_id: str,
        embedding: List[float],
        metadata: ExhibitMetadata,
        title_embedding: Optional[List[float]] = None,
        title_text: Optional[str] = None,
        desc_embedding: Optional[List[float]] = None,
        desc_text: Optional[str] = None,
    ) -> None:
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
            add_kw: Dict[str, Any] = {
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
        embedding: List[float],
    ) -> None:
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

    # --------------------------------------------------------------- search

    def search_exhibit(
        self,
        image_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        display_threshold: Optional[float] = None,
    ) -> List[ExhibitSearchResult]:
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
        query_embedding: List[float],
        variant: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        display_threshold: Optional[float] = None,
        query_text: Optional[str] = None,
    ) -> List[ExhibitSearchResult]:
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

        search_results: List[ExhibitSearchResult] = []
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
        question_embedding: List[float],
        exhibit_id: str,
        limit: int = 3,
        score_threshold: float = 0.0,
        display_threshold: Optional[float] = None,
        query_text: Optional[str] = None,
    ) -> List[FAQSearchResult]:
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
        meta_by_id: Dict[str, Dict] = {}
        for i, fid in enumerate(faq_lookup["ids"]):
            meta_by_id[fid] = faq_lookup["metadatas"][i]

        search_results: List[FAQSearchResult] = []
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

    def get_exhibit_metadata(self, exhibit_id: str) -> Optional[ExhibitMetadata]:
        results = self.exhibits_collection.get(ids=[exhibit_id])
        if not results["ids"]:
            return None
        try:
            return self._deserialize_metadata(results["metadatas"][0])
        except Exception:
            return None

    def _hybrid_search(
        self,
        collection: chromadb.Collection,
        query_embedding: List[float],
        query_text: str,
        n_results: int,
        where: Optional[Dict] = None,
        exhibit_id_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Hybrid search combining Semantic Search with BM25 using RRF.

        Returns ``(doc_id, similarity_score)`` sorted by RRF score.
        """
        normalized = self._normalize_embedding(query_embedding)
        total = collection.count()
        if total == 0:
            return []

        fetch_n = min(total, n_results * 3)

        # 1. Semantic Search
        query_kw: Dict[str, Any] = {
            "query_embeddings": [normalized],
            "n_results": fetch_n,
        }
        if where:
            query_kw["where"] = where

        vec_results = collection.query(**query_kw)

        rrf_scores: Dict[str, float] = {}
        similarities: Dict[str, float] = {}

        if vec_results["ids"] and vec_results["ids"][0]:
            for rank, (doc_id, dist) in enumerate(
                zip(vec_results["ids"][0], vec_results["distances"][0])
            ):
                rrf_scores[doc_id] = 1.0 / (_RRF_K + rank + 1)
                similarities[doc_id] = 1.0 - dist

        # 2. BM25 Search
        bm25_ranked = self._bm25_rank(
            collection, query_text, fetch_n, exhibit_id_filter
        )

        for rank, (doc_id, _bm25_score) in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

        # Compute similarity for BM25-only documents
        bm25_only = [did for did, _ in bm25_ranked if did not in similarities]
        if bm25_only:
            extra_sims = self._compute_similarities(collection, normalized, bm25_only)
            similarities.update(extra_sims)

        # 3. Merge via RRF
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (doc_id, similarities.get(doc_id, 0.0))
            for doc_id, _ in ranked[: n_results * 2]
        ]

    def _vector_search(
        self,
        collection: chromadb.Collection,
        query_embedding: List[float],
        n_results: int,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """Pure vector search. Returns ``(doc_id, similarity)`` pairs."""
        normalized = self._normalize_embedding(query_embedding)
        total = collection.count()
        if total == 0:
            return []

        fetch_n = min(total, n_results)
        query_kw: Dict[str, Any] = {
            "query_embeddings": [normalized],
            "n_results": fetch_n,
        }
        if where:
            query_kw["where"] = where

        results = collection.query(**query_kw)

        pairs: List[Tuple[str, float]] = []
        if results["ids"] and results["ids"][0]:
            for doc_id, dist in zip(results["ids"][0], results["distances"][0]):
                pairs.append((doc_id, 1.0 - dist))
        return pairs

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

        doc_ids: List[str] = data["ids"]
        tokenized = [self._tokenize(doc or "") for doc in data["documents"]]

        if not any(tokenized):
            return

        bm25 = BM25Okapi(tokenized)

        meta_map: Dict[str, Dict] = {}
        if data.get("metadatas"):
            for i, did in enumerate(doc_ids):
                meta_map[did] = data["metadatas"][i] or {}

        self._bm25_cache[name] = (bm25, doc_ids, meta_map)

        path = self._bm25_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"bm25": bm25, "doc_ids": doc_ids, "meta_map": meta_map}, f)
        logger.info("Built BM25 index for '%s': %d documents.", name, len(doc_ids))

    def _load_bm25_indexes(self) -> None:
        """Load pre-built BM25 indexes from disk."""
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
                self._bm25_cache[collection.name] = (
                    payload["bm25"],
                    payload["doc_ids"],
                    payload["meta_map"],
                )
                logger.info(
                    "Loaded BM25 index for '%s': %d documents.",
                    collection.name,
                    len(payload["doc_ids"]),
                )
            except Exception:
                logger.exception("Failed to load BM25 index for '%s'.", collection.name)

    def _invalidate_bm25(self, collection: chromadb.Collection) -> None:
        """Drop in-memory BM25 cache."""
        self._bm25_cache.pop(collection.name, None)

    def _bm25_rank(
        self,
        collection: chromadb.Collection,
        query_text: str,
        top_k: int,
        exhibit_id_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Rank documents in collection by BM25 relevance to query_text."""
        cached = self._bm25_cache.get(collection.name)
        if cached is None:
            return []

        bm25, doc_ids, meta_map = cached
        tokens = self._tokenize(query_text)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1]

        results: List[Tuple[str, float]] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc_id = doc_ids[idx]
            if exhibit_id_filter:
                meta = meta_map.get(doc_id, {})
                if meta.get("exhibit_id") != exhibit_id_filter:
                    continue
            results.append((doc_id, float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _compute_similarities(
        collection: chromadb.Collection,
        query_embedding: List[float],
        doc_ids: List[str],
    ) -> Dict[str, float]:
        """Compute cosine similarity for specific documents against a query."""
        if not doc_ids:
            return {}

        data = collection.get(ids=doc_ids, include=["embeddings"])
        query_arr = np.asarray(query_embedding, dtype=np.float32)

        result: Dict[str, float] = {}
        for i, doc_id in enumerate(data["ids"]):
            doc_emb = np.asarray(data["embeddings"][i], dtype=np.float32)
            result[doc_id] = float(np.dot(query_arr, doc_emb))
        return result

    # ------------------------------------------------------------ utilities

    def _fetch_exhibit_metadata(self, exhibit_ids: List[str]) -> Dict[str, Dict]:
        """Fetch metadata from the exhibits collection."""
        if not exhibit_ids:
            return {}
        lookup = self.exhibits_collection.get(ids=exhibit_ids)
        meta_by_id: Dict[str, Dict] = {}
        for i, eid in enumerate(lookup["ids"]):
            meta_by_id[eid] = lookup["metadatas"][i]
        return meta_by_id

    @staticmethod
    def _normalize_embedding(embedding: List[float]) -> List[float]:
        arr = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    @staticmethod
    def _serialize_metadata(metadata: ExhibitMetadata) -> Dict[str, Any]:
        """Convert ExhibitMetadata to a flat dict."""
        raw = metadata.model_dump()
        processed: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                processed[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, list):
                processed[key] = json.dumps(value, ensure_ascii=False)
            else:
                processed[key] = value
        return processed

    @staticmethod
    def _deserialize_metadata(meta_dict: Dict[str, Any]) -> ExhibitMetadata:
        """Restore ExhibitMetadata from a metadata dict."""
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
    ) -> List[ExhibitSearchResult]:
        search_results: List[ExhibitSearchResult] = []
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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25."""
        return [_stemmer.stem(w) for w in _WORD_RE.findall(text.lower()) if len(w) >= 2]
