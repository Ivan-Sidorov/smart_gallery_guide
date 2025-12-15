import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from config.config import FAISS_STORAGE_DIR
from database.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult

logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self, storage_dir: str = FAISS_STORAGE_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.exhibit_index_path = self.storage_dir / "exhibits.index"
        self.exhibit_meta_path = self.storage_dir / "exhibits.json"
        self.faq_index_path = self.storage_dir / "faq.index"
        self.faq_meta_path = self.storage_dir / "faq.json"
        self.title_index_path = self.storage_dir / "title.index"
        self.title_meta_path = self.storage_dir / "title.json"
        self.desc_index_path = self.storage_dir / "desc.index"
        self.desc_meta_path = self.storage_dir / "desc.json"

        self.exhibit_index: Optional[faiss.IndexFlatIP] = None
        self.exhibit_dim: Optional[int] = None
        self.exhibit_ids: List[str] = []
        self.exhibit_metadata: Dict[str, Dict] = {}

        self.faq_index: Optional[faiss.IndexFlatIP] = None
        self.faq_dim: Optional[int] = None
        self.faq_items: List[Dict] = []

        self.title_index: Optional[faiss.IndexFlatIP] = None
        self.title_dim: Optional[int] = None
        self.title_ids: List[str] = []

        self.desc_index: Optional[faiss.IndexFlatIP] = None
        self.desc_dim: Optional[int] = None
        self.desc_ids: List[str] = []

        self._load_exhibits()
        self._load_faq()
        self._load_title_index()
        self._load_desc_index()

    def add_exhibit(
        self,
        exhibit_id: str,
        embedding: List[float],
        metadata: ExhibitMetadata,
        title_embedding: Optional[List[float]] = None,
        desc_embedding: Optional[List[float]] = None,
    ) -> None:
        vector = self._normalize(embedding)
        dim = vector.shape[1]

        if self.exhibit_index is None:
            self.exhibit_index = faiss.IndexFlatIP(dim)
            self.exhibit_dim = dim
        elif dim != self.exhibit_dim:
            raise ValueError("Dimension of embedding does not match the index.")

        if exhibit_id in self.exhibit_ids:
            logger.warning(
                "Exhibit %s already exists in the index, skipping.", exhibit_id
            )
            return

        self.exhibit_index.add(vector)
        self.exhibit_ids.append(exhibit_id)
        self.exhibit_metadata[exhibit_id] = metadata.model_dump()

        self._save_exhibits()

        if title_embedding is not None:
            self._add_title_embedding(exhibit_id, title_embedding)
        if desc_embedding is not None:
            self._add_desc_embedding(exhibit_id, desc_embedding)

    def add_faq(
        self,
        question: str,
        answer: str,
        exhibit_id: str,
        embedding: List[float],
        category: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        vector = self._normalize(embedding)
        dim = vector.shape[1]

        if self.faq_index is None:
            self.faq_index = faiss.IndexFlatIP(dim)
            self.faq_dim = dim
        elif dim != self.faq_dim:
            raise ValueError("Dimension of embedding does not match the FAQ index.")

        self.faq_index.add(vector)
        self.faq_items.append(
            {
                "question": question,
                "answer": answer,
                "exhibit_id": exhibit_id,
                "category": category,
                "metadata": metadata or {},
            }
        )

        self._save_faq()

    def search_exhibit(
        self,
        image_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        display_threshold: Optional[float] = None,
    ) -> List[ExhibitSearchResult]:
        if self.exhibit_index is None or self.exhibit_index.ntotal == 0:
            return []

        query = self._normalize(image_embedding)
        if query.shape[1] != self.exhibit_dim:
            raise ValueError("Dimension of query does not match the index.")

        k = min(self.exhibit_index.ntotal, max(limit * 3, limit))
        scores, idxs = self.exhibit_index.search(query, k)

        cutoff = display_threshold if display_threshold is not None else score_threshold

        results: List[ExhibitSearchResult] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if score < cutoff:
                continue

            exhibit_id = self.exhibit_ids[idx]
            metadata_dict = self.exhibit_metadata.get(exhibit_id)
            if not metadata_dict:
                continue

            metadata = ExhibitMetadata(**metadata_dict)
            results.append(
                ExhibitSearchResult(
                    exhibit_id=exhibit_id,
                    title=metadata.title,
                    similarity_score=float(score),
                    metadata=metadata,
                )
            )
            if len(results) >= limit:
                break

        return results

    def search_text(
        self,
        query_embedding: List[float],
        variant: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        display_threshold: Optional[float] = None,
    ) -> List[ExhibitSearchResult]:
        if variant == "title":
            index = self.title_index
            ids = self.title_ids
            dim = self.title_dim
        elif variant == "desc":
            index = self.desc_index
            ids = self.desc_ids
            dim = self.desc_dim
        else:
            raise ValueError("variant must be 'title' or 'desc'")

        if index is None or index.ntotal == 0:
            return []

        query = self._normalize(query_embedding)
        if query.shape[1] != dim:
            raise ValueError("Dimension of query does not match the text index.")

        k = min(index.ntotal, max(limit * 3, limit))
        scores, idxs = index.search(query, k)

        cutoff = display_threshold if display_threshold is not None else score_threshold

        results: List[ExhibitSearchResult] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if score < cutoff:
                continue

            exhibit_id = ids[idx]
            metadata_dict = self.exhibit_metadata.get(exhibit_id)
            if not metadata_dict:
                continue

            metadata = ExhibitMetadata(**metadata_dict)
            results.append(
                ExhibitSearchResult(
                    exhibit_id=exhibit_id,
                    title=metadata.title,
                    similarity_score=float(score),
                    metadata=metadata,
                )
            )
            if len(results) >= limit:
                break

        return results

    def search_faq(
        self,
        question_embedding: List[float],
        exhibit_id: str,
        limit: int = 3,
        score_threshold: float = 0.0,
        display_threshold: Optional[float] = None,
    ) -> List[FAQSearchResult]:
        if self.faq_index is None or self.faq_index.ntotal == 0:
            return []

        query = self._normalize(question_embedding)
        if query.shape[1] != self.faq_dim:
            raise ValueError("Dimension of query does not match the FAQ index.")

        k = self.faq_index.ntotal
        scores, idxs = self.faq_index.search(query, k)

        cutoff = display_threshold if display_threshold is not None else score_threshold

        results: List[FAQSearchResult] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue

            item = self.faq_items[idx]
            if item["exhibit_id"] != exhibit_id:
                continue
            if score < cutoff:
                continue

            results.append(
                FAQSearchResult(
                    question=item["question"],
                    answer=item["answer"],
                    exhibit_id=item["exhibit_id"],
                    similarity_score=float(score),
                )
            )
            if len(results) >= limit:
                break

        return results

    def get_exhibit_metadata(self, exhibit_id: str) -> Optional[ExhibitMetadata]:
        metadata_dict = self.exhibit_metadata.get(exhibit_id)
        if not metadata_dict:
            return None
        return ExhibitMetadata(**metadata_dict)

    def _normalize(self, vector: List[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(arr)
        return arr

    def _load_exhibits(self) -> None:
        if not (self.exhibit_index_path.exists() and self.exhibit_meta_path.exists()):
            return
        try:
            with open(self.exhibit_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.exhibit_ids = meta.get("ids", [])
            self.exhibit_metadata = meta.get("metadata", {})
            self.exhibit_dim = meta.get("dimension")

            self.exhibit_index = faiss.read_index(str(self.exhibit_index_path))
            if self.exhibit_dim is None:
                self.exhibit_dim = self.exhibit_index.d
            if self.exhibit_index.ntotal != len(self.exhibit_ids):
                logger.warning(
                    "Size of exhibit index does not match the metadata. Resetting index."
                )
                self._reset_exhibits()
        except Exception:
            logger.exception("Failed to load exhibit index, creating new one.")
            self._reset_exhibits()

    def _load_faq(self) -> None:
        if not (self.faq_index_path.exists() and self.faq_meta_path.exists()):
            return
        try:
            with open(self.faq_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.faq_items = meta.get("items", [])
            self.faq_dim = meta.get("dimension")

            self.faq_index = faiss.read_index(str(self.faq_index_path))
            if self.faq_dim is None:
                self.faq_dim = self.faq_index.d
            if self.faq_index.ntotal != len(self.faq_items):
                logger.warning(
                    "Size of FAQ index does not match the metadata. Resetting index."
                )
                self._reset_faq()
        except Exception:
            logger.exception("Failed to load FAQ index, creating new one.")
            self._reset_faq()

    def _save_exhibits(self) -> None:
        if self.exhibit_index is None:
            return

        meta = {
            "dimension": self.exhibit_dim,
            "ids": self.exhibit_ids,
            "metadata": self.exhibit_metadata,
        }
        with open(self.exhibit_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        faiss.write_index(self.exhibit_index, str(self.exhibit_index_path))

    def _add_title_embedding(self, exhibit_id: str, embedding: List[float]) -> None:
        vector = self._normalize(embedding)
        dim = vector.shape[1]

        if self.title_index is None:
            self.title_index = faiss.IndexFlatIP(dim)
            self.title_dim = dim
        elif dim != self.title_dim:
            raise ValueError("Dimension of text index (title) does not match.")

        self.title_index.add(vector)
        self.title_ids.append(exhibit_id)
        self._save_title_index()

    def _add_desc_embedding(self, exhibit_id: str, embedding: List[float]) -> None:
        vector = self._normalize(embedding)
        dim = vector.shape[1]

        if self.desc_index is None:
            self.desc_index = faiss.IndexFlatIP(dim)
            self.desc_dim = dim
        elif dim != self.desc_dim:
            raise ValueError("Dimension of text index (desc) does not match.")

        self.desc_index.add(vector)
        self.desc_ids.append(exhibit_id)
        self._save_desc_index()

    def _save_faq(self) -> None:
        if self.faq_index is None:
            return

        meta = {"dimension": self.faq_dim, "items": self.faq_items}
        with open(self.faq_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        faiss.write_index(self.faq_index, str(self.faq_index_path))

    def _save_title_index(self) -> None:
        if self.title_index is None:
            return

        meta = {"dimension": self.title_dim, "ids": self.title_ids}
        with open(self.title_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        faiss.write_index(self.title_index, str(self.title_index_path))

    def _save_desc_index(self) -> None:
        if self.desc_index is None:
            return

        meta = {"dimension": self.desc_dim, "ids": self.desc_ids}
        with open(self.desc_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        faiss.write_index(self.desc_index, str(self.desc_index_path))

    def _reset_exhibits(self) -> None:
        self.exhibit_index = None
        self.exhibit_dim = None
        self.exhibit_ids = []
        self.exhibit_metadata = {}
        self._reset_title_index()
        self._reset_desc_index()

    def _reset_faq(self) -> None:
        self.faq_index = None
        self.faq_dim = None
        self.faq_items = []

    def _load_title_index(self) -> None:
        if not (self.title_index_path.exists() and self.title_meta_path.exists()):
            return
        try:
            with open(self.title_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.title_ids = meta.get("ids", [])
            self.title_dim = meta.get("dimension")

            self.title_index = faiss.read_index(str(self.title_index_path))
            if self.title_dim is None:
                self.title_dim = self.title_index.d
            if self.title_index.ntotal != len(self.title_ids):
                logger.warning(
                    "Size of text index (title) does not match the metadata. Resetting index."
                )
                self._reset_title_index()
        except Exception:
            logger.exception("Failed to load text index title, creating new one.")
            self._reset_title_index()

    def _load_desc_index(self) -> None:
        if not (self.desc_index_path.exists() and self.desc_meta_path.exists()):
            return
        try:
            with open(self.desc_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.desc_ids = meta.get("ids", [])
            self.desc_dim = meta.get("dimension")

            self.desc_index = faiss.read_index(str(self.desc_index_path))
            if self.desc_dim is None:
                self.desc_dim = self.desc_index.d
            if self.desc_index.ntotal != len(self.desc_ids):
                logger.warning(
                    "Size of text index (desc) does not match the metadata. Resetting index."
                )
                self._reset_desc_index()
        except Exception:
            logger.exception("Failed to load text index desc, creating new one.")
            self._reset_desc_index()

    def _reset_title_index(self) -> None:
        self.title_index = None
        self.title_dim = None
        self.title_ids = []

    def _reset_desc_index(self) -> None:
        self.desc_index = None
        self.desc_dim = None
        self.desc_ids = []
