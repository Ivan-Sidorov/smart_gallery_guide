"""BM25 ranking utilities and Reciprocal Rank Fusion (RRF) merge."""

import re

from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

# RRF constant used both for hybrid fusion and as a public default.
RRF_K: int = 60

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_stemmer = SnowballStemmer("russian")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, words ≥ 2 chars, Russian stemming."""
    return [_stemmer.stem(w) for w in _WORD_RE.findall(text.lower()) if len(w) >= 2]


class BM25Index:
    """`rank_bm25.BM25Okapi` index wrapper."""

    __slots__ = ("bm25", "doc_ids", "meta_map")

    def __init__(
        self,
        bm25: BM25Okapi,
        doc_ids: list[str],
        meta_map: dict[str, dict],
    ) -> None:
        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.meta_map = meta_map

    @classmethod
    def build(
        cls,
        doc_ids: list[str],
        documents: list[str],
        meta_map: dict[str, dict] | None = None,
    ) -> "BM25Index | None":
        """Build an index from raw documents. Returns ``None`` if nothing to index."""
        tokenized = [tokenize(doc or "") for doc in documents]
        if not any(tokenized):
            return None
        return cls(BM25Okapi(tokenized), doc_ids, meta_map or {})

    def rank(
        self,
        query_text: str,
        top_k: int,
        exhibit_id_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Rank documents by BM25 relevance to ``query_text``.

        Args:
            query_text: Query string.
            top_k: Maximum number of returned ``(doc_id, score)`` pairs.
            exhibit_id_filter: Optional filter - keep only documents whose
                ``meta_map[doc_id]["exhibit_id"]`` matches this value.

        Returns:
            ``(doc_id, bm25_score)`` pairs sorted by score descending.
        """
        tokens = tokenize(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        results: list[tuple[str, float]] = []
        for idx in scores.argsort()[::-1]:
            score = float(scores[idx])
            if score <= 0:
                break
            doc_id = self.doc_ids[idx]
            if (
                exhibit_id_filter
                and self.meta_map.get(doc_id, {}).get("exhibit_id") != exhibit_id_filter
            ):
                continue
            results.append((doc_id, score))
            if len(results) >= top_k:
                break
        return results


def rrf_merge(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists into one via Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked lists.
        k: RRF constant.

    Returns:
        List of tuples of ``(doc_id, rrf_score)`` sorted by score descending.
    """
    rrf_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
