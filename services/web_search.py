import asyncio
import logging
from dataclasses import dataclass
from typing import List

from duckduckgo_search import DDGS

from config.config import WEB_SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title: str
    snippet: str
    url: str


class WebSearchService:
    """DuckDuckGo web search wrapper for enriching exhibit context."""

    def __init__(self, max_results: int | None = None):
        self.max_results = max_results or WEB_SEARCH_MAX_RESULTS

    async def search(self, query: str) -> List[WebSearchResult]:
        """Run a web search asynchronously (offloads blocking I/O to thread)."""
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, query: str) -> List[WebSearchResult]:
        try:
            with DDGS() as ddgs:
                raw = list(
                    ddgs.text(query, region="ru-ru", max_results=self.max_results)
                )
            results = [
                WebSearchResult(
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                    url=r.get("href", ""),
                )
                for r in raw
            ]
            logger.info("Web search for %r returned %d results", query, len(results))
            return results
        except Exception as e:
            logger.error("Web search failed for %r: %s", query, e, exc_info=True)
            return []

    @staticmethod
    def format_results(results: List[WebSearchResult]) -> str:
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"{i}. {r.title}\n{r.snippet}\nИсточник: {r.url}")
        return "\n\n".join(parts)
