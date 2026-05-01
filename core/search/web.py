"""Web search wrapper around DDGS used to enrich VLM context."""

import asyncio
import logging
from dataclasses import dataclass

from ddgs import DDGS

from core.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """A single web search hit returned by :class:`WebSearchService`."""

    title: str
    snippet: str
    url: str


class WebSearchService:
    """DDGS web metasearch wrapper for enriching VLM context."""

    def __init__(self, max_results: int | None = None):
        self.max_results = max_results or get_settings().web_search_max_results

    async def search(self, query: str) -> list[WebSearchResult]:
        """Run a web search asynchronously.

        Args:
            query: Search query string.

        Returns:
            List of web search results.
        """
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, query: str) -> list[WebSearchResult]:
        try:
            raw = DDGS().text(
                query, region="ru-ru", max_results=self.max_results, backend="auto"
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
    def format_results(results: list[WebSearchResult]) -> str:
        """Prepare a list of web search results.

        Args:
            results: List of web search results.

        Returns:
            Formatted string of the results.
        """
        if not results:
            return ""
        parts = [
            f"{i}. {r.title}\n{r.snippet}\nИсточник: {r.url}"
            for i, r in enumerate(results, 1)
        ]
        return "\n\n".join(parts)
