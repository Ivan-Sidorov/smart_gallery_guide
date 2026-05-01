"""Legacy import path for the ``agent.agent`` module."""

from core.agent import GuideAgent
from core.settings import get_settings

# Re-exported for monkeypatch-style tests.
WEB_SEARCH_ENABLED: bool = get_settings().web_search_enabled

__all__ = ["GuideAgent", "WEB_SEARCH_ENABLED"]
