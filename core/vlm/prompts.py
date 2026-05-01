"""System prompts used by the VLM client."""

from core.settings import get_settings


def base_system_prompt() -> str:
    """Default museum-guide system prompt for direct VLM answers."""
    return get_settings().vllm_system_prompt


def search_evaluation_system_prompt() -> str:
    """Prompt that asks VLM to either answer directly or request web search."""
    return get_settings().vllm_search_eval_system_prompt


def enriched_system_prompt() -> str:
    """Prompt for the second VLM call after web search results have been fetched."""
    return get_settings().vllm_enriched_system_prompt
