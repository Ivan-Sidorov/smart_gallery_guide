"""Legacy import path for the ``config.config`` module."""

from core.settings import PROJECT_ROOT, get_settings

_settings = get_settings()

# --------------------------------------------------------------------- Telegram
TELEGRAM_BOT_TOKEN: str = _settings.telegram_bot_token

# ---------------------------------------------------------------------- Chroma
CHROMA_PERSIST_DIR: str = _settings.chroma_persist_dir
CHROMA_COLLECTION_EXHIBITS: str = _settings.chroma_collection_exhibits
CHROMA_COLLECTION_TITLE: str = _settings.chroma_collection_title
CHROMA_COLLECTION_DESC: str = _settings.chroma_collection_desc
CHROMA_COLLECTION_FAQ: str = _settings.chroma_collection_faq

# ------------------------------------------------------------------------ vLLM
VLLM_API_BASE_URL: str = _settings.vllm_api_base_url
VLLM_VLM_MODEL: str = _settings.vllm_vlm_model
VLLM_VLM_MAX_TOKENS: int = _settings.vllm_vlm_max_tokens
VLLM_VLM_TEMPERATURE: float = _settings.vllm_vlm_temperature
VLLM_API_KEY: str = _settings.vllm_api_key
VLLM_SYSTEM_PROMPT: str = _settings.vllm_system_prompt
VLLM_SEARCH_EVAL_SYSTEM_PROMPT: str = _settings.vllm_search_eval_system_prompt
VLLM_ENRICHED_SYSTEM_PROMPT: str = _settings.vllm_enriched_system_prompt

# ------------------------------------------------------------------- Encoders
VISION_ENCODER_MODEL: str = _settings.vision_encoder_model
TEXT_ENCODER_MODEL: str = _settings.text_encoder_model

# ----------------------------------------------------------------- Thresholds
EXHIBIT_MATCH_THRESHOLD: float = _settings.exhibit_match_threshold
FAQ_RELEVANCE_THRESHOLD: float = _settings.faq_relevance_threshold
DISPLAY_SCORE_THRESHOLD: float = _settings.display_score_threshold

# ----------------------------------------------------------------- Web search
WEB_SEARCH_ENABLED: bool = _settings.web_search_enabled
WEB_SEARCH_MAX_RESULTS: int = _settings.web_search_max_results

# ----------------------------------------------------------------- Data paths
DATA_DIR = _settings.data_dir
EXHIBITS_DIR = _settings.exhibits_dir
METADATA_DIR = _settings.metadata_dir
FAQ_DIR = _settings.faq_dir

__all__ = [
    "PROJECT_ROOT",
    "TELEGRAM_BOT_TOKEN",
    "CHROMA_PERSIST_DIR",
    "CHROMA_COLLECTION_EXHIBITS",
    "CHROMA_COLLECTION_TITLE",
    "CHROMA_COLLECTION_DESC",
    "CHROMA_COLLECTION_FAQ",
    "VLLM_API_BASE_URL",
    "VLLM_VLM_MODEL",
    "VLLM_VLM_MAX_TOKENS",
    "VLLM_VLM_TEMPERATURE",
    "VLLM_API_KEY",
    "VLLM_SYSTEM_PROMPT",
    "VLLM_SEARCH_EVAL_SYSTEM_PROMPT",
    "VLLM_ENRICHED_SYSTEM_PROMPT",
    "VISION_ENCODER_MODEL",
    "TEXT_ENCODER_MODEL",
    "EXHIBIT_MATCH_THRESHOLD",
    "FAQ_RELEVANCE_THRESHOLD",
    "DISPLAY_SCORE_THRESHOLD",
    "WEB_SEARCH_ENABLED",
    "WEB_SEARCH_MAX_RESULTS",
    "DATA_DIR",
    "EXHIBITS_DIR",
    "METADATA_DIR",
    "FAQ_DIR",
]
