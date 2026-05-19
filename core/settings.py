"""Application settings for the Smart Gallery Guide service."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the Smart Gallery Guide service."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ Telegram
    telegram_bot_token: str = Field(default="", validation_alias="TELEGRAM_BOT_TOKEN")

    # -------------------------------------------------------------------- Chroma
    chroma_persist_dir: str = Field(
        default_factory=lambda: str(PROJECT_ROOT / "chroma_store"),
        validation_alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection_exhibits: str = Field(
        default="exhibits", validation_alias="CHROMA_COLLECTION_EXHIBITS"
    )
    chroma_collection_title: str = Field(
        default="exhibits_title", validation_alias="CHROMA_COLLECTION_TITLE"
    )
    chroma_collection_desc: str = Field(
        default="exhibits_desc", validation_alias="CHROMA_COLLECTION_DESC"
    )
    chroma_collection_faq: str = Field(
        default="faq", validation_alias="CHROMA_COLLECTION_FAQ"
    )

    # ----------------------------------------------------------------------- vLLM
    vllm_api_base_url: str = Field(
        default="http://localhost:2828/v1", validation_alias="VLLM_API_BASE_URL"
    )
    vllm_vlm_model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct", validation_alias="VLLM_VLM_MODEL"
    )
    vllm_vlm_max_tokens: int = Field(
        default=6144, validation_alias="VLLM_VLM_MAX_TOKENS"
    )
    vllm_vlm_temperature: float = Field(
        default=0.7, validation_alias="VLLM_VLM_TEMPERATURE"
    )
    # When True, pass enable_thinking=False to vLLM (Qwen3 reasoning models).
    vllm_vlm_disable_thinking: bool = Field(
        default=True, validation_alias="VLLM_VLM_DISABLE_THINKING"
    )
    vllm_api_key: str = Field(default="", validation_alias="VLLM_API_KEY")

    # --------------------------------------------------------------- Encoder models
    vision_encoder_model: str = Field(
        default="google/siglip-base-patch16-224",
        validation_alias="VISION_ENCODER_MODEL",
    )
    text_encoder_model: str = Field(
        default="deepvk/USER-bge-m3", validation_alias="TEXT_ENCODER_MODEL"
    )
    asr_encoder_model: str = Field(
        default="openai/whisper-small", validation_alias="ASR_ENCODER_MODEL"
    )
    asr_encoder_language: str = Field(
        default="russian", validation_alias="ASR_ENCODER_LANGUAGE"
    )

    # -------------------------------------------------------------- Retrieval thresholds
    exhibit_match_threshold: float = Field(
        default=0.6, validation_alias="EXHIBIT_MATCH_THRESHOLD"
    )
    faq_relevance_threshold: float = Field(
        default=0.6, validation_alias="FAQ_RELEVANCE_THRESHOLD"
    )
    display_score_threshold: float = Field(
        default=0.5, validation_alias="DISPLAY_SCORE_THRESHOLD"
    )

    # -------------------------------------------------------------------- Web search
    web_search_enabled: bool = Field(
        default=True, validation_alias="WEB_SEARCH_ENABLED"
    )
    web_search_max_results: int = Field(
        default=5, validation_alias="WEB_SEARCH_MAX_RESULTS"
    )

    # ---------------------------------------------------------------------- PostgreSQL
    database_url: str = Field(
        default="postgresql+asyncpg://smart_guide:smart_guide@localhost:5432/smart_guide",
        validation_alias="DATABASE_URL",
    )
    database_echo: bool = Field(default=False, validation_alias="DATABASE_ECHO")
    database_pool_size: int = Field(default=10, validation_alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(
        default=20, validation_alias="DATABASE_MAX_OVERFLOW"
    )

    # ---------------------------------------------------------------------- VLM worker
    # How often the worker polls inference_tasks when the queue is empty
    worker_poll_interval_s: float = Field(
        default=1.0, validation_alias="WORKER_POLL_INTERVAL_S"
    )
    # Max number of VLM tasks one worker processes concurrently
    worker_concurrency: int = Field(default=2, validation_alias="WORKER_CONCURRENCY")
    # Running tasks older than this are considered stuck and re-queued.
    worker_stale_timeout_s: int = Field(
        default=300, validation_alias="WORKER_STALE_TIMEOUT_S"
    )
    # How often the recovery sweep runs.
    worker_stale_check_interval_s: int = Field(
        default=60, validation_alias="WORKER_STALE_CHECK_INTERVAL_S"
    )

    # ---------------------------------------------------------------------- FastAPI
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8081, validation_alias="API_PORT")
    api_log_level: str = Field(default="info", validation_alias="API_LOG_LEVEL")
    api_request_id_header: str = Field(
        default="X-Request-Id", validation_alias="API_REQUEST_ID_HEADER"
    )
    # When True, lifespan loads TextEncoder + VectorDatabase
    api_load_ml: bool = Field(default=True, validation_alias="API_LOAD_ML")
    # Default top-K for search endpoints
    api_default_top_k: int = Field(default=5, validation_alias="API_DEFAULT_TOP_K")
    api_max_top_k: int = Field(default=20, validation_alias="API_MAX_TOP_K")

    # ---------------------------------------------------------------- Telegram adapter
    # Base URL of the FastAPI backend the Telegram adapter calls over HTTP
    adapter_backend_url: str = Field(
        default="http://localhost:8081", validation_alias="ADAPTER_BACKEND_URL"
    )
    # HTTP timeout (seconds) for adapter -> API calls
    adapter_http_timeout_s: float = Field(
        default=30.0, validation_alias="ADAPTER_HTTP_TIMEOUT_S"
    )
    # Initial delay between polling /v1/tasks/{id}
    adapter_task_poll_initial_s: float = Field(
        default=0.5, validation_alias="ADAPTER_TASK_POLL_INITIAL_S"
    )
    # Maximum delay between polls
    adapter_task_poll_max_s: float = Field(
        default=3.0, validation_alias="ADAPTER_TASK_POLL_MAX_S"
    )
    # Multiplier applied to the polling delay after each empty/pending poll
    adapter_task_poll_factor: float = Field(
        default=1.5, validation_alias="ADAPTER_TASK_POLL_FACTOR"
    )
    # Total time for waiting on a single VLM task before cancelling
    adapter_task_poll_timeout_s: float = Field(
        default=120.0, validation_alias="ADAPTER_TASK_POLL_TIMEOUT_S"
    )
    # Header the adapter uses to propagate request-id to the API
    adapter_request_id_header: str = Field(
        default="X-Request-Id", validation_alias="ADAPTER_REQUEST_ID_HEADER"
    )
    # Logging level for the adapter process
    adapter_log_level: str = Field(default="info", validation_alias="ADAPTER_LOG_LEVEL")

    # -------------------------------------------------------------------- Data paths
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def exhibits_dir(self) -> Path:
        return self.data_dir / "exhibits"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def faq_dir(self) -> Path:
        return self.data_dir / "faq"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide singleton of `Settings` object."""
    return Settings()


def _ensure_data_dirs(settings: Settings) -> None:
    """Create data directories if they do not exist."""
    for path in (
        settings.data_dir,
        settings.exhibits_dir,
        settings.metadata_dir,
        settings.faq_dir,
    ):
        path.mkdir(exist_ok=True)


_ensure_data_dirs(get_settings())
