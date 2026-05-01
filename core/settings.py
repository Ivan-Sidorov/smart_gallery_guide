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
        default="http://localhost:8000/v1", validation_alias="VLLM_API_BASE_URL"
    )
    vllm_vlm_model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct", validation_alias="VLLM_VLM_MODEL"
    )
    vllm_vlm_max_tokens: int = Field(
        default=500, validation_alias="VLLM_VLM_MAX_TOKENS"
    )
    vllm_vlm_temperature: float = Field(
        default=0.7, validation_alias="VLLM_VLM_TEMPERATURE"
    )
    vllm_api_key: str = Field(default="", validation_alias="VLLM_API_KEY")
    vllm_system_prompt: str = Field(
        default="Ты – музейный гид. Отвечай кратко и по делу (1–3 предложения). ",
        validation_alias="VLLM_SYSTEM_PROMPT",
    )

    # --------------------------------------------------------------- Encoder models
    vision_encoder_model: str = Field(
        default="google/siglip-base-patch16-224",
        validation_alias="VISION_ENCODER_MODEL",
    )
    text_encoder_model: str = Field(
        default="deepvk/USER-bge-m3", validation_alias="TEXT_ENCODER_MODEL"
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

    # -------------------------------------------------------------- System prompts
    @property
    def vllm_search_eval_system_prompt(self) -> str:
        """Prompt that asks VLM to either answer or formulate a web search query."""
        return (
            "Ты – музейный гид-эксперт. Тебе дан вопрос посетителя о музейном экспонате, "
            "изображение экспоната и справочная информация из базы данных музея.\n\n"
            "Определи, можешь ли ты дать точный и полный ответ, опираясь ТОЛЬКО на "
            "предоставленные данные и изображение. Ни в каком случае не отвечай из собственных "
            "знаний, даже если уверен в ответе. Запрещено использовать факты, не апедставленные "
            "в контексте.Истинной считается только информация, которая есть в предоставленном "
            "контексте. Если предоставленной информации недостаточно, отвечай, что не знаешь.\n\n"
            "Если ДА – ответь в формате:\nANSWER: <твой ответ, 1–3 предложения>\n\n"
            "Если для точного ответа нужна дополнительная информация – сформулируй один "
            "поисковый запрос на русском языке в формате:\nSEARCH: <поисковый запрос>\n\n"
            "Поисковый запрос должен включать название произведения и/или имя автора "
            "вместе с темой вопроса. Используй строго один из двух форматов ответа."
        )

    @property
    def vllm_enriched_system_prompt(self) -> str:
        """System prompt for the second VLM call enriched with web search snippets."""
        return (
            self.vllm_system_prompt + "\n\n"
            "Используй информацию из базы данных музея как основной источник. "
            "Дополнительные сведения из интернета используй как вспомогательный источник. "
            "Если источники противоречат друг другу, отдавай приоритет музейным данным."
        )


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
