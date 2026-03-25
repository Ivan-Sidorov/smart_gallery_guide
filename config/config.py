import os
from pathlib import Path

from dotenv import load_dotenv

# Load env
load_dotenv()

# Project dir
PROJECT_ROOT = Path(__file__).parent.parent

# Telegram bot config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_store"))
CHROMA_COLLECTION_EXHIBITS = os.getenv("CHROMA_COLLECTION_EXHIBITS", "exhibits")
CHROMA_COLLECTION_TITLE = os.getenv("CHROMA_COLLECTION_TITLE", "exhibits_title")
CHROMA_COLLECTION_DESC = os.getenv("CHROMA_COLLECTION_DESC", "exhibits_desc")
CHROMA_COLLECTION_FAQ = os.getenv("CHROMA_COLLECTION_FAQ", "faq")

# vLLM config
VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "http://localhost:8000/v1")
VLLM_VLM_MODEL = os.getenv("VLLM_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VLLM_VLM_MAX_TOKENS = int(os.getenv("VLLM_VLM_MAX_TOKENS", "500"))
VLLM_VLM_TEMPERATURE = float(os.getenv("VLLM_VLM_TEMPERATURE", "0.7"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
VLLM_SYSTEM_PROMPT = os.getenv(
    "VLLM_SYSTEM_PROMPT",
    "Ты – музейный гид. Отвечай кратко и по делу (1–3 предложения). ",
)

VLLM_SEARCH_EVAL_SYSTEM_PROMPT = (
    "Ты – музейный гид-эксперт. Тебе дан вопрос посетителя о музейном экспонате, "
    "изображение экспоната и справочная информация из базы данных музея.\n\n"
    "Определи, можешь ли ты дать точный и полный ответ, опираясь ТОЛЬКО на "
    "предоставленные данные и изображение.\n\n"
    "Если ДА – ответь в формате:\nANSWER: <твой ответ, 1–3 предложения>\n\n"
    "Если для точного ответа нужна дополнительная информация – сформулируй один "
    "поисковый запрос на русском языке в формате:\nSEARCH: <поисковый запрос>\n\n"
    "Поисковый запрос должен включать название произведения и/или имя автора "
    "вместе с темой вопроса. Используй строго один из двух форматов ответа."
)

VLLM_ENRICHED_SYSTEM_PROMPT = (
    VLLM_SYSTEM_PROMPT + "\n\n"
    "Используй информацию из базы данных музея как основной источник. "
    "Дополнительные сведения из интернета используй как вспомогательный источник. "
    "Если источники противоречат друг другу, отдавай приоритет музейным данным."
)

# Encoder models
VISION_ENCODER_MODEL = os.getenv(
    "VISION_ENCODER_MODEL", "google/siglip-base-patch16-224"
)
TEXT_ENCODER_MODEL = os.getenv("TEXT_ENCODER_MODEL", "deepvk/USER-bge-m3")

# Thresholds
EXHIBIT_MATCH_THRESHOLD = float(os.getenv("EXHIBIT_MATCH_THRESHOLD", "0.6"))
FAQ_RELEVANCE_THRESHOLD = float(os.getenv("FAQ_RELEVANCE_THRESHOLD", "0.6"))
DISPLAY_SCORE_THRESHOLD = float(os.getenv("DISPLAY_SCORE_THRESHOLD", "0.5"))

# Web search
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in ("true", "1")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# Data dirs
DATA_DIR = PROJECT_ROOT / "data"
EXHIBITS_DIR = DATA_DIR / "exhibits"
METADATA_DIR = DATA_DIR / "metadata"
FAQ_DIR = DATA_DIR / "faq"

DATA_DIR.mkdir(exist_ok=True)
EXHIBITS_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)
FAQ_DIR.mkdir(exist_ok=True)
