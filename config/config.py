import os
from pathlib import Path

from dotenv import load_dotenv

# Load env
load_dotenv()

# Project dir
PROJECT_ROOT = Path(__file__).parent.parent

# Telegram bot config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ChromaDB config
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION_EXHIBITS = os.getenv("CHROMA_COLLECTION_EXHIBITS", "exhibits")
CHROMA_COLLECTION_FAQ = os.getenv("CHROMA_COLLECTION_FAQ", "faq")

# vLLM config
VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "http://localhost:8000/v1")
VLLM_VLM_MODEL = os.getenv("VLLM_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VLLM_ASR_MODEL = os.getenv("VLLM_ASR_MODEL", "openai/whisper-large-v3")
VLLM_TTS_MODEL = os.getenv("VLLM_TTS_MODEL", "coqui/XTTS-v2")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")  # Optional API key if needed

# Encoder models
VISION_ENCODER_MODEL = os.getenv("VISION_ENCODER_MODEL", "openai/clip-vit-base-patch32")
TEXT_ENCODER_MODEL = os.getenv("TEXT_ENCODER_MODEL", "deepvk/USER-bge-m3")

# Thresholds
EXHIBIT_MATCH_THRESHOLD = float(os.getenv("EXHIBIT_MATCH_THRESHOLD", "0.7"))
FAQ_RELEVANCE_THRESHOLD = float(os.getenv("FAQ_RELEVANCE_THRESHOLD", "0.6"))

# Audio config
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", "wav")

# Data dirs
DATA_DIR = PROJECT_ROOT / "data"
EXHIBITS_DIR = DATA_DIR / "exhibits"
METADATA_DIR = DATA_DIR / "metadata"
TEMP_AUDIO_DIR = PROJECT_ROOT / "temp_audio"

DATA_DIR.mkdir(exist_ok=True)
EXHIBITS_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
