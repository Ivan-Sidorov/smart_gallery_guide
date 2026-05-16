"""ASR transcription service."""

import asyncio
import logging

from core.encoders.asr import ASREncoder
from core.settings import Settings

logger = logging.getLogger(__name__)


class ASRService:
    """Transcribe speech audio to text."""

    def __init__(
        self,
        asr_encoder: ASREncoder | None,
        settings: Settings,
    ) -> None:
        self._asr_encoder = asr_encoder
        self._settings = settings

    @property
    def is_available(self) -> bool:
        """Return whether the ASR model was loaded at startup."""
        return self._asr_encoder is not None

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Convert speech audio to text in a worker thread."""
        if self._asr_encoder is None:
            logger.warning("[asr_service] transcribe called but ASR is not loaded")
            return ""

        _ = self._settings
        return await asyncio.to_thread(self._asr_encoder.transcribe, audio_bytes)
