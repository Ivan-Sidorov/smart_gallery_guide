"""ASR encoder for converting speech audio to text."""

import io
import logging
from typing import Any

import numpy as np
import torch
from transformers import pipeline

from core.device import resolve_pipeline_device
from core.settings import get_settings

logger = logging.getLogger(__name__)


class ASREncoder:
    """Speech-to-text using a HuggingFace automatic-speech-recognition pipeline."""

    def __init__(
        self,
        model_name: str | None = None,
        language: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.asr_encoder_model
        self.language = (
            language if language is not None else settings.asr_encoder_language
        )
        self.device: int | str = resolve_pipeline_device(None)
        pipeline_kwargs: dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": self.model_name,
            "device": self.device,
        }
        self._pipeline = pipeline(**pipeline_kwargs)
        self._generate_kwargs: dict[str, Any] | None = None
        if self.language:
            self._generate_kwargs = {
                "language": self.language,
                "task": "transcribe",
            }

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int | None = None,
    ) -> str:
        """Transcribe raw audio bytes to plain text.

        Args:
            audio_bytes: Encoded audio bytes.
            sample_rate: Target sampling rate for the model (default 16 kHz).

        Returns:
            Recognised text, or an empty string when input is empty.
        """
        if not audio_bytes:
            return ""

        target_sr = sample_rate or 16_000
        try:
            import librosa

            audio_array, _sr = librosa.load(
                io.BytesIO(audio_bytes),
                sr=target_sr,
                mono=True,
            )
        except Exception as exc:
            logger.warning("[ASREncoder] failed to decode audio: %s", exc)
            raise ValueError("Unsupported or corrupted audio") from exc

        inputs: dict[str, Any] = {
            "array": np.asarray(audio_array, dtype=np.float32),
            "sampling_rate": target_sr,
        }
        if self._generate_kwargs:
            result = self._pipeline(inputs, generate_kwargs=self._generate_kwargs)
        else:
            result = self._pipeline(inputs)

        return _extract_text(result)

    def close(self) -> None:
        """Release model resources."""
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _extract_text(result: object) -> str:
    """Normalise pipeline output to a single string."""
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("text") or "").strip()
        return str(first).strip()
    return str(result).strip()
