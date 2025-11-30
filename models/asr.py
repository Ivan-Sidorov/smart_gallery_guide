from pathlib import Path
from typing import Optional, Union


class ASR:
    """Automatic Speech Recognition client."""

    def __init__(self):
        # TODO: asr initialization
        pass

    async def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = "ru",
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path (Union[str, Path]): Path to audio file
            language (Optional[str]): Language code. Defaults to "ru".

        Returns:
            str: Transcribed text
        """
        # TODO: audio transcription
        raise NotImplementedError("ASR transcription not implemented yet")

    async def close(self):
        """Close ASR client."""
        # TODO: cleanup
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
