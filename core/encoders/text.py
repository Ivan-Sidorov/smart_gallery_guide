"""Text encoder for generating text embeddings."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from core.settings import get_settings


class TextEncoder:
    """Text encoder for generating text embeddings."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.text_encoder_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name).to(self.device)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single string into a 1D embedding.

        Args:
            text: String to encode.

        Returns:
            Embedding of the string.
        """
        return self.model.encode(text, convert_to_numpy=True)

    def encode_texts_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple strings into embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            Embeddings of the strings.
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def close(self) -> None:
        """Release GPU memory held by the underlying model."""
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.to("cpu")
        except Exception:
            pass
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
