from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.config import TEXT_ENCODER_MODEL


class TextEncoder:
    """Text encoder for generating text embeddings."""

    def __init__(self, model_name: str = TEXT_ENCODER_MODEL):
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding.

        Args:
            text (str): Text to encode

        Returns:
            np.ndarray: Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts (List[str]): List of texts to encode

        Returns:
            np.ndarray: Array of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
