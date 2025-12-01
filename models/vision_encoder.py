from typing import List, Union

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from config.config import VISION_ENCODER_MODEL


class VisionEncoder:
    """Vision encoder for generating image embeddings."""

    def __init__(self, model_name: str = VISION_ENCODER_MODEL):
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def encode_image(self, image: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        """
        Encode image into embedding.

        Args:
            image (Union[Image.Image, str, np.ndarray]): Image to encode

        Returns:
            np.ndarray: Embedding vector
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        embedding = self.model.encode(image, convert_to_numpy=True)
        return embedding

    def encode_images_batch(self, images: List[Union[Image.Image, str]]) -> np.ndarray:
        """
        Encode multiple images into embeddings.

        Args:
            images (List[Union[Image.Image, str]]): List of images to encode

        Returns:
            np.ndarray: Array of embedding vectors
        """
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)

        # Generate embeddings
        embeddings = self.model.encode(
            pil_images, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding using CLIP.

        Args:
            text (str): Text to encode

        Returns:
            np.ndarray: Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into embeddings.

        Args:
            texts (List[str]): List of texts to encode

        Returns:
            np.ndarray: Array of embedding vectors
        """
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings
