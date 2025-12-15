from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config.config import VISION_ENCODER_MODEL


class VisionEncoder:
    """Vision encoder for generating image and text embeddings."""

    def __init__(self, model_name: str = VISION_ENCODER_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @staticmethod
    def _to_pil(image: Union[Image.Image, str, np.ndarray]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image)
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    def encode_image(self, image: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        """
        Encode image into embedding.

        Args:
            image (Union[Image.Image, str, np.ndarray]): Image to encode

        Returns:
            np.ndarray: Embedding vector
        """
        pil_image = self._to_pil(image)

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy()[0]

    def encode_images_batch(
        self, images: List[Union[Image.Image, str, np.ndarray]]
    ) -> np.ndarray:
        """
        Encode multiple images into embeddings.

        Args:
            images (List[Union[Image.Image, str, np.ndarray]]): List of images to encode

        Returns:
            np.ndarray: Array of embedding vectors
        """
        pil_images = [self._to_pil(img) for img in images]

        inputs = self.processor(
            images=pil_images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding using CLIP/SigLIP.

        Args:
            text (str): Text to encode

        Returns:
            np.ndarray: Embedding vector
        """
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features.cpu().numpy()[0]

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into embeddings.

        Args:
            texts (List[str]): List of texts to encode

        Returns:
            np.ndarray: Array of embedding vectors
        """
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features.cpu().numpy()
