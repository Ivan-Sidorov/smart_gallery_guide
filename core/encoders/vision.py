"""Vision encoder for images and short captions (default: SigLIP base)."""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from core.settings import get_settings


class VisionEncoder:
    """Vision encoder for generating image and text embeddings."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.vision_encoder_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    @staticmethod
    def _to_pil(image: Image.Image | str | np.ndarray) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image)
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    def encode_image(self, image: Image.Image | str | np.ndarray) -> np.ndarray:
        """Encode a single image into a 1D embedding.

        Args:
            image: Image to encode.

        Returns:
            Embedding of the image.
        """
        pil_image = self._to_pil(image)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy()[0]

    def encode_images_batch(
        self, images: list[Image.Image | str | np.ndarray]
    ) -> np.ndarray:
        """Encode multiple images into embeddings.

        Args:
            images: List of images to encode.

        Returns:
            Embeddings of the images.
        """
        pil_images = [self._to_pil(img) for img in images]
        inputs = self.processor(
            images=pil_images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single string into a 1D embedding.

        Args:
            text: String to encode.

        Returns:
            Embedding of the string.
        """
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features.cpu().numpy()[0]

    def encode_texts_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple strings into embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            Embeddings of the strings.
        """
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features.cpu().numpy()

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
