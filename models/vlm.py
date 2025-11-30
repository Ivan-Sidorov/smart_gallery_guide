"""Vision Language Model client using vLLM API."""

import base64
from io import BytesIO
from typing import Optional

from openai import AsyncOpenAI
from PIL import Image

from config.config import (
    VLLM_API_BASE_URL,
    VLLM_API_KEY,
    VLLM_VLM_MAX_TOKENS,
    VLLM_VLM_MODEL,
    VLLM_VLM_TEMPERATURE,
)


class VLM:
    """Vision Language Model client using vLLM API."""

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize VLM client.

        Args:
            api_base_url: vLLM API base URL. Defaults to config.
            model_name: Model name. Defaults to config.
            api_key: API key. Defaults to config.
        """
        self.api_base_url = api_base_url or VLLM_API_BASE_URL
        self.model_name = model_name or VLLM_VLM_MODEL
        api_key = api_key or VLLM_API_KEY

        self.client = AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=api_key if api_key else "mock_key",
        )

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image

        Returns:
            Base64 encoded image string
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    async def answer_question(
        self,
        image: Image.Image,
        question: str,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image (Image.Image): Image to analyze
            question (str): Question about the image
            context (Optional[str]): Context about the exhibit
            max_tokens (Optional[int]): Maximum tokens in response. Defaults to config.
            temperature (Optional[float]): Sampling temperature. Defaults to config.

        Returns:
            str: Answer string
        """
        max_tokens = max_tokens if max_tokens is not None else VLLM_VLM_MAX_TOKENS
        temperature = temperature if temperature is not None else VLLM_VLM_TEMPERATURE

        # Convert image to base64
        image_base64 = self._image_to_base64(image)
        image_url = f"data:image/png;base64,{image_base64}"

        # Build prompt with context if provided
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        try:
            # Use OpenAI-compatible API format for vision models
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract answer from response
            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content
                return (
                    answer.strip() if answer else "Не удалось получить ответ от модели."
                )
            else:
                return "Не удалось получить ответ от модели."
        except Exception as e:
            return f"Ошибка при обращении к VLM API: {str(e)}"

    async def close(self):
        """Close VLM client."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
