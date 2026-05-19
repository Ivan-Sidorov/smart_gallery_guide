"""Async VLM client for OpenAI-compatible endpoints."""

import base64
import re
from dataclasses import dataclass
from io import BytesIO

from openai import AsyncOpenAI
from PIL import Image

from core.settings import get_settings
from core.vlm.prompts import base_system_prompt, search_evaluation_system_prompt
from core.vlm.response import strip_vlm_reasoning


class VLMError(RuntimeError):
    """VLM API call failed."""


VLM_NO_ANSWER_TEXT = "Не удалось получить ответ от модели."


@dataclass
class SearchEvaluation:
    """Result of VLM deciding whether external search is needed."""

    needs_search: bool
    search_query: str = ""
    answer: str = ""


class VLM:
    """Vision Language Model client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        self.api_base_url = api_base_url or settings.vllm_api_base_url
        self.model_name = model_name or settings.vllm_vlm_model
        resolved_key = api_key if api_key is not None else settings.vllm_api_key

        self.client = AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=resolved_key if resolved_key else "mock_key",
        )

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    async def answer_question(
        self,
        image: Image.Image,
        question: str,
        context: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Answer a question about an image.

        Args:
            image: Image to answer the question about.
            question: Question to answer.
            context: Context to use for the answer.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature to use for the answer.
            system_prompt: System prompt to use for the answer.

        Returns:
            Answer to the question.
        """
        settings = get_settings()
        max_tokens = (
            max_tokens if max_tokens is not None else settings.vllm_vlm_max_tokens
        )
        temperature = (
            temperature if temperature is not None else settings.vllm_vlm_temperature
        )
        system_prompt = (
            system_prompt if system_prompt is not None else base_system_prompt()
        )

        image_url = f"data:image/png;base64,{self._image_to_base64(image)}"

        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content
                if not answer:
                    return VLM_NO_ANSWER_TEXT
                return strip_vlm_reasoning(answer)
            return VLM_NO_ANSWER_TEXT
        except Exception as e:
            raise VLMError(str(e)) from e

    async def evaluate_search_need(
        self,
        image: Image.Image,
        question: str,
        context: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> SearchEvaluation:
        """Ask VLM whether it can answer from local context or needs web search.

        Args:
            image: Image to answer the question about.
            question: Question to answer.
            context: Context to use for the answer.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature to use for the answer.

        Returns:
            SearchEvaluation object containing the decision and search query.
        """
        settings = get_settings()
        max_tokens = (
            max_tokens if max_tokens is not None else settings.vllm_vlm_max_tokens
        )
        temperature = (
            temperature if temperature is not None else settings.vllm_vlm_temperature
        )

        image_url = f"data:image/png;base64,{self._image_to_base64(image)}"

        prompt = (
            f"Контекст: {context}\n\nВопрос: {question}"
            if context
            else f"Вопрос: {question}"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": search_evaluation_system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            raw = ""
            if response.choices and len(response.choices) > 0:
                raw = strip_vlm_reasoning(response.choices[0].message.content or "")
            return self._parse_search_evaluation(raw)
        except Exception as e:
            raise VLMError(str(e)) from e

    @staticmethod
    def _parse_search_evaluation(raw: str) -> SearchEvaluation:
        """Parse VLM output into SearchEvaluation (ANSWER: / SEARCH: format)."""
        search_match = re.match(r"(?i)^SEARCH:\s*(.+)", raw, re.DOTALL)
        if search_match:
            return SearchEvaluation(
                needs_search=True, search_query=search_match.group(1).strip()
            )

        answer_match = re.match(r"(?i)^ANSWER:\s*(.+)", raw, re.DOTALL)
        if answer_match:
            return SearchEvaluation(
                needs_search=False,
                answer=strip_vlm_reasoning(answer_match.group(1)),
            )

        return SearchEvaluation(needs_search=False, answer=strip_vlm_reasoning(raw))

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> "VLM":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
