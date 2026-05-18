"""Async HTTP client for the Smart Gallery Guide FastAPI backend."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextvars import ContextVar
from typing import Any

import httpx

from api.schemas.asr import TranscribeResponse
from api.schemas.exhibits import ExhibitDTO, ExhibitSearchResultDTO
from api.schemas.faq import FAQSearchResultDTO
from api.schemas.feedback import FeedbackDTO
from api.schemas.messages import BotReplyCreateRequest, MessageDTO
from api.schemas.qa import QAResponse
from api.schemas.sessions import SessionDTO
from api.schemas.tasks import TaskDTO

logger = logging.getLogger(__name__)

# Optional request id propagated from the adapter to the API
current_request_id: ContextVar[str | None] = ContextVar(
    "adapter_request_id", default=None
)


class APIClientError(RuntimeError):
    """Raised on non-2xx responses from the backend."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"API error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class TaskTimeoutError(RuntimeError):
    """Raised when waiting for a task exceeds the configured limit."""


class APIClient:
    """Wrapper around `httpx.AsyncClient` for the backend HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float,
        request_id_header: str,
        poll_initial_s: float,
        poll_max_s: float,
        poll_factor: float,
        poll_timeout_s: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._request_id_header = request_id_header
        self._poll_initial_s = poll_initial_s
        self._poll_max_s = poll_max_s
        self._poll_factor = poll_factor
        self._poll_timeout_s = poll_timeout_s
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            trust_env=False,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> APIClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Construct headers for the API request."""
        headers: dict[str, str] = {}
        rid = current_request_id.get()
        if rid:
            headers[self._request_id_header] = rid
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Raise an exception for non-2xx responses."""
        if response.is_success:
            return
        detail: str
        try:
            payload = response.json()
            detail = str(payload.get("detail") or payload)
        except Exception:
            detail = response.text
        raise APIClientError(response.status_code, detail)

    async def start_or_resume_session(
        self,
        *,
        user_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        locale: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> SessionDTO:
        """Open or reuse an active session for a Telegram user."""
        payload: dict[str, Any] = {"user_id": user_id}
        if username is not None:
            payload["username"] = username
        if first_name is not None:
            payload["first_name"] = first_name
        if last_name is not None:
            payload["last_name"] = last_name
        if locale is not None:
            payload["locale"] = locale
        if context is not None:
            payload["context"] = context
        response = await self._client.post(
            "/v1/sessions", json=payload, headers=self._headers()
        )
        self._raise_for_status(response)
        return SessionDTO.model_validate(response.json())

    async def update_session_context(
        self, session_id: uuid.UUID, context: dict[str, Any]
    ) -> SessionDTO:
        """Replace session context."""
        response = await self._client.patch(
            f"/v1/sessions/{session_id}/context",
            json={"context": context},
            headers=self._headers(),
        )
        self._raise_for_status(response)
        return SessionDTO.model_validate(response.json())

    async def get_exhibit(self, exhibit_id: str) -> ExhibitDTO | None:
        """Fetch an exhibit by id."""
        response = await self._client.get(
            f"/v1/exhibits/{exhibit_id}", headers=self._headers()
        )
        if response.status_code == 404:
            return None
        self._raise_for_status(response)
        return ExhibitDTO.model_validate(response.json())

    async def search_exhibits(
        self,
        query: str,
        top_k: int | None = None,
        *,
        user_id: int | None = None,
        session_id: uuid.UUID | None = None,
    ) -> list[ExhibitSearchResultDTO]:
        """Hybrid text search."""
        payload: dict[str, Any] = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        if user_id is not None:
            payload["user_id"] = user_id
        if session_id is not None:
            payload["session_id"] = str(session_id)
        response = await self._client.post(
            "/v1/exhibits/search", json=payload, headers=self._headers()
        )
        self._raise_for_status(response)
        return [ExhibitSearchResultDTO.model_validate(item) for item in response.json()]

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "voice.ogg",
        content_type: str = "audio/ogg",
    ) -> TranscribeResponse:
        """Transcribe a voice or audio attachment to plain text."""
        files = {"audio": (filename, audio_bytes, content_type)}
        response = await self._client.post(
            "/v1/asr/transcribe",
            files=files,
            headers=self._headers(),
        )
        self._raise_for_status(response)
        return TranscribeResponse.model_validate(response.json())

    async def recognize_exhibit(
        self,
        image_bytes: bytes,
        *,
        filename: str = "photo.jpg",
        content_type: str = "image/jpeg",
        top_k: int | None = None,
        user_id: int | None = None,
        session_id: uuid.UUID | None = None,
    ) -> list[ExhibitSearchResultDTO]:
        """Recognise an exhibit from a user photo."""
        files = {"image": (filename, image_bytes, content_type)}
        data: dict[str, Any] = {}
        if top_k is not None:
            data["top_k"] = str(top_k)
        if user_id is not None:
            data["user_id"] = str(user_id)
        if session_id is not None:
            data["session_id"] = str(session_id)
        response = await self._client.post(
            "/v1/exhibits/recognize",
            files=files,
            data=data,
            headers=self._headers(),
        )
        self._raise_for_status(response)
        return [ExhibitSearchResultDTO.model_validate(item) for item in response.json()]

    async def search_faq(
        self,
        *,
        exhibit_id: str,
        question: str,
        top_k: int | None = None,
    ) -> list[FAQSearchResultDTO]:
        """FAQ search inside one exhibit."""
        payload: dict[str, Any] = {"exhibit_id": exhibit_id, "question": question}
        if top_k is not None:
            payload["top_k"] = top_k
        response = await self._client.post(
            "/v1/faq/search", json=payload, headers=self._headers()
        )
        self._raise_for_status(response)
        return [FAQSearchResultDTO.model_validate(item) for item in response.json()]

    async def log_exhibit_event(
        self,
        *,
        session_id: uuid.UUID,
        user_id: int,
        exhibit_id: str,
        event: str,
        content: str | None = None,
    ) -> MessageDTO:
        """Persist a user–exhibit interaction (select or question)."""
        payload: dict[str, Any] = {
            "session_id": str(session_id),
            "user_id": user_id,
            "exhibit_id": exhibit_id,
            "event": event,
        }
        if content is not None:
            payload["content"] = content
        response = await self._client.post(
            "/v1/messages", json=payload, headers=self._headers()
        )
        self._raise_for_status(response)
        return MessageDTO.model_validate(response.json())

    async def log_bot_reply(
        self,
        *,
        session_id: uuid.UUID,
        user_id: int,
        content: str,
        exhibit_id: str | None = None,
        api_task_id: uuid.UUID | None = None,
    ) -> MessageDTO:
        """Persist an outbound bot answer."""
        payload = BotReplyCreateRequest(
            session_id=session_id,
            user_id=user_id,
            content=content,
            exhibit_id=exhibit_id,
            api_task_id=api_task_id,
        )
        response = await self._client.post(
            "/v1/messages/bot-reply",
            json=payload.model_dump(mode="json"),
            headers=self._headers(),
        )
        self._raise_for_status(response)
        return MessageDTO.model_validate(response.json())

    async def submit_feedback(
        self,
        *,
        message_id: int,
        user_id: int,
        rating: int,
        comment: str | None = None,
    ) -> FeedbackDTO:
        """Submit like/dislike for a bot reply."""
        payload: dict[str, Any] = {
            "message_id": message_id,
            "user_id": user_id,
            "rating": rating,
        }
        if comment is not None:
            payload["comment"] = comment
        response = await self._client.post(
            "/v1/feedback", json=payload, headers=self._headers()
        )
        self._raise_for_status(response)
        return FeedbackDTO.model_validate(response.json())

    async def qa_exhibit(
        self,
        *,
        exhibit_id: str,
        question: str,
        user_id: int | None = None,
        session_id: uuid.UUID | None = None,
    ) -> QAResponse:
        """FAQ-first Q&A; VLM fallback returns a task handle."""
        payload: dict[str, Any] = {"exhibit_id": exhibit_id, "question": question}
        if user_id is not None:
            payload["user_id"] = user_id
        if session_id is not None:
            payload["session_id"] = str(session_id)
        response = await self._client.post(
            "/v1/qa/exhibit", json=payload, headers=self._headers()
        )
        self._raise_for_status(response)
        return QAResponse.model_validate(response.json())

    async def qa_image(
        self,
        *,
        image_bytes: bytes,
        question: str,
        exhibit_id: str | None = None,
        user_id: int | None = None,
        session_id: uuid.UUID | None = None,
        filename: str = "photo.jpg",
        content_type: str = "image/jpeg",
    ) -> QAResponse:
        """VLM Q&A over a user photo."""
        files = {"image": (filename, image_bytes, content_type)}
        data: dict[str, Any] = {"question": question}
        if exhibit_id is not None:
            data["exhibit_id"] = exhibit_id
        if user_id is not None:
            data["user_id"] = str(user_id)
        if session_id is not None:
            data["session_id"] = str(session_id)
        response = await self._client.post(
            "/v1/qa/image", files=files, data=data, headers=self._headers()
        )
        self._raise_for_status(response)
        return QAResponse.model_validate(response.json())

    async def get_task(self, task_id: uuid.UUID) -> TaskDTO | None:
        """Fetch a task by id."""
        response = await self._client.get(
            f"/v1/tasks/{task_id}", headers=self._headers()
        )
        if response.status_code == 404:
            return None
        self._raise_for_status(response)
        return TaskDTO.model_validate(response.json())

    async def wait_for_task(
        self,
        task_id: uuid.UUID,
        *,
        on_status: Any | None = None,
    ) -> TaskDTO:
        """Poll the task until terminal status."""
        deadline = asyncio.get_running_loop().time() + self._poll_timeout_s
        delay = self._poll_initial_s
        last_status: str | None = None

        while True:
            task = await self.get_task(task_id)
            if task is None:
                raise APIClientError(404, f"Task {task_id} not found")

            if task.status != last_status:
                last_status = task.status
                if on_status is not None:
                    try:
                        await on_status(task)
                    except Exception:
                        logger.exception(
                            "[APIClient] on_status callback failed (task=%s)", task_id
                        )

            if task.status in {"done", "error"}:
                return task

            if asyncio.get_running_loop().time() >= deadline:
                raise TaskTimeoutError(
                    f"Task {task_id} did not finish within {self._poll_timeout_s:.1f}s"
                )

            await asyncio.sleep(delay)
            delay = min(self._poll_max_s, delay * self._poll_factor)
