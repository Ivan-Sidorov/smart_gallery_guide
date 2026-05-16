"""Smoke tests for the Telegram adapter's HTTP client."""

import uuid
from datetime import datetime, timezone

import httpx
import pytest

from adapters.telegram.api_client import (
    APIClient,
    APIClientError,
    TaskTimeoutError,
    current_request_id,
)
from api.schemas.qa import QAResponse


def _build_client(handler) -> APIClient:
    """Construct an APIClient (underlying transport is mocked)."""
    client = APIClient(
        base_url="http://api.test",
        timeout_s=5.0,
        request_id_header="X-Request-Id",
        poll_initial_s=0.001,
        poll_max_s=0.01,
        poll_factor=2.0,
        poll_timeout_s=1.0,
    )
    transport = httpx.MockTransport(handler)
    client._client = httpx.AsyncClient(
        base_url="http://api.test", transport=transport, trust_env=False
    )
    return client


async def test_start_or_resume_session_roundtrip() -> None:
    """Adapter sends the user payload and parses the SessionDTO response."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = request.read().decode()
        body = {
            "id": "11111111-1111-1111-1111-111111111111",
            "user_id": 42,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "context": {"current_exhibit_id": "ex-1"},
        }
        return httpx.Response(200, json=body)

    async with _build_client(handler) as client:
        session = await client.start_or_resume_session(
            user_id=42, username="ivan", first_name="Иван"
        )

    assert captured["method"] == "POST"
    assert captured["url"] == "http://api.test/v1/sessions"
    assert '"user_id":42' in str(captured["body"]).replace(" ", "")
    assert session.user_id == 42
    assert session.context["current_exhibit_id"] == "ex-1"


async def test_get_exhibit_404_returns_none() -> None:
    """404 -> None, not an exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Exhibit not found"})

    async with _build_client(handler) as client:
        result = await client.get_exhibit("missing")

    assert result is None


async def test_transcribe_audio_sends_multipart() -> None:
    """Voice bytes go to /v1/asr/transcribe as multipart."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"text": "привет"})

    async with _build_client(handler) as client:
        result = await client.transcribe_audio(b"\x00\x01", filename="voice.ogg")

    assert captured["url"] == "http://api.test/v1/asr/transcribe"
    assert "multipart/form-data" in str(captured["content_type"])
    assert result.text == "привет"


async def test_recognize_exhibit_sends_multipart() -> None:
    """Photo bytes go to /v1/exhibits/recognize as multipart."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(
            200,
            json=[
                {
                    "exhibit_id": "ex-1",
                    "title": "T",
                    "similarity_score": 0.7,
                    "metadata": {},
                }
            ],
        )

    async with _build_client(handler) as client:
        hits = await client.recognize_exhibit(b"\x89PNG", filename="p.png")

    assert captured["url"] == "http://api.test/v1/exhibits/recognize"
    assert "multipart/form-data" in str(captured["content_type"])
    assert hits[0].exhibit_id == "ex-1"


async def test_qa_exhibit_returns_response() -> None:
    """qa_exhibit deserialises into a QAResponse."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "task",
                "answer": None,
                "task_id": "22222222-2222-2222-2222-222222222222",
            },
        )

    async with _build_client(handler) as client:
        response = await client.qa_exhibit(exhibit_id="ex-1", question="q")

    assert isinstance(response, QAResponse)
    assert response.mode == "task"
    assert str(response.task_id) == "22222222-2222-2222-2222-222222222222"


async def test_wait_for_task_polls_until_done() -> None:
    """wait_for_task polls /v1/tasks/{id} until status='done'."""
    tid = uuid.UUID("33333333-3333-3333-3333-333333333333")
    states = iter(["pending", "running", "done"])
    seen_statuses: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(states, "done")
        seen_statuses.append(status)
        return httpx.Response(
            200,
            json={
                "id": str(tid),
                "type": "vlm_qa",
                "status": status,
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "finished_at": None,
                "request": {},
                "result": {"answer": "ok"} if status == "done" else None,
                "error": None,
            },
        )

    callback_invocations: list[str] = []

    async def _cb(task) -> None:
        callback_invocations.append(task.status)

    async with _build_client(handler) as client:
        result = await client.wait_for_task(tid, on_status=_cb)

    assert result.status == "done"
    assert seen_statuses == ["pending", "running", "done"]
    assert callback_invocations == ["pending", "running", "done"]
    assert result.result == {"answer": "ok"}


async def test_wait_for_task_timeout() -> None:
    """Tasks that never finish raise TaskTimeoutError."""
    tid = uuid.UUID("44444444-4444-4444-4444-444444444444")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": str(tid),
                "type": "vlm_qa",
                "status": "running",
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "finished_at": None,
                "request": {},
                "result": None,
                "error": None,
            },
        )

    client = _build_client(handler)
    # Small timeout for the test.
    client._poll_timeout_s = 0.05
    client._poll_initial_s = 0.01
    client._poll_max_s = 0.02
    async with client:
        with pytest.raises(TaskTimeoutError):
            await client.wait_for_task(tid)


async def test_api_client_error_on_5xx() -> None:
    """Non-2xx responses raise APIClientError with the parsed detail."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "VLM down"})

    async with _build_client(handler) as client:
        with pytest.raises(APIClientError) as exc_info:
            await client.qa_exhibit(exhibit_id="ex-1", question="q")

    assert exc_info.value.status_code == 503
    assert "VLM down" in str(exc_info.value)


async def test_request_id_header_is_propagated() -> None:
    """When a request-id is bound, it is sent in the configured header."""
    seen_header: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_header["value"] = request.headers.get("X-Request-Id")
        return httpx.Response(
            200,
            json={
                "id": "55555555-5555-5555-5555-555555555555",
                "user_id": 1,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "ended_at": None,
                "context": {},
            },
        )

    token = current_request_id.set("tg-update-77")
    try:
        async with _build_client(handler) as client:
            await client.start_or_resume_session(user_id=1)
    finally:
        current_request_id.reset(token)

    assert seen_header["value"] == "tg-update-77"
