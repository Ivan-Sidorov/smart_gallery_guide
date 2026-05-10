"""Smoke tests for the FastAPI backend without Postgres /Redis/Chroma."""

import os
import uuid
from datetime import datetime, timezone

import pytest

# Disable ML loading
os.environ["API_LOAD_ML"] = "false"
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
for _proxy_var in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
):
    os.environ.pop(_proxy_var, None)

from asgi_lifespan import LifespanManager  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from api.deps import (  # noqa: E402
    get_db_session,
    get_exhibit_service,
    get_faq_service,
    get_qa_service,
    get_session_service,
    get_task_service,
)
from api.main import create_app  # noqa: E402
from api.schemas.exhibits import ExhibitDTO, ExhibitSearchResultDTO  # noqa: E402
from api.schemas.faq import FAQSearchResultDTO  # noqa: E402
from api.schemas.qa import QAResponse  # noqa: E402
from api.schemas.sessions import SessionDTO  # noqa: E402
from api.schemas.tasks import TaskDTO  # noqa: E402
from core.settings import get_settings  # noqa: E402

get_settings.cache_clear()


class _FakeExhibitService:
    async def get(self, exhibit_id: str) -> ExhibitDTO | None:
        if exhibit_id == "missing":
            return None
        return ExhibitDTO(
            exhibit_id=exhibit_id,
            title="Звёздная ночь",
            author="Ван Гог",
            year="1889",
            description="ночь",
            image_path=f"data/exhibits/{exhibit_id}.jpg",
            extra={"techniq": "масло"},
        )

    async def list(self, limit: int = 100, offset: int = 0) -> list[ExhibitDTO]:
        """List exhibits ordered by created_at desc."""
        return [await self.get("ex-1"), await self.get("ex-2")]

    async def search_by_text(
        self, query: str, top_k=None, score_threshold=None
    ) -> list[ExhibitSearchResultDTO]:
        """Cascading text search: title –> description –> image-text."""
        if not query:
            return []
        dto = await self.get("ex-1")
        assert dto is not None
        return [
            ExhibitSearchResultDTO(
                exhibit_id=dto.exhibit_id,
                title=dto.title,
                similarity_score=0.91,
                metadata={"exhibit_id": dto.exhibit_id, "title": dto.title},
            )
        ]

    async def recognize_by_image(
        self, image_bytes: bytes, top_k=None, score_threshold=None
    ) -> list[ExhibitSearchResultDTO]:
        """Recognise an exhibit from a user photo (SigLIP + Chroma)."""
        if not image_bytes:
            return []
        dto = await self.get("ex-1")
        assert dto is not None
        return [
            ExhibitSearchResultDTO(
                exhibit_id=dto.exhibit_id,
                title=dto.title,
                similarity_score=0.84,
                metadata={"exhibit_id": dto.exhibit_id, "title": dto.title},
            )
        ]


class _FakeFAQService:
    def __init__(self, hits: list[FAQSearchResultDTO] | None = None) -> None:
        self.hits = hits or []

    async def search(
        self, exhibit_id: str, question: str, top_k=None, score_threshold=None
    ) -> list[FAQSearchResultDTO]:
        """Search FAQ items inside a single exhibit by question text."""
        return list(self.hits)


class _FakeTaskService:
    def __init__(self) -> None:
        self.created: list[TaskDTO] = []

    async def get(self, task_id: uuid.UUID) -> TaskDTO | None:
        """Fetch the status/result of a queued inference task."""
        for task in self.created:
            if task.id == task_id:
                return task
        return None

    async def enqueue(
        self, *, type, request, user_id=None, session_id=None, model=None
    ) -> TaskDTO:
        """Enqueue a new inference task."""
        dto = TaskDTO(
            id=uuid.uuid4(),
            type=type.value,
            status="pending",
            queued_at=datetime.now(timezone.utc),
            request=request,
        )
        self.created.append(dto)
        return dto


class _FakeQAService:
    def __init__(self, faq: _FakeFAQService, tasks: _FakeTaskService) -> None:
        self._faq = faq
        self._tasks = tasks

    async def answer_about_exhibit(
        self, *, exhibit_id, question, user_id=None, session_id=None
    ) -> QAResponse:
        """FAQ lookup with VLM fallback."""
        hits = await self._faq.search(exhibit_id, question, top_k=1)
        if hits:
            return QAResponse(mode="faq", answer=hits[0].answer)
        from db.models import TaskType

        task = await self._tasks.enqueue(
            type=TaskType.VLM_QA,
            request={"exhibit_id": exhibit_id, "question": question},
            user_id=user_id,
            session_id=session_id,
        )
        return QAResponse(mode="task", task_id=task.id)

    async def answer_about_image(
        self, *, question, image_bytes, exhibit_id=None, user_id=None, session_id=None
    ) -> QAResponse:
        """VLM Q&A over a user image."""
        from db.models import TaskType

        task = await self._tasks.enqueue(
            type=TaskType.VLM_QA,
            request={"question": question, "image_size_bytes": len(image_bytes)},
        )
        return QAResponse(mode="task", task_id=task.id)


class _FakeSessionService:
    def __init__(self) -> None:
        self.sessions: dict[uuid.UUID, SessionDTO] = {}

    async def start_or_resume(
        self,
        *,
        user_id,
        username=None,
        first_name=None,
        last_name=None,
        locale=None,
        context=None,
    ) -> SessionDTO:
        """Open/resume a session for a Telegram user."""
        for s in self.sessions.values():
            if s.user_id == user_id and s.ended_at is None:
                return s
        dto = SessionDTO(
            id=uuid.uuid4(),
            user_id=user_id,
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            context=dict(context or {}),
        )
        self.sessions[dto.id] = dto
        return dto

    async def get(self, session_id: uuid.UUID) -> SessionDTO | None:
        """Fetch a session by id."""
        return self.sessions.get(session_id)

    async def update_context(
        self, session_id: uuid.UUID, context: dict
    ) -> SessionDTO | None:
        """Replace a session's context."""
        existing = self.sessions.get(session_id)
        if existing is None:
            return None
        updated = existing.model_copy(update={"context": dict(context)})
        self.sessions[session_id] = updated
        return updated


@pytest.fixture
def fake_services():
    """Build a fresh fake-service for each test."""
    exhibits = _FakeExhibitService()
    faq = _FakeFAQService()
    tasks = _FakeTaskService()
    qa = _FakeQAService(faq=faq, tasks=tasks)
    sessions = _FakeSessionService()
    return {
        "exhibits": exhibits,
        "faq": faq,
        "tasks": tasks,
        "qa": qa,
        "sessions": sessions,
    }


@pytest.fixture
async def client(fake_services):
    """Yield an HTTP client bound to a fresh FastAPI app with fake services."""
    app = create_app()

    async def _no_db():
        yield None

    app.dependency_overrides[get_db_session] = _no_db
    app.dependency_overrides[get_exhibit_service] = lambda: fake_services["exhibits"]
    app.dependency_overrides[get_faq_service] = lambda: fake_services["faq"]
    app.dependency_overrides[get_task_service] = lambda: fake_services["tasks"]
    app.dependency_overrides[get_qa_service] = lambda: fake_services["qa"]
    app.dependency_overrides[get_session_service] = lambda: fake_services["sessions"]

    async with LifespanManager(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            trust_env=False,
        ) as ac:
            yield ac


# ---------------------------------------------------------------------- health


async def test_liveness(client):
    """Liveness probe."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_request_id_echoed(client):
    """Request-id echoed in the response headers."""
    response = await client.get("/healthz", headers={"X-Request-Id": "trace-42"})
    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "trace-42"


async def test_request_id_generated(client):
    """Request-id generated if not provided."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    rid = response.headers.get("X-Request-Id")
    assert rid and len(rid) >= 16


# -------------------------------------------------------------------- exhibits


async def test_get_exhibit_ok(client):
    """Fetch an exhibit by id."""
    response = await client.get("/v1/exhibits/ex-1")
    assert response.status_code == 200
    body = response.json()
    assert body["exhibit_id"] == "ex-1"
    assert body["title"] == "Звёздная ночь"


async def test_get_exhibit_404(client):
    """404 when exhibit not found."""
    response = await client.get("/v1/exhibits/missing")
    assert response.status_code == 404


async def test_search_exhibits_ok(client):
    """Text search: title –> description –> image-text."""
    response = await client.post(
        "/v1/exhibits/search",
        json={"query": "ночь", "top_k": 3},
    )
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["exhibit_id"] == "ex-1"
    assert items[0]["similarity_score"] == pytest.approx(0.91)


async def test_search_exhibits_validation_error(client):
    """Validation error when query is empty."""
    response = await client.post("/v1/exhibits/search", json={"query": ""})
    assert response.status_code == 422


async def test_recognize_exhibit_ok(client):
    """Recognise an exhibit from a user photo (SigLIP + Chroma)."""
    response = await client.post(
        "/v1/exhibits/recognize",
        files={"image": ("photo.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["exhibit_id"] == "ex-1"
    assert items[0]["similarity_score"] == pytest.approx(0.84)


async def test_recognize_exhibit_no_image(client):
    """Validation error when no image is provided."""
    response = await client.post("/v1/exhibits/recognize")
    assert response.status_code == 422


# ------------------------------------------------------------------------- faq


async def test_faq_search_empty(client):
    """Empty result when question is empty."""
    response = await client.post(
        "/v1/faq/search",
        json={"exhibit_id": "ex-1", "question": "когда написана?"},
    )
    assert response.status_code == 200
    assert response.json() == []


async def test_faq_search_with_hit(client, fake_services):
    """FAQ hit when question is found."""
    fake_services["faq"].hits = [
        FAQSearchResultDTO(
            exhibit_id="ex-1",
            question="когда написана?",
            answer="1889 год.",
            similarity_score=0.87,
        )
    ]
    response = await client.post(
        "/v1/faq/search",
        json={"exhibit_id": "ex-1", "question": "когда?"},
    )
    assert response.status_code == 200
    items = response.json()
    assert items[0]["answer"] == "1889 год."


# -------------------------------------------------------------------------- qa


async def test_qa_exhibit_faq_hit(client, fake_services):
    """FAQ hit when question is found."""
    fake_services["faq"].hits = [
        FAQSearchResultDTO(
            exhibit_id="ex-1",
            question="когда?",
            answer="1889.",
            similarity_score=0.95,
        )
    ]
    response = await client.post(
        "/v1/qa/exhibit",
        json={"exhibit_id": "ex-1", "question": "когда?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "faq"
    assert body["answer"] == "1889."
    assert body["task_id"] is None


async def test_qa_exhibit_falls_back_to_task(client, fake_services):
    """Falls back to VLM when FAQ is not found."""
    response = await client.post(
        "/v1/qa/exhibit",
        json={"exhibit_id": "ex-1", "question": "что-то редкое?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "task"
    assert body["task_id"]
    assert len(fake_services["tasks"].created) == 1


async def test_qa_image_creates_task(client, fake_services):
    """Create a VLM task when no FAQ is found."""
    response = await client.post(
        "/v1/qa/image",
        files={"image": ("photo.jpg", b"fake-bytes", "image/jpeg")},
        data={"question": "что это?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "task"
    assert body["task_id"]


# ------------------------------------------------------------------------ tasks


async def test_get_task_404(client):
    """404 when task not found."""
    response = await client.get(f"/v1/tasks/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_get_task_ok(client, fake_services):
    """Fetch a task by id."""
    enqueued = await fake_services["tasks"].enqueue(
        type=__import__("db.models", fromlist=["TaskType"]).TaskType.VLM_QA,
        request={"q": "x"},
    )
    response = await client.get(f"/v1/tasks/{enqueued.id}")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


# ---------------------------------------------------------------------- sessions


async def test_session_start_and_get(client):
    """Start a session and fetch it by id."""
    response = await client.post(
        "/v1/sessions",
        json={"user_id": 12345, "username": "ivan"},
    )
    assert response.status_code == 200
    sid = response.json()["id"]

    fetched = await client.get(f"/v1/sessions/{sid}")
    assert fetched.status_code == 200
    assert fetched.json()["user_id"] == 12345


async def test_session_update_context(client):
    """Update a session's context."""
    response = await client.post(
        "/v1/sessions",
        json={"user_id": 999},
    )
    sid = response.json()["id"]

    patched = await client.patch(
        f"/v1/sessions/{sid}/context",
        json={"context": {"current_exhibit_id": "ex-7"}},
    )
    assert patched.status_code == 200
    assert patched.json()["context"]["current_exhibit_id"] == "ex-7"


async def test_session_404(client):
    """404 when session not found."""
    response = await client.get(f"/v1/sessions/{uuid.uuid4()}")
    assert response.status_code == 404
