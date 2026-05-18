"""FastAPI dependency providers."""

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.services import (
    ASRService,
    ExhibitService,
    FAQService,
    FeedbackService,
    MessageService,
    QAService,
    SessionService,
    TaskService,
)
from core.encoders.asr import ASREncoder
from core.encoders.text import TextEncoder
from core.encoders.vision import VisionEncoder
from core.settings import Settings, get_settings
from core.vector_db import VectorDatabase
from db.session import get_sessionmaker


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield a per-request async session."""
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_app_settings() -> Settings:
    """Return the singleton ``Settings`` instance."""
    return get_settings()


def get_text_encoder(request: Request) -> TextEncoder | None:
    """Return the lifespan-loaded ``TextEncoder``, or ``None`` if disabled."""
    return getattr(request.app.state, "text_encoder", None)


def get_vision_encoder(request: Request) -> VisionEncoder | None:
    """Return the lifespan-loaded ``VisionEncoder``, or ``None`` if disabled."""
    return getattr(request.app.state, "vision_encoder", None)


def get_asr_encoder(request: Request) -> ASREncoder | None:
    """Return the lifespan-loaded ``ASREncoder``, or ``None`` if disabled."""
    return getattr(request.app.state, "asr_encoder", None)


def get_vector_db(request: Request) -> VectorDatabase | None:
    """Return the lifespan-loaded ``VectorDatabase``, or ``None`` if disabled."""
    return getattr(request.app.state, "vector_db", None)


def get_exhibit_service(
    session: AsyncSession = Depends(get_db_session),
    vector_db: VectorDatabase | None = Depends(get_vector_db),
    text_encoder: TextEncoder | None = Depends(get_text_encoder),
    vision_encoder: VisionEncoder | None = Depends(get_vision_encoder),
    settings: Settings = Depends(get_app_settings),
) -> ExhibitService:
    """Construct ``ExhibitService`` for this request."""
    return ExhibitService(
        session=session,
        vector_db=vector_db,
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        settings=settings,
    )


def get_asr_service(
    asr_encoder: ASREncoder | None = Depends(get_asr_encoder),
    settings: Settings = Depends(get_app_settings),
) -> ASRService:
    """Construct ``ASRService`` for this request."""
    return ASRService(asr_encoder=asr_encoder, settings=settings)


def get_faq_service(
    vector_db: VectorDatabase | None = Depends(get_vector_db),
    text_encoder: TextEncoder | None = Depends(get_text_encoder),
    settings: Settings = Depends(get_app_settings),
) -> FAQService:
    """Construct ``FAQService`` for this request."""
    return FAQService(vector_db=vector_db, text_encoder=text_encoder, settings=settings)


def get_task_service(
    session: AsyncSession = Depends(get_db_session),
) -> TaskService:
    """Construct ``TaskService`` for this request."""
    return TaskService(session=session)


def get_qa_service(
    session: AsyncSession = Depends(get_db_session),
    faq: FAQService = Depends(get_faq_service),
    tasks: TaskService = Depends(get_task_service),
    settings: Settings = Depends(get_app_settings),
) -> QAService:
    """Construct ``QAService`` for this request."""
    return QAService(
        session=session, faq_service=faq, task_service=tasks, settings=settings
    )


def get_session_service(
    session: AsyncSession = Depends(get_db_session),
) -> SessionService:
    """Construct ``SessionService`` for this request."""
    return SessionService(session=session)


def get_message_service(
    session: AsyncSession = Depends(get_db_session),
) -> MessageService:
    """Construct `MessageService` for this request."""
    return MessageService(session=session)


def get_feedback_service(
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackService:
    """Construct `FeedbackService` for this request."""
    return FeedbackService(session=session)
