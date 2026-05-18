"""FastAPI application factory + lifespan."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy import text

from api.logging import bind_request_id, configure_logging, reset_request_id
from api.routers import asr, exhibits, faq, health, messages, qa, sessions, tasks
from core.settings import Settings, get_settings
from db.session import get_engine

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("api.access")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize singletons on startup and release them on shutdown."""
    settings: Settings = get_settings()
    configure_logging(settings.api_log_level)
    logger.info("[API lifespan] starting (load_ml=%s)", settings.api_load_ml)

    app.state.settings = settings
    app.state.text_encoder = None
    app.state.vision_encoder = None
    app.state.asr_encoder = None
    app.state.vector_db = None

    if settings.api_load_ml:
        from core.encoders.asr import ASREncoder
        from core.encoders.text import TextEncoder
        from core.encoders.vision import VisionEncoder
        from core.vector_db import VectorDatabase

        logger.info(
            "[API lifespan] loading TextEncoder (%s)", settings.text_encoder_model
        )
        app.state.text_encoder = TextEncoder()
        logger.info(
            "[API lifespan] loading VisionEncoder (%s)", settings.vision_encoder_model
        )
        app.state.vision_encoder = VisionEncoder()
        logger.info(
            "[API lifespan] loading ASREncoder (%s)", settings.asr_encoder_model
        )
        app.state.asr_encoder = ASREncoder()
        logger.info(
            "[API lifespan] opening VectorDatabase at %s", settings.chroma_persist_dir
        )
        app.state.vector_db = VectorDatabase()

    engine = get_engine()
    app.state.db_engine = engine
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            await conn.execute(text("SELECT 1 FROM exhibits LIMIT 0"))
        logger.info("[API lifespan] Postgres ping and schema check successful")
    except Exception as exc:
        logger.error(
            "[API lifespan] Postgres connection or schema check failed: %s. "
            "Ensure migrations are applied.",
            exc,
        )
        raise

    try:
        yield
    finally:
        logger.info("[API lifespan] shutting down")
        for attr_name in ("text_encoder", "vision_encoder", "asr_encoder"):
            component = getattr(app.state, attr_name, None)
            if component is None:
                continue
            try:
                component.close()
            except Exception as exc:
                logger.warning("[API lifespan] error closing %s: %s", attr_name, exc)
        try:
            await engine.dispose()
        except Exception as exc:
            logger.warning("[API lifespan] error destroying DB engine: %s", exc)


def create_app() -> FastAPI:
    """Build the FastAPI app."""
    settings = get_settings()
    app = FastAPI(
        title="Smart Gallery Guide API",
        version="0.2.0",
        description=(
            "HTTP backend for the Smart Gallery Guide. Owns text retrieval "
            "(bge-m3 + Chroma + BM25), speech-to-text (Whisper ASR), "
            "image-based exhibit recognition (SigLIP), "
            "exhibit metadata, sessions and the inference-task ledger. Heavy "
            "VLM Q&A is delegated to a background worker."
        ),
        lifespan=lifespan,
    )

    request_id_header = settings.api_request_id_header

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        """Emit one structured log line per request with method/path/status/latency."""
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            access_logger.exception(
                "[API access] request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        access_logger.info(
            "[API access] request handled",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )
        return response

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        """Bind X-Request-Id to the request context and echo it on the response."""
        incoming = request.headers.get(request_id_header)
        rid = incoming or uuid.uuid4().hex
        token = bind_request_id(rid)
        request.state.request_id = rid
        try:
            response = await call_next(request)
        finally:
            reset_request_id(token)
        response.headers[request_id_header] = rid
        return response

    app.include_router(health.router)
    app.include_router(asr.router)
    app.include_router(exhibits.router)
    app.include_router(faq.router)
    app.include_router(qa.router)
    app.include_router(tasks.router)
    app.include_router(sessions.router)
    app.include_router(messages.router)

    return app


app = create_app()
