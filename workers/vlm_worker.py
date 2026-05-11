"""VLM worker: claims tasks one at a time, calls the VLM, and writes the result back."""

import asyncio
import base64
import io
import logging
import os
import signal
import socket
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from core.search.web import WebSearchService
from core.settings import Settings, get_settings
from core.vlm.client import VLM
from core.vlm.prompts import enriched_system_prompt
from db.models import TaskType
from db.repositories import ExhibitRepository, InferenceTaskRepository
from db.session import get_engine, session_scope

logger = logging.getLogger(__name__)


def _worker_id() -> str:
    """Stable identifier written to `inference_tasks.worker`."""
    return f"vlm-worker@{socket.gethostname()}:{os.getpid()}"


def _build_exhibit_context(exhibit: Any) -> str:
    """Render exhibit row into a context block for the VLM."""
    parts = [f"Название: {exhibit.title}"]
    if exhibit.author:
        parts.append(f"Автор: {exhibit.author}")
    if exhibit.year:
        parts.append(f"Год: {exhibit.year}")

    extra = dict(exhibit.extra or {})
    for key, label in (
        ("material", "Материал"),
        ("school", "Школа/страна"),
        ("techniq", "Техника"),
        ("place", "Место"),
        ("epoque", "Эпоха"),
    ):
        value = extra.get(key)
        if value:
            parts.append(f"{label}: {value}")

    if exhibit.description:
        parts.append(f"Описание: {exhibit.description}")
    if anotation := extra.get("anotation"):
        parts.append(f"Экспертный комментарий: {anotation}")
    return "\n".join(parts)


def _decode_image(image_b64: str) -> Image.Image:
    """Decode base64-encoded image."""
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _load_exhibit_image(
    image_path: str | None, settings: Settings
) -> Image.Image | None:
    """Load exhibit image from the data volume."""
    if not image_path:
        return None
    path = Path(image_path)
    if not path.is_absolute():
        path = settings.project_root / path
    if not path.exists():
        logger.warning("[vlm-worker] exhibit image not found at %s", path)
        return None
    with Image.open(path) as img:
        return img.convert("RGB")


async def _resolve_qa_inputs(
    session: AsyncSession,
    request: dict[str, Any],
    settings: Settings,
) -> tuple[Image.Image | None, str, str]:
    """Pull image + context + question from the task payload."""
    question = str(request.get("question", "")).strip()
    exhibit_id = request.get("exhibit_id")

    image: Image.Image | None = None
    if request.get("image_b64"):
        image = _decode_image(request["image_b64"])

    context = ""
    if exhibit_id:
        repo = ExhibitRepository(session)
        exhibit = await repo.get(str(exhibit_id))
        if exhibit is None:
            logger.warning(
                "[vlm-worker] exhibit_id=%s referenced by task but not in DB",
                exhibit_id,
            )
        else:
            context = _build_exhibit_context(exhibit)
            if image is None:
                image = _load_exhibit_image(exhibit.image_path, settings)

    return image, context, question


async def _run_vlm(
    image: Image.Image,
    question: str,
    context: str,
    *,
    web_search: WebSearchService | None,
) -> str:
    """Call VLM with optional two-step web-search enrichment."""
    async with VLM() as vlm:
        if web_search is None:
            return await vlm.answer_question(
                image=image, question=question, context=context or None
            )

        evaluation = await vlm.evaluate_search_need(
            image=image, question=question, context=context or None
        )
        if not evaluation.needs_search:
            return evaluation.answer

        logger.info("[vlm-worker] web search requested: %r", evaluation.search_query)
        results = await web_search.search(evaluation.search_query)
        web_context = WebSearchService.format_results(results)
        enriched = (
            f"{context}\n\nДополнительная информация из интернета:\n{web_context}"
            if web_context
            else context
        )
        return await vlm.answer_question(
            image=image,
            question=question,
            context=enriched or None,
            system_prompt=enriched_system_prompt(),
        )


async def _process_task(
    task_id: uuid.UUID,
    settings: Settings,
    web_search: WebSearchService | None,
) -> None:
    """Execute a single claimed task and persist its result."""
    try:
        async with session_scope() as session:
            task = await InferenceTaskRepository(session).get(task_id)
            if task is None:
                logger.error("[vlm-worker] task %s vanished before processing", task_id)
                return
            if task.type != TaskType.VLM_QA:
                await InferenceTaskRepository(session).mark_error(
                    task_id, f"unsupported task type: {task.type.value}"
                )
                return
            request = dict(task.request or {})
            image, context, question = await _resolve_qa_inputs(
                session, request, settings
            )

        if not question:
            await _mark_error(task_id, "empty question")
            return
        if image is None:
            await _mark_error(task_id, "no image available")
            return

        answer = await _run_vlm(
            image=image, question=question, context=context, web_search=web_search
        )

        async with session_scope() as session:
            await InferenceTaskRepository(session).mark_done(
                task_id, result={"answer": answer}
            )
        logger.info("[vlm-worker] task %s done", task_id)

    except Exception as exc:
        logger.exception("[vlm-worker] task %s failed", task_id)
        await _mark_error(task_id, f"{type(exc).__name__}: {exc}")


async def _mark_error(task_id: uuid.UUID, message: str) -> None:
    """Mark a task as errored."""
    try:
        async with session_scope() as session:
            await InferenceTaskRepository(session).mark_error(task_id, message)
    except Exception:
        logger.exception("[vlm-worker] failed to mark task %s as error", task_id)


async def _claim_one(worker_id: str) -> uuid.UUID | None:
    """Claim the oldest pending task."""
    async with session_scope() as session:
        task = await InferenceTaskRepository(session).claim_pending(worker=worker_id)
        return task.id if task is not None else None


async def _recovery_loop(settings: Settings, stop: asyncio.Event) -> None:
    """Periodically re-queue tasks stuck in `running`."""
    interval = max(1, settings.worker_stale_check_interval_s)
    timeout = timedelta(seconds=max(1, settings.worker_stale_timeout_s))
    while not stop.is_set():
        try:
            async with session_scope() as session:
                requeued = await InferenceTaskRepository(session).requeue_stale_running(
                    older_than=timeout
                )
            if requeued:
                logger.warning(
                    "[vlm-worker] re-queued %d stale running task(s)", requeued
                )
        except Exception:
            logger.exception("[vlm-worker] recovery sweep failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def _main_loop(stop: asyncio.Event) -> None:
    settings = get_settings()
    worker_id = _worker_id()
    logger.info(
        "[vlm-worker] starting (concurrency=%d, poll_interval=%.2fs)",
        settings.worker_concurrency,
        settings.worker_poll_interval_s,
    )

    web_search = WebSearchService() if settings.web_search_enabled else None
    semaphore = asyncio.Semaphore(max(1, settings.worker_concurrency))
    inflight: set[asyncio.Task[None]] = set()
    recovery = asyncio.create_task(_recovery_loop(settings, stop), name="recovery")

    try:
        while not stop.is_set():
            await semaphore.acquire()
            if stop.is_set():
                semaphore.release()
                break

            try:
                task_id = await _claim_one(worker_id)
            except Exception:
                logger.exception("[vlm-worker] claim failed; backing off")
                semaphore.release()
                await asyncio.sleep(settings.worker_poll_interval_s)
                continue

            if task_id is None:
                semaphore.release()
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=settings.worker_poll_interval_s
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            async def _runner(tid: uuid.UUID = task_id) -> None:
                try:
                    await _process_task(tid, settings, web_search)
                finally:
                    semaphore.release()

            t = asyncio.create_task(_runner(), name=f"task-{task_id}")
            inflight.add(t)
            t.add_done_callback(inflight.discard)

        logger.info(
            "[vlm-worker] stop signal received; draining %d task(s)", len(inflight)
        )
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
    finally:
        recovery.cancel()
        try:
            await recovery
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await get_engine().dispose()
        except Exception:
            logger.exception("[vlm-worker] engine dispose failed")
        logger.info("[vlm-worker] stopped")


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop, stop: asyncio.Event
) -> None:
    def _handler() -> None:
        if not stop.is_set():
            logger.info("[vlm-worker] received shutdown signal")
            stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _handler())


def main() -> None:
    """CLI entrypoint: `python -m workers.vlm_worker`."""
    logging.basicConfig(
        level=os.environ.get("WORKER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    async def _runner() -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        _install_signal_handlers(loop, stop)
        await _main_loop(stop)

    asyncio.run(_runner())


if __name__ == "__main__":
    main()
