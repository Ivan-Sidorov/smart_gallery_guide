"""Formatting helpers and Telegram I/O wrappers used by the adapter."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.ext import ContextTypes

from api.schemas.exhibits import ExhibitDTO, ExhibitSearchResultDTO
from api.schemas.faq import FAQSearchResultDTO
from core.vlm.client import VLM_NO_ANSWER_TEXT

logger = logging.getLogger(__name__)

# Telegram shows the typing indicator for ~5 seconds per sendChatAction call.
TYPING_ACTION_REFRESH_SECONDS = 4.0


@asynccontextmanager
async def typing_while(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    action: ChatAction = ChatAction.TYPING,
) -> AsyncIterator[None]:
    """Keep the chat action alive until the wrapped block finishes."""
    stop = asyncio.Event()

    async def _refresh() -> None:
        while not stop.is_set():
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=action)
            except BadRequest as exc:
                logger.warning("[utils] send_chat_action failed: %s", exc)
                return
            except (TimedOut, NetworkError) as exc:
                logger.warning("[utils] send_chat_action network error: %s", exc)
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=TYPING_ACTION_REFRESH_SECONDS
                )
            except asyncio.TimeoutError:
                continue

    task = asyncio.create_task(_refresh())
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def download_audio_bytes(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> tuple[bytes, str, str] | None:
    """Download a voice message or audio attachment.

    Returns:
        Tuple of (raw bytes, filename, MIME type), or ``None`` on failure.
    """
    try:
        message = update.message
        if message is None:
            logger.warning("[utils] No message in update")
            return None

        if message.voice is not None:
            attachment = message.voice
            filename = "voice.ogg"
            content_type = "audio/ogg"
        elif message.audio is not None:
            attachment = message.audio
            filename = attachment.file_name or "audio.mp3"
            content_type = attachment.mime_type or "audio/mpeg"
        else:
            logger.warning("[utils] No voice/audio in message")
            return None

        file = await context.bot.get_file(attachment.file_id)
        data = await file.download_as_bytearray()
        logger.info(
            "[utils] Downloaded audio: %s, size: %d bytes",
            attachment.file_id,
            len(data),
        )
        return bytes(data), filename, content_type

    except Exception:
        logger.exception("[utils] Error downloading audio")
        return None


async def download_photo_bytes(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bytes | None:
    """Download the largest photo attachment as raw bytes."""
    try:
        if not update.message or not update.message.photo:
            logger.warning("[utils] No photo in message")
            return None

        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        data = await file.download_as_bytearray()
        logger.info(
            "[utils] Downloaded photo: %s, size: %d bytes", photo.file_id, len(data)
        )
        return bytes(data)

    except Exception:
        logger.exception("[utils] Error downloading photo")
        return None


async def safe_edit_text(
    text: str,
    *,
    callback_query: Any = None,
    message: Any = None,
    update: Update | None = None,
    reply_markup: Any = None,
    parse_mode: Any = None,
) -> None:
    """Safely edit a Telegram message, with fallback to sending a new message."""

    def _fallback_message() -> Any:
        if callback_query is not None and getattr(callback_query, "message", None):
            return callback_query.message
        if update is not None and getattr(update, "effective_message", None):
            return update.effective_message
        if message is not None:
            return message
        return None

    for attempt in range(3):
        try:
            if callback_query is not None:
                await callback_query.edit_message_text(
                    text, reply_markup=reply_markup, parse_mode=parse_mode
                )
            elif message is not None:
                await message.edit_text(
                    text, reply_markup=reply_markup, parse_mode=parse_mode
                )
            else:
                fallback_msg = _fallback_message()
                if fallback_msg is not None:
                    await safe_reply_text(
                        fallback_msg,
                        text,
                        reply_markup=reply_markup,
                        parse_mode=parse_mode,
                    )
                elif callback_query is not None:
                    await callback_query.answer()
            return
        except BadRequest as e:
            err = str(e)
            if "Message is not modified" in err:
                if callback_query is not None:
                    await callback_query.answer()
                return
            if "Message can't be edited" in err or "message to edit not found" in err:
                fallback_msg = _fallback_message()
                if fallback_msg is not None:
                    await safe_reply_text(
                        fallback_msg,
                        text,
                        reply_markup=reply_markup,
                        parse_mode=parse_mode,
                    )
                elif callback_query is not None:
                    await callback_query.answer()
                return
            raise
        except (TimedOut, NetworkError) as e:
            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            logger.warning("[utils] Telegram API timeout while editing message: %s", e)
            fallback_msg = _fallback_message()
            if fallback_msg is not None:
                await safe_reply_text(
                    fallback_msg,
                    text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                )
            return


async def safe_delete_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_id: int,
) -> bool:
    """Delete a chat message; return False on expected Telegram errors."""
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        return True
    except BadRequest as exc:
        logger.warning("[utils] delete_message failed: %s", exc)
        return False
    except (TimedOut, NetworkError) as exc:
        logger.warning("[utils] delete_message network error: %s", exc)
        return False


async def safe_reply_text(
    message: Any,
    text: str,
    reply_markup: Any = None,
    parse_mode: Any = None,
) -> None:
    """Send a message with retries on Telegram network timeouts."""
    for attempt in range(3):
        try:
            await message.reply_text(
                text, reply_markup=reply_markup, parse_mode=parse_mode
            )
            return
        except (TimedOut, NetworkError) as e:
            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            logger.warning("[utils] Telegram API timeout while sending message: %s", e)
            return


async def safe_reply_photo(
    message: Any,
    photo_path: str,
    caption: str = "",
    reply_markup: Any = None,
    parse_mode: Any = None,
) -> bool:
    """Send a local photo file with retries on Telegram timeouts."""
    path = Path(photo_path)
    if not path.exists():
        logger.warning("[utils] Exhibit image not found: %s", photo_path)
        return False

    for attempt in range(3):
        try:
            with path.open("rb") as f:
                await message.reply_photo(
                    photo=f,
                    caption=caption,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                )
            return True
        except (TimedOut, NetworkError) as e:
            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            logger.warning("[utils] Telegram API timeout while sending photo: %s", e)
            return False
        except BadRequest as e:
            logger.warning("[utils] Telegram API BadRequest while sending photo: %s", e)
            return False
        except Exception:
            logger.exception("[utils] Unexpected error while sending photo")
            return False


def _extra(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return the `extra` sub-dict from metadata or an empty dict."""
    if not metadata:
        return {}
    extra = metadata.get("extra")
    return dict(extra) if isinstance(extra, dict) else {}


def format_exhibit_dto(exhibit: ExhibitDTO) -> str:
    """Render an ExhibitDTO into a Markdown card."""
    return _format_exhibit_card(
        title=exhibit.title,
        author=exhibit.author,
        year=exhibit.year,
        description=exhibit.description,
        extra=dict(exhibit.extra or {}),
    )


def format_exhibit_search_result(result: ExhibitSearchResultDTO) -> str:
    """Render a single search hit into Markdown."""
    metadata = dict(result.metadata or {})
    return _format_exhibit_card(
        title=result.title,
        author=metadata.get("author") or metadata.get("artist"),
        year=metadata.get("year"),
        description=metadata.get("description") or metadata.get("display_description"),
        extra=_extra(metadata) or metadata,
    )


def _format_exhibit_card(
    *,
    title: str,
    author: str | None,
    year: str | None,
    description: str | None,
    extra: dict[str, Any],
) -> str:
    """Render an exhibit card into Markdown."""
    lines = [f"*{title}*"]

    if author:
        lines.append(f"Автор: {author}")
    if year:
        lines.append(f"Год: {year}")

    mat_parts: list[str] = []
    if material := extra.get("material"):
        mat_parts.append(str(material))
    if techniq := extra.get("techniq"):
        mat_parts.append(str(techniq))
    if mat_parts:
        lines.append(f"Материал/техника: {', '.join(mat_parts)}")

    geo_parts: list[str] = []
    if school := extra.get("school"):
        geo_parts.append(str(school))
    if place := extra.get("place"):
        geo_parts.append(str(place))
    if geo_parts:
        lines.append(f"Происхождение: {', '.join(geo_parts)}")

    if collection := extra.get("collection_name"):
        lines.append(f"Коллекция: {collection}")

    if description:
        lines.append(f"\n{description}")

    if anotation := extra.get("anotation"):
        lines.append(f"\n*Экспертный комментарий:*\n{anotation}")

    return "\n".join(lines)


def format_exhibit_search_results(results: list[ExhibitSearchResultDTO]) -> str:
    """Render a list of search hits as a numbered list."""
    if not results:
        return "Экспонаты не найдены."

    if len(results) == 1:
        return format_exhibit_search_result(results[0])

    lines = [f"Найдено экспонатов: {len(results)}\n"]
    for i, result in enumerate(results, 1):
        score_percent = int(result.similarity_score * 100)
        metadata = dict(result.metadata or {})
        author = metadata.get("author") or metadata.get("artist")
        line = f"{i}. *{result.title}*"
        if author:
            line += f" — {author}"
        line += f" (совпадение: {score_percent}%)"
        lines.append(line)

    return "\n".join(lines)


def format_faq_results(results: list[FAQSearchResultDTO]) -> str:
    """Render FAQ hits as a Markdown list."""
    if not results:
        return "Похожие вопросы не найдены в базе часто задаваемых вопросов."

    lines = ["*Похожие вопросы:*\n"]
    for i, result in enumerate(results, 1):
        lines.append(f"{i}. *Вопрос:* {result.question}")
        lines.append(f"   *Ответ:* {result.answer}")
        if i < len(results):
            lines.append("")

    return "\n".join(lines)


UNAVAILABLE_ANSWER_TEXT = "Сейчас не получается ответить на вопрос. Попробуйте позже."

# Older user-facing stubs that must not be shown.
_UNAVAILABLE_LEGACY_TEXTS = frozenset(
    {
        VLM_NO_ANSWER_TEXT,
        "Сервер сейчас не отвечает. Попробуйте позже.",
        "Ответ занимает слишком много времени. Попробуйте позже.",
        "Произошла сетевая ошибка. Попробуйте ещё раз.",
        "Произошла ошибка при обработке изображения. Попробуйте ещё раз.",
        "Произошла ошибка при обработке аудио. Попробуйте ещё раз.",
        "Произошла ошибка при поиске. Попробуйте ещё раз.",
        "Произошла ошибка. Попробуйте ещё раз.",
        "Произошла ошибка. Пожалуйста, попробуйте ещё раз.",
    }
)

_UNAVAILABLE_ANSWER_MARKERS = (
    "Ошибка при обращении к VLM API",
    '"detail"',
    "'detail'",
)


def is_unusable_model_answer(answer: str | None) -> bool:
    """True when the model output should not be shown to the user."""
    if not answer or not answer.strip():
        return True
    text = answer.strip()
    if text in _UNAVAILABLE_LEGACY_TEXTS:
        return True
    if text.startswith("{") and "detail" in text:
        return True
    return any(marker in text for marker in _UNAVAILABLE_ANSWER_MARKERS)


def format_vlm_answer(answer: str | None) -> str:
    """Render the VLM answer for display."""
    if is_unusable_model_answer(answer):
        return UNAVAILABLE_ANSWER_TEXT
    return answer.strip()


def truncate_text(text: str, max_length: int = 4096) -> str:
    """Truncate text to Telegram's 4096 character limit."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
