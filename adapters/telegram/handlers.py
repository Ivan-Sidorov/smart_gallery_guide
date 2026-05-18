"""Tanslates Telegram updates into API calls and renders the responses back to the chat."""

import logging
import uuid
from typing import Any

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import NetworkError, TimedOut
from telegram.ext import ContextTypes

from adapters.telegram.api_client import (
    APIClient,
    APIClientError,
    TaskTimeoutError,
    current_request_id,
)
from adapters.telegram.keyboards import (
    get_back_to_exhibit_keyboard,
    get_exhibits_list_keyboard,
    get_main_keyboard,
)
from adapters.telegram.utils import (
    download_audio_bytes,
    download_photo_bytes,
    format_exhibit_dto,
    format_exhibit_search_result,
    format_exhibit_search_results,
    format_faq_results,
    format_vlm_answer,
    safe_edit_text,
    safe_reply_photo,
    safe_reply_text,
    truncate_text,
)
from api.schemas.exhibits import ExhibitDTO

logger = logging.getLogger(__name__)

# Bot
API_CLIENT_KEY = "api_client"

# Session state
SESSION_ID_KEY = "session_id"
SESSION_CONTEXT_KEY = "session_context"

CURRENT_EXHIBIT_KEY = "current_exhibit_id"
WAITING_FOR_QUESTION_KEY = "waiting_for_question"
SEARCH_MODE_KEY = "search_mode"


def _api(context: ContextTypes.DEFAULT_TYPE) -> APIClient:
    """Return the shared APIClient stored in `bot_data`."""
    client = context.bot_data.get(API_CLIENT_KEY)
    if client is None:
        raise RuntimeError("APIClient is not initialised in bot_data")
    return client


def _user_id(update: Update) -> int | None:
    """Return the user id from the update."""
    return update.effective_user.id if update.effective_user else None


async def _bind_request_id(update: Update) -> None:
    """Bind X-Request-Id for outbound API calls."""
    update_id = getattr(update, "update_id", None)
    if update_id is None:
        rid = uuid.uuid4().hex
    else:
        rid = f"tg-{update_id}"
    current_request_id.set(rid)


async def _ensure_session(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> tuple[uuid.UUID, dict[str, Any]] | None:
    """Make sure a session exists for the current user."""
    user = update.effective_user
    if user is None:
        return None

    cached_sid = context.user_data.get(SESSION_ID_KEY)
    cached_ctx = context.user_data.get(SESSION_CONTEXT_KEY)
    if isinstance(cached_sid, uuid.UUID) and isinstance(cached_ctx, dict):
        return cached_sid, cached_ctx

    api = _api(context)
    try:
        session = await api.start_or_resume_session(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            locale=user.language_code,
        )
    except APIClientError as exc:
        logger.error(
            "[telegram:handlers] can't start session for user_id=%s: %s", user.id, exc
        )
        return None

    context.user_data[SESSION_ID_KEY] = session.id
    context.user_data[SESSION_CONTEXT_KEY] = dict(session.context or {})
    return session.id, context.user_data[SESSION_CONTEXT_KEY]


async def _update_session_context(
    context: ContextTypes.DEFAULT_TYPE,
    session_id: uuid.UUID,
    session_ctx: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Merge `patch` into `session_ctx` and PATCH the backend."""
    merged = {**session_ctx, **patch}
    for key, value in list(merged.items()):
        if value is None:
            merged.pop(key, None)
    api = _api(context)
    try:
        updated = await api.update_session_context(session_id, merged)
    except APIClientError as exc:
        logger.warning("[telegram:handlers] session context update failed: %s", exc)
        context.user_data[SESSION_CONTEXT_KEY] = merged
        return merged
    new_ctx = dict(updated.context or {})
    context.user_data[SESSION_CONTEXT_KEY] = new_ctx
    return new_ctx


async def _log_exhibit_event(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: tuple[uuid.UUID, dict[str, Any]] | None,
    *,
    exhibit_id: str,
    event: str,
    content: str | None = None,
) -> None:
    """Persist a user–exhibit interaction when session and user are known."""
    if session is None:
        return
    user_id = _user_id(update)
    if user_id is None:
        return
    sid, _ = session
    api = _api(context)
    try:
        await api.log_exhibit_event(
            session_id=sid,
            user_id=user_id,
            exhibit_id=exhibit_id,
            event=event,
            content=content,
        )
    except APIClientError as exc:
        logger.warning("[telegram:handlers] log exhibit event failed: %s", exc)


async def _send_exhibit_card(
    message: Any,
    exhibit: ExhibitDTO,
) -> None:
    """Send a card with the exhibit image and Markdown info."""
    full_text = format_exhibit_dto(exhibit)
    sent_photo = False
    if exhibit.image_path:
        sent_photo = await safe_reply_photo(
            message,
            photo_path=exhibit.image_path,
            caption="",
            parse_mode=ParseMode.MARKDOWN,
        )
    if not sent_photo:
        await safe_reply_text(
            message,
            truncate_text(full_text),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    await safe_reply_text(
        message,
        truncate_text(full_text),
        parse_mode=ParseMode.MARKDOWN,
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    await _bind_request_id(update)
    welcome_message = (
        "Это умный гид для художественного музея!\n\n"
        "Я помогу вам узнать больше об экспонатах и ответить на ваши вопросы.\n\n"
        "Что я умею:\n"
        "• Распознавать экспонаты по фотографии\n"
        "• Искать экспонаты по текстовому или голосовому запросу\n"
        "• Отвечать на вопросы об экспонатах\n"
        "Отправьте фото экспоната, текст, голосовое сообщение "
        "или используйте кнопки меню."
    )

    if update.message is not None:
        await update.message.reply_text(
            welcome_message,
            reply_markup=get_main_keyboard(),
            parse_mode=ParseMode.MARKDOWN,
        )

    session = await _ensure_session(update, context)
    if session is not None:
        sid, ctx = session
        await _update_session_context(
            context,
            sid,
            ctx,
            {
                CURRENT_EXHIBIT_KEY: None,
                SEARCH_MODE_KEY: None,
                WAITING_FOR_QUESTION_KEY: None,
            },
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command."""
    await _bind_request_id(update)
    help_message = (
        "*Помощь*\n\n"
        "*Как использовать бота:*\n\n"
        "1. *Распознавание экспоната:*\n"
        "   Отправьте фотографию экспоната, и я найду его в базе.\n\n"
        "2. *Поиск по тексту или голосу:*\n"
        "   Напишите или продиктуйте название или описание экспоната.\n\n"
        "3. *Вопросы об экспонате:*\n"
        "   После распознавания экспоната вы можете задать вопрос, "
        "просто отправив сообщение с вопросом.\n\n"
        "*Команды:*\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать эту справку"
    )

    if update.message is not None:
        await update.message.reply_text(
            help_message,
            reply_markup=get_main_keyboard(),
            parse_mode=ParseMode.MARKDOWN,
        )


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a photo message: try to recognise the exhibit."""
    await _bind_request_id(update)
    if not update.message:
        return

    session = await _ensure_session(update, context)

    processing_msg = await update.message.reply_text(
        "Обрабатываю изображение...",
        reply_markup=get_main_keyboard(),
    )

    try:
        image_bytes = await download_photo_bytes(update, context)
        if not image_bytes:
            await safe_edit_text(
                "Не удалось загрузить изображение.",
                update=update,
                message=processing_msg,
            )
            return

        await safe_edit_text(
            "Распознаю экспонат...", update=update, message=processing_msg
        )
        api = _api(context)
        user_id = _user_id(update)
        session_id = session[0] if session is not None else None
        results = await api.recognize_exhibit(
            image_bytes,
            user_id=user_id,
            session_id=session_id,
        )

        if not results:
            await safe_edit_text(
                "Экспонат не найден. Попробуйте другое изображение или "
                "используйте текстовый поиск.",
                update=update,
                message=processing_msg,
            )
            return

        if len(results) == 1:
            result = results[0]
            exhibit_id = result.exhibit_id
            if session is not None:
                sid, ctx = session
                await _update_session_context(
                    context,
                    sid,
                    ctx,
                    {
                        CURRENT_EXHIBIT_KEY: exhibit_id,
                        WAITING_FOR_QUESTION_KEY: None,
                        SEARCH_MODE_KEY: None,
                    },
                )
            await _log_exhibit_event(
                update, context, session, exhibit_id=exhibit_id, event="select"
            )

            await safe_edit_text(
                "Нашёл экспонат — отправляю карточку ниже.",
                update=update,
                message=processing_msg,
            )
            await _render_exhibit(context, processing_msg, exhibit_id)
        else:
            if session is not None:
                sid, ctx = session
                await _update_session_context(
                    context,
                    sid,
                    ctx,
                    {SEARCH_MODE_KEY: None, WAITING_FOR_QUESTION_KEY: None},
                )
            text = format_exhibit_search_results(results)
            await safe_edit_text(
                truncate_text(text),
                update=update,
                message=processing_msg,
                reply_markup=get_exhibits_list_keyboard(results),
                parse_mode=ParseMode.MARKDOWN,
            )

    except APIClientError as exc:
        logger.error("[telegram:handlers] photo_handler API error: %s", exc)
        await safe_edit_text(
            "Сервер сейчас не отвечает. Попробуйте позже.",
            update=update,
            message=processing_msg,
        )
    except (TimedOut, NetworkError) as exc:
        logger.warning("[telegram:handlers] photo_handler network error: %s", exc)
        await safe_edit_text(
            "Произошла сетевая ошибка. Попробуйте ещё раз.",
            update=update,
            message=processing_msg,
        )
    except Exception:
        logger.exception("[telegram:handlers] photo_handler unexpected error")
        await safe_edit_text(
            "Произошла ошибка при обработке изображения. Попробуйте ещё раз.",
            update=update,
            message=processing_msg,
        )


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice/audio: transcribe via ASR, then reuse the text pipeline."""
    await _bind_request_id(update)
    if not update.message:
        return

    processing_msg = await update.message.reply_text(
        "Распознаю речь...",
        reply_markup=get_main_keyboard(),
    )

    try:
        downloaded = await download_audio_bytes(update, context)
        if not downloaded:
            await safe_edit_text(
                "Не удалось загрузить аудио.",
                update=update,
                message=processing_msg,
            )
            return

        audio_bytes, filename, content_type = downloaded
        api = _api(context)
        try:
            result = await api.transcribe_audio(
                audio_bytes,
                filename=filename,
                content_type=content_type,
            )
        except APIClientError as exc:
            logger.error("[telegram:handlers] transcribe_audio API error: %s", exc)
            await safe_edit_text(
                "Сервер сейчас не отвечает. Попробуйте позже.",
                update=update,
                message=processing_msg,
            )
            return

        text = result.text.strip()
        if not text:
            await safe_edit_text(
                "Не удалось распознать речь. Попробуйте ещё раз.",
                update=update,
                message=processing_msg,
            )
            return

        await safe_edit_text(
            f"Распознано: _{truncate_text(text, max_length=500)}_",
            update=update,
            message=processing_msg,
            parse_mode=ParseMode.MARKDOWN,
        )
        await _handle_user_text(update, context, text)

    except (TimedOut, NetworkError) as exc:
        logger.warning("[telegram:handlers] voice_handler network error: %s", exc)
        await safe_edit_text(
            "Произошла сетевая ошибка. Попробуйте ещё раз.",
            update=update,
            message=processing_msg,
        )
    except Exception:
        logger.exception("[telegram:handlers] voice_handler unexpected error")
        await safe_edit_text(
            "Произошла ошибка при обработке аудио. Попробуйте ещё раз.",
            update=update,
            message=processing_msg,
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a text message: menu commands, search mode, or question to VLM."""
    await _bind_request_id(update)
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    await _handle_user_text(update, context, text)


async def _handle_user_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
) -> None:
    """Route recognised or typed user text through search / Q&A flows."""
    if not update.message:
        return

    session = await _ensure_session(update, context)
    ctx_data: dict[str, Any] = {}
    sid: uuid.UUID | None = None
    if session is not None:
        sid, ctx_data = session

    if text == "Поиск экспонатов":
        if sid is not None:
            await _update_session_context(
                context,
                sid,
                ctx_data,
                {SEARCH_MODE_KEY: True, WAITING_FOR_QUESTION_KEY: None},
            )
        await update.message.reply_text(
            "Отправьте новое фото экспоната или напишите название/описание для "
            "поиска другого экспоната.",
            reply_markup=get_main_keyboard(),
        )
        return

    if text == "Помощь":
        await help_command(update, context)
        return

    if text == "Отмена":
        if sid is not None:
            await _update_session_context(
                context,
                sid,
                ctx_data,
                {SEARCH_MODE_KEY: None, WAITING_FOR_QUESTION_KEY: None},
            )
        await update.message.reply_text("Отменено.", reply_markup=get_main_keyboard())
        return

    search_mode = bool(ctx_data.get(SEARCH_MODE_KEY))
    current_exhibit_id = ctx_data.get(CURRENT_EXHIBIT_KEY)

    if search_mode:
        await _do_text_search(update, context, text, session)
        return

    if not current_exhibit_id:
        await update.message.reply_text(
            "Сначала выберите экспонат: отправьте его фото или нажмите "
            "«Поиск экспонатов».",
            reply_markup=get_main_keyboard(),
        )
        return

    await _do_exhibit_question(update, context, text, str(current_exhibit_id), session)


async def _do_text_search(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    session: tuple[uuid.UUID, dict[str, Any]] | None,
) -> None:
    """Run a text search for exhibits."""
    message = update.message
    if message is None:
        logger.warning("[handlers] _do_text_search called without message")
        return

    processing_msg = await message.reply_text(
        "Ищу экспонаты...",
        reply_markup=get_main_keyboard(),
    )

    try:
        api = _api(context)
        user_id = _user_id(update)
        session_id = session[0] if session is not None else None
        results = await api.search_exhibits(
            query, user_id=user_id, session_id=session_id
        )

        if not results:
            await safe_edit_text(
                "Экспонаты не найдены. Попробуйте другой запрос или отправьте фото.",
                update=update,
                message=processing_msg,
            )
            return

        if len(results) == 1:
            result = results[0]
            exhibit_id = result.exhibit_id
            if session is not None:
                sid, ctx_data = session
                await _update_session_context(
                    context,
                    sid,
                    ctx_data,
                    {
                        CURRENT_EXHIBIT_KEY: exhibit_id,
                        WAITING_FOR_QUESTION_KEY: None,
                        SEARCH_MODE_KEY: None,
                    },
                )
            await _log_exhibit_event(
                update, context, session, exhibit_id=exhibit_id, event="select"
            )
            await safe_edit_text(
                "Нашёл экспонат — отправляю карточку ниже.",
                update=update,
                message=processing_msg,
            )
            await _render_exhibit(context, processing_msg, exhibit_id, fallback=result)
        else:
            text_result = format_exhibit_search_results(results)
            await safe_edit_text(
                truncate_text(text_result),
                update=update,
                message=processing_msg,
                reply_markup=get_exhibits_list_keyboard(results),
                parse_mode=ParseMode.MARKDOWN,
            )

    except APIClientError as exc:
        logger.error("[handlers] _do_text_search API error: %s", exc)
        await safe_edit_text(
            "Сервер сейчас не отвечает. Попробуйте позже.",
            update=update,
            message=processing_msg,
        )
    except Exception:
        logger.exception("[handlers] _do_text_search unexpected error")
        await safe_edit_text(
            "Произошла ошибка при поиске. Попробуйте ещё раз.",
            update=update,
            message=processing_msg,
        )


async def _do_exhibit_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question: str,
    exhibit_id: str,
    session: tuple[uuid.UUID, dict[str, Any]] | None,
) -> None:
    """Ask a question about an exhibit."""
    message = update.message
    if message is None:
        logger.warning("[handlers] _do_exhibit_question called without message")
        return

    processing_msg = await message.reply_text(
        "Думаю...",
        reply_markup=get_main_keyboard(),
    )

    user_id = _user_id(update)
    sid = session[0] if session is not None else None

    await _log_exhibit_event(
        update,
        context,
        session,
        exhibit_id=exhibit_id,
        event="question",
        content=question,
    )

    api = _api(context)
    try:
        response = await api.qa_exhibit(
            exhibit_id=exhibit_id,
            question=question,
            user_id=user_id,
            session_id=sid,
        )
    except APIClientError as exc:
        logger.error("[handlers] qa_exhibit API error: %s", exc)
        await safe_edit_text(
            "Сервер сейчас не отвечает. Попробуйте позже.",
            update=update,
            message=processing_msg,
        )
        return

    if response.mode == "faq" and response.answer is not None:
        if session is not None:
            sid, ctx_data = session
            await _update_session_context(
                context, sid, ctx_data, {WAITING_FOR_QUESTION_KEY: None}
            )
        await safe_edit_text(
            truncate_text(format_vlm_answer(response.answer)),
            update=update,
            message=processing_msg,
            reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
        )
        return

    if response.task_id is None:
        await safe_edit_text(
            "Не удалось обработать вопрос. Попробуйте позже.",
            update=update,
            message=processing_msg,
        )
        return

    await _wait_and_render_task(
        update, context, processing_msg, response.task_id, exhibit_id=exhibit_id
    )

    if session is not None:
        sid, ctx_data = session
        await _update_session_context(
            context, sid, ctx_data, {WAITING_FOR_QUESTION_KEY: None}
        )


async def _wait_and_render_task(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    processing_msg: Any,
    task_id: uuid.UUID,
    *,
    exhibit_id: str | None = None,
) -> None:
    """Poll the task to completion and render the result back to chat."""
    api = _api(context)

    async def _on_status(task: Any) -> None:
        if task.status == "running":
            await safe_edit_text(
                "Готовлю ответ...",
                update=update,
                message=processing_msg,
            )

    try:
        task = await api.wait_for_task(task_id, on_status=_on_status)
    except TaskTimeoutError:
        await safe_edit_text(
            "Ответ занимает слишком много времени. Попробуйте позже.",
            update=update,
            message=processing_msg,
        )
        return
    except APIClientError as exc:
        logger.error("[handlers] wait_for_task API error: %s", exc)
        await safe_edit_text(
            "Сервер сейчас не отвечает. Попробуйте позже.",
            update=update,
            message=processing_msg,
        )
        return

    if task.status == "error":
        logger.warning("[handlers] task %s errored: %s", task_id, task.error)
        await safe_edit_text(
            "Не удалось получить ответ. Попробуйте ещё раз.",
            update=update,
            message=processing_msg,
            reply_markup=(
                get_back_to_exhibit_keyboard(exhibit_id) if exhibit_id else None
            ),
        )
        return

    answer = ""
    if isinstance(task.result, dict):
        answer = str(task.result.get("answer") or "").strip()
    await safe_edit_text(
        truncate_text(format_vlm_answer(answer)),
        update=update,
        message=processing_msg,
        reply_markup=(get_back_to_exhibit_keyboard(exhibit_id) if exhibit_id else None),
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries from buttons."""
    await _bind_request_id(update)
    if not update.callback_query:
        return

    query = update.callback_query
    await query.answer()

    callback_data = query.data
    if not callback_data:
        return

    try:
        action, exhibit_id = callback_data.split(":", 1)
    except ValueError:
        return

    try:
        if action == "select_exhibit":
            await _handle_exhibit_selection(update, context, exhibit_id)
        elif action == "exhibit_info":
            await _handle_exhibit_info(update, context, exhibit_id)
        elif action == "ask_question":
            await _handle_ask_question(update, context, exhibit_id)
        elif action == "search_faq":
            await _handle_search_faq(update, context, exhibit_id)
    except Exception:
        logger.exception("[handlers] callback_handler unexpected error")
        try:
            await query.edit_message_text("Произошла ошибка. Попробуйте ещё раз.")
        except Exception:
            pass
        if query.message:
            await query.message.reply_text(
                "Открыл главное меню — используйте кнопки ниже.",
                reply_markup=get_main_keyboard(),
            )


async def _render_exhibit(
    context: ContextTypes.DEFAULT_TYPE,
    message: Any,
    exhibit_id: str,
    *,
    fallback: Any = None,
) -> None:
    """Fetch an exhibit by id from the API and render it."""
    api = _api(context)
    try:
        exhibit = await api.get_exhibit(exhibit_id)
    except APIClientError as exc:
        logger.warning("[handlers] get_exhibit(%s) failed: %s", exhibit_id, exc)
        exhibit = None

    if exhibit is not None:
        await _send_exhibit_card(message, exhibit)
        return

    if fallback is not None:
        await safe_reply_text(
            message,
            truncate_text(format_exhibit_search_result(fallback)),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    await safe_reply_text(
        message,
        "Экспонат не найден.",
        reply_markup=get_main_keyboard(),
    )


async def _handle_exhibit_selection(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """User picked an exhibit from a search-result list."""
    callback_query = update.callback_query
    if callback_query is None:
        logger.warning("[handlers] _handle_exhibit_selection called without callback")
        return

    session = await _ensure_session(update, context)
    if session is not None:
        sid, ctx_data = session
        await _update_session_context(
            context,
            sid,
            ctx_data,
            {
                CURRENT_EXHIBIT_KEY: exhibit_id,
                SEARCH_MODE_KEY: None,
                WAITING_FOR_QUESTION_KEY: None,
            },
        )

    api = _api(context)
    exhibit = await api.get_exhibit(exhibit_id)
    if exhibit is None:
        await safe_edit_text("Экспонат не найден.", callback_query=callback_query)
        if callback_query.message:
            await callback_query.message.reply_text(
                "Открыл главное меню — используйте кнопки ниже.",
                reply_markup=get_main_keyboard(),
            )
        return

    await _log_exhibit_event(
        update, context, session, exhibit_id=exhibit_id, event="select"
    )
    await safe_edit_text(
        "Открыл карточку экспоната ниже.",
        callback_query=callback_query,
    )
    if callback_query.message:
        await _send_exhibit_card(callback_query.message, exhibit)


async def _handle_exhibit_info(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """User requested to re-open the exhibit card."""
    callback_query = update.callback_query
    if callback_query is None:
        logger.warning(
            "[telegram:handlers] _handle_exhibit_info called without callback"
        )
        return

    api = _api(context)
    exhibit = await api.get_exhibit(exhibit_id)
    if exhibit is None:
        await safe_edit_text("Экспонат не найден.", callback_query=callback_query)
        if callback_query.message:
            await callback_query.message.reply_text(
                "Открыл главное меню — используйте кнопки ниже.",
                reply_markup=get_main_keyboard(),
            )
        return

    await safe_edit_text(
        "Отправляю полную карточку экспоната ниже.",
        callback_query=callback_query,
    )
    if callback_query.message:
        await _send_exhibit_card(callback_query.message, exhibit)


async def _handle_ask_question(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """User pressed 'ask a question': put the exhibit and wait for text input."""
    callback_query = update.callback_query
    if callback_query is None:
        logger.warning(
            "[telegram:handlers] _handle_ask_question called without callback"
        )
        return

    session = await _ensure_session(update, context)
    if session is not None:
        sid, ctx_data = session
        await _update_session_context(
            context,
            sid,
            ctx_data,
            {
                CURRENT_EXHIBIT_KEY: exhibit_id,
                WAITING_FOR_QUESTION_KEY: True,
                SEARCH_MODE_KEY: None,
            },
        )

    await safe_edit_text(
        "Задайте вопрос об экспонате текстом.",
        callback_query=callback_query,
        reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
    )


async def _handle_search_faq(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """User pressed 'search FAQ': do FAQ search using the last message text."""
    callback_query = update.callback_query
    if callback_query is None:
        logger.warning("[telegram:handlers] _handle_search_faq called without callback")
        return

    await callback_query.answer("Ищу в базе часто задаваемых вопросов...")

    message_text = ""
    if callback_query.message and callback_query.message.text:
        message_text = callback_query.message.text

    if not message_text:
        await safe_edit_text(
            "Напишите ваш вопрос для поиска в базе часто задаваемых вопросов.",
            callback_query=callback_query,
            reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
        )
        return

    question = message_text.split("\n", 1)[0].strip()[:200]

    api = _api(context)
    try:
        results = await api.search_faq(exhibit_id=exhibit_id, question=question)
    except APIClientError as exc:
        logger.error("[telegram:handlers] search_faq API error: %s", exc)
        await safe_edit_text(
            "Сервер сейчас не отвечает. Попробуйте позже.",
            callback_query=callback_query,
            reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
        )
        return

    await safe_edit_text(
        truncate_text(format_faq_results(results)),
        callback_query=callback_query,
        reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
        parse_mode=ParseMode.MARKDOWN,
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(
        "[telegram:handlers] Exception while handling an update: %s",
        context.error,
        exc_info=context.error,
    )

    if isinstance(context.error, (TimedOut, NetworkError)):
        return

    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Произошла ошибка. Пожалуйста, попробуйте ещё раз."
            )
        except Exception:
            pass
