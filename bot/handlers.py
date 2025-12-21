import logging
import asyncio
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.ext import ContextTypes

from agent.agent import GuideAgent
from bot.keyboards import (
    get_back_to_exhibit_keyboard,
    get_exhibit_keyboard,
    get_exhibits_list_keyboard,
    get_main_keyboard,
)
from bot.utils import (
    download_photo,
    format_exhibit_info,
    format_exhibit_search_results,
    format_faq_results,
    format_vlm_answer,
    truncate_text,
)
from database.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


AGENT_KEY = "agent"
CURRENT_EXHIBIT_KEY = "current_exhibit_id"
WAITING_FOR_QUESTION_KEY = "waiting_for_question"
SEARCH_MODE_KEY = "search_mode"


async def safe_edit_text(
    text: str,
    *,
    callback_query=None,
    message=None,
    update: Optional[Update] = None,
    reply_markup=None,
    parse_mode=None,
) -> None:
    """
    Safely edit a Telegram message, with fallback to sending a new message.
    """

    def _fallback_message():
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
            logger.warning("Telegram API timeout while editing message: %s", e)
            fallback_msg = _fallback_message()
            if fallback_msg is not None:
                await safe_reply_text(
                    fallback_msg,
                    text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                )
            return


async def safe_reply_text(
    message, text: str, reply_markup=None, parse_mode=None
) -> None:
    """
    Send a message with retries on Telegram network timeouts.
    """
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
            logger.warning("Telegram API timeout while sending message: %s", e)
            return


def get_agent(context: ContextTypes.DEFAULT_TYPE) -> GuideAgent:
    """
    Get or create GuideAgent instance from context.
    """
    if AGENT_KEY not in context.bot_data:
        vector_db = VectorDatabase()
        context.bot_data[AGENT_KEY] = GuideAgent(vector_db=vector_db)
    return context.bot_data[AGENT_KEY]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start command.
    """
    welcome_message = (
        "Это умный гид для картинной галереи!\n\n"
        "Я помогу вам узнать больше об экспонатах и ответить на ваши вопросы.\n\n"
        "Что я умею:\n"
        "• Распознавать экспонаты по фотографии\n"
        "• Искать экспонаты по текстовому запросу\n"
        "• Отвечать на вопросы об экспонатах\n"
        "Отправьте фото экспоната или используйте кнопки меню."
    )

    await update.message.reply_text(
        welcome_message,
        reply_markup=get_main_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
    )

    if update.effective_user:
        user_id = update.effective_user.id
        if user_id in context.user_data:
            context.user_data[user_id].clear()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /help command.
    """
    help_message = (
        "*Помощь*\n\n"
        "*Как использовать бота:*\n\n"
        "1. *Распознавание экспоната:*\n"
        "   Отправьте фотографию экспоната, и я найду его в базе.\n\n"
        "2. *Поиск по тексту:*\n"
        "   Напишите название или описание экспоната для поиска.\n\n"
        "3. *Вопросы об экспонате:*\n"
        "   После распознавания экспоната вы можете задать вопрос, "
        "отправив фото с подписью или используя кнопку 'Задать вопрос'.\n\n"
        "*Команды:*\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать эту справку"
    )

    await update.message.reply_text(
        help_message,
        reply_markup=get_main_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
    )


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle photo messages - recognize exhibit.
    """
    if not update.message:
        return

    processing_msg = await update.message.reply_text(
        "Обрабатываю изображение...",
        reply_markup=get_main_keyboard(),
    )

    try:
        image = await download_photo(update, context)
        if not image:
            await safe_edit_text(
                "Не удалось загрузить изображение.",
                update=update,
                message=processing_msg,
            )
            return

        agent = get_agent(context)

        await safe_edit_text(
            "Распознаю экспонат...", update=update, message=processing_msg
        )
        results = await agent.recognize_exhibit(image)

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

            if update.effective_user:
                user_id = update.effective_user.id
                context.user_data[user_id] = context.user_data.get(user_id, {})
                context.user_data[user_id][CURRENT_EXHIBIT_KEY] = exhibit_id
                context.user_data[user_id].pop(WAITING_FOR_QUESTION_KEY, None)
                context.user_data[user_id].pop(SEARCH_MODE_KEY, None)

            text = format_exhibit_info(result.metadata)
            await safe_edit_text(
                truncate_text(text),
                update=update,
                message=processing_msg,
                reply_markup=get_exhibit_keyboard(exhibit_id),
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            if update.effective_user:
                user_id = update.effective_user.id
                user_data = context.user_data.get(user_id, {})
                user_data.pop(SEARCH_MODE_KEY, None)
                user_data.pop(WAITING_FOR_QUESTION_KEY, None)
            text = format_exhibit_search_results(results)
            await safe_edit_text(
                truncate_text(text),
                update=update,
                message=processing_msg,
                reply_markup=get_exhibits_list_keyboard(results),
                parse_mode=ParseMode.MARKDOWN,
            )

    except Exception as e:
        logger.error(f"Error in photo_handler: {e}", exc_info=True)
        await safe_edit_text(
            "Произошла ошибка при обработке изображения. Попробуйте еще раз.",
            update=update,
            message=processing_msg,
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle text messages - search exhibits or handle questions.
    """
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    user_id = update.effective_user.id if update.effective_user else None
    user_data = {}
    if user_id:
        user_data = context.user_data.get(user_id, {})
        context.user_data[user_id] = user_data

    if text == "Поиск экспонатов":
        if user_id:
            user_data[SEARCH_MODE_KEY] = True
            user_data.pop(WAITING_FOR_QUESTION_KEY, None)
        await update.message.reply_text(
            "Отправьте новое фото экспоната или напишите название/описание для поиска "
            "другого экспоната.",
            reply_markup=get_main_keyboard(),
        )
        return

    if text == "Помощь":
        await help_command(update, context)
        return

    if text == "Отмена":
        if user_id:
            user_data.pop(WAITING_FOR_QUESTION_KEY, None)
            user_data.pop(SEARCH_MODE_KEY, None)
        await update.message.reply_text("Отменено.", reply_markup=get_main_keyboard())
        return

    search_mode = user_data.get(SEARCH_MODE_KEY, False) if user_id else False
    current_exhibit_id = user_data.get(CURRENT_EXHIBIT_KEY) if user_id else None

    if search_mode:
        processing_msg = await update.message.reply_text(
            "Ищу экспонаты...",
            reply_markup=get_main_keyboard(),
        )

        try:
            agent = get_agent(context)
            results = await agent.search_exhibits_by_text(text)

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

                if user_id:
                    context.user_data[user_id] = context.user_data.get(user_id, {})
                    context.user_data[user_id][CURRENT_EXHIBIT_KEY] = exhibit_id
                    context.user_data[user_id].pop(WAITING_FOR_QUESTION_KEY, None)
                    context.user_data[user_id].pop(SEARCH_MODE_KEY, None)

                text_result = format_exhibit_info(result.metadata)
                await safe_edit_text(
                    truncate_text(text_result),
                    update=update,
                    message=processing_msg,
                    reply_markup=get_exhibit_keyboard(exhibit_id),
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                text_result = format_exhibit_search_results(results)
                await safe_edit_text(
                    truncate_text(text_result),
                    update=update,
                    message=processing_msg,
                    reply_markup=get_exhibits_list_keyboard(results),
                    parse_mode=ParseMode.MARKDOWN,
                )

        except Exception as e:
            logger.error(f"Error in text_handler: {e}", exc_info=True)
            await safe_edit_text(
                "Произошла ошибка при поиске. Попробуйте еще раз.",
                update=update,
                message=processing_msg,
            )
        return

    if not current_exhibit_id:
        await update.message.reply_text(
            "Сначала выберите экспонат: отправьте его фото или нажмите «Поиск экспонатов».",
            reply_markup=get_main_keyboard(),
        )
        return

    processing_msg = await update.message.reply_text(
        "Отвечаю на вопрос...",
        reply_markup=get_main_keyboard(),
    )

    try:
        agent = get_agent(context)
        answer = await agent.answer_question_about_exhibit(
            question=text, exhibit_id=current_exhibit_id
        )

        if user_id:
            user_data.pop(WAITING_FOR_QUESTION_KEY, None)

        await safe_edit_text(
            truncate_text(format_vlm_answer(answer)),
            update=update,
            message=processing_msg,
            reply_markup=get_back_to_exhibit_keyboard(current_exhibit_id),
        )

    except Exception as e:
        logger.error(f"Error in text_handler: {e}", exc_info=True)
        await safe_edit_text(
            "Произошла ошибка при обработке вопроса. Попробуйте еще раз.",
            update=update,
            message=processing_msg,
        )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle callback queries from inline buttons.
    """
    if not update.callback_query:
        return

    query = update.callback_query
    await query.answer()

    callback_data = query.data
    if not callback_data:
        return

    try:
        exhibit_id = callback_data.split(":", 1)[1]
        if callback_data.startswith("select_exhibit:"):
            await handle_exhibit_selection(update, context, exhibit_id)

        elif callback_data.startswith("exhibit_info:"):
            await handle_exhibit_info(update, context, exhibit_id)

        elif callback_data.startswith("ask_question:"):
            await handle_ask_question(update, context, exhibit_id)

        elif callback_data.startswith("search_faq:"):
            await handle_search_faq(update, context, exhibit_id)

    except Exception as e:
        logger.error(f"Error in callback_handler: {e}", exc_info=True)
        await query.edit_message_text("Произошла ошибка. Попробуйте еще раз.")
        if query.message:
            await query.message.reply_text(
                "Открыл главное меню — используйте кнопки ниже.",
                reply_markup=get_main_keyboard(),
            )


async def handle_exhibit_selection(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """
    Handle exhibit selection from list.
    """
    if not update.callback_query:
        return

    if update.effective_user:
        user_id = update.effective_user.id
        context.user_data[user_id] = context.user_data.get(user_id, {})
        context.user_data[user_id][CURRENT_EXHIBIT_KEY] = exhibit_id
        context.user_data[user_id].pop(SEARCH_MODE_KEY, None)
        context.user_data[user_id].pop(WAITING_FOR_QUESTION_KEY, None)

    agent = get_agent(context)
    metadata = agent.get_exhibit_info(exhibit_id)

    if not metadata:
        await safe_edit_text(
            "Экспонат не найден.", callback_query=update.callback_query
        )
        if update.callback_query.message:
            await update.callback_query.message.reply_text(
                "Открыл главное меню — используйте кнопки ниже.",
                reply_markup=get_main_keyboard(),
            )
        return

    text = format_exhibit_info(metadata)
    await safe_edit_text(
        truncate_text(text),
        callback_query=update.callback_query,
        reply_markup=get_exhibit_keyboard(exhibit_id),
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_exhibit_info(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """
    Handle exhibit info request.
    """
    if not update.callback_query:
        return

    agent = get_agent(context)
    metadata = agent.get_exhibit_info(exhibit_id)

    if not metadata:
        await safe_edit_text(
            "Экспонат не найден.", callback_query=update.callback_query
        )
        if update.callback_query.message:
            await update.callback_query.message.reply_text(
                "Открыл главное меню — используйте кнопки ниже.",
                reply_markup=get_main_keyboard(),
            )
        return

    text = format_exhibit_info(metadata)
    await safe_edit_text(
        truncate_text(text),
        callback_query=update.callback_query,
        reply_markup=get_exhibit_keyboard(exhibit_id),
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_ask_question(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """
    Handle ask question request.
    """
    if not update.callback_query:
        return

    if update.effective_user:
        user_id = update.effective_user.id
        context.user_data[user_id] = context.user_data.get(user_id, {})
        context.user_data[user_id][CURRENT_EXHIBIT_KEY] = exhibit_id
        context.user_data[user_id][WAITING_FOR_QUESTION_KEY] = True
        context.user_data[user_id].pop(SEARCH_MODE_KEY, None)

    await safe_edit_text(
        "Задайте вопрос об экспонате текстом.",
        callback_query=update.callback_query,
        reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
    )


async def handle_search_faq(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """
    Handle FAQ search request.
    """
    if not update.callback_query:
        return

    await update.callback_query.answer("Ищу в FAQ...")

    message_text = ""
    if update.callback_query.message and update.callback_query.message.text:
        message_text = update.callback_query.message.text

    if not message_text:
        await safe_edit_text(
            "Напишите ваш вопрос для поиска в FAQ.",
            callback_query=update.callback_query,
            reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
        )
        return

    question = message_text.split("\n")[0].strip()
    if len(question) > 200:
        question = question[:200]

    try:
        agent = get_agent(context)
        results = await agent.search_faq(question=question, exhibit_id=exhibit_id)

        text = format_faq_results(results)
        await safe_edit_text(
            truncate_text(text),
            callback_query=update.callback_query,
            reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
            parse_mode=ParseMode.MARKDOWN,
        )

    except Exception as e:
        logger.error(f"Error in handle_search_faq: {e}", exc_info=True)
        await safe_edit_text(
            "Произошла ошибка при поиске в FAQ.",
            callback_query=update.callback_query,
            reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
        )


async def error_handler(
    update: Optional[Update], context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle errors.
    """
    logger.error(
        f"Exception while handling an update: {context.error}", exc_info=context.error
    )

    if isinstance(context.error, (TimedOut, NetworkError)):
        return

    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Произошла ошибка. Пожалуйста, попробуйте еще раз."
            )
        except Exception:
            pass
