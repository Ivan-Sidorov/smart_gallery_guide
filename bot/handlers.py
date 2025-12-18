import logging
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
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


async def safe_edit_message_text(
    callback_query, text: str, reply_markup=None, parse_mode=None
) -> None:
    """
    Safely edit message text.

    Args:
        callback_query: Telegram callback query object
        text (str): Message text
        reply_markup: Optional reply markup
        parse_mode: Optional parse mode
    """
    try:
        await callback_query.edit_message_text(
            text, reply_markup=reply_markup, parse_mode=parse_mode
        )
    except BadRequest as e:
        if "Message is not modified" in str(e):
            await callback_query.answer()
        else:
            raise


def get_agent(context: ContextTypes.DEFAULT_TYPE) -> GuideAgent:
    """
    Get or create GuideAgent instance from context.

    Args:
        context (ContextTypes.DEFAULT_TYPE): Bot context

    Returns:
        GuideAgent: Agent instance
    """
    if AGENT_KEY not in context.bot_data:
        vector_db = VectorDatabase()
        context.bot_data[AGENT_KEY] = GuideAgent(vector_db=vector_db)
    return context.bot_data[AGENT_KEY]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start command.

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
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

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
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
        "4. *FAQ:*\n"
        "   Используйте кнопку 'Поиск в FAQ' для поиска похожих вопросов.\n\n"
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

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
    """
    if not update.message:
        return

    processing_msg = await update.message.reply_text("Обрабатываю изображение...")

    try:
        image = await download_photo(update, context)
        if not image:
            await processing_msg.edit_text("Не удалось загрузить изображение.")
            return

        agent = get_agent(context)

        await processing_msg.edit_text("Распознаю экспонат...")
        results = await agent.recognize_exhibit(image)

        if not results:
            await processing_msg.edit_text(
                "Экспонат не найден. Попробуйте другое изображение или "
                "используйте текстовый поиск."
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
            await processing_msg.edit_text(
                truncate_text(text),
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
            await processing_msg.edit_text(
                truncate_text(text),
                reply_markup=get_exhibits_list_keyboard(results),
                parse_mode=ParseMode.MARKDOWN,
            )

    except Exception as e:
        logger.error(f"Error in photo_handler: {e}", exc_info=True)
        await processing_msg.edit_text(
            "Произошла ошибка при обработке изображения. Попробуйте еще раз."
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle text messages - search exhibits or handle questions.

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
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
            "другого экспоната."
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
        processing_msg = await update.message.reply_text("Ищу экспонаты...")

        try:
            agent = get_agent(context)
            results = await agent.search_exhibits_by_text(text)

            if not results:
                await processing_msg.edit_text(
                    "Экспонаты не найдены. Попробуйте другой запрос или отправьте фото."
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
                await processing_msg.edit_text(
                    truncate_text(text_result),
                    reply_markup=get_exhibit_keyboard(exhibit_id),
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                text_result = format_exhibit_search_results(results)
                await processing_msg.edit_text(
                    truncate_text(text_result),
                    reply_markup=get_exhibits_list_keyboard(results),
                    parse_mode=ParseMode.MARKDOWN,
                )

        except Exception as e:
            logger.error(f"Error in text_handler: {e}", exc_info=True)
            await processing_msg.edit_text(
                "Произошла ошибка при поиске. Попробуйте еще раз."
            )
        return

    if not current_exhibit_id:
        await update.message.reply_text(
            "Сначала выберите экспонат: отправьте его фото или нажмите «Поиск экспонатов»."
        )
        return

    processing_msg = await update.message.reply_text("Отвечаю на вопрос...")

    try:
        agent = get_agent(context)
        answer = await agent.answer_question_about_exhibit(
            question=text, exhibit_id=current_exhibit_id
        )

        if user_id:
            user_data.pop(WAITING_FOR_QUESTION_KEY, None)

        await processing_msg.edit_text(
            truncate_text(format_vlm_answer(answer)),
            reply_markup=get_back_to_exhibit_keyboard(current_exhibit_id),
        )

    except Exception as e:
        logger.error(f"Error in text_handler: {e}", exc_info=True)
        await processing_msg.edit_text(
            "Произошла ошибка при обработке вопроса. Попробуйте еще раз."
        )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle callback queries from inline buttons.

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
    """
    if not update.callback_query:
        return

    query = update.callback_query
    await query.answer()

    callback_data = query.data
    if not callback_data:
        return

    try:
        if callback_data.startswith("select_exhibit:"):
            exhibit_id = callback_data.split(":", 1)[1]
            await handle_exhibit_selection(update, context, exhibit_id)

        elif callback_data.startswith("exhibit_info:"):
            exhibit_id = callback_data.split(":", 1)[1]
            await handle_exhibit_info(update, context, exhibit_id)

        elif callback_data.startswith("ask_question:"):
            exhibit_id = callback_data.split(":", 1)[1]
            await handle_ask_question(update, context, exhibit_id)

        elif callback_data.startswith("search_faq:"):
            exhibit_id = callback_data.split(":", 1)[1]
            await handle_search_faq(update, context, exhibit_id)

    except Exception as e:
        logger.error(f"Error in callback_handler: {e}", exc_info=True)
        await query.edit_message_text("Произошла ошибка. Попробуйте еще раз.")


async def handle_exhibit_selection(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """
    Handle exhibit selection from list.

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
        exhibit_id (str): Selected exhibit ID
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
        await safe_edit_message_text(update.callback_query, "Экспонат не найден.")
        return

    text = format_exhibit_info(metadata)
    await safe_edit_message_text(
        update.callback_query,
        truncate_text(text),
        reply_markup=get_exhibit_keyboard(exhibit_id),
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_exhibit_info(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """
    Handle exhibit info request.

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
        exhibit_id (str): Exhibit ID
    """
    if not update.callback_query:
        return

    agent = get_agent(context)
    metadata = agent.get_exhibit_info(exhibit_id)

    if not metadata:
        await safe_edit_message_text(update.callback_query, "Экспонат не найден.")
        return

    text = format_exhibit_info(metadata)
    await safe_edit_message_text(
        update.callback_query,
        truncate_text(text),
        reply_markup=get_exhibit_keyboard(exhibit_id),
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_ask_question(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """
    Handle ask question request.

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
        exhibit_id (str): Exhibit ID
    """
    if not update.callback_query:
        return

    if update.effective_user:
        user_id = update.effective_user.id
        context.user_data[user_id] = context.user_data.get(user_id, {})
        context.user_data[user_id][CURRENT_EXHIBIT_KEY] = exhibit_id
        context.user_data[user_id][WAITING_FOR_QUESTION_KEY] = True
        context.user_data[user_id].pop(SEARCH_MODE_KEY, None)

    await safe_edit_message_text(
        update.callback_query,
        "Задайте вопрос об экспонате текстом.",
        reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
    )


async def handle_search_faq(
    update: Update, context: ContextTypes.DEFAULT_TYPE, exhibit_id: str
) -> None:
    """
    Handle FAQ search request.

    Args:
        update (Update): Telegram update object
        context (ContextTypes.DEFAULT_TYPE): Bot context
        exhibit_id (str): Exhibit ID
    """
    if not update.callback_query:
        return

    await update.callback_query.answer("Ищу в FAQ...")

    message_text = ""
    if update.callback_query.message and update.callback_query.message.text:
        message_text = update.callback_query.message.text

    if not message_text:
        await safe_edit_message_text(
            update.callback_query,
            "Напишите ваш вопрос для поиска в FAQ.",
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
        await safe_edit_message_text(
            update.callback_query,
            truncate_text(text),
            reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
            parse_mode=ParseMode.MARKDOWN,
        )

    except Exception as e:
        logger.error(f"Error in handle_search_faq: {e}", exc_info=True)
        await safe_edit_message_text(
            update.callback_query,
            "Произошла ошибка при поиске в FAQ.",
            reply_markup=get_back_to_exhibit_keyboard(exhibit_id),
        )


async def error_handler(
    update: Optional[Update], context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle errors.

    Args:
        update (Optional[Update]): Telegram update object (may be None)
        context (ContextTypes.DEFAULT_TYPE): Bot context
    """
    logger.error(
        f"Exception while handling an update: {context.error}", exc_info=context.error
    )

    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Произошла ошибка. Пожалуйста, попробуйте еще раз."
            )
        except Exception:
            pass
