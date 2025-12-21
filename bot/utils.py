import io
import logging
from typing import List, Optional

from PIL import Image
from telegram import Update
from telegram.ext import ContextTypes

from database.schemas import ExhibitMetadata, ExhibitSearchResult, FAQSearchResult

logger = logging.getLogger(__name__)


async def download_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> Optional[Image.Image]:
    """
    Download photo from Telegram message.
    """
    try:
        if not update.message or not update.message.photo:
            logger.warning("No photo in message")
            return None

        photo = update.message.photo[-1]

        file = await context.bot.get_file(photo.file_id)
        file_data = await file.download_as_bytearray()

        image = Image.open(io.BytesIO(file_data))

        logger.info(f"Downloaded photo: {photo.file_id}, size: {len(file_data)} bytes")
        return image

    except Exception as e:
        logger.error(f"Error downloading photo: {e}", exc_info=True)
        return None


def format_exhibit_info(metadata: ExhibitMetadata) -> str:
    """
    Format exhibit metadata into a readable message.
    """
    lines = [f"*{metadata.title}*"]

    if metadata.artist:
        lines.append(f"Художник: {metadata.artist}")

    if metadata.year:
        lines.append(f"Год: {metadata.year}")

    lines.append(f"\n{metadata.description}")

    if metadata.interesting_facts:
        lines.append("\n*Интересные факты:*")
        for i, fact in enumerate(metadata.interesting_facts, 1):
            lines.append(f"{i}. {fact}")

    return "\n".join(lines)


def format_exhibit_search_results(results: List[ExhibitSearchResult]) -> str:
    """
    Format exhibit search results into a readable message.
    """
    if not results:
        return "Экспонаты не найдены."

    if len(results) == 1:
        result = results[0]
        return format_exhibit_info(result.metadata)

    lines = [f"Найдено экспонатов: {len(results)}\n"]
    for i, result in enumerate(results, 1):
        score_percent = int(result.similarity_score * 100)
        line = f"{i}. *{result.title}*"
        if result.metadata.artist:
            line += f" — {result.metadata.artist}"
        line += f" (совпадение: {score_percent}%)"
        lines.append(line)

    return "\n".join(lines)


def format_faq_results(results: List[FAQSearchResult]) -> str:
    """
    Format FAQ search results into a readable message.
    """
    if not results:
        return "Похожие вопросы не найдены в базе FAQ."

    lines = ["*Похожие вопросы:*\n"]

    for i, result in enumerate(results, 1):
        lines.append(f"{i}. *Вопрос:* {result.question}")
        lines.append(f"   *Ответ:* {result.answer}")
        if i < len(results):
            lines.append("")

    return "\n".join(lines)


def format_vlm_answer(answer: str) -> str:
    """
    Format VLM answer for display.
    """
    if not answer:
        return "Не удалось получить ответ."

    return answer


def truncate_text(text: str, max_length: int = 4096) -> str:
    """
    Truncate text to maximum length for Telegram message.
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."
