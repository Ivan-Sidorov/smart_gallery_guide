import logging
import sys

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from agent.agent import GuideAgent
from bot.handlers import (
    AGENT_KEY,
    callback_handler,
    error_handler,
    help_command,
    photo_handler,
    start_command,
    text_handler,
)
from database.vector_db import VectorDatabase
from models.text_encoder import TextEncoder
from models.vision_encoder import VisionEncoder
from config.config import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)

    logger.info("Preloading vector DB and models...")
    vector_db = VectorDatabase()
    vision_encoder = VisionEncoder()
    text_encoder = TextEncoder()
    agent = GuideAgent(
        vector_db=vector_db,
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
    )

    request = HTTPXRequest(
        connect_timeout=10.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0,
    )
    application = (
        Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()
    )
    application.bot_data[AGENT_KEY] = agent

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
    )

    application.add_handler(CallbackQueryHandler(callback_handler))

    application.add_error_handler(error_handler)

    logger.info("Bot handlers registered. Starting bot...")

    try:
        application.run_polling(
            drop_pending_updates=True,
        )
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
