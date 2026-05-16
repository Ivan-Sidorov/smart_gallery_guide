"""Telegram adapter entry point."""

import asyncio
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

from adapters.telegram.api_client import APIClient
from adapters.telegram.handlers import (
    API_CLIENT_KEY,
    callback_handler,
    error_handler,
    help_command,
    photo_handler,
    start_command,
    text_handler,
    voice_handler,
)
from adapters.telegram.settings import AdapterSettings, load_adapter_settings

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level.upper(),
    )


def _build_api_client(settings: AdapterSettings) -> APIClient:
    return APIClient(
        base_url=settings.backend_url,
        timeout_s=settings.http_timeout_s,
        request_id_header=settings.request_id_header,
        poll_initial_s=settings.task_poll_initial_s,
        poll_max_s=settings.task_poll_max_s,
        poll_factor=settings.task_poll_factor,
        poll_timeout_s=settings.task_poll_timeout_s,
    )


def build_application(settings: AdapterSettings) -> Application:
    """Construct a python-telegram-bot `Application` connected to API client."""
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    request = HTTPXRequest(
        connect_timeout=10.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0,
    )

    api_client = _build_api_client(settings)

    async def _post_init(app: Application) -> None:
        app.bot_data[API_CLIENT_KEY] = api_client
        logger.info("[adapter] API client ready (backend=%s)", settings.backend_url)

    async def _post_shutdown(app: Application) -> None:
        client = app.bot_data.get(API_CLIENT_KEY)
        if client is not None:
            await client.aclose()
            logger.info("[adapter] API client closed")

    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .request(request)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, voice_handler)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
    )
    application.add_handler(CallbackQueryHandler(callback_handler))
    application.add_error_handler(error_handler)

    return application


def main() -> None:
    """CLI entrypoint: `python -m adapters.telegram.app`."""
    settings = load_adapter_settings()
    _configure_logging(settings.log_level)

    if not settings.telegram_bot_token:
        logger.error("[adapter] TELEGRAM_BOT_TOKEN is not set; nothing to do.")
        sys.exit(1)

    application = build_application(settings)
    logger.info("[adapter] starting Telegram polling")

    try:
        application.run_polling(drop_pending_updates=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("[adapter] stopped by user")
    except Exception as exc:
        logger.exception("[adapter] fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
