"""Inline/reply keyboards for the Telegram adapter."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

from api.schemas.exhibits import ExhibitSearchResultDTO


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Main reply keyboard shown under the input field."""
    keyboard = [
        ["Поиск экспонатов"],
        ["Помощь"],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def get_exhibits_list_keyboard(
    exhibits: list[ExhibitSearchResultDTO],
) -> InlineKeyboardMarkup:
    """Inline keyboard with buttons for selecting an exhibit from search results."""
    keyboard = []
    for i, exhibit in enumerate(exhibits, 1):
        title = exhibit.title
        if len(title) > 50:
            title = title[:47] + "..."
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"{i}. {title}",
                    callback_data=f"select_exhibit:{exhibit.exhibit_id}",
                )
            ]
        )

    return InlineKeyboardMarkup(keyboard)


def get_answer_keyboard(
    message_id: int | None,
    exhibit_id: str | None = None,
    *,
    show_rating: bool = True,
    show_comment_prompt: bool = False,
) -> InlineKeyboardMarkup | None:
    """Inline keyboard with like/dislike and optional back-to-exhibit button."""
    if message_id is None and exhibit_id is None:
        return None

    keyboard: list[list[InlineKeyboardButton]] = []
    if message_id is not None:
        if show_rating:
            keyboard.append(
                [
                    InlineKeyboardButton("👍", callback_data=f"fb:{message_id}:1"),
                    InlineKeyboardButton("👎", callback_data=f"fb:{message_id}:-1"),
                ]
            )
        elif show_comment_prompt:
            keyboard.append(
                [
                    InlineKeyboardButton(
                        "✏️ Оставить комментарий",
                        callback_data=f"fb_comment:{message_id}",
                    )
                ]
            )
    if exhibit_id:
        keyboard.append(
            [
                InlineKeyboardButton(
                    "Назад к экспонату", callback_data=f"exhibit_info:{exhibit_id}"
                )
            ]
        )
    return InlineKeyboardMarkup(keyboard) if keyboard else None


def get_back_to_exhibit_keyboard(exhibit_id: str) -> InlineKeyboardMarkup:
    """Single 'back to exhibit card' button."""
    keyboard = [
        [
            InlineKeyboardButton(
                "Назад к экспонату", callback_data=f"exhibit_info:{exhibit_id}"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_cancel_keyboard() -> ReplyKeyboardMarkup:
    """Reply keyboard with a single 'Cancel' button."""
    keyboard = [["Отмена"]]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        one_time_keyboard=True,
    )
