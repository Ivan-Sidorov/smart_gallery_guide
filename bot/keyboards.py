from typing import List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

from database.schemas import ExhibitSearchResult


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """
    Get main menu keyboard.
    """
    keyboard = [
        ["Поиск экспонатов"],
        ["Помощь"],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def get_exhibit_keyboard(exhibit_id: str) -> InlineKeyboardMarkup:
    """
    Get keyboard for a specific exhibit.
    """
    keyboard = [
        [
            InlineKeyboardButton(
                "Задать вопрос", callback_data=f"ask_question:{exhibit_id}"
            ),
        ],
        [
            InlineKeyboardButton(
                "Полная информация", callback_data=f"exhibit_info:{exhibit_id}"
            ),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_exhibits_list_keyboard(
    exhibits: List[ExhibitSearchResult],
) -> InlineKeyboardMarkup:
    """
    Get keyboard for selecting an exhibit from search results.
    """
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


def get_back_to_exhibit_keyboard(exhibit_id: str) -> InlineKeyboardMarkup:
    """
    Get keyboard with back button to return to exhibit view.
    """
    keyboard = [
        [
            InlineKeyboardButton(
                "Назад к экспонату", callback_data=f"exhibit_info:{exhibit_id}"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_cancel_keyboard() -> ReplyKeyboardMarkup:
    """
    Get keyboard with cancel button.
    """
    keyboard = [["Отмена"]]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        one_time_keyboard=True,
    )
