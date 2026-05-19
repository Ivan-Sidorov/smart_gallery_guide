"""Tests for Telegram adapter text helpers."""

from adapters.telegram.utils import (
    UNAVAILABLE_ANSWER_TEXT,
    format_vlm_answer,
    is_unusable_model_answer,
)


def test_is_unusable_model_answer_detects_api_json() -> None:
    assert is_unusable_model_answer('{"detail":"model overloaded"}')


def test_is_unusable_model_answer_detects_vlm_error_prefix() -> None:
    assert is_unusable_model_answer(
        'Ошибка при обращении к VLM API: {"detail":"bad request"}'
    )


def test_is_unusable_model_answer_detects_vlm_no_answer_sentinel() -> None:
    assert is_unusable_model_answer("Не удалось получить ответ от модели.")


def test_format_vlm_answer_replaces_unusable() -> None:
    text = format_vlm_answer('{"detail":"down"}')
    assert "Сейчас не получается ответить" in text
    assert "detail" not in text


def test_format_vlm_answer_replaces_vlm_no_answer_sentinel() -> None:
    text = format_vlm_answer("Не удалось получить ответ от модели.")
    assert text == UNAVAILABLE_ANSWER_TEXT
    assert "Не удалось получить" not in text


def test_is_unusable_model_answer_detects_legacy_server_stub() -> None:
    assert is_unusable_model_answer("Сервер сейчас не отвечает. Попробуйте позже.")


def test_format_vlm_answer_replaces_legacy_server_stub() -> None:
    text = format_vlm_answer("Сервер сейчас не отвечает. Попробуйте позже.")
    assert text == UNAVAILABLE_ANSWER_TEXT
