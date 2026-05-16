"""Unit tests for ASR encoder helpers."""

from core.encoders.asr import _extract_text


def test_extract_text_from_dict() -> None:
    assert _extract_text({"text": "  привет  "}) == "привет"


def test_extract_text_from_list() -> None:
    assert _extract_text([{"text": "ответ"}]) == "ответ"


def test_extract_text_empty_dict() -> None:
    assert _extract_text({}) == ""
