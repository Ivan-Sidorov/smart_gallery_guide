"""Tests for VLM response normalization."""

from core.vlm.response import strip_vlm_reasoning


def test_strip_redacted_thinking_block() -> None:
    raw = "<think>\n" "Сначала подумаю о картине.\n" "</think>\n\n" "Иван Айвазовский"
    assert strip_vlm_reasoning(raw) == "Иван Айвазовский"


def test_strip_thinking_before_answer_tag() -> None:
    raw = "<think>hidden</think>" "<answer>Морской пейзаж</answer>"
    assert strip_vlm_reasoning(raw) == "Морской пейзаж"
