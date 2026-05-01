"""Shared pytest fixtures and helpers for smart gallery guide tests."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.schemas import ExhibitMetadata, ExhibitSearchResult


def build_metadata(exhibit_id: str = "ex-1", **overrides: Any) -> ExhibitMetadata:
    """Construct a valid ExhibitMetadata with sensible defaults for tests."""
    defaults: dict[str, Any] = dict(
        exhibit_id=exhibit_id,
        title="Звёздная ночь",
        artist="Винсент Ван Гог",
        year="1889",
        image_path=f"data/exhibits/{exhibit_id}.jpg",
    )
    defaults.update(overrides)
    return ExhibitMetadata(**defaults)


def build_search_result(score: float, exhibit_id: str = "ex-1") -> ExhibitSearchResult:
    """Construct an ExhibitSearchResult wrapping a synthetic metadata object."""
    md = build_metadata(exhibit_id)
    return ExhibitSearchResult(
        exhibit_id=exhibit_id,
        title=md.title,
        similarity_score=score,
        metadata=md,
    )


@pytest.fixture
def text_encoder_mock() -> MagicMock:
    enc = MagicMock(name="TextEncoder")
    enc.encode_text.return_value = np.zeros(8, dtype=np.float32)
    return enc


@pytest.fixture
def vision_encoder_mock() -> MagicMock:
    enc = MagicMock(name="VisionEncoder")
    enc.encode_text.return_value = np.zeros(8, dtype=np.float32)
    enc.encode_image.return_value = np.zeros(8, dtype=np.float32)
    return enc


@pytest.fixture
def vector_db_mock() -> MagicMock:
    return MagicMock(name="VectorDatabase")


@pytest.fixture
def guide_agent(
    vector_db_mock: MagicMock,
    vision_encoder_mock: MagicMock,
    text_encoder_mock: MagicMock,
):
    """Build a GuideAgent with all heavy dependencies replaced by mocks.

    Web search is explicitly disabled by passing ``web_search=None`` so the
    agent becomes deterministic and isolated from the runtime
    ``web_search_enabled`` setting.
    """
    from core.agent import GuideAgent

    return GuideAgent(
        vector_db=vector_db_mock,
        vision_encoder=vision_encoder_mock,
        text_encoder=text_encoder_mock,
        web_search=None,
    )
