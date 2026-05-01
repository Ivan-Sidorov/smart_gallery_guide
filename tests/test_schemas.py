"""Unit tests for Pydantic schemas that carry non-trivial logic."""

from tests.conftest import build_metadata


def test_display_description_prefers_primary_description() -> None:
    """Test that the display description prefers the primary description."""
    md = build_metadata(
        description="Основное описание",
        description_perplexity="Описание Perplexity",
        antic_art_description="Antique art description",
    )
    assert md.display_description == "Основное описание"


def test_display_description_falls_back_to_perplexity() -> None:
    """Test that the display description falls back to perplexity when the primary description is not present."""
    md = build_metadata(
        description=None,
        description_perplexity="Описание Perplexity",
    )
    assert md.display_description == "Описание Perplexity"


def test_display_description_builds_from_fields_when_no_text_present() -> None:
    """Test that the display description builds from fields when no text is present."""
    md = build_metadata(
        description=None,
        description_perplexity=None,
        antic_art_description=None,
        material="холст, масло",
        additional_info={
            "techniq": "постимпрессионизм",
            "place": "Сен-Реми-де-Прованс",
            "epoque": "XIX век",
        },
    )
    desc = md.display_description

    assert "Звёздная ночь" in desc
    assert "Ван Гог" in desc
    assert "1889" in desc
    assert "холст, масло" in desc
    assert "постимпрессионизм" in desc
    assert "Сен-Реми-де-Прованс" in desc
    assert "XIX век" in desc
