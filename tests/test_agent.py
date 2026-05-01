"""Unit tests for GuideAgent with mocked dependencies."""

from unittest.mock import MagicMock

from tests.conftest import build_metadata, build_search_result


async def test_search_by_text_returns_title_hits_when_confident(
    guide_agent, vector_db_mock: MagicMock, text_encoder_mock: MagicMock
) -> None:
    """Test that the agent returns title hits when confident."""
    title_hit = build_search_result(score=0.92)
    vector_db_mock.search_text.return_value = [title_hit]

    results = await guide_agent.search_exhibits_by_text("ван гог")

    assert results == [title_hit]
    text_encoder_mock.encode_text.assert_called_once_with("ван гог")

    variants_tried = [
        call.kwargs.get("variant") for call in vector_db_mock.search_text.call_args_list
    ]
    assert variants_tried == ["title"]
    vector_db_mock.search_exhibit.assert_not_called()


async def test_search_by_text_falls_back_to_description_index(
    guide_agent, vector_db_mock: MagicMock
) -> None:
    """Test that the agent falls back to the description index when the title index is not confident."""
    weak_title = build_search_result(score=0.45)
    strong_desc = build_search_result(score=0.80, exhibit_id="ex-2")

    def search_text_side_effect(*_args, **kwargs):
        return [weak_title] if kwargs.get("variant") == "title" else [strong_desc]

    vector_db_mock.search_text.side_effect = search_text_side_effect

    results = await guide_agent.search_exhibits_by_text("импрессионизм")

    assert results == [strong_desc]
    variants_tried = [
        call.kwargs.get("variant") for call in vector_db_mock.search_text.call_args_list
    ]
    assert variants_tried == ["title", "desc"]
    vector_db_mock.search_exhibit.assert_not_called()


async def test_search_by_text_falls_back_to_vision_cascade(
    guide_agent, vector_db_mock: MagicMock, vision_encoder_mock: MagicMock
) -> None:
    """Test that the agent falls back to the vision index when the title and description indexes are not confident."""
    weak_title = build_search_result(score=0.40)
    weak_desc = build_search_result(score=0.45)
    vision_hit = build_search_result(score=0.70, exhibit_id="ex-3")

    vector_db_mock.search_text.side_effect = [[weak_title], [weak_desc]]
    vector_db_mock.search_exhibit.return_value = [vision_hit]

    results = await guide_agent.search_exhibits_by_text("что-то редкое")

    assert results == [vision_hit]
    vision_encoder_mock.encode_text.assert_called_once_with("что-то редкое")
    vector_db_mock.search_exhibit.assert_called_once()


async def test_search_by_text_returns_empty_on_encoder_failure(
    guide_agent, text_encoder_mock: MagicMock
) -> None:
    """Test that the agent returns empty list when the text encoder fails."""
    text_encoder_mock.encode_text.side_effect = RuntimeError("encoder down")

    results = await guide_agent.search_exhibits_by_text("anything")

    assert results == []


async def test_close_is_idempotent_and_delegates_to_components(
    vector_db_mock: MagicMock,
    vision_encoder_mock: MagicMock,
    text_encoder_mock: MagicMock,
) -> None:
    """Test that the agent is idempotent and delegates to components."""
    from core.agent import GuideAgent

    vector_db_mock.close = MagicMock()
    agent = GuideAgent(
        vector_db=vector_db_mock,
        vision_encoder=vision_encoder_mock,
        text_encoder=text_encoder_mock,
        web_search=None,
    )

    await agent.close()
    await agent.close()

    assert vector_db_mock.close.call_count == 2


async def test_close_survives_failing_component(
    vector_db_mock: MagicMock,
    vision_encoder_mock: MagicMock,
    text_encoder_mock: MagicMock,
) -> None:
    """Test that the agent survives a failing component."""
    from core.agent import GuideAgent

    vector_db_mock.close = MagicMock(side_effect=RuntimeError("boom"))
    vision_encoder_mock.close = MagicMock()

    agent = GuideAgent(
        vector_db=vector_db_mock,
        vision_encoder=vision_encoder_mock,
        text_encoder=text_encoder_mock,
        web_search=None,
    )
    await agent.close()

    vector_db_mock.close.assert_called_once()
    vision_encoder_mock.close.assert_called_once()


async def test_async_context_manager_closes_on_exit(guide_agent) -> None:
    """Test that the agent closes on exit."""
    close_called = []

    async def tracking_close() -> None:
        close_called.append(True)

    guide_agent.close = tracking_close

    async with guide_agent as a:
        assert a is guide_agent

    assert close_called == [True]


def test_build_exhibit_context_contains_primary_fields(guide_agent) -> None:
    """Test that the agent builds a context string with primary fields."""
    md = build_metadata(
        artist="Ван Гог",
        year="1889",
        material="холст, масло",
        school="Нидерланды",
        description="Описание экспоната.",
        additional_info={"techniq": "масло", "place": "Сен-Реми", "epoque": "XIX век"},
    )

    ctx = guide_agent._build_exhibit_context(md)

    for expected in [
        "Название: Звёздная ночь",
        "Автор: Ван Гог",
        "Год: 1889",
        "Материал: холст, масло",
        "Школа/страна: Нидерланды",
        "Техника: масло",
        "Место: Сен-Реми",
        "Эпоха: XIX век",
        "Описание: Описание экспоната.",
    ]:
        assert expected in ctx, f"Context is missing line: {expected!r}"


def test_build_exhibit_context_omits_absent_fields(guide_agent) -> None:
    """Test that the agent builds a context string without absent fields."""
    md = build_metadata(artist=None, year=None, material=None, school=None)
    ctx = guide_agent._build_exhibit_context(md)

    assert "Автор:" not in ctx
    assert "Год:" not in ctx
    assert "Материал:" not in ctx
    assert "Школа/страна:" not in ctx
    assert "Название: Звёздная ночь" in ctx
