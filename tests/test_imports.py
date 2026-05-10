"""Smoke test: every first-class module must import cleanly."""

import importlib

import pytest

MODULES = [
    "config.config",
    "database.schemas",
    "database.vector_db",
    "models.text_encoder",
    "models.vision_encoder",
    "models.vlm",
    "services.web_search",
    "agent.agent",
    "bot.handlers",
    "bot.bot",
    "db",
    "db.models",
    "db.repositories",
    "db.session",
    "core.sync",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None
