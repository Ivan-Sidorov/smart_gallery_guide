"""Smoke test: every first-class module must import cleanly."""

import importlib

import pytest

MODULES = [
    "core.settings",
    "core.schemas",
    "core.search.bm25",
    "core.search.web",
    "core.vlm.client",
    "core.vector_db",
    "core.sync",
    "api",
    "api.main",
    "db",
    "db.models",
    "db.repositories",
    "db.session",
    "adapters.telegram.api_client",
    "adapters.telegram.keyboards",
    "adapters.telegram.utils",
    "adapters.telegram.settings",
    "adapters.telegram.handlers",
    "adapters.telegram.app",
    "workers.vlm_worker",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None
