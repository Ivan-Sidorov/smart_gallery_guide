"""FastAPI backend for the Smart Gallery Guide service.

The `create_app` function is exposed lazily so that Telegram adapter and scripts that only need
`api.schemas` do not pull in heavy dependencies.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from api.main import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:  # noqa: D401
    if name == "create_app":
        from importlib import import_module

        module = import_module("api.main")
        value = module.create_app
        globals()[name] = value
        return value
    raise AttributeError(f"module 'api' has no attribute {name!r}")
