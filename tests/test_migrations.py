"""Integration test: migration applies cleanly and matches ORM models."""

import asyncio
import os

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from db import models  # noqa: F401
from db.base import Base

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    TEST_DATABASE_URL is None,
    reason="TEST_DATABASE_URL is not set, integration test skipped",
)


def _alembic_config() -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL or "")
    return cfg


def _drop_everything(url: str) -> None:
    """Drop all tables, indexes and enum types."""

    async def _run() -> None:
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
        await engine.dispose()

    asyncio.run(_run())


def _schema_diff(url: str) -> list:
    """Return the diff between live DB schema and ORM ``Base.metadata``."""

    def _diff(sync_conn: Connection) -> list:
        mc = MigrationContext.configure(
            sync_conn,
            opts={"compare_type": True, "compare_server_default": True},
        )
        return list(compare_metadata(mc, Base.metadata))

    async def _run() -> list:
        engine = create_async_engine(url)
        async with engine.connect() as conn:
            result = await conn.run_sync(_diff)
        await engine.dispose()
        return result

    return asyncio.run(_run())


def test_upgrade_then_downgrade_round_trip() -> None:
    """`alembic upgrade head` then `downgrade base` runs without errors."""
    assert TEST_DATABASE_URL is not None
    _drop_everything(TEST_DATABASE_URL)

    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")


def test_models_match_migration_state() -> None:
    """After `upgrade head`, ORM metadata must match the live DB schema.

    Equivalent to ``alembic check`` — catches forgotten columns, mistyped
    enum names, missing indexes, etc.
    """
    assert TEST_DATABASE_URL is not None
    _drop_everything(TEST_DATABASE_URL)

    cfg = _alembic_config()
    command.upgrade(cfg, "head")

    diff = _schema_diff(TEST_DATABASE_URL)
    assert (
        diff == []
    ), "ORM models and DB schema diverged after `upgrade head`:\n" + "\n".join(
        repr(d) for d in diff
    )


def test_migration_chain_is_linear() -> None:
    """All migration revisions form a single linear chain (no branches/orphans)."""
    cfg = _alembic_config()
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"Expected exactly one head, got {heads}"
