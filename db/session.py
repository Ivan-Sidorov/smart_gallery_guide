"""Async SQLAlchemy engine."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Process-wide singleton async engine."""
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Singleton ``async_sessionmaker``."""
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
        class_=AsyncSession,
    )


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Async context manager that yields a session and commits/rolls back on exit."""
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
