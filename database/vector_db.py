"""Legacy import path for the ``database.vector_db`` module."""

from core.vector_db import HNSW_CONFIG, VectorDatabase

__all__ = ["VectorDatabase", "HNSW_CONFIG"]
