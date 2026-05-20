"""Inference task ORM model."""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class TaskType(str, enum.Enum):
    """Inference task type."""

    VLM_QA = "vlm_qa"


class TaskStatus(str, enum.Enum):
    """Inference task status."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    """Return enum values for SQLAlchemy native enums."""
    return [member.value for member in enum_cls]


class InferenceTask(Base):
    """Queued/processed ML inference task."""

    __tablename__ = "inference_tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    type: Mapped[TaskType] = mapped_column(
        Enum(TaskType, name="task_type", values_callable=_enum_values), nullable=False
    )
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, name="task_status", values_callable=_enum_values),
        nullable=False,
        default=TaskStatus.PENDING,
        server_default=TaskStatus.PENDING.value,
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL")
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL")
    )
    request: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker: Mapped[str | None] = mapped_column(String(128))
    model: Mapped[str | None] = mapped_column(String(256))

    __table_args__ = (
        Index(
            "ix_inference_tasks_status_queued_partial",
            "status",
            "queued_at",
            postgresql_where=text("status in ('pending','running')"),
        ),
    )
