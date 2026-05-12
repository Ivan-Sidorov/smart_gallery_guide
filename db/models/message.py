"""Dialogue message ORM model."""

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
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class MessageDirection(str, enum.Enum):
    """Direction of a dialogue message."""

    IN = "in"
    OUT = "out"


class MessageType(str, enum.Enum):
    """Message type."""

    TEXT = "text"
    PHOTO = "photo"
    CALLBACK = "callback"
    SYSTEM = "system"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    """Return enum values for SQLAlchemy native enums."""
    return [member.value for member in enum_cls]


class Message(Base):
    """A single message exchanged with the user."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    direction: Mapped[MessageDirection] = mapped_column(
        Enum(MessageDirection, name="message_direction", values_callable=_enum_values),
        nullable=False,
    )
    type: Mapped[MessageType] = mapped_column(
        Enum(MessageType, name="message_type", values_callable=_enum_values),
        nullable=False,
    )
    content: Mapped[str | None] = mapped_column(Text)
    attachments: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    api_task_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
        Index("ix_messages_user_created", "user_id", "created_at"),
    )
