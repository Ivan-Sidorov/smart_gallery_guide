"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-01 20:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(length=64)),
        sa.Column("first_name", sa.String(length=128)),
        sa.Column("last_name", sa.String(length=128)),
        sa.Column("locale", sa.String(length=16)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )

    op.create_table(
        "exhibits",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("author", sa.String(length=256)),
        sa.Column("year", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("image_path", sa.String(length=1024)),
        sa.Column(
            "extra",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("source", sa.String(length=128)),
        sa.Column("checksum", sa.String(length=128)),
        sa.Column(
            "embedding_version", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "needs_reindex",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_exhibits")),
    )
    op.create_index(
        "ix_exhibits_needs_reindex_partial",
        "exhibits",
        ["needs_reindex"],
        postgresql_where=sa.text("needs_reindex = true"),
    )

    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column(
            "context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_sessions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sessions")),
    )
    op.create_index(op.f("ix_sessions_user_id"), "sessions", ["user_id"])

    op.create_table(
        "faq_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("exhibit_id", sa.String(length=128), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=128)),
        sa.Column(
            "embedding_version", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["exhibit_id"],
            ["exhibits.id"],
            name=op.f("fk_faq_items_exhibit_id_exhibits"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_faq_items")),
    )
    op.create_index(op.f("ix_faq_items_exhibit_id"), "faq_items", ["exhibit_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "direction",
            sa.Enum("in", "out", name="message_direction"),
            nullable=False,
        ),
        sa.Column(
            "type",
            sa.Enum("text", "photo", "callback", "system", name="message_type"),
            nullable=False,
        ),
        sa.Column("content", sa.Text()),
        sa.Column(
            "attachments",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("api_task_id", postgresql.UUID(as_uuid=True)),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name=op.f("fk_messages_session_id_sessions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_messages_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
    )
    op.create_index(
        "ix_messages_session_created", "messages", ["session_id", "created_at"]
    )
    op.create_index("ix_messages_user_created", "messages", ["user_id", "created_at"])

    op.create_table(
        "inference_tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "type",
            sa.Enum("vlm_qa", name="task_type"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "done", "error", name="task_status"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("user_id", sa.BigInteger()),
        sa.Column("session_id", postgresql.UUID(as_uuid=True)),
        sa.Column(
            "request",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("error", sa.Text()),
        sa.Column(
            "queued_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("worker", sa.String(length=128)),
        sa.Column("model", sa.String(length=256)),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_inference_tasks_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name=op.f("fk_inference_tasks_session_id_sessions"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_tasks")),
    )
    op.create_index(
        "ix_inference_tasks_status_queued_partial",
        "inference_tasks",
        ["status", "queued_at"],
        postgresql_where=sa.text("status in ('pending','running')"),
    )

    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", sa.BigInteger(), nullable=False),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_feedback_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
    )
    op.create_index(op.f("ix_feedback_message_id"), "feedback", ["message_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_feedback_message_id"), table_name="feedback")
    op.drop_table("feedback")

    op.drop_index(
        "ix_inference_tasks_status_queued_partial", table_name="inference_tasks"
    )
    op.drop_table("inference_tasks")

    op.drop_index("ix_messages_user_created", table_name="messages")
    op.drop_index("ix_messages_session_created", table_name="messages")
    op.drop_table("messages")

    op.drop_index(op.f("ix_faq_items_exhibit_id"), table_name="faq_items")
    op.drop_table("faq_items")

    op.drop_index(op.f("ix_sessions_user_id"), table_name="sessions")
    op.drop_table("sessions")

    op.drop_index("ix_exhibits_needs_reindex_partial", table_name="exhibits")
    op.drop_table("exhibits")

    op.drop_table("users")

    sa.Enum(name="task_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="task_type").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="message_type").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="message_direction").drop(op.get_bind(), checkfirst=True)
