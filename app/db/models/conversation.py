"""conversations / messages / evidence (Data/Schema spec Doc 4 sec 15-17).

`evidence` is the table that makes EduRAG's central promise mechanical: it
is the join between an answer and the exact chunk/sentence/source/timestamp
that backs it (spec sec 32-33's lineage requirement)."""

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPKMixin

MESSAGE_ROLES = ("USER", "ASSISTANT", "SYSTEM")
GROUNDING_STATUS_VALUES = ("GROUNDED", "PARTIALLY_GROUNDED", "UNVERIFIED", "ABSTAINED")


class Conversation(UUIDPKMixin, TimestampMixin, Base):
    __tablename__ = "conversations"

    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)


class Message(UUIDPKMixin, Base):
    __tablename__ = "messages"

    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    message_order: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    grounding_status: Mapped[str | None] = mapped_column(String, nullable=True)


class Evidence(UUIDPKMixin, Base):
    __tablename__ = "evidence"

    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), index=True)
    sentence_id: Mapped[str | None] = mapped_column(ForeignKey("sentences.id"), nullable=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    start_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reranker_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    nli_score: Mapped[float | None] = mapped_column(Float, nullable=True)
