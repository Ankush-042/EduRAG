"""sources + source_artifacts (Data/Schema spec Doc 4 sec 7-8).

A source belongs to exactly one session (no cross-session retrieval, per
sec 24's mandatory isolation rule). Large files never live in Postgres/
SQLite — source_artifacts stores only their metadata/paths."""

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPKMixin

SOURCE_TYPES = ("YOUTUBE", "LOCAL_VIDEO", "PDF", "DOCUMENT")

SOURCE_STATUS_VALUES = (
    "QUEUED", "DOWNLOADING", "EXTRACTING", "TRANSCRIBING", "PROCESSING",
    "INDEXING", "READY", "FAILED", "CANCELLED", "DELETED",
)

ARTIFACT_TYPES = (
    "ORIGINAL_MEDIA", "ORIGINAL_DOCUMENT", "AUDIO", "RAW_TRANSCRIPT",
    "NORMALIZED_TRANSCRIPT", "TIMESTAMP_TRANSCRIPT", "EXTRACTED_TEXT",
    "PROCESSING_LOG",
)


class Source(UUIDPKMixin, TimestampMixin, Base):
    __tablename__ = "sources"

    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    source_type: Mapped[str] = mapped_column(String)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String, nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String, default="QUEUED", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class SourceArtifact(UUIDPKMixin, Base):
    __tablename__ = "source_artifacts"

    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), index=True)
    artifact_type: Mapped[str] = mapped_column(String)
    storage_path: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String, nullable=True)
