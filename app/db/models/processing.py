"""processing_jobs — async ingestion state machine (TRD Doc 2 sec 40,
Data/Schema spec Doc 4 sec 18). One row per pipeline stage run so the UI
can show real progress, not a fake percentage (UI/UX spec Doc 3 sec 10)."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPKMixin

JOB_TYPES = (
    "DOWNLOAD", "AUDIO_EXTRACTION", "TRANSCRIPTION", "NORMALIZATION",
    "SECTIONING", "CHUNKING", "CONTEXTUAL_ENRICHMENT", "EMBEDDING",
    "BM25_INDEXING", "VECTOR_INDEXING", "FULL_PIPELINE",
)
JOB_STATUS_VALUES = ("PENDING", "RUNNING", "COMPLETED", "FAILED")


class ProcessingJob(UUIDPKMixin, Base):
    __tablename__ = "processing_jobs"

    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), index=True)
    job_type: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="PENDING", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
