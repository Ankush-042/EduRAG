"""retrieval_runs / verification_results — query-time diagnostics used for
evaluation and failure diagnosis (AI/RAG spec Doc 5 sec 58, 72-74; Data/
Schema spec Doc 4 sec 19, 21). Not shown to normal users — developer/eval
mode only (UI/UX spec Doc 3 sec 42-44)."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPKMixin, _utcnow

NLI_VERDICTS = ("ENTAILMENT", "NOT_SUPPORTED")


class RetrievalRun(UUIDPKMixin, Base):
    __tablename__ = "retrieval_runs"

    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), index=True)
    query_text: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_scope: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dense_candidates: Mapped[int] = mapped_column(Integer, default=0)
    bm25_candidates: Mapped[int] = mapped_column(Integer, default=0)
    fused_candidates: Mapped[int] = mapped_column(Integer, default=0)
    reranked_candidates: Mapped[int] = mapped_column(Integer, default=0)
    final_evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    retrieval_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reranking_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generation_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verification_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class VerificationResult(UUIDPKMixin, Base):
    __tablename__ = "verification_results"

    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), index=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id"), index=True)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"))
    verdict: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
