"""sections / chunks / sentences / claims — the hierarchical content
representation from Data/Schema spec Doc 4 sec 5, 10-14:

  Source -> Section -> Chunk -> Sentence -> Claim

`chunks.text` is the original evidence (authoritative for citation);
`chunks.contextualized_text` is the retrieval-only enriched representation
(TRD Doc 2 sec 11, AI/RAG spec Doc 5 sec 15-16) — never shown as evidence.
"""

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPKMixin

CLAIM_TYPES = ("SOURCE_CLAIM", "GENERATED_CLAIM")


class Section(UUIDPKMixin, Base):
    __tablename__ = "sections"

    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), index=True)
    parent_section_id: Mapped[str | None] = mapped_column(ForeignKey("sections.id"), nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    section_order: Mapped[int] = mapped_column(Integer, default=0)


class Chunk(UUIDPKMixin, Base):
    __tablename__ = "chunks"

    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), index=True)
    section_id: Mapped[str] = mapped_column(ForeignKey("sections.id"), index=True)
    parent_chunk_id: Mapped[str | None] = mapped_column(ForeignKey("chunks.id"), nullable=True, index=True)
    chunk_order: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    contextualized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Sentence(UUIDPKMixin, Base):
    __tablename__ = "sentences"

    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), index=True)
    sentence_order: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    start_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)


class Claim(UUIDPKMixin, Base):
    __tablename__ = "claims"

    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.id"), nullable=True)
    sentence_id: Mapped[str | None] = mapped_column(ForeignKey("sentences.id"), nullable=True)
    message_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id"), nullable=True, index=True)
    claim_text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String)
