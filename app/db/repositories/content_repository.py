"""Persistence for sections/chunks/sentences (Data/Schema spec Doc 4 sec
5, 10-14) — kept separate from the ORM models and from the structuring
service, matching the split already used for sources/jobs/sessions."""

from sqlalchemy.orm import Session as DbSession

from app.db.models.content import Chunk, Section, Sentence
from app.db.models.source import Source


def create_section(
    db: DbSession,
    *,
    source_id: str,
    section_order: int = 0,
    title: str | None = None,
    summary: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Section:
    section = Section(
        source_id=source_id,
        section_order=section_order,
        title=title,
        summary=summary,
        start_time=start_time,
        end_time=end_time,
    )
    db.add(section)
    db.flush()
    return section


def create_chunk(
    db: DbSession,
    *,
    source_id: str,
    section_id: str,
    chunk_order: int,
    text: str,
    start_time: float | None = None,
    end_time: float | None = None,
    token_count: int | None = None,
    language: str | None = None,
    content_hash: str | None = None,
) -> Chunk:
    chunk = Chunk(
        source_id=source_id,
        section_id=section_id,
        chunk_order=chunk_order,
        text=text,
        start_time=start_time,
        end_time=end_time,
        token_count=token_count,
        language=language,
        content_hash=content_hash,
    )
    db.add(chunk)
    db.flush()
    return chunk


def create_sentence(
    db: DbSession,
    *,
    chunk_id: str,
    sentence_order: int,
    text: str,
    start_time: float | None = None,
    end_time: float | None = None,
    content_hash: str | None = None,
) -> Sentence:
    sentence = Sentence(
        chunk_id=chunk_id,
        sentence_order=sentence_order,
        text=text,
        start_time=start_time,
        end_time=end_time,
        content_hash=content_hash,
    )
    db.add(sentence)
    db.flush()
    return sentence


def count_chunks_for_source(db: DbSession, source_id: str) -> int:
    return db.query(Chunk).filter(Chunk.source_id == source_id).count()


def list_chunks_for_source(db: DbSession, source_id: str) -> list[Chunk]:
    return (
        db.query(Chunk)
        .filter(Chunk.source_id == source_id)
        .order_by(Chunk.chunk_order.asc())
        .all()
    )


def get_section(db: DbSession, section_id: str) -> Section | None:
    return db.get(Section, section_id)


def set_chunk_embedding(
    db: DbSession,
    chunk: Chunk,
    *,
    contextualized_text: str,
    embedding_model: str,
    embedding_dimension: int,
) -> Chunk:
    chunk.contextualized_text = contextualized_text
    chunk.embedding_model = embedding_model
    chunk.embedding_dimension = embedding_dimension
    db.flush()
    return chunk


def list_chunks_for_session(db: DbSession, session_id: str) -> list[Chunk]:
    """All chunks across every (non-deleted) source in a session — the
    corpus a session's BM25/sparse index is built over. A join rather than
    a session_id column on Chunk itself: sections/chunks/sentences already
    key off source_id only (Data spec Doc 4 sec 5), and session isolation
    (sec 24) is enforced at the Source level everywhere else in this
    codebase too — duplicating session_id onto every content table would
    just be another place for it to drift out of sync."""
    return (
        db.query(Chunk)
        .join(Source, Chunk.source_id == Source.id)
        .filter(Source.session_id == session_id, Source.status != "DELETED")
        .order_by(Chunk.source_id.asc(), Chunk.chunk_order.asc())
        .all()
    )
