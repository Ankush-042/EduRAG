"""Persistence for sources + source_artifacts. Kept separate from the ORM
models themselves (TRD Doc 2 sec 38) and from the ingestion service (which
owns *what* happens; this owns *how it's stored*)."""

from sqlalchemy.orm import Session as DbSession

from app.db.models.source import Source, SourceArtifact


def create_source(
    db: DbSession,
    *,
    session_id: str,
    source_type: str,
    title: str | None = None,
    original_name: str | None = None,
    source_url: str | None = None,
    mime_type: str | None = None,
) -> Source:
    source = Source(
        session_id=session_id,
        source_type=source_type,
        title=title,
        original_name=original_name,
        source_url=source_url,
        mime_type=mime_type,
        status="QUEUED",
    )
    db.add(source)
    db.flush()
    return source


def get_source(db: DbSession, source_id: str) -> Source | None:
    return db.get(Source, source_id)


def list_sources_for_session(db: DbSession, session_id: str) -> list[Source]:
    return (
        db.query(Source)
        .filter(Source.session_id == session_id, Source.status != "DELETED")
        .order_by(Source.created_at.asc())
        .all()
    )


def update_source_status(
    db: DbSession, source: Source, status: str, *, error_message: str | None = None
) -> Source:
    source.status = status
    source.error_message = error_message
    db.flush()
    return source


def set_source_metadata(
    db: DbSession,
    source: Source,
    *,
    title: str | None = None,
    duration_seconds: int | None = None,
    file_size_bytes: int | None = None,
    content_hash: str | None = None,
    language: str | None = None,
) -> Source:
    if title is not None:
        source.title = title
    if duration_seconds is not None:
        source.duration_seconds = duration_seconds
    if file_size_bytes is not None:
        source.file_size_bytes = file_size_bytes
    if content_hash is not None:
        source.content_hash = content_hash
    if language is not None:
        source.language = language
    db.flush()
    return source


def get_latest_artifact(
    db: DbSession, *, source_id: str, artifact_type: str
) -> SourceArtifact | None:
    return (
        db.query(SourceArtifact)
        .filter(SourceArtifact.source_id == source_id, SourceArtifact.artifact_type == artifact_type)
        .order_by(SourceArtifact.id.desc())
        .first()
    )


def add_artifact(
    db: DbSession,
    *,
    source_id: str,
    artifact_type: str,
    storage_path: str,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    checksum: str | None = None,
) -> SourceArtifact:
    artifact = SourceArtifact(
        source_id=source_id,
        artifact_type=artifact_type,
        storage_path=storage_path,
        mime_type=mime_type,
        size_bytes=size_bytes,
        checksum=checksum,
    )
    db.add(artifact)
    db.flush()
    return artifact
