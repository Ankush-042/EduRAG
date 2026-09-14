"""Sprint 4 (part 2) — contextual enrichment + indexing: the chunks
Sprint 3 built become an actually-searchable corpus here (TRD Doc 2 sec
11, 14; AI/RAG spec Doc 5 sec 15-16) — dense vectors in Qdrant, a sparse
BM25 index alongside it. Retrieval itself (hybrid search, RRF fusion,
reranking) is Sprint 5's job; this module's output is what that pipeline
will read from.

Scope decision (Doc 6 authority): real contextual enrichment (an LLM call
per chunk describing how it fits the surrounding document, per Anthropic's
"contextual retrieval" technique that Doc 5 sec 15-16 references) needs a
generation provider actually configured — but GROQ_API_KEY is unset by
default (generation isn't wired up until a later sprint), and making
indexing depend on an API key/network call would turn "add a source" into
something that can fail on a missing key or a flaky request, for a
pipeline that's been reliable and fully local end-to-end so far. A
deterministic, local, non-LLM contextualization stands in for now:
prepend the source title and section context to each chunk before
embedding. This is a documented, reversible simplification, same pattern
as Sprint 3's one-Section-per-source call — revisit once generation is
wired up (the eval set in Sprint 11 can then compare LLM-contextualized
vs. this against retrieval quality directly).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from sqlalchemy.orm import Session as DbSession

from app.core.config import get_settings
from app.db.models.processing import ProcessingJob
from app.db.models.source import Source
from app.db.repositories import content_repository as content
from app.db.repositories import processing_job_repository as jobs
from app.db.repositories import source_repository as sources
from app.services.embedding import get_embedder

settings = get_settings()

# Resolved once, here, to an absolute path -- same reasoning as the
# transcript/artifact path fixes elsewhere in this codebase: Streamlit is
# always launched from the project root today, but a relative path stored
# or reused anywhere is fragile against that assumption ever changing.
_QDRANT_DIR = Path(settings.qdrant_path).resolve()

# Module-level singleton — see get_vector_store_client for why this isn't
# opened/closed per call.
_VECTOR_STORE_CLIENT = None


class IndexingError(Exception):
    """Raised for any indexing failure; the message is what the UI/DB shows."""


def get_vector_store_client():
    """Embedded/local Qdrant — a path on disk, not a server (see
    qdrant_path in config.py: zero external service to run, same call
    already made for SQLite over Postgres). Cached at module level and
    reused for the life of the process: qdrant-client's local mode holds
    an exclusive lock on the path while open, so repeatedly opening and
    closing it across Streamlit reruns would fight itself for that lock —
    one client, opened once, avoids that entirely."""
    global _VECTOR_STORE_CLIENT
    if _VECTOR_STORE_CLIENT is None:
        from qdrant_client import QdrantClient

        _QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        _VECTOR_STORE_CLIENT = QdrantClient(path=str(_QDRANT_DIR))
    return _VECTOR_STORE_CLIENT


def _ensure_collection(client, name: str, dimension: int) -> None:
    # get_collections()/create_collection() rather than the newer
    # collection_exists() helper -- these two have been stable across
    # qdrant-client versions for far longer, and this module can't be
    # tested against the real package before it reaches your machine (see
    # the standing sandbox-dependency note), so the more conservative,
    # longer-established API is the safer bet here.
    from qdrant_client.models import Distance, VectorParams

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )


def _format_timestamp(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _contextualize_chunk(chunk, section, source: Source) -> str:
    """Deterministic stand-in for LLM-based contextual retrieval — see
    module docstring. Prepends title/section/position so the embedded
    text carries document-level context a bare chunk wouldn't, without
    ever touching chunk.text itself (that stays the untouched original —
    per content.py's own docstring, contextualized_text is retrieval-only
    and must never be shown as evidence)."""
    bits = []
    if source.title:
        bits.append(f'From "{source.title}"')
    if section is not None and section.title and section.title != source.title:
        bits.append(f'section "{section.title}"')
    start = _format_timestamp(chunk.start_time)
    end = _format_timestamp(chunk.end_time)
    if start and end:
        bits.append(f"[{start}-{end}]")
    prefix = ", ".join(bits)
    return f"{prefix}: {chunk.text}" if prefix else chunk.text


_TOKEN_RE = re.compile(r"[a-z0-9]+")


# tokenize() and bm25_index_path() are public (not underscore-prefixed)
# because Sprint 5's retrieval.py needs the exact same tokenizer and the
# exact same on-disk index location to read back what this module wrote —
# they're shared contract between indexing and retrieval, not private
# internals, so they're named and exposed that way rather than reached
# into across a module boundary that implies they're private.


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokens — no stemming/lemmatization in
    v1 (documented simplification, same spirit as Sprint 3's ASR-segment-
    as-sentence call); revisit if the eval set shows BM25 recall
    suffering from it."""
    return _TOKEN_RE.findall(text.lower())


def bm25_index_path(session_id: str) -> Path:
    return _QDRANT_DIR.parent / "bm25" / f"{session_id}.pkl"


def _rebuild_bm25_index(db: DbSession, session_id: str) -> None:
    """Rebuilds the WHOLE session's BM25 index from scratch from every
    chunk currently in the DB for that session, rather than incrementally
    patching one. rank_bm25 has no incremental-update API to patch safely
    anyway, and a full rebuild is cheap at the corpus sizes a single
    session's worth of educational sources realistically reaches (this
    isn't web-scale search) — so the simple, always-correct option is
    also the practical one here, not just the easy one."""
    from rank_bm25 import BM25Okapi

    chunks = content.list_chunks_for_session(db, session_id)
    if not chunks:
        return

    chunk_ids = [c.id for c in chunks]
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    path = bm25_index_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"chunk_ids": chunk_ids, "bm25": bm25}, f)


def index_source(db: DbSession, source: Source, job: ProcessingJob) -> None:
    """Embeds + indexes every chunk for this source (dense -> Qdrant,
    sparse -> a rebuilt session-wide BM25 pickle), then moves the source
    to READY — the last stage of the ingestion pipeline. Raises
    IndexingError on any failure — callers (ingestion.py) already wrap
    this in a try/except that marks the source FAILED."""
    jobs.start_job(db, job, stage="INDEXING")

    chunks = content.list_chunks_for_source(db, source.id)
    if not chunks:
        raise IndexingError("No chunks found for this source — content structuring may have failed.")

    section_cache: dict[str, object] = {}

    def _section_for(chunk):
        if chunk.section_id not in section_cache:
            section_cache[chunk.section_id] = content.get_section(db, chunk.section_id)
        return section_cache[chunk.section_id]

    contextualized_texts = [
        _contextualize_chunk(chunk, _section_for(chunk), source) for chunk in chunks
    ]

    jobs.update_progress(db, job, progress=0.3, stage="EMBEDDING")
    # Commit before embedding -- can take a real while for a full lecture's
    # worth of chunks, and (see app/db/session.py) holding this write open
    # for that whole duration is exactly what caused the live "database is
    # locked" crash elsewhere in the pipeline.
    db.commit()

    embedder = get_embedder()
    try:
        vectors = embedder.embed_documents(contextualized_texts)
        dimension = embedder.dimension()
    except Exception as exc:  # sentence-transformers/torch raise their own types
        raise IndexingError(f"Embedding failed: {exc}") from exc

    jobs.update_progress(db, job, progress=0.6, stage="WRITING_VECTOR_INDEX")
    db.commit()  # same reasoning as above -- the qdrant upsert below is a separate store, no reason to hold SQLite's writer lock through it

    try:
        from qdrant_client.models import PointStruct

        client = get_vector_store_client()
        _ensure_collection(client, settings.qdrant_collection, dimension)

        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "source_id": source.id,
                    "session_id": source.session_id,
                    "chunk_id": chunk.id,
                    "section_id": chunk.section_id,
                    "text": chunk.text,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "chunk_order": chunk.chunk_order,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
    except Exception as exc:  # qdrant-client raises its own types too
        raise IndexingError(f"Writing to the vector store failed: {exc}") from exc

    for chunk, contextualized_text in zip(chunks, contextualized_texts):
        content.set_chunk_embedding(
            db,
            chunk,
            contextualized_text=contextualized_text,
            embedding_model=settings.embedding_model,
            embedding_dimension=dimension,
        )

    jobs.update_progress(db, job, progress=0.85, stage="BUILDING_SPARSE_INDEX")
    db.commit()

    try:
        _rebuild_bm25_index(db, source.session_id)
    except Exception as exc:  # rank_bm25/pickle raise their own types
        raise IndexingError(f"Building the keyword (BM25) index failed: {exc}") from exc

    jobs.update_progress(db, job, progress=0.95, stage="FINALIZING")

    sources.update_source_status(db, source, "READY")
