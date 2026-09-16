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

Sprint 9 update: generation IS wired up now, and GROQ_API_KEY is already
a hard requirement for Q&A to work at all -- so the "don't make indexing
depend on a key that isn't configured yet" concern above no longer
applies by the time a real deployment reaches this code. _contextualize_chunks
below now does real per-chunk LLM contextualization (a short "how does
this excerpt fit the lecture" blurb from a small, fast Groq model,
prepended to the chunk before embedding -- the same shape as Anthropic's
contextual retrieval technique, just windowed to the chunk's immediate
neighbors rather than the whole document, to keep each call small and
this pipeline stage fast even on a long lecture) when
settings.enable_llm_contextualization and a key are both present, run
concurrently across chunks so a full lecture's chunk count doesn't turn
into that many sequential network round-trips. The original deterministic
prefix (_contextualize_chunk) is kept, unconditionally, as the fallback:
for the whole source when contextualization is disabled or no key is
configured, and per-chunk for any individual call that errors, times out,
or comes back empty -- one flaky Groq request degrades that one chunk's
retrieval quality, never the pipeline's reliability, matching this
codebase's established fallback-rather-than-block pattern (e.g. the GPU
-> CPU ASR fallback in transcription.py).
"""

from __future__ import annotations

import concurrent.futures
import pickle
import re
import threading
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

# Self-audit finding (post-Sprint-11): the check-then-set below had no
# lock. ingestion.py spawns one background pipeline thread per source, so
# two sources reaching indexing close together (a very natural thing to
# try -- add two sources back to back) could both see
# _VECTOR_STORE_CLIENT is None and both try to open the same local Qdrant
# path, which Qdrant refuses to do twice concurrently -- the loser gets a
# confusing FAILED status on a source whose content was never actually
# the problem.
_VECTOR_STORE_CLIENT_LOCK = threading.Lock()


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
        with _VECTOR_STORE_CLIENT_LOCK:
            # Re-check inside the lock -- another thread may have already
            # opened the client while this thread was waiting for the lock.
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


# Deliberately NOT given the whole document (the way Anthropic's own
# contextual-retrieval writeup does it) -- that's one prompt per chunk
# with the entire transcript as input every time, which for a full
# lecture would mean re-sending tens of thousands of tokens per chunk and
# would make ingestion slower, not "real quick," on exactly the long
# sources this app is built for. A short window of the immediately
# neighboring chunks' text is nearly as informative for "what is this
# excerpt part of" and keeps every call's input small and fast.
_CONTEXTUALIZATION_SYSTEM_PROMPT = (
    "You situate short lecture excerpts for a search index. You will be "
    "given a lecture's title, optionally a section title, the text just "
    "before this excerpt, the excerpt itself, and the text just after it. "
    "Write EXACTLY ONE short sentence (max 25 words) stating what topic or "
    "step in the lecture this excerpt is part of -- for example: 'Part of "
    "the lecture's explanation of variable swapping, covering why the "
    "naive one-line approach loses the original value.' Do not summarize "
    "or repeat the excerpt's own content, do not add any other commentary "
    "or preamble, and output nothing but that one sentence."
)

# Neighboring text is truncated before going into the prompt -- this is
# context FOR the excerpt, not more content to contextualize, so a long
# neighboring chunk shouldn't be allowed to dominate a call meant to be
# short and fast.
_NEIGHBOR_CHARS = 400


def _build_contextualization_prompt(chunk, prev_chunk, next_chunk, section, source: Source) -> str:
    lines = [f'Lecture title: "{source.title}"' if source.title else "Lecture title: (untitled)"]
    if section is not None and section.title and section.title != source.title:
        lines.append(f'Section: "{section.title}"')
    if prev_chunk is not None:
        lines.append(f"Text immediately before this excerpt: {prev_chunk.text[-_NEIGHBOR_CHARS:]}")
    lines.append(f"THIS EXCERPT: {chunk.text}")
    if next_chunk is not None:
        lines.append(f"Text immediately after this excerpt: {next_chunk.text[:_NEIGHBOR_CHARS]}")
    return "\n\n".join(lines)


def _llm_contextualize_chunk(client, model_name: str, chunk, prev_chunk, next_chunk, section, source: Source) -> str:
    """One Groq call producing a short contextual blurb for `chunk`. Raises
    on any failure (bad response, timeout, rate limit, empty content) --
    the caller (_contextualize_chunks) catches per-chunk and keeps that
    chunk's deterministic fallback instead, so this never needs to be
    defensive about its own errors."""
    from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

    prompt = _build_contextualization_prompt(chunk, prev_chunk, next_chunk, section, source)
    # Same reasoning as generation.py's Sprint 10 hardening: retry only
    # genuinely transient errors, bounded and explicit, rather than
    # relying on (or stacking on top of) the Groq SDK's own implicit
    # retry. Worth doing here even though a per-chunk fallback already
    # exists -- a chunk that recovers on retry gets the real LLM
    # contextualization instead of silently settling for the
    # deterministic fallback over what would've been a transient blip.
    retryer = Retrying(
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
        ),
        stop=stop_after_attempt(settings.contextualization_max_attempts),
        wait=wait_exponential(multiplier=0.5, max=settings.contextualization_retry_max_wait_s),
        reraise=True,
    )
    response = retryer(
        client.chat.completions.create,
        model=model_name,
        messages=[
            {"role": "system", "content": _CONTEXTUALIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=80,
        timeout=settings.contextualization_timeout_s,
    )
    choice = response.choices[0] if response.choices else None
    content = choice.message.content if choice and choice.message else None
    if not content or not content.strip():
        raise ValueError("empty contextualization response")
    return content.strip()


def _contextualize_chunks(chunks, section_for, source: Source) -> list[str]:
    """Produces one contextualization string per chunk, same order as
    `chunks`. See module docstring (Sprint 9 update) for the full
    reasoning -- in short: real per-chunk LLM contextualization when
    enabled and a Groq key is configured, run concurrently; the original
    deterministic prefix as the unconditional fallback, both for the
    whole source (disabled / no key) and per-chunk (any individual call
    that fails)."""
    deterministic = [_contextualize_chunk(chunk, section_for(chunk), source) for chunk in chunks]

    if not settings.enable_llm_contextualization or not settings.groq_api_key:
        return deterministic

    try:
        from groq import Groq
    except ImportError:
        return deterministic

    # max_retries=0: retry policy is owned explicitly by the tenacity
    # Retrying inside _llm_contextualize_chunk (same reasoning as
    # generation.py's Sprint 10 hardening) rather than also left active
    # here underneath it.
    client = Groq(api_key=settings.groq_api_key, max_retries=0)
    # Start from the safe fallback for every chunk; each successful LLM
    # call below overwrites just its own index. A total failure of this
    # whole function (e.g. Groq unreachable) still leaves a fully valid,
    # fully deterministic result -- nothing here can make indexing worse
    # than it was before this feature existed.
    results = list(deterministic)

    def _task(i: int) -> str:
        prev_chunk = chunks[i - 1] if i > 0 else None
        next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
        return _llm_contextualize_chunk(
            client,
            settings.contextualization_model,
            chunks[i],
            prev_chunk,
            next_chunk,
            section_for(chunks[i]),
            source,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.contextualization_max_workers) as pool:
        future_to_index = {pool.submit(_task, i): i for i in range(len(chunks))}
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                blurb = future.result()
            except Exception:
                # Leave results[i] as the deterministic fallback already
                # in place -- see the module docstring on why a per-chunk
                # failure here is a quality degradation, never a pipeline
                # failure.
                continue
            results[i] = f"{blurb}\n\n{chunks[i].text}"

    return results


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
    # Self-audit finding (post-Sprint-11): this commit used to sit after
    # _contextualize_chunks below instead of before it. start_job() only
    # flushes (app/db/repositories/processing_job_repository.py), so that
    # write stayed open across the ENTIRE contextualization step -- real
    # Groq network calls, concurrent, with retries -- reproducing the
    # exact "hold a write transaction open across a slow operation" bug
    # class app/db/session.py's docstring documents fixing everywhere
    # else in this pipeline. Moved here, before any slow step, matching
    # every other stage in this same function below.
    db.commit()

    chunks = content.list_chunks_for_source(db, source.id)
    if not chunks:
        raise IndexingError("No chunks found for this source — content structuring may have failed.")

    section_cache: dict[str, object] = {}

    def _section_for(chunk):
        if chunk.section_id not in section_cache:
            section_cache[chunk.section_id] = content.get_section(db, chunk.section_id)
        return section_cache[chunk.section_id]

    contextualized_texts = _contextualize_chunks(chunks, _section_for, source)

    # Whole-pipeline percentage, not a private per-function scale -- see
    # the matching comment in ingestion.py's _extract_audio.
    jobs.update_progress(db, job, progress=0.85, stage="EMBEDDING")
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

    jobs.update_progress(db, job, progress=0.90, stage="WRITING_VECTOR_INDEX")
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

    jobs.update_progress(db, job, progress=0.95, stage="BUILDING_SPARSE_INDEX")
    db.commit()

    try:
        _rebuild_bm25_index(db, source.session_id)
    except Exception as exc:  # rank_bm25/pickle raise their own types
        raise IndexingError(f"Building the keyword (BM25) index failed: {exc}") from exc

    jobs.update_progress(db, job, progress=0.98, stage="FINALIZING")

    sources.update_source_status(db, source, "READY")
