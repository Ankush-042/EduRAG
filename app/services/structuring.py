"""Sprint 3 — content structuring: turns a flat transcript into the
Source -> Section -> Chunk -> Sentence hierarchy retrieval will actually
search over (Data/Schema spec Doc 4 sec 5, 10-14; TRD Doc 2 sec 9-10).

Scope decision (Doc 6 authority), UPDATED post-Sprint-11: the original
call here was that real topic-boundary detection wasn't available yet --
that job naturally wants embeddings (Sprint 4) to measure where topics
actually shift, which didn't exist before this step ran at the time this
was written. That's no longer true: embeddings have existed since Sprint
4, and _group_segments now uses them (see its own docstring, and
enable_semantic_chunking in config.py) to cut chunk boundaries where the
topic actually shifts, not just where a word-count ceiling happens to
land. Still one Section per source, and "sentence" == one ASR segment --
Whisper's own VAD-based segmentation already produces natural utterance-
level breaks with accurate timestamps, which remains a reasonable stand-in
for a real sentence splitter that would otherwise have to re-interpolate
timestamps by hand from word-level data. Revisit if the eval set (Sprint
8's scripts/eval_answers.py) shows this hurting retrieval or citation
granularity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy.orm import Session as DbSession

from app.core.config import get_settings
from app.db.models.processing import ProcessingJob
from app.db.models.source import Source
from app.db.repositories import content_repository as content
from app.db.repositories import processing_job_repository as jobs
from app.db.repositories import source_repository as sources

settings = get_settings()


class StructuringError(Exception):
    """Raised for any structuring failure; the message is what the UI/DB shows."""


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Both operands come from Embedder.embed_documents(), which always
    returns L2-normalized vectors (embedding.py: normalize_embeddings=True)
    -- so the plain dot product already IS the cosine similarity, no norm
    division needed. Kept as its own function anyway so the "these vectors
    are already normalized" assumption is documented in exactly one place
    rather than inlined and easy to get wrong if that assumption ever
    changes."""
    return sum(x * y for x, y in zip(a, b))


def _group_segments(
    segments: list[dict],
    target_words: int,
    min_words: int = 0,
    embeddings: list[list[float]] | None = None,
    similarity_threshold: float = 0.5,
) -> list[list[dict]]:
    """Packs segments into chunk-groups, never splitting a segment (our
    sentence unit) across two chunks. A single unusually long segment
    becomes its own (oversized) chunk rather than being cut mid-sentence.

    Two independent reasons to end a group and start a new one:
      1. Hard cap (always active): adding the next segment would push the
         group past target_words. This is the original Sprint 3 behavior
         and remains a hard ceiling regardless of topic similarity -- a
         long run of on-topic segments still can't grow one chunk without
         bound.
      2. Topic boundary (only when embeddings is not None): the cosine
         similarity between this segment's embedding and the previous
         segment's embedding drops below similarity_threshold, AND the
         current group already has at least min_words -- so a topic dip
         across just one or two short segments can't fragment a chunk
         down to near-nothing. This is the new, semantic half; passing
         embeddings=None (or leaving min_words at its 0 default) reduces
         this function exactly to the original word-count-only behavior,
         which is what enable_semantic_chunking=False (config.py) relies
         on.

    embeddings, when given, must be the same length as segments and in
    the same order -- embeddings[i] is segments[i]'s vector. The
    similarity check compares embeddings[i-1] to embeddings[i], i.e.
    "how similar is this segment to the one right before it", which is
    what a topic-shift-at-this-point signal should measure.
    """
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_words = 0
    for i, seg in enumerate(segments):
        seg_words = len(seg["text"].split())

        hit_hard_cap = bool(current) and current_words + seg_words > target_words

        hit_topic_boundary = (
            not hit_hard_cap
            and embeddings is not None
            and current
            and current_words >= min_words
            and _cosine_similarity(embeddings[i - 1], embeddings[i]) < similarity_threshold
        )

        if hit_hard_cap or hit_topic_boundary:
            groups.append(current)
            current = []
            current_words = 0

        current.append(seg)
        current_words += seg_words
    if current:
        groups.append(current)
    return groups


def structure_source(db: DbSession, source: Source, job: ProcessingJob) -> None:
    """Reads the normalized transcript, builds one Section + chunked
    Sentences under it, and moves the source to INDEXING. Raises
    StructuringError on any failure — callers (ingestion.py) already wrap
    this in a try/except that marks the source FAILED."""
    jobs.start_job(db, job, stage="STRUCTURING")

    transcript_artifact = sources.get_latest_artifact(
        db, source_id=source.id, artifact_type="NORMALIZED_TRANSCRIPT"
    )
    if transcript_artifact is None:
        raise StructuringError(
            "No normalized transcript found for this source — transcription may have failed."
        )

    try:
        payload = json.loads(Path(transcript_artifact.storage_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuringError(f"Couldn't read the transcript artifact: {exc}") from exc

    segments = [seg for seg in payload.get("segments", []) if seg.get("text", "").strip()]
    if not segments:
        raise StructuringError("The transcript has no usable text to structure.")

    # Whole-pipeline percentage, not a private per-function scale -- see
    # the matching comment in ingestion.py's _extract_audio.
    jobs.update_progress(db, job, progress=0.68, stage="CHUNKING")

    section = content.create_section(
        db,
        source_id=source.id,
        section_order=0,
        title=source.title,
        start_time=segments[0]["start"],
        end_time=segments[-1]["end"],
    )

    segment_embeddings: list[list[float]] | None = None
    if settings.enable_semantic_chunking:
        try:
            from app.services.embedding import get_embedder

            segment_embeddings = get_embedder().embed_documents(
                [seg["text"] for seg in segments]
            )
        except Exception:
            # Same fallback shape as indexing.py's LLM contextualization:
            # semantic chunking is an enhancement over the word-count
            # packer, not a dependency of it. If the embedding model
            # can't load or errors out here (missing torch install, OOM,
            # whatever), fall back to the exact pre-existing word-count-
            # only behavior (embeddings=None) rather than failing the
            # whole source -- a source can still be structured and
            # indexed without this, just with less precise chunk
            # boundaries. The embedder gets called again moments later in
            # indexing.py for the real per-chunk vectors regardless, so
            # this isn't the only place a bad embedding setup would
            # surface -- it just shouldn't be the place structuring itself
            # dies.
            segment_embeddings = None

    groups = _group_segments(
        segments,
        settings.chunk_target_words,
        min_words=settings.chunk_min_words,
        embeddings=segment_embeddings,
        similarity_threshold=settings.semantic_chunk_similarity_threshold,
    )

    for chunk_order, group in enumerate(groups):
        chunk_text = " ".join(seg["text"] for seg in group)
        chunk = content.create_chunk(
            db,
            source_id=source.id,
            section_id=section.id,
            chunk_order=chunk_order,
            text=chunk_text,
            start_time=group[0]["start"],
            end_time=group[-1]["end"],
            # Word count, not a real tokenizer count — a cheap proxy until
            # Sprint 4 wires up the actual embedding model's tokenizer.
            token_count=len(chunk_text.split()),
            language=source.language,
            content_hash=_content_hash(chunk_text),
        )
        for sentence_order, seg in enumerate(group):
            content.create_sentence(
                db,
                chunk_id=chunk.id,
                sentence_order=sentence_order,
                text=seg["text"],
                start_time=seg["start"],
                end_time=seg["end"],
                content_hash=_content_hash(seg["text"]),
            )

    jobs.update_progress(db, job, progress=0.70, stage="AWAITING_INDEXING")

    # Hand off to Sprint 4 (contextual enrichment, embeddings, indexing).
    sources.update_source_status(db, source, "INDEXING")
