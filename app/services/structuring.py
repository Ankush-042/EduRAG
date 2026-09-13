"""Sprint 3 — content structuring: turns a flat transcript into the
Source -> Section -> Chunk -> Sentence hierarchy retrieval will actually
search over (Data/Schema spec Doc 4 sec 5, 10-14; TRD Doc 2 sec 9-10).

Scope decision (Doc 6 authority): real topic-boundary detection isn't
available yet — no topic-segmentation model is in the locked v1 set, and
that job naturally wants embeddings (Sprint 4) to measure where topics
actually shift, which don't exist before this step runs. Until then: one
Section per source, and "sentence" == one ASR segment. Whisper's own
VAD-based segmentation already produces natural utterance-level breaks
with accurate timestamps, which is a reasonable stand-in for a real
sentence splitter that would otherwise have to re-interpolate timestamps
by hand from word-level data. Both are documented, reversible
simplifications — revisit if the eval set (Sprint 11) shows either one
hurting retrieval or citation granularity.
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


def _group_segments(segments: list[dict], target_words: int) -> list[list[dict]]:
    """Packs segments into chunk-groups up to ~target_words each, never
    splitting a segment (our sentence unit) across two chunks. A single
    unusually long segment becomes its own (oversized) chunk rather than
    being cut mid-sentence."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_words = 0
    for seg in segments:
        seg_words = len(seg["text"].split())
        if current and current_words + seg_words > target_words:
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

    jobs.update_progress(db, job, progress=0.3, stage="CHUNKING")

    section = content.create_section(
        db,
        source_id=source.id,
        section_order=0,
        title=source.title,
        start_time=segments[0]["start"],
        end_time=segments[-1]["end"],
    )

    groups = _group_segments(segments, settings.chunk_target_words)

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

    jobs.update_progress(db, job, progress=0.9, stage="AWAITING_INDEXING")

    # Hand off to Sprint 4 (contextual enrichment, embeddings, indexing).
    sources.update_source_status(db, source, "INDEXING")
