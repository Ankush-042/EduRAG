"""Sprint 2 — transcription: faster-whisper ASR with word-level timestamps,
text normalization, and transcript artifact storage (TRD Doc 2 sec 6,
AI/RAG spec Doc 5 principle 9; artifact shape per Data spec Doc 4 sec 8).

Picks up exactly where Sprint 1 (ingestion) left off: a source sitting in
TRANSCRIBING with an AUDIO artifact on disk. This module runs the ASR
model, writes three artifacts (raw / normalized / word-timestamps), and
hands the source off to PROCESSING for Sprint 3 (content structuring —
sections/chunks/sentences) to pick up.

Kept independent of app/services/ingestion.py: ingestion owns download/
save/extract, this owns ASR — the model-abstraction split from
app/core/interfaces.py applies here too (concrete engine is swappable
without touching the pipeline that calls it).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from sqlalchemy.orm import Session as DbSession

from app.core.config import get_settings
from app.core.interfaces import Transcriber, Transcript, TranscriptSegment, WordTimestamp
from app.db.models.processing import ProcessingJob
from app.db.models.source import Source
from app.db.repositories import processing_job_repository as jobs
from app.db.repositories import source_repository as sources

settings = get_settings()

# Loaded models are expensive (seconds to load, hundreds of MB) — cache by
# size so re-transcribing within the same process reuses the same model.
_MODEL_CACHE: dict[str, object] = {}


class TranscriptionError(Exception):
    """Raised for any ASR failure; the message is what the UI/DB shows."""


def _model_size(asr_model_spec: str) -> str:
    """settings.asr_model is "faster-whisper:base" — this is the only
    engine wired up right now, so we just take the size after the colon."""
    return asr_model_spec.split(":", 1)[1] if ":" in asr_model_spec else asr_model_spec


def _load_model(size: str):
    if size not in _MODEL_CACHE:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed (pip install -r requirements.txt)."
            ) from exc
        # CPU + int8: the "consumer laptop feasible" constraint (PRD Doc 1
        # sec 36) applies to the ASR model choice as much as it did to the
        # SQLite default in app/core/config.py — no CUDA setup required.
        _MODEL_CACHE[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return _MODEL_CACHE[size]


def _normalize_text(text: str) -> str:
    """Unicode-normalize and collapse whitespace. Deliberately NOT
    lowercasing or stripping punctuation — those would lose information
    the generation/citation layers need from the evidence text later."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_language_hint(language: str | None) -> str | None:
    """Source metadata (e.g. yt-dlp's reported language) comes back as
    locale-style codes like "en-US" or "pt-BR", but faster-whisper only
    accepts its own short list of bare ISO 639-1 codes ("en") and raises
    ValueError on anything else. Take just the base subtag; if there's
    nothing usable, return None and let Whisper auto-detect instead of
    guessing wrong."""
    if not language:
        return None
    base = re.split(r"[-_]", language, maxsplit=1)[0].strip().lower()
    return base or None


class WhisperTranscriber(Transcriber):
    """Concrete Transcriber (app/core/interfaces.py) backed by
    faster-whisper. Swapping ASR engines later means adding another class
    here, not touching this module's callers."""

    def __init__(self, model_spec: str):
        self._size = _model_size(model_spec)

    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
        model = _load_model(self._size)
        language = normalize_language_hint(language)

        try:
            segments, info = self._run(model, audio_path, language)
        except ValueError as exc:
            if language is not None and "not a valid language code" in str(exc):
                # The normalized hint still wasn't one faster-whisper
                # recognizes (a handful of yt-dlp/locale codes don't map
                # 1:1 onto Whisper's list) — auto-detect rather than
                # failing the whole source over a metadata quirk.
                segments, info = self._run(model, audio_path, None)
            else:
                raise TranscriptionError(f"Transcription failed: {exc}") from exc
        except Exception as exc:  # faster-whisper/ctranslate2 raise their own types
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        return Transcript(language=info.language, segments=segments)

    @staticmethod
    def _run(model, audio_path: str, language: str | None):
        segments_iter, info = model.transcribe(
            audio_path, language=language, word_timestamps=True, vad_filter=True
        )
        segments = []
        for seg in segments_iter:
            words = [
                WordTimestamp(word=w.word.strip(), start=w.start, end=w.end)
                for w in (seg.words or [])
            ]
            segments.append(
                TranscriptSegment(text=seg.text.strip(), start=seg.start, end=seg.end, words=words)
            )
        return segments, info


def _get_transcriber() -> WhisperTranscriber:
    return WhisperTranscriber(settings.asr_model)


def transcribe_source(db: DbSession, source: Source, job: ProcessingJob) -> None:
    """Runs ASR on the source's AUDIO artifact, writes transcript
    artifacts, and moves the source to PROCESSING. Raises
    TranscriptionError on any failure — callers (ingestion.py) already
    wrap this in a try/except that marks the source FAILED."""
    jobs.start_job(db, job, stage="TRANSCRIBING")

    audio_artifact = sources.get_latest_artifact(db, source_id=source.id, artifact_type="AUDIO")
    if audio_artifact is None:
        raise TranscriptionError("No audio artifact found for this source — audio extraction may have failed.")

    transcript = _get_transcriber().transcribe(audio_artifact.storage_path, language=source.language)
    if not transcript.segments:
        raise TranscriptionError("No speech was detected in the audio.")

    jobs.update_progress(db, job, progress=0.6, stage="NORMALIZING_TRANSCRIPT")

    raw_payload = {
        "language": transcript.language,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words],
            }
            for seg in transcript.segments
        ],
    }
    normalized_payload = {
        "language": transcript.language,
        "segments": [
            {"text": _normalize_text(seg.text), "start": seg.start, "end": seg.end}
            for seg in transcript.segments
        ],
    }
    timestamps_payload = {
        "language": transcript.language,
        "words": [
            {"word": w.word, "start": w.start, "end": w.end}
            for seg in transcript.segments
            for w in seg.words
        ],
    }

    # Resolved to absolute here, at write time — settings.transcript_dir is
    # relative by default, and a relative path stored in the DB would only
    # resolve correctly again if a later process happens to share the same
    # cwd (true today since the app is always launched from the project
    # root, but not a safe thing to depend on for paths read back later).
    transcript_dir = Path(settings.transcript_dir).resolve()
    transcript_dir.mkdir(parents=True, exist_ok=True)

    raw_path = transcript_dir / f"{source.id}.raw.json"
    normalized_path = transcript_dir / f"{source.id}.normalized.json"
    timestamps_path = transcript_dir / f"{source.id}.timestamps.json"

    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    normalized_path.write_text(json.dumps(normalized_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timestamps_path.write_text(json.dumps(timestamps_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sources.add_artifact(
        db, source_id=source.id, artifact_type="RAW_TRANSCRIPT",
        storage_path=str(raw_path), mime_type="application/json", size_bytes=raw_path.stat().st_size,
    )
    sources.add_artifact(
        db, source_id=source.id, artifact_type="NORMALIZED_TRANSCRIPT",
        storage_path=str(normalized_path), mime_type="application/json", size_bytes=normalized_path.stat().st_size,
    )
    sources.add_artifact(
        db, source_id=source.id, artifact_type="TIMESTAMP_TRANSCRIPT",
        storage_path=str(timestamps_path), mime_type="application/json", size_bytes=timestamps_path.stat().st_size,
    )

    if not source.language:
        sources.set_source_metadata(db, source, language=transcript.language)

    jobs.update_progress(db, job, progress=0.9, stage="AWAITING_CONTENT_STRUCTURING")

    # Hand off to Sprint 3 (content structuring: sections/chunks/sentences).
    sources.update_source_status(db, source, "PROCESSING")
