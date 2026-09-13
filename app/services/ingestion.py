"""Sprint 1 — source ingestion: validation, download, audio extraction, and
the state-machine transitions from TRD Doc 2 sec 40 / Data spec Doc 4 sec 7:

    QUEUED -> DOWNLOADING -> EXTRACTING -> TRANSCRIBING -> PROCESSING -> ...

This module drives sources through download/save + audio extraction, then
(as of Sprint 2) straight into transcription via app.services.transcription
— there's no background job queue in this MVP (Doc 6 authority: no Celery/
Redis), so the full pipeline runs synchronously per add_*_source call and
lands the source in PROCESSING, ready for Sprint 3 (content structuring).
Every step is wrapped so a failure lands the source in FAILED with a
readable error_message rather than leaving it stuck (TRD Doc 2 sec 24).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session as DbSession

from app.core.config import get_settings
from app.db.models.source import Source
from app.db.repositories import processing_job_repository as jobs
from app.db.repositories import source_repository as sources
from app.services.transcription import normalize_language_hint, transcribe_source

settings = get_settings()


class IngestionError(Exception):
    """Raised for any ingestion failure; the message is what the UI/DB shows."""


@dataclass
class UploadedFile:
    """Minimal shape the service needs from a Streamlit UploadedFile-like
    object, so this module has no direct Streamlit dependency."""

    name: str
    read_bytes: bytes
    mime_type: str | None = None


def _ensure_dirs() -> None:
    for d in (settings.media_dir, settings.audio_dir, settings.transcript_dir):
        Path(d).mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_audio_ffmpeg(input_path: Path, output_path: Path) -> None:
    """16kHz mono WAV — the format ASR models expect, produced once here
    rather than re-decoded on every transcription attempt."""
    if shutil.which("ffmpeg") is None:
        raise IngestionError(
            "ffmpeg is not installed or not on PATH. Install it (e.g. `winget install ffmpeg` "
            "or `choco install ffmpeg` on Windows) and try again."
        )
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ac", "1", "-ar", "16000",
            str(output_path),
        ],
        capture_output=True,
        # Explicit UTF-8 rather than text=True's default of
        # locale.getpreferredencoding() — on Windows that's usually cp1252,
        # which raises UnicodeDecodeError the moment ffmpeg echoes a
        # non-ASCII video title (Hindi, etc.) to stderr. errors="replace"
        # so a still-undecodable byte can never crash the error path itself.
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise IngestionError(f"Audio extraction failed: {result.stderr[-800:]}")


class SourceIngestionService:
    def __init__(self, db: DbSession):
        self.db = db
        _ensure_dirs()

    # -- YouTube ---------------------------------------------------------

    def add_youtube_source(self, session_id: str, url: str) -> Source:
        url = url.strip()
        if not url or ("youtube.com" not in url and "youtu.be" not in url):
            raise IngestionError("That doesn't look like a YouTube URL.")

        source = sources.create_source(
            self.db, session_id=session_id, source_type="YOUTUBE", source_url=url
        )
        job = jobs.create_job(self.db, source_id=source.id, job_type="FULL_PIPELINE")

        try:
            self._download_youtube(source, job)
            self._extract_audio(source, job)
            transcribe_source(self.db, source, job)
            jobs.complete_job(self.db, job)
        except Exception as exc:  # noqa: BLE001 — deliberately broad: any
            # failure here must land the source in FAILED, never half-done.
            sources.update_source_status(self.db, source, "FAILED", error_message=str(exc))
            jobs.fail_job(self.db, job, error_message=str(exc))
        return source

    def _download_youtube(self, source: Source, job) -> None:
        try:
            import yt_dlp
        except ImportError as exc:
            raise IngestionError("yt-dlp is not installed (pip install -r requirements.txt).") from exc

        sources.update_source_status(self.db, source, "DOWNLOADING")
        jobs.start_job(self.db, job, stage="DOWNLOADING")

        out_template = str(Path(settings.media_dir) / f"{source.id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source.source_url, download=True)
                media_path = Path(ydl.prepare_filename(info)).resolve()
        except Exception as exc:  # yt-dlp raises its own exception types
            hint = ""
            if "403" in str(exc):
                # YouTube regularly changes its throttling/cipher scheme;
                # this is near-always an out-of-date yt-dlp, not a real
                # permissions issue, and updating almost always fixes it
                # without changing anything else about the request.
                hint = " (this is usually an outdated yt-dlp — try: python -m pip install -U yt-dlp, then retry)"
            raise IngestionError(f"Could not download the video: {exc}{hint}") from exc

        if not media_path.exists():
            raise IngestionError("Download reported success but the media file is missing.")

        content_hash = _sha256_file(media_path)
        sources.set_source_metadata(
            self.db,
            source,
            title=info.get("title"),
            duration_seconds=int(info.get("duration") or 0) or None,
            file_size_bytes=media_path.stat().st_size,
            content_hash=content_hash,
            language=normalize_language_hint(info.get("language")),
        )
        sources.add_artifact(
            self.db,
            source_id=source.id,
            artifact_type="ORIGINAL_MEDIA",
            storage_path=str(media_path),
            size_bytes=media_path.stat().st_size,
            checksum=content_hash,
        )
        self._media_path = media_path  # handed to _extract_audio below

    # -- Local video -------------------------------------------------------

    def add_local_video_source(self, session_id: str, upload: UploadedFile) -> Source:
        source = sources.create_source(
            self.db,
            session_id=session_id,
            source_type="LOCAL_VIDEO",
            original_name=upload.name,
            mime_type=upload.mime_type,
        )
        job = jobs.create_job(self.db, source_id=source.id, job_type="FULL_PIPELINE")

        try:
            self._save_local_upload(source, job, upload)
            self._extract_audio(source, job)
            transcribe_source(self.db, source, job)
            jobs.complete_job(self.db, job)
        except Exception as exc:  # noqa: BLE001
            sources.update_source_status(self.db, source, "FAILED", error_message=str(exc))
            jobs.fail_job(self.db, job, error_message=str(exc))
        return source

    def _save_local_upload(self, source: Source, job, upload: UploadedFile) -> None:
        sources.update_source_status(self.db, source, "EXTRACTING")
        jobs.start_job(self.db, job, stage="SAVING_UPLOAD")

        suffix = Path(upload.name).suffix or ".mp4"
        media_path = (Path(settings.media_dir) / f"{source.id}{suffix}").resolve()
        media_path.write_bytes(upload.read_bytes)

        content_hash = _sha256_file(media_path)
        sources.set_source_metadata(
            self.db,
            source,
            title=Path(upload.name).stem,
            file_size_bytes=media_path.stat().st_size,
            content_hash=content_hash,
        )
        sources.add_artifact(
            self.db,
            source_id=source.id,
            artifact_type="ORIGINAL_MEDIA",
            storage_path=str(media_path),
            mime_type=upload.mime_type,
            size_bytes=media_path.stat().st_size,
            checksum=content_hash,
        )
        self._media_path = media_path

    # -- Shared: audio extraction -----------------------------------------

    def _extract_audio(self, source: Source, job) -> None:
        sources.update_source_status(self.db, source, "EXTRACTING")
        jobs.update_progress(self.db, job, progress=0.5, stage="EXTRACTING_AUDIO")

        audio_path = (Path(settings.audio_dir) / f"{source.id}.wav").resolve()
        _extract_audio_ffmpeg(self._media_path, audio_path)

        sources.add_artifact(
            self.db,
            source_id=source.id,
            artifact_type="AUDIO",
            storage_path=str(audio_path),
            mime_type="audio/wav",
            size_bytes=audio_path.stat().st_size if audio_path.exists() else None,
        )

        # transcribe_source() (Sprint 2) picks up from here and carries the
        # job/status the rest of the way to PROCESSING.
        sources.update_source_status(self.db, source, "TRANSCRIBING")
        jobs.update_progress(self.db, job, progress=0.4, stage="TRANSCRIBING")
