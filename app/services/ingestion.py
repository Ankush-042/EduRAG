"""Sprint 1 — source ingestion: validation, download, audio extraction, and
the state-machine transitions from TRD Doc 2 sec 40 / Data spec Doc 4 sec 7:

    QUEUED -> DOWNLOADING -> EXTRACTING -> TRANSCRIBING -> PROCESSING -> INDEXING -> READY

This module drives sources through download/save + audio extraction, then
straight into transcription (Sprint 2), content structuring (Sprint 3),
and indexing (Sprint 4) — there's no background job queue in this MVP
(Doc 6 authority: no Celery/Redis), so the full pipeline runs
synchronously per add_*_source call and lands the source in READY,
searchable by the retrieval pipeline Sprint 5 builds. Every step is
wrapped so a failure lands the source in FAILED with a readable
error_message rather than leaving it stuck (TRD Doc 2 sec 24).
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
from app.services.indexing import index_source
from app.services.structuring import structure_source
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
            structure_source(self.db, source, job)
            index_source(self.db, source, job)
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
            # As of yt-dlp's 2026 YouTube extractor, solving YouTube's JS
            # challenge (needed to get real, downloadable format URLs at
            # all) requires an external JS runtime + the small "EJS" script
            # that drives it (see yt-dlp wiki: "EJS"). `yt-dlp[default]`
            # (requirements.txt) already bundles the yt-dlp-ejs package, so
            # this normally needs nothing further -- but if that package is
            # ever missing/stale, letting yt-dlp fetch the script itself
            # from GitHub is the documented, supported fallback rather than
            # a hard failure. It only downloads a small script, on demand,
            # and does nothing when the bundled package already works.
            "remote_components": ["ejs:github"],
        }
        # NOTE: deliberately NOT forcing a specific player_client (e.g.
        # "tv") here. That was tried and reverted -- YouTube's tv client
        # extraction is itself in an actively broken state right now
        # (yt-dlp issue #17389, "tv_downgraded ... UNPLAYABLE"), so forcing
        # it can turn a video that would download fine on yt-dlp's own
        # default client selection into a hard failure. yt-dlp's current
        # default already includes its own client-fallback logic (per the
        # maintainers, it "doesn't solely rely on tv_downgraded ... isn't
        # anymore"), and un-restricted videos are reported to typically
        # work with zero special config. Cookies remain the one thing this
        # app forces explicitly, and only when configured, because they're
        # the one mechanism that's actually necessary for genuinely
        # age-/login-restricted content rather than a workaround for a
        # moving target.
        if settings.youtube_cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (settings.youtube_cookies_from_browser,)
        elif settings.youtube_cookies_file:
            ydl_opts["cookiefile"] = settings.youtube_cookies_file

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source.source_url, download=True)
                media_path = Path(ydl.prepare_filename(info)).resolve()
        except Exception as exc:  # yt-dlp raises its own exception types
            hint = ""
            exc_str = str(exc)
            if "403" in exc_str:
                # YouTube regularly changes its throttling/cipher scheme;
                # this is near-always an out-of-date yt-dlp, not a real
                # permissions issue, and updating almost always fixes it
                # without changing anything else about the request.
                hint = " (this is usually an outdated yt-dlp — try: python -m pip install -U yt-dlp, then retry)"
            elif "Sign in to confirm" in exc_str or "not a bot" in exc_str:
                # YouTube's bot-check. The "tv" client fallback above
                # already tries to avoid this without cookies — seeing it
                # anyway means this specific video/IP needs real account
                # cookies. Set YOUTUBE_COOKIES_FROM_BROWSER (or
                # YOUTUBE_COOKIES_FILE) in .env — see .env.example.
                hint = (
                    " (YouTube is bot-checking this request even via the TV client — "
                    "set YOUTUBE_COOKIES_FROM_BROWSER or YOUTUBE_COOKIES_FILE in .env, "
                    "restart the app, and retry — see .env.example)"
                )
            elif "Requested format is not available" in exc_str:
                # As of yt-dlp's 2026 YouTube extractor, this almost always
                # means no JS runtime could be found to solve YouTube's JS
                # challenge (see yt-dlp wiki: "EJS") -- extraction silently
                # comes back with no real, downloadable formats at all, not
                # just a missing audio-only one. Confirmed against yt-dlp's
                # own EJS documentation and multiple 2026 upstream issues
                # (e.g. yt-dlp/yt-dlp#16350) rather than assumed.
                hint = (
                    " (this almost always means yt-dlp has no JavaScript runtime "
                    "to solve YouTube's challenge with -- install Deno "
                    "[winget install DenoLand.Deno], make sure `pip install -U "
                    "\"yt-dlp[default]\"` has been run so the yt-dlp-ejs helper "
                    "package is present, then restart the app and retry)"
                )
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
            structure_source(self.db, source, job)
            index_source(self.db, source, job)
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
