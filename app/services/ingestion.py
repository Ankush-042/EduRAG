"""Sprint 1 — source ingestion: validation, download, audio extraction, and
the state-machine transitions from TRD Doc 2 sec 40 / Data spec Doc 4 sec 7:

    QUEUED -> DOWNLOADING -> EXTRACTING -> TRANSCRIBING -> PROCESSING -> INDEXING -> READY

This module drives sources through download/save + audio extraction, then
straight into transcription (Sprint 2), content structuring (Sprint 3),
and indexing (Sprint 4). Every step is wrapped so a failure lands the
source in FAILED with a readable error_message rather than leaving it
stuck (TRD Doc 2 sec 24).

Sprint 7 (background execution): there's still no real job queue (no
Celery/Redis, per Doc 6 authority) — but the whole point of a 1-hour
lecture needing to "process real quick" is that the UI can't sit blocked
for the full pipeline duration either. add_youtube_source /
add_local_video_source now only do the fast, synchronous part (validate,
create the Source + ProcessingJob rows) and hand the actual download ->
extract -> transcribe -> structure -> index pipeline off to a daemon
thread (_run_*_pipeline below), returning immediately with the source in
QUEUED. The UI polls source/job status on its own rerun cycle (already
DB-backed, so this needed no new plumbing) instead of blocking on a
st.spinner for the real duration of ingestion.

The one thing that changes because of this: a SQLAlchemy Session is not
thread-safe, so the background thread can't reuse the caller's `self.db`
— it opens its own SessionLocal() (app/db/session.py already sets
check_same_thread=False for exactly this) and looks the Source/
ProcessingJob back up by id rather than being handed the ORM objects
across the thread boundary.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import threading
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


def _spawn_pipeline_thread(target, *args) -> None:
    """daemon=True so a still-running ingestion never blocks process
    exit -- it's re-run from scratch on next launch, same as any other
    interrupted-and-restarted job in this MVP (no resume-from-partial
    logic exists at any stage, so there's nothing extra to clean up)."""
    threading.Thread(target=target, args=args, daemon=True).start()


def _run_youtube_pipeline(source_id: str, job_id: str) -> None:
    """Entry point for the background thread (Sprint 7) — opens its own
    DB session/service instance and runs the same steps
    add_youtube_source used to run inline. source.source_url is already
    persisted by the time this runs, so it doesn't need to be passed in."""
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        source = sources.get_source(db, source_id)
        job = jobs.get_job(db, job_id)
        service = SourceIngestionService(db)
        try:
            # Committing after every stage (rather than once at the end) is
            # what makes the progress bar in the UI actually move -- the UI
            # reads through a completely separate SessionLocal()/connection
            # (main thread), so it only ever sees whatever this thread has
            # committed, never what's merely flushed within an open
            # transaction here.
            service._download_youtube(source, job)
            db.commit()
            service._extract_audio(source, job)
            db.commit()
            transcribe_source(db, source, job)
            db.commit()
            structure_source(db, source, job)
            db.commit()
            index_source(db, source, job)  # index_source itself moves the source to READY
            db.commit()
            jobs.complete_job(db, job)
            db.commit()
        except Exception as exc:  # noqa: BLE001 — must land in FAILED, never half-done.
            db.rollback()
            source = sources.get_source(db, source_id)
            job = jobs.get_job(db, job_id)
            sources.update_source_status(db, source, "FAILED", error_message=str(exc))
            jobs.fail_job(db, job, error_message=str(exc))
            db.commit()


def _run_local_video_pipeline(source_id: str, job_id: str, upload: "UploadedFile") -> None:
    """Same as _run_youtube_pipeline but for a local upload — the raw
    bytes (`upload`) have to be passed through directly since, unlike a
    YouTube URL, they were never persisted to the DB."""
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        source = sources.get_source(db, source_id)
        job = jobs.get_job(db, job_id)
        service = SourceIngestionService(db)
        try:
            # See _run_youtube_pipeline above for why this commits after
            # every stage instead of once at the end.
            service._save_local_upload(source, job, upload)
            db.commit()
            service._extract_audio(source, job)
            db.commit()
            transcribe_source(db, source, job)
            db.commit()
            structure_source(db, source, job)
            db.commit()
            index_source(db, source, job)  # index_source itself moves the source to READY
            db.commit()
            jobs.complete_job(db, job)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            source = sources.get_source(db, source_id)
            job = jobs.get_job(db, job_id)
            sources.update_source_status(db, source, "FAILED", error_message=str(exc))
            jobs.fail_job(db, job, error_message=str(exc))
            db.commit()


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
        """Sprint 7: only the fast, synchronous part happens here now
        (validate + create the Source/ProcessingJob rows) — the actual
        download/transcribe/index pipeline runs on a background thread
        (_run_youtube_pipeline) so this returns in milliseconds instead of
        blocking on however long the real video takes. The caller is
        responsible for committing self.db right after this returns,
        same as before (the background thread needs the row committed
        before it can see it from its own session)."""
        url = url.strip()
        if not url or ("youtube.com" not in url and "youtu.be" not in url):
            raise IngestionError("That doesn't look like a YouTube URL.")

        source = sources.create_source(
            self.db, session_id=session_id, source_type="YOUTUBE", source_url=url
        )
        job = jobs.create_job(self.db, source_id=source.id, job_type="FULL_PIPELINE")
        self.db.commit()

        _spawn_pipeline_thread(_run_youtube_pipeline, source.id, job.id)
        return source

    def _download_youtube(self, source: Source, job) -> None:
        try:
            import yt_dlp
        except ImportError as exc:
            raise IngestionError("yt-dlp is not installed (pip install -r requirements.txt).") from exc

        sources.update_source_status(self.db, source, "DOWNLOADING")
        jobs.start_job(self.db, job, stage="DOWNLOADING")
        # Commit BEFORE the slow yt-dlp call below, not after -- the real
        # root cause of the live "database is locked" crash (confirmed on
        # real hardware, mid-download): without this, the write above
        # stays open in an uncommitted transaction for the ENTIRE download
        # (SQLite allows exactly one open writer at a time, even under
        # WAL), blocking every other write in the app -- including the
        # UI's own per-rerun session touch -- for however long the
        # download takes, not just for the instant of this one write.
        self.db.commit()

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
        """Sprint 7: same split as add_youtube_source — create the rows
        synchronously, hand the pipeline to a background thread. The raw
        upload bytes (`upload`) are passed directly into the thread since
        they aren't persisted anywhere the thread's own session could
        look them back up from."""
        source = sources.create_source(
            self.db,
            session_id=session_id,
            source_type="LOCAL_VIDEO",
            original_name=upload.name,
            mime_type=upload.mime_type,
        )
        job = jobs.create_job(self.db, source_id=source.id, job_type="FULL_PIPELINE")
        self.db.commit()

        _spawn_pipeline_thread(_run_local_video_pipeline, source.id, job.id, upload)
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
        # Self-audit finding (post-Sprint-11): every progress=X value across
        # ingestion.py/transcription.py/structuring.py/indexing.py used to
        # be that FUNCTION's own private 0-~0.9 scale, with nothing
        # rescaling between stages -- so the one job row the UI reads
        # (app/ui/main.py's st.progress) visibly jumped backward at every
        # stage boundary (e.g. 90% at the end of transcription -> 30% at
        # the start of structuring). These are now one shared whole-
        # pipeline percentage, monotonically increasing end to end:
        # EXTRACTING_AUDIO 15% -> TRANSCRIBING 20% -> ... -> FINALIZING
        # 98% -> complete_job's 100%. See each call site below/in the
        # other three files for its place in that same scale.
        jobs.update_progress(self.db, job, progress=0.15, stage="EXTRACTING_AUDIO")
        self.db.commit()  # release the write lock before the ffmpeg subprocess runs

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
        jobs.update_progress(self.db, job, progress=0.20, stage="TRANSCRIBING")
