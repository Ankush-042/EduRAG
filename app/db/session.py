"""Engine + session factory. SQLite needs check_same_thread=False for
Streamlit's threaded execution model; everything else is standard.

Sprint 7's background-thread pipeline exposed a real chain of SQLite
concurrency issues on actual hardware. The full, now-confirmed story,
in order:

1. SQLite's default journal mode (rollback journal) blocks ALL readers
   for the whole duration of any write. Fixed with WAL mode (below) --
   confirmed active (data/edurag.db-wal and -shm exist).

2. WAL still only allows one writer at a time. The real reason writers
   were actually colliding wasn't a fluke race -- it was this module's
   own ingestion.py: each pipeline stage function (download, extract,
   transcribe, index) writes a status/progress row at its START, then
   does its slow work (a multi-minute yt-dlp download, ffmpeg, ASR,
   embedding) *inside the same still-open transaction*, and only
   commits after the whole stage returns. That holds SQLite's one write
   lock for the ENTIRE stage duration, not just the instant of the
   write -- so the UI thread's own periodic writes (touch_session on
   every rerun) were blocked for however long a download or
   transcription took, which is exactly the "database is locked" seen
   live, mid-download. The real fix is in ingestion.py/transcription.py/
   indexing.py: commit immediately after each small write, before
   starting the slow operation that follows it, so the write lock is
   only ever held for milliseconds.

3. An earlier attempt at this file wrapped Session.flush()/commit() in
   a Python-level retry loop as a defense-in-depth on top of (1)+(2).
   That was itself a bug: SQLAlchemy puts a Session into a "deactive,
   needs rollback" state the moment ANY flush/commit fails, and calling
   commit() again on a still-deactive session raises PendingRollbackError
   instead of retrying anything -- confirmed on real hardware, not
   guessed. A correct retry would need to roll back and REDO the
   mutation, which a generic Session-level wrapper can't do without
   knowing what the caller was trying to write. Removed rather than
   patched further: with (2) actually fixed, the write-lock window is
   now milliseconds, and SQLite's own busy_timeout (below) -- which
   operates underneath SQLAlchemy, with no session-state pitfalls -- is
   the correct tool for smoothing over a window that small.
"""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_is_sqlite = settings.database_url.startswith("sqlite")
_connect_args = {"check_same_thread": False, "timeout": 30} if _is_sqlite else {}


def _ensure_sqlite_dir(database_url: str) -> None:
    """SQLite (unlike a real DB server) never creates its own parent
    directory — it just fails with 'unable to open database file' if the
    folder isn't there yet. Nothing else in this app is guaranteed to run
    before the engine connects, so this has to happen right here."""
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
        Path(url.database).resolve().parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir(settings.database_url)

engine = create_engine(settings.database_url, connect_args=_connect_args)


if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            # NORMAL (not the default FULL) is WAL mode's own documented
            # pairing -- WAL already guarantees consistency after a crash,
            # FULL's extra fsync is for the journal mode this app no
            # longer uses.
            cursor.execute("PRAGMA synchronous=NORMAL")
            # Belt-and-suspenders alongside the `timeout` connect arg --
            # this is the actual SQLite-level busy timeout in milliseconds;
            # the connect arg sets the same thing through Python's sqlite3
            # module, but setting the PRAGMA directly means it's not
            # depending on that mapping being right for every driver.
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
