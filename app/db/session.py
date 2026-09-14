"""Engine + session factory. SQLite needs check_same_thread=False for
Streamlit's threaded execution model; everything else is standard.

Sprint 7 hit real "database is locked" errors on real hardware (not
guessed) the moment the background-thread ingestion pipeline landed --
even on a plain app launch with nothing actively ingesting. Root cause:
SQLite's DEFAULT journal mode (rollback journal, not WAL) takes a
database-wide lock for the whole duration of a write transaction, and
blocks ALL readers while that lock is held -- a `timeout` connect arg
only controls how long a blocked connection waits before giving up, it
doesn't reduce the actual contention. That was fine when this app only
ever had one connection open at a time (a single Streamlit script run,
start to finish); it stopped being fine the moment a background pipeline
thread and the UI's polling thread both hold connections open
concurrently, which is exactly Sprint 7's whole point.

Fix, part 1: WAL (write-ahead log) mode, SQLite's own documented answer
to this -- readers no longer block on a writer (and vice versa) at all,
they just read the last-committed snapshot from the WAL file. Confirmed
actually taking effect on real hardware (data/edurag.db-wal and
-shm exist and are non-empty -- SQLite only creates those files after a
successful `PRAGMA journal_mode=WAL`, so this part IS active).

Fix, part 2: WAL mode still only allows ONE writer at a time -- it
removes reader/writer blocking, not writer/writer blocking. That
residual case (two threads' sessions both trying to write within the
same short window -- exactly what a background pipeline thread and the
UI's polling thread can do) still raised "database is locked" even with
WAL active and busy_timeout set, on real hardware, immediately, with no
visible wait -- meaning something about how that specific contention
surfaced wasn't being smoothed over by the busy-timeout wait at all.
Rather than keep guessing at the exact interleaving (double Streamlit
script runs? a stale thread from an earlier interrupted run? something
else?) with no way to attach a debugger to the actual failure, this
makes the fix independent of the exact mechanism: every Session this
app hands out automatically retries a flush/commit that fails with
"database is locked", with exponential backoff, before giving up. SQLite
write locks are inherently transient (the other writer finishes in
milliseconds under normal operation) -- if this app's OWN code is the
only thing ever writing to this file, no genuine deadlock is possible,
so a bounded retry is a correctness fix here, not a band-aid over a
real problem it can't resolve. If retries are ever exhausted, that's
new, real evidence (something is actually stuck, not just slow) worth
looking at properly rather than something to retry forever."""

import time
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
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


def _retry_on_locked(fn, retries: int = 8, base_delay: float = 0.05):
    """Retries fn() on a SQLite "database is locked" OperationalError with
    exponential backoff (~6s worst case across 8 attempts), re-raising
    immediately for any other error and re-raising the lock error itself
    once retries run out. Safe to retry blindly: fn is always exactly
    Session.flush/Session.commit, and a failed flush/commit never
    partially applies -- there's nothing to undo before trying again."""
    for attempt in range(retries):
        try:
            return fn()
        except OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt == retries - 1:
                raise
            time.sleep(base_delay * (2**attempt))


class _RetryingSession(Session):
    """A Session whose flush()/commit() absorb transient SQLite lock
    contention instead of surfacing it as a crash -- see the module
    docstring's "Fix, part 2". Every SessionLocal() in this app gets this
    behavior automatically; no call site needs to know about it."""

    def flush(self, *args, **kwargs):
        return _retry_on_locked(lambda: Session.flush(self, *args, **kwargs))

    def commit(self):
        return _retry_on_locked(lambda: Session.commit(self))


SessionLocal = sessionmaker(bind=engine, class_=_RetryingSession, autoflush=False, autocommit=False)


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
