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

Fix: WAL (write-ahead log) mode, SQLite's own documented answer to this
-- readers no longer block on a writer (and vice versa) at all, they
just read the last-committed snapshot from the WAL file. This is set via
PRAGMA on every new connection (WAL is persisted in the DB file itself
once set, but setting it defensively on every connect is cheap and
guards against a stale journal-mode DB file predating this fix). The
`timeout` connect arg stays as a second line of defense for the one
case WAL doesn't fully remove -- two connections trying to WRITE at the
literal same instant."""

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
