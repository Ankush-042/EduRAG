"""Engine + session factory. SQLite needs check_same_thread=False for
Streamlit's threaded execution model; everything else is standard.

Sprint 7: ingestion now runs on a background thread with its own session
(app/services/ingestion.py) while the Streamlit request thread keeps
polling with its own reads — two threads touching the same SQLite file
concurrently. SQLite serializes writes at the file level and raises
"database is locked" if a writer can't get the lock within its busy
timeout; the default timeout (5s) is tight for that pattern, so it's
raised here rather than waiting to see the error on real hardware."""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_connect_args = (
    {"check_same_thread": False, "timeout": 30} if settings.database_url.startswith("sqlite") else {}
)


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
