"""Persistence for the temporary, no-login session (Data spec Doc 4 sec 6).
One row per browser session; everything else hangs off session_id."""

from datetime import timedelta

from sqlalchemy.orm import Session as DbSession

from app.core.config import get_settings
from app.db.base import _utcnow
from app.db.models.session import SessionModel

settings = get_settings()


def create_session(db: DbSession) -> SessionModel:
    now = _utcnow()
    session = SessionModel(
        created_at=now,
        last_active_at=now,
        expires_at=now + timedelta(hours=settings.session_ttl_hours),
        status="ACTIVE",
    )
    db.add(session)
    db.flush()
    return session


def get_active_session(db: DbSession, session_id: str) -> SessionModel | None:
    session = db.get(SessionModel, session_id)
    if session is None or session.status != "ACTIVE":
        return None
    if session.expires_at < _utcnow():
        session.status = "EXPIRED"
        db.flush()
        return None
    return session


def touch_session(db: DbSession, session: SessionModel) -> SessionModel:
    session.last_active_at = _utcnow()
    session.expires_at = _utcnow() + timedelta(hours=settings.session_ttl_hours)
    db.flush()
    return session
