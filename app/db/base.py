"""SQLAlchemy declarative base + a shared UUID/timestamp mixin.

Every table in the Data/Schema spec (Doc 4) uses a UUID primary key and
created_at/updated_at bookkeeping — centralized here so models stay short.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Naive UTC, deliberately — not tz-aware. SQLite (our default DB, see
    app/core/config.py) has no real datetime type: DateTime(timezone=True)
    round-trips as a naive datetime once read back from a row, no matter
    what tzinfo it was written with. If this returned an aware datetime,
    any comparison against a value freshly loaded from the DB (e.g. an
    expiry check on a session created in an earlier run) would crash with
    "can't compare offset-naive and offset-aware datetimes" the moment the
    two sides came from different origins. Staying naive-UTC everywhere
    sidesteps that regardless of backend; the column keeps timezone=True
    since that's still correct if this ever points at real Postgres."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class UUIDPKMixin:
    id: Mapped[str] = mapped_column(primary_key=True, default=new_uuid)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
