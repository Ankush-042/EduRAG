"""sessions — one temporary EduRAG workspace (Data/Schema spec Doc 4 sec 6).
No accounts: everything else hangs off this table, and its expiry drives
cleanup of sources, indexes and conversations."""

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPKMixin, new_uuid

STATUS_VALUES = ("ACTIVE", "EXPIRED", "CLEANUP_PENDING", "DELETED")


class SessionModel(UUIDPKMixin, Base):
    __tablename__ = "sessions"

    session_token: Mapped[str] = mapped_column(String, unique=True, default=new_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, default="ACTIVE")
