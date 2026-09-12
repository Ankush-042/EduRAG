"""Importing this package registers every table on Base.metadata — required
before Base.metadata.create_all() or an Alembic autogenerate, since the
model files reference each other's tables by string (e.g. Claim.message_id
-> "messages.id") and SQLAlchemy only resolves those once all mapped
classes have been loaded."""

from app.db.models.session import SessionModel  # noqa: F401
from app.db.models.source import Source, SourceArtifact  # noqa: F401
from app.db.models.content import Section, Chunk, Sentence, Claim  # noqa: F401
from app.db.models.conversation import Conversation, Message, Evidence  # noqa: F401
from app.db.models.processing import ProcessingJob  # noqa: F401
from app.db.models.evaluation import RetrievalRun, VerificationResult  # noqa: F401
