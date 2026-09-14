"""Persistence for conversations / messages / evidence / claims /
verification_results (Data/Schema spec Doc 4 sec 15-21) — the answer ->
citation lineage Sprint 6's answering.py writes to on every turn. Kept
separate from the ORM models and from answering.py itself, the same split
already used everywhere else in this codebase (sources, jobs, sessions,
content)."""

from __future__ import annotations

from sqlalchemy.orm import Session as DbSession

from app.db.models.content import Claim
from app.db.models.conversation import Conversation, Evidence, Message
from app.db.models.evaluation import VerificationResult


def create_conversation(db: DbSession, *, session_id: str, title: str | None = None) -> Conversation:
    conversation = Conversation(session_id=session_id, title=title)
    db.add(conversation)
    db.flush()
    return conversation


def get_conversation(db: DbSession, conversation_id: str) -> Conversation | None:
    return db.get(Conversation, conversation_id)


def get_latest_conversation_for_session(db: DbSession, session_id: str) -> Conversation | None:
    return (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.created_at.desc())
        .first()
    )


def get_or_create_conversation(db: DbSession, session_id: str) -> Conversation:
    """One conversation per session is all the current UI exposes (a
    single "Ask" panel, not a conversation list/switcher yet) — so reuse
    the session's most recent conversation rather than creating a fresh
    one per question. Multiple conversations per session are already fully
    supported by the schema (session_id is a plain indexed foreign key,
    not unique), so this is a UI-layer simplification, not a schema
    limitation, for whenever the UI grows a real conversation list."""
    existing = get_latest_conversation_for_session(db, session_id)
    if existing is not None:
        return existing
    return create_conversation(db, session_id=session_id)


def next_message_order(db: DbSession, conversation_id: str) -> int:
    last = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.message_order.desc())
        .first()
    )
    return (last.message_order + 1) if last is not None else 0


def create_message(
    db: DbSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    message_order: int,
    latency_ms: int | None = None,
    grounding_status: str | None = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        message_order=message_order,
        latency_ms=latency_ms,
        grounding_status=grounding_status,
    )
    db.add(message)
    db.flush()
    return message


def update_message(
    db: DbSession,
    message: Message,
    *,
    content: str | None = None,
    grounding_status: str | None = None,
    latency_ms: int | None = None,
) -> Message:
    if content is not None:
        message.content = content
    if grounding_status is not None:
        message.grounding_status = grounding_status
    if latency_ms is not None:
        message.latency_ms = latency_ms
    db.flush()
    return message


def list_messages_for_conversation(db: DbSession, conversation_id: str) -> list[Message]:
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.message_order.asc())
        .all()
    )


def add_evidence(
    db: DbSession,
    *,
    message_id: str,
    chunk_id: str,
    source_id: str,
    rank: int,
    sentence_id: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    retrieval_score: float | None = None,
    reranker_score: float | None = None,
    nli_score: float | None = None,
) -> Evidence:
    evidence = Evidence(
        message_id=message_id,
        chunk_id=chunk_id,
        sentence_id=sentence_id,
        source_id=source_id,
        rank=rank,
        start_time=start_time,
        end_time=end_time,
        retrieval_score=retrieval_score,
        reranker_score=reranker_score,
        nli_score=nli_score,
    )
    db.add(evidence)
    db.flush()
    return evidence


def update_evidence_nli_score(db: DbSession, evidence: Evidence, *, nli_score: float) -> Evidence:
    evidence.nli_score = nli_score
    db.flush()
    return evidence


def list_evidence_for_message(db: DbSession, message_id: str) -> list[Evidence]:
    return (
        db.query(Evidence)
        .filter(Evidence.message_id == message_id)
        .order_by(Evidence.rank.asc())
        .all()
    )


def create_claim(
    db: DbSession,
    *,
    claim_text: str,
    claim_type: str,
    message_id: str | None = None,
    source_id: str | None = None,
    sentence_id: str | None = None,
) -> Claim:
    claim = Claim(
        source_id=source_id,
        sentence_id=sentence_id,
        message_id=message_id,
        claim_text=claim_text,
        claim_type=claim_type,
    )
    db.add(claim)
    db.flush()
    return claim


def add_verification_result(
    db: DbSession, *, message_id: str, claim_id: str, evidence_id: str, verdict: str, score: float
) -> VerificationResult:
    result = VerificationResult(
        message_id=message_id,
        claim_id=claim_id,
        evidence_id=evidence_id,
        verdict=verdict,
        score=score,
    )
    db.add(result)
    db.flush()
    return result


def list_verification_results_for_message(db: DbSession, message_id: str) -> list[VerificationResult]:
    """Sprint 11 (UI retrieval-details panel): Claim has no order column of
    its own (UUIDPKMixin only -- no created_at, no explicit sequence), but
    answering.py writes one VerificationResult per claim in the exact same
    order split_into_claims produced them, and VerificationResult DOES
    carry created_at -- so ordering by that reconstructs claim order
    without needing a schema change just for a diagnostics display."""
    return (
        db.query(VerificationResult)
        .filter(VerificationResult.message_id == message_id)
        .order_by(VerificationResult.created_at.asc())
        .all()
    )


def get_claims_by_ids(db: DbSession, claim_ids: list[str]) -> dict[str, Claim]:
    if not claim_ids:
        return {}
    rows = db.query(Claim).filter(Claim.id.in_(claim_ids)).all()
    return {claim.id: claim for claim in rows}
