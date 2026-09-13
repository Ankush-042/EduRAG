"""Sprint 6 (part 4) — the orchestration entry point: retrieval ->
generation -> claim splitting -> per-claim NLI verification ->
grounding_status -> persistence (TRD Doc 2 sec 21-30). This is the single
function the UI calls for one turn; nothing below this layer touches the
Conversation/Message/Evidence/Claim/VerificationResult tables — retrieval.py,
generation.py, claims.py and grounding.py are all single-purpose services
that don't know this schema exists, matching the split already used
throughout this codebase (transcription vs. structuring, embedding vs.
indexing, retrieval vs. reranking).

grounding_status (Data/Schema spec Doc 4 sec 16; GROUNDING_STATUS_VALUES in
app/db/models/conversation.py) is computed, per assistant message, as:

  ABSTAINED           No evidence was retrieved at all -- the generator is
                       never even called (see _ABSTAIN_MESSAGE below). The
                       safest possible response to "we have nothing
                       relevant indexed" is to say so plainly, not to let a
                       generator improvise an answer from its own training
                       data with nothing to cite.
  GROUNDED            An answer was generated and every extracted claim is
                       entailed by at least one evidence passage.
  PARTIALLY_GROUNDED  An answer was generated and at least one claim is
                       entailed, but at least one other claim isn't.
  UNVERIFIED          An answer was generated but zero claims came back
                       entailed -- including the degenerate case of zero
                       extractable claims. Produced, but nothing in it
                       could be confirmed against the evidence, which is
                       treated as no better than "can't verify" rather than
                       silently upgraded to PARTIALLY_GROUNDED.

A CONTRADICTION verdict is recorded at the claim level in
verification_results either way (so it's inspectable later) but never
lifts grounding_status toward GROUNDED/PARTIALLY_GROUNDED on its own --
only an ENTAILMENT verdict does that. This mapping never marks a
contradicted claim's message as more grounded than the evidence actually
supports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session as DbSession

from app.db.models.content import CLAIM_TYPES
from app.db.repositories import content_repository as content
from app.db.repositories import conversation_repository as conversations
from app.services.claims import split_into_claims, strip_citation_markers
from app.services.generation import GenerationError, get_generator
from app.services.grounding import get_verifier
from app.services.retrieval import RetrievalError, retrieve

assert "GENERATED_CLAIM" in CLAIM_TYPES  # guards the literal used below against schema drift

_ABSTAIN_MESSAGE = (
    "I couldn't find anything about this in your sources, so I don't have "
    "a grounded answer to give you. Try rephrasing the question, or add a "
    "source that covers this topic."
)


@dataclass
class EvidenceView:
    """UI-facing view of one Evidence row plus the chunk text it points
    at -- lets the UI render a citation list without querying
    content_repository/conversation_repository itself."""

    rank: int
    chunk_id: str
    text: str
    start_time: float | None
    reranker_score: float


@dataclass
class AnsweredMessage:
    message_id: str
    content: str
    grounding_status: str
    evidence: list[EvidenceView] = field(default_factory=list)


class AnsweringError(Exception):
    """Raised when a turn can't be produced at all (retrieval or
    generation itself failed) -- distinct from an ABSTAINED answer, which
    is a legitimate, successfully-produced result, not a failure."""


def _build_context(evidence_rows: list[tuple]) -> str:
    """Numbered context blocks the generation prompt instructs the model
    to cite by number, e.g. "[2] ...". Plain text, no markdown -- it goes
    directly into generation.py's user prompt."""
    return "\n\n".join(f"[{i}] {chunk.text}" for i, (chunk, _candidate) in enumerate(evidence_rows, start=1))


def _make_abstained(db: DbSession, conversation_id: str, order: int, t_start: float) -> AnsweredMessage:
    latency_ms = int((time.monotonic() - t_start) * 1000)
    message = conversations.create_message(
        db,
        conversation_id=conversation_id,
        role="ASSISTANT",
        content=_ABSTAIN_MESSAGE,
        message_order=order,
        latency_ms=latency_ms,
        grounding_status="ABSTAINED",
    )
    return AnsweredMessage(message_id=message.id, content=_ABSTAIN_MESSAGE, grounding_status="ABSTAINED")


def answer(db: DbSession, session_id: str, question: str) -> AnsweredMessage:
    """Runs one full question-answering turn for `session_id` and persists
    every step of it (the user message, the assistant message, its
    evidence, its claims, and their verification results). Always returns
    an AnsweredMessage on success -- an ABSTAINED answer is a complete,
    successful result, not an error; AnsweringError is reserved for turns
    that genuinely couldn't be produced (retrieval or generation raised)."""
    question = question.strip()
    if not question:
        raise AnsweringError("Question is empty.")

    t_start = time.monotonic()
    conversation = conversations.get_or_create_conversation(db, session_id)

    try:
        reranked = retrieve(db, session_id, question)
    except RetrievalError as exc:
        raise AnsweringError(str(exc)) from exc

    user_order = conversations.next_message_order(db, conversation.id)
    conversations.create_message(
        db, conversation_id=conversation.id, role="USER", content=question, message_order=user_order,
    )
    assistant_order = user_order + 1

    if not reranked:
        return _make_abstained(db, conversation.id, assistant_order, t_start)

    chunk_ids = [candidate.chunk_id for candidate in reranked]
    chunks_by_id = content.get_chunks_by_ids(db, chunk_ids)
    evidence_rows = [
        (chunks_by_id[candidate.chunk_id], candidate)
        for candidate in reranked
        if candidate.chunk_id in chunks_by_id  # defensively skip a chunk deleted since retrieval ran
    ]
    if not evidence_rows:
        return _make_abstained(db, conversation.id, assistant_order, t_start)

    context = _build_context(evidence_rows)

    try:
        generator = get_generator()
        raw_answer = generator.generate(question, context)
    except GenerationError as exc:
        raise AnsweringError(str(exc)) from exc

    # Placeholder grounding_status -- overwritten once claim verification
    # below actually runs. Never left as UNVERIFIED by accident: the final
    # update_message call always runs before this function returns.
    assistant_message = conversations.create_message(
        db,
        conversation_id=conversation.id,
        role="ASSISTANT",
        content=raw_answer,
        message_order=assistant_order,
        grounding_status="UNVERIFIED",
    )

    evidence_records = []
    for rank, (chunk, candidate) in enumerate(evidence_rows, start=1):
        evidence = conversations.add_evidence(
            db,
            message_id=assistant_message.id,
            chunk_id=chunk.id,
            source_id=chunk.source_id,
            rank=rank,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            retrieval_score=candidate.retrieval_score,
            reranker_score=candidate.score,
        )
        evidence_records.append(evidence)

    claim_texts = split_into_claims(raw_answer)
    verifier = get_verifier()
    verdicts: list[str] = []

    for claim_text in claim_texts:
        claim_row = conversations.create_claim(
            db, claim_text=claim_text, claim_type="GENERATED_CLAIM", message_id=assistant_message.id,
        )
        claim_for_nli = strip_citation_markers(claim_text)
        if not claim_for_nli:
            continue

        # Check this one claim against every evidence passage and keep the
        # single best-supporting match: prefer an ENTAILMENT verdict over
        # any other, and among same-verdict matches prefer the higher
        # score. This is O(claims x evidence) NLI calls per turn (typically
        # a handful x up to top_k_evidence) -- deliberately exhaustive
        # rather than only checking the passage the claim happened to cite,
        # since a model can (and does) mis-cite a passage number while
        # still being grounded in a *different* retrieved passage.
        best_evidence = None
        best_result = None
        for evidence, (chunk, _candidate) in zip(evidence_records, evidence_rows):
            result = verifier.verify(claim_for_nli, chunk.text)
            is_better = (
                best_result is None
                or (result.verdict == "ENTAILMENT" and best_result.verdict != "ENTAILMENT")
                or (result.verdict == best_result.verdict and result.score > best_result.score)
            )
            if is_better:
                best_evidence, best_result = evidence, result

        if best_evidence is None or best_result is None:
            continue  # unreachable given the evidence_rows guard above, but never crash on it

        conversations.add_verification_result(
            db,
            message_id=assistant_message.id,
            claim_id=claim_row.id,
            evidence_id=best_evidence.id,
            verdict=best_result.verdict,
            score=best_result.score,
        )
        if best_result.verdict == "ENTAILMENT":
            conversations.update_evidence_nli_score(db, best_evidence, nli_score=best_result.score)
        verdicts.append(best_result.verdict)

    if not verdicts:
        grounding_status = "UNVERIFIED"
    elif all(v == "ENTAILMENT" for v in verdicts):
        grounding_status = "GROUNDED"
    elif any(v == "ENTAILMENT" for v in verdicts):
        grounding_status = "PARTIALLY_GROUNDED"
    else:
        grounding_status = "UNVERIFIED"

    latency_ms = int((time.monotonic() - t_start) * 1000)
    conversations.update_message(
        db, assistant_message, grounding_status=grounding_status, latency_ms=latency_ms
    )

    return AnsweredMessage(
        message_id=assistant_message.id,
        content=raw_answer,
        grounding_status=grounding_status,
        evidence=[
            EvidenceView(
                rank=rank,
                chunk_id=chunk.id,
                text=chunk.text,
                start_time=chunk.start_time,
                reranker_score=candidate.score,
            )
            for rank, (chunk, candidate) in enumerate(evidence_rows, start=1)
        ],
    )
