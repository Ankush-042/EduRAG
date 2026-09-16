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

A NOT_SUPPORTED verdict (app/services/grounding.py -- claim scored below
the supported/hallucinated threshold against its best-matching evidence)
is recorded at the claim level in verification_results either way (so
it's inspectable later) but never lifts grounding_status toward
GROUNDED/PARTIALLY_GROUNDED on its own -- only an ENTAILMENT verdict does
that. This mapping never marks an unsupported claim's message as more
grounded than the evidence actually supports.

Sprint 9 finding (real, evidence-based -- three separate live answers
showed the same pattern before this fix): a claim that legitimately
synthesizes facts from MULTIPLE retrieved passages -- e.g. "the lecture
covers integers, floats, and booleans [1] and also strings [4]", where
ints/floats/bools are stated in passage 1 and strings only in passage 4
-- was scored against each passage ONE AT A TIME, and no single passage
fully entails a claim that's true only across their union. That's not a
verifier defect; it's an artifact of only ever checking one premise at a
time. Every one of the three real false negatives found in Sprint 8's
eval run had this exact shape. Fix: when no single evidence passage
entails a claim, also check it against the full concatenated context
(all retrieved passages together) before giving up -- see the
`combined_result` fallback below. This can only ever raise a verdict
(make grounding_status more accurate), never lower one, since it's
strictly an additional check on top of the existing per-passage loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session as DbSession

from app.core.interfaces import EntailmentResult
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


# Self-audit finding (post-Sprint-11): unlike retrieve()/generator.generate()
# above, verifier.verify() had NO exception handling anywhere in this file
# -- a genuinely likely failure point (it's the first real run of HHEM-2.1
# in this exact process; grounding.py's own docstring already flags the
# .to(device) call and trust_remote_code loading as unverified outside this
# sandbox) would raise straight out of the per-claim loop below, past the
# assistant message that's ALREADY been persisted as "UNVERIFIED", and
# past the UI's `except answering.AnsweringError` (main.py), crashing the
# whole Streamlit script with a raw traceback instead of this app's own
# "never crash the UI" standard (see generation.py/indexing.py's Groq
# error handling for the standard this was missing). NOT_SUPPORTED is
# already a fully legitimate, correctly-handled verdict everywhere else in
# this function (it never raises grounding_status on its own) -- so on any
# verifier failure, this degrades that ONE claim to "not supported" rather
# than crashing the whole answer. A person still gets their answer; that
# one claim just doesn't count toward GROUNDED/PARTIALLY_GROUNDED, which is
# the honest, safe outcome when grounding genuinely couldn't be checked.
def _safe_verify(verifier, claim: str, evidence_text: str) -> EntailmentResult:
    try:
        return verifier.verify(claim, evidence_text)
    except Exception:
        return EntailmentResult(verdict="NOT_SUPPORTED", score=0.0)


def _build_context(evidence_rows: list[tuple]) -> str:
    """Numbered context blocks the generation prompt instructs the model
    to cite by number, e.g. "[2] ...". Plain text, no markdown -- it goes
    directly into generation.py's user prompt."""
    return "\n\n".join(f"[{i}] {chunk.text}" for i, (chunk, _candidate) in enumerate(evidence_rows, start=1))


def _build_verification_context(evidence_rows: list[tuple]) -> str:
    """Same passages as _build_context, but WITHOUT the "[n] " numbering --
    a self-review catch on the Sprint 9 combined-context check below: that
    check reused _build_context's numbered string as the NLI premise, but
    claims.py's own docstring already established the reason evidence
    text is never fed to the verifier with citation syntax attached (the
    NLI model was never trained on it and it's noise, not signal, to it).
    The per-passage loop above was never affected -- it verifies against
    chunk.text directly -- only the combined-context fallback had this
    inconsistency, since it's the one place a freshly-built multi-passage
    string gets passed to verifier.verify() instead of raw chunk text."""
    return "\n\n".join(chunk.text for chunk, _candidate in evidence_rows)


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
    except Exception as exc:
        # Self-audit finding: retrieve() itself has real unguarded failure
        # points beneath it (embedder.embed_query, the cross-encoder
        # reranker, an unpickle of a corrupted BM25 index) that don't raise
        # RetrievalError -- those must not reach the UI as a raw traceback
        # either. Same "never crash, always surface a clear message"
        # standard the Groq paths already got in generation.py/indexing.py.
        raise AnsweringError(f"Something went wrong while searching your sources: {exc}") from exc

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
    # Separate, citation-marker-free copy for the combined-context NLI
    # check further down -- see _build_verification_context's docstring.
    # The generator still gets the numbered `context` above; only the
    # verifier gets this one.
    verification_context = _build_verification_context(evidence_rows)

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
            result = _safe_verify(verifier, claim_for_nli, chunk.text)
            is_better = (
                best_result is None
                or (result.verdict == "ENTAILMENT" and best_result.verdict != "ENTAILMENT")
                or (result.verdict == best_result.verdict and result.score > best_result.score)
            )
            if is_better:
                best_evidence, best_result = evidence, result

        if best_evidence is None or best_result is None:
            continue  # unreachable given the evidence_rows guard above, but never crash on it

        # Sprint 9 fix (see module docstring): a claim that's only true
        # across MULTIPLE passages combined will never entail against any
        # one of them alone. Only spend the extra NLI call when the
        # per-passage loop above didn't already find a clean ENTAILMENT --
        # this is strictly a second chance, never a way to downgrade a
        # verdict the per-passage loop already got right.
        if best_result.verdict != "ENTAILMENT":
            combined_result = _safe_verify(verifier, claim_for_nli, verification_context)
            if combined_result.verdict == "ENTAILMENT":
                # Still cite the single passage the claim scored highest
                # against (best_evidence) -- that's the most useful pointer
                # for a person checking the citation -- but record the
                # verdict/score that actually reflects why this claim is
                # considered grounded (the full retrieved context, not that
                # one passage in isolation).
                best_result = combined_result

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
