"""
Model abstraction layer (TRD Doc 2 sec 44, AI/RAG spec Doc 5 principle 9).

Every AI capability EduRAG depends on is defined here as an interface, not
a concrete library call. Concrete implementations (faster-whisper, a
sentence-transformers embedder, a cross-encoder reranker, Groq generation,
a local NLI model, ...) live in app/services/ and are wired up through
app/core/config.py — so replacing, say, the reranker never means touching
the retrieval pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[TranscriptSegment]


@dataclass
class RetrievedCandidate:
    chunk_id: str
    text: str
    score: float
    retrieval_method: str  # "dense" | "bm25" | "fused"


@dataclass
class RerankedCandidate:
    chunk_id: str
    text: str
    score: float
    # The pre-rerank (RRF-fused) score this candidate carried in, kept
    # alongside the reranker's own score rather than discarded -- Sprint 6's
    # Evidence table (Data/Schema spec Doc 4 sec 17) has separate
    # retrieval_score/reranker_score columns, and losing the first one here
    # would mean it could never be populated. Optional/defaulted so nothing
    # that already constructs a RerankedCandidate without it breaks.
    retrieval_score: float | None = None


@dataclass
class EntailmentResult:
    verdict: str  # "ENTAILMENT" | "NOT_SUPPORTED" -- see app/services/grounding.py
    score: float


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class Transcriber(ABC):
    """Local ASR with word-level timestamps (TRD Doc 2 sec 6)."""

    @abstractmethod
    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
        ...


class Embedder(ABC):
    """Dense embedding model, ingestion- and query-time (TRD Doc 2 sec 13)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        ...


class Reranker(ABC):
    """Cross-encoder reranker over fused candidates (TRD Doc 2 sec 20)."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedCandidate]) -> list[RerankedCandidate]:
        ...


class Generator(ABC):
    """Grounded answer generation (TRD Doc 2 sec 25-26). Implementations
    include the primary fast-API provider and the local fallback; both
    conform to this interface so the pipeline doesn't care which is used."""

    @abstractmethod
    def generate(self, question: str, context: str, language: str = "en") -> str:
        ...


class EntailmentVerifier(ABC):
    """Local NLI/entailment model for claim-level grounding verification
    (TRD Doc 2 sec 27-30) — deliberately not a second generative LLM call."""

    @abstractmethod
    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        ...
