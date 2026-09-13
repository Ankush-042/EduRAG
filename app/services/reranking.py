"""Sprint 5 (part 2) — cross-encoder reranking over fused hybrid-search
candidates (TRD Doc 2 sec 20). A cross-encoder scores each (query, chunk)
pair jointly rather than comparing independently-computed embeddings, so
it catches relevance a bi-encoder's dense similarity alone misses — the
standard reason hybrid retrieval pipelines rerank the fused candidate set
instead of just trusting RRF's blended ranking directly.

Kept independent of retrieval.py, same split used throughout this
codebase (embedding vs. indexing, transcription vs. structuring): this
module only knows how to score (query, text) pairs, not how the
candidates it's scoring were produced.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.interfaces import RerankedCandidate, Reranker, RetrievedCandidate

settings = get_settings()

# Same reasoning as embedding.py's _MODEL_CACHE: cross-encoder models are
# expensive to load, cheap to reuse within a process.
_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from sentence_transformers import CrossEncoder

    # device explicitly pinned — same reasoning as embedding.py's
    # _get_model: avoids PyTorch opening a second CUDA context alongside
    # ctranslate2's in the same process (config.py's torch_model_device).
    model = CrossEncoder(model_name, device=settings.torch_model_device)
    _MODEL_CACHE[model_name] = model
    return model


class CrossEncoderReranker(Reranker):
    """Concrete Reranker (app/core/interfaces.py) backed by a
    sentence-transformers CrossEncoder. Swapping rerankers later means
    adding another class here, not touching retrieval.py."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    def rerank(self, query: str, candidates: list[RetrievedCandidate]) -> list[RerankedCandidate]:
        if not candidates:
            return []

        model = _get_model(self._model_name)
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [
            RerankedCandidate(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=float(score),
                retrieval_score=candidate.score,
            )
            for candidate, score in scored
        ]


def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(settings.reranker_model)
