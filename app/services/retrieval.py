"""Sprint 5 (part 3) — hybrid retrieval: dense (Qdrant) + sparse (BM25)
search over one session's indexed chunks, fused with Reciprocal Rank
Fusion, then reranked (TRD Doc 2 sec 14-20; AI/RAG spec Doc 5 principle
9's hybrid-over-dense-only call). This is what Sprint 6's generation step
will call for evidence — nothing here produces an answer, only ranked,
attributable chunk text.

Session isolation (Data spec Doc 4 sec 24, the same mandatory rule
list_chunks_for_session's docstring already cites): dense search is
filtered by session_id in the Qdrant payload; sparse search reads the
per-session BM25 pickle indexing.py already builds. Neither path can see
another session's chunks even if they scored higher.
"""

from __future__ import annotations

import pickle

from sqlalchemy.orm import Session as DbSession

from app.core.config import get_settings
from app.core.interfaces import RerankedCandidate, RetrievedCandidate
from app.db.repositories import content_repository as content
from app.services.embedding import get_embedder
from app.services.indexing import bm25_index_path, get_vector_store_client, tokenize
from app.services.reranking import get_reranker

settings = get_settings()


class RetrievalError(Exception):
    """Raised for any retrieval failure; the message is what the UI shows."""


def _dense_search(query: str, session_id: str, top_k: int) -> list[str]:
    """Returns chunk ids ranked by dense (embedding) similarity, most
    relevant first. Empty list (not an error) whenever nothing's been
    indexed yet for this session -- an empty result set is a completely
    normal, expected state before any source finishes indexing, not a
    failure."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_vector_store_client()
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    session_filter = Filter(
        must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    )
    try:
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            query_filter=session_filter,
            limit=top_k,
            with_payload=False,
        )
    except Exception:
        # The collection may not exist yet -- no source in this (or any)
        # session has finished indexing. That's a legitimate "no results"
        # state, not a real retrieval failure, so it degrades to an empty
        # ranking rather than raising and blocking the sparse side too.
        return []
    return [str(point.id) for point in results]


def _sparse_search(query: str, session_id: str, top_k: int) -> list[str]:
    """Returns chunk ids ranked by BM25 keyword score, most relevant
    first. Empty list whenever this session has no BM25 index yet (same
    "nothing indexed yet" reasoning as _dense_search above)."""
    path = bm25_index_path(session_id)
    if not path.exists():
        return []

    with open(path, "rb") as f:
        data = pickle.load(f)

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = data["bm25"].get_scores(tokenized_query)
    chunk_ids = data["chunk_ids"]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunk_ids[i] for i in ranked[:top_k] if scores[i] > 0]


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Standard RRF: score(id) = sum over rankings of 1/(k + rank), rank
    1-indexed. Chunks that show up in both the dense and sparse rankings
    accumulate score from both, which is the entire point -- rewarding
    agreement between two retrieval methods that make different kinds of
    mistakes, rather than trusting either alone."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def retrieve(db: DbSession, session_id: str, query: str) -> list[RerankedCandidate]:
    """The full hybrid-retrieval pipeline for one query: dense + sparse
    search -> RRF fusion -> cross-encoder rerank -> top_k_evidence
    candidates. Returns an empty list (not an error) when this session
    has nothing indexed yet or the query matches nothing -- that's a
    legitimate, expected outcome the caller (eventually Sprint 6's
    generation step) needs to handle as "no evidence found", not treat as
    a crash."""
    query = query.strip()
    if not query:
        raise RetrievalError("Query is empty.")

    dense_ids = _dense_search(query, session_id, settings.max_retrieval_candidates)
    sparse_ids = _sparse_search(query, session_id, settings.max_retrieval_candidates)

    fused = _reciprocal_rank_fusion([dense_ids, sparse_ids], settings.rrf_k)
    fused = fused[: settings.max_retrieval_candidates]
    if not fused:
        return []

    chunks_by_id = content.get_chunks_by_ids(db, [chunk_id for chunk_id, _ in fused])

    candidates = [
        RetrievedCandidate(
            chunk_id=chunk_id, text=chunks_by_id[chunk_id].text, score=score, retrieval_method="fused"
        )
        for chunk_id, score in fused
        if chunk_id in chunks_by_id  # a chunk indexed then later deleted, defensively skipped
    ]
    if not candidates:
        return []

    reranker = get_reranker()
    reranked = reranker.rerank(query, candidates)
    return reranked[: settings.top_k_evidence]
