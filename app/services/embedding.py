"""Sprint 4 (part 1) — dense embeddings: turns chunk text into vectors for
the vector store, and queries into the same vector space at retrieval
time (TRD Doc 2 sec 13; AI/RAG spec Doc 5 principle 9's model-abstraction
rule — this is the Embedder interface's only concrete implementation for
now, swappable later without touching indexing.py or the retrieval
pipeline that will consume it in Sprint 5).

Kept independent of app/services/indexing.py, same split as transcription
vs. structuring: this module only knows how to turn text into vectors,
not what to do with them.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.interfaces import Embedder

settings = get_settings()

# Loaded models are expensive (seconds, hundreds of MB) — cached by model
# name so re-embedding within the same process reuses the same instance,
# same reasoning as transcription.py's _MODEL_CACHE.
_MODEL_CACHE: dict[str, object] = {}

# BAAI/bge-* models were trained with an asymmetric convention: passages
# are embedded as-is, but queries need a fixed instruction prefix prepended
# or retrieval quality measurably drops (this is documented by the model
# authors, not an EduRAG-specific quirk). Most other sentence-transformers
# models (e5, MiniLM, ...) don't need this, so it's applied conditionally
# on the configured model name rather than unconditionally.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _needs_bge_query_instruction(model_name: str) -> bool:
    return model_name.lower().startswith("baai/bge")


def _get_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from sentence_transformers import SentenceTransformer

    # No explicit device= here (unlike transcription.py's GPU-first
    # dance): embedding models at this size (bge-small is 33M params) run
    # fast enough on CPU that chasing GPU wiring for this step isn't worth
    # the added failure surface — sentence-transformers already picks CUDA
    # automatically via torch when it's usable, so GPU is used for free
    # wherever the transcription GPU setup already made torch/CUDA work,
    # with no extra code needed here.
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


class SentenceTransformerEmbedder(Embedder):
    """Concrete Embedder (app/core/interfaces.py) backed by
    sentence-transformers. Swapping embedding models/providers later means
    adding another class here, not touching indexing.py or retrieval."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_model(self._model_name)
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        model = _get_model(self._model_name)
        text = query
        if _needs_bge_query_instruction(self._model_name):
            text = _BGE_QUERY_INSTRUCTION + query
        vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def dimension(self) -> int:
        """The vector size this model produces — needed once, up front,
        to create the Qdrant collection with the right size (Qdrant
        rejects vectors that don't match a collection's declared
        dimension). Cheap: sentence-transformers models publish this
        without needing to actually embed anything."""
        model = _get_model(self._model_name)
        return int(model.get_sentence_embedding_dimension())


def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings.embedding_model)
