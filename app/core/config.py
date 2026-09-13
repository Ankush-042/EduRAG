"""
Central configuration for EduRAG.

All model/provider/infra choices are read from environment (.env) so that
any component (ASR, embedder, reranker, generator, NLI verifier, vector
store) can be swapped without touching application code — this is the
"model abstraction" principle from the TRD (Doc 2, sections 44-45) and the
AI/RAG spec (Doc 5, principle 9).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database -----------------------------------------------------
    # SQLite by default: zero external service to run, matches the
    # "consumer-laptop feasible" constraint (PRD Doc 1 sec 36, TRD Doc 2
    # sec 46). Swapping to Postgres later is a one-line env change since
    # everything goes through SQLAlchemy.
    database_url: str = "sqlite:///./data/edurag.db"

    # --- Vector store ---------------------------------------------------
    # Embedded/local Qdrant (a path on disk, not a server) is the default —
    # zero external service to run, same "consumer-laptop feasible" call
    # already made for SQLite over Postgres (PRD Doc 1 sec 36). qdrant_url
    # is kept for later production deployments against a real Qdrant
    # server (a one-line swap, per the model-abstraction principle above);
    # it's simply unused while qdrant_path is set, which is the only mode
    # this MVP actually exercises.
    qdrant_path: str = "./data/qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "edurag_chunks"

    # --- Models (interchangeable, see app/core/interfaces.py) ---------
    asr_model: str = "faster-whisper:base"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # --- Generation -----------------------------------------------------
    # Fast external inference is the primary generation path (TRD Doc 2
    # sec 25/48); a local model is the fallback only (sec 49), never the
    # default, per the accuracy-over-local-purity call made earlier.
    generation_provider: str = "groq"
    generation_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""
    local_generation_model: str = ""

    # --- Retrieval tuning ------------------------------------------------
    max_retrieval_candidates: int = 30
    top_k_evidence: int = 6

    # --- Content structuring (Sprint 3) -----------------------------------
    # Target words per chunk -- sentences (ASR segments) are packed into a
    # chunk until adding the next one would exceed this, never splitting a
    # sentence across chunks. ~180 words is a common small-to-big sweet
    # spot; revisit if the eval set (Sprint 11) says otherwise.
    chunk_target_words: int = 180

    # --- Session ----------------------------------------------------------
    session_ttl_hours: int = 6

    # --- YouTube download auth (optional) ---------------------------------
    # YouTube increasingly serves "Sign in to confirm you're not a bot" to
    # yt-dlp on some videos/IPs even for public, unrestricted content. Both
    # are optional and unset by default — most videos don't need either;
    # set ONE of them (browser cookies are the easier path — just needs to
    # already be logged into YouTube in that browser) only once a specific
    # video actually hits the bot-check.
    #   YOUTUBE_COOKIES_FROM_BROWSER=chrome   (or edge/firefox/brave/...)
    #   YOUTUBE_COOKIES_FILE=./cookies.txt    (exported via a browser extension)
    youtube_cookies_from_browser: str = ""
    youtube_cookies_file: str = ""

    # --- Local storage (ingestion artifacts) ------------------------------
    # Never committed (see .gitignore) — original media/audio stay on disk
    # only, per the local-first / no-large-blobs-in-the-DB rule (Data spec
    # Doc 4 sec 3, 9).
    media_dir: str = "./data/media"
    audio_dir: str = "./data/audio"
    transcript_dir: str = "./data/transcripts"


@lru_cache
def get_settings() -> Settings:
    return Settings()
