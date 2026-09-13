"""EduRAG — Streamlit entry point.

Sprint 1: sources can actually be added now (YouTube URL or a local video
upload) and go through real ingestion (download/save -> audio extraction),
with their state-machine status shown per UI/UX spec Doc 3 sec 9-13.
Sprint 2 adds real transcription to that same pipeline, plus a transcript
preview so it's visible that ASR actually ran. The question/answer
workspace and evidence rendering still land in later sprints — this file
keeps growing into `AppShell` incrementally rather than being scaffolded
as dead UI upfront (Doc 3 sec 40).
"""

import json
import sys
from pathlib import Path

# Streamlit executes this file directly, so sys.path[0] is this file's own
# folder (app/ui), not the project root — `import app...` fails otherwise.
# This makes `streamlit run app/ui/app.py` work regardless of cwd/OS.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.db import models  # noqa: F401  (registers tables before create_all)
from app.db.repositories import (
    content_repository,
    conversation_repository,
    session_repository,
    source_repository,
)
from app.services import answering
from app.services.ingestion import IngestionError, SourceIngestionService, UploadedFile

st.set_page_config(page_title="EduRAG", page_icon=None, layout="centered")

STATUS_LABELS = {
    "QUEUED": "Queued",
    "DOWNLOADING": "Downloading",
    "EXTRACTING": "Extracting audio",
    "TRANSCRIBING": "Transcribing",
    "PROCESSING": "Structuring content",
    "INDEXING": "Embedding & indexing",
    "READY": "Ready",
    "FAILED": "Couldn't process this source",
    "CANCELLED": "Cancelled",
}

# Statuses reached only after transcription has actually completed —
# safe to look for a transcript artifact at these stages.
_TRANSCRIBED_STATUSES = {"PROCESSING", "INDEXING", "READY"}

# Statuses reached only after content structuring has actually completed —
# safe to look for chunks at these stages.
_STRUCTURED_STATUSES = {"INDEXING", "READY"}


def _ensure_schema() -> None:
    # Sprint 0 convenience only — real migrations run through Alembic
    # (TRD Doc 2 sec 39); this just means a fresh clone works immediately.
    Base.metadata.create_all(bind=engine)


def _bootstrap_session_id() -> str:
    """One temporary, no-login session per browser tab (Data spec Doc 4
    sec 6) — created once and reused across Streamlit reruns."""
    if "session_id" in st.session_state:
        with SessionLocal() as db:
            existing = session_repository.get_active_session(db, st.session_state["session_id"])
            if existing is not None:
                session_repository.touch_session(db, existing)
                db.commit()
                return existing.id

    with SessionLocal() as db:
        session = session_repository.create_session(db)
        db.commit()
        st.session_state["session_id"] = session.id
        return session.id


def _format_duration(seconds: int | None) -> str:
    if not seconds:
        return ""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def render_header() -> None:
    st.markdown("## EduRAG")
    st.markdown("Learn from your material. Ask anything. Verify the answer.")
    st.divider()


def render_add_source(session_id: str) -> None:
    st.markdown("### Start with your learning material")
    st.markdown("Add a YouTube lecture or upload a local video to begin.")

    tab_youtube, tab_local = st.tabs(["YouTube", "Local video"])

    with tab_youtube:
        with st.form("add_youtube_form", clear_on_submit=True):
            url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            submitted = st.form_submit_button("Add source")
        if submitted and url:
            with st.spinner("Downloading and extracting audio — this can take a while for long videos..."):
                with SessionLocal() as db:
                    service = SourceIngestionService(db)
                    try:
                        source = service.add_youtube_source(session_id, url)
                        db.commit()
                        if source.status == "FAILED":
                            st.error(f"Couldn't process this source: {source.error_message}")
                    except IngestionError as exc:
                        db.rollback()
                        st.error(str(exc))
            st.rerun()

    with tab_local:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mkv", "mov", "webm", "avi"])
        if uploaded is not None and st.button("Add this video"):
            with st.spinner("Saving and extracting audio..."):
                with SessionLocal() as db:
                    service = SourceIngestionService(db)
                    try:
                        upload = UploadedFile(
                            name=uploaded.name,
                            read_bytes=uploaded.getvalue(),
                            mime_type=uploaded.type,
                        )
                        source = service.add_local_video_source(session_id, upload)
                        db.commit()
                        if source.status == "FAILED":
                            st.error(f"Couldn't process this source: {source.error_message}")
                    except IngestionError as exc:
                        db.rollback()
                        st.error(str(exc))
            st.rerun()


def render_source_list(session_id: str) -> None:
    with SessionLocal() as db:
        current_sources = source_repository.list_sources_for_session(db, session_id)

    if not current_sources:
        return

    st.divider()
    st.markdown("### Your sources")
    for source in current_sources:
        with st.container(border=True):
            title = source.title or source.original_name or source.source_url or "Untitled source"
            meta_bits = [source.source_type.replace("_", " ").title()]
            duration = _format_duration(source.duration_seconds)
            if duration:
                meta_bits.append(duration)
            if source.language:
                meta_bits.append(f"Language: {source.language.upper()}")
            st.markdown(f"**{title}**")
            st.caption(" · ".join(meta_bits))

            status_label = STATUS_LABELS.get(source.status, source.status)
            if source.status == "FAILED":
                st.error(f"{status_label}: {source.error_message or 'unknown error'}")
                if st.button("Remove", key=f"remove_{source.id}"):
                    with SessionLocal() as db:
                        s = source_repository.get_source(db, source.id)
                        source_repository.update_source_status(db, s, "DELETED")
                        db.commit()
                    st.rerun()
            elif source.status == "READY":
                st.success(status_label)
            else:
                st.info(status_label)

            # Sprint 2 proof-of-work: once transcription has actually run,
            # show a snippet so it's visible in the UI, not just in job logs.
            if source.status in _TRANSCRIBED_STATUSES:
                with SessionLocal() as db:
                    artifact = source_repository.get_latest_artifact(
                        db, source_id=source.id, artifact_type="NORMALIZED_TRANSCRIPT"
                    )
                if artifact is not None:
                    with st.expander("Transcript preview"):
                        try:
                            payload = json.loads(Path(artifact.storage_path).read_text(encoding="utf-8"))
                            device = payload.get("asr_device")
                            if device:
                                st.caption(f"Transcribed on: {device.upper()}")
                            preview_text = " ".join(seg["text"] for seg in payload["segments"])
                            st.write(preview_text[:1500] + ("…" if len(preview_text) > 1500 else ""))
                        except (OSError, json.JSONDecodeError, KeyError) as exc:
                            st.caption(f"Couldn't load transcript preview: {exc}")

            # Sprint 3/4 proof-of-work: once structuring has actually run,
            # show how many chunks came out of it -- and once indexing has
            # actually completed (READY), that they're searchable, not
            # just chunked.
            if source.status in _STRUCTURED_STATUSES:
                with SessionLocal() as db:
                    chunk_count = content_repository.count_chunks_for_source(db, source.id)
                if chunk_count:
                    plural = "s" if chunk_count != 1 else ""
                    if source.status == "READY":
                        st.caption(f"{chunk_count} chunk{plural} indexed and searchable")
                    else:
                        st.caption(f"{chunk_count} chunk{plural} ready for indexing")


_GROUNDING_LABELS = {
    "GROUNDED": ("Grounded in your sources", "success"),
    "PARTIALLY_GROUNDED": ("Partially grounded", "warning"),
    "UNVERIFIED": ("Could not be verified", "warning"),
    "ABSTAINED": ("No answer found", "info"),
}


def _render_grounding_badge(status: str | None) -> None:
    if not status:
        return
    label, kind = _GROUNDING_LABELS.get(status, (status.title(), "info"))
    getattr(st, kind)(label)


def _render_evidence(db, message_id: str) -> None:
    evidence_rows = conversation_repository.list_evidence_for_message(db, message_id)
    if not evidence_rows:
        return
    chunks_by_id = content_repository.get_chunks_by_ids(db, [e.chunk_id for e in evidence_rows])
    with st.expander(f"Evidence ({len(evidence_rows)})"):
        for evidence in evidence_rows:
            chunk = chunks_by_id.get(evidence.chunk_id)
            meta_bits = [f"#{evidence.rank}"]
            start = _format_duration(int(evidence.start_time)) if evidence.start_time else None
            if start:
                meta_bits.append(f"at {start}")
            st.caption(" · ".join(meta_bits))
            if chunk is not None:
                st.write(chunk.text)


def render_ask(session_id: str) -> None:
    """The question/answer workspace: retrieval -> generation -> claim-level
    grounding verification -> a persisted, cited answer (TRD Doc 2 sec
    21-30). Renders the session's whole conversation (not just the latest
    turn), since messages/evidence are now actually persisted rather than
    being a single-shot preview."""
    with SessionLocal() as db:
        has_ready_source = any(
            s.status == "READY" for s in source_repository.list_sources_for_session(db, session_id)
        )
    if not has_ready_source:
        return

    st.divider()
    st.markdown("### Ask")
    st.caption("Answers are generated only from your sources, with every claim checked against the evidence.")

    with SessionLocal() as db:
        conversation = conversation_repository.get_latest_conversation_for_session(db, session_id)
        history = (
            conversation_repository.list_messages_for_conversation(db, conversation.id)
            if conversation is not None
            else []
        )
        for message in history:
            if message.role == "USER":
                with st.chat_message("user"):
                    st.write(message.content)
            elif message.role == "ASSISTANT":
                with st.chat_message("assistant"):
                    _render_grounding_badge(message.grounding_status)
                    st.write(message.content)
                    _render_evidence(db, message.id)

    with st.form("ask_form", clear_on_submit=True):
        query = st.text_input("Question", placeholder="e.g. What is a comment in Python?")
        submitted = st.form_submit_button("Ask")

    if not submitted or not query:
        return

    with st.spinner("Thinking..."):
        with SessionLocal() as db:
            try:
                answering.answer(db, session_id, query)
                db.commit()
            except answering.AnsweringError as exc:
                db.rollback()
                st.error(str(exc))
                return
    st.rerun()


def main() -> None:
    _ensure_schema()
    session_id = _bootstrap_session_id()
    render_header()
    render_add_source(session_id)
    render_source_list(session_id)
    render_ask(session_id)


if __name__ == "__main__":
    main()
