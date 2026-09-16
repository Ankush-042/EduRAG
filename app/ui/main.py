"""EduRAG — Streamlit entry point.

Sprint 1: sources can actually be added now (YouTube URL or a local video
upload) and go through real ingestion (download/save -> audio extraction),
with their state-machine status shown per UI/UX spec Doc 3 sec 9-13.
Sprint 2 adds real transcription to that same pipeline, plus a transcript
preview so it's visible that ASR actually ran. The question/answer
workspace and evidence rendering still land in later sprints — this file
keeps growing into `AppShell` incrementally rather than being scaffolded
as dead UI upfront (Doc 3 sec 40).

Sprint 11 (UI polish, explicitly the last priority per his own ordering —
accuracy/grounding came first): this file's actual logic (DB access,
ingestion calls, the answering pipeline, the background-thread status
polling) is UNCHANGED from Sprint 1-10. Everything below is presentation
only:
  - Sources moved into a persistent sidebar (render_sidebar) instead of
    sitting in the main scroll above the Q&A -- the UI/UX spec's own
    source -> question -> answer -> evidence hierarchy (Doc 3) reads
    better as "sources live on the side, the workspace is the answer,"
    not "scroll past your library every time."
  - st.chat_message bubbles are gone. The spec's own non-goals list (Doc
    3 sec listing "not a ChatGPT clone") already ruled out a generic
    chat-bubble look; this renders each turn as an editorial Q&A block
    instead (question as a heading, answer as prose, citations as small
    superscripts) via hand-written HTML/CSS through st.markdown(...,
    unsafe_allow_html=True) -- content Claude fully controls, not a
    dependency on Streamlit's internal (version-fragile) widget classes.
  - Grounding status is a small custom pill badge instead of Streamlit's
    default st.success/st.warning/st.info/st.error boxes, which read as
    generic dev-tool alerts, not an "editorial, premium" product surface.
  - New: a per-answer "Retrieval & grounding details" panel exposing
    latency, per-passage retrieval/rerank/NLI scores, and per-claim
    verdicts -- the actual technical work (contextual retrieval, hybrid
    rerank, claim-level NLI verification) made visible instead of living
    only in the DB.
Global color/font baseline lives in .streamlit/config.toml (safe, stable,
officially documented keys). Anything more specific than that base theme
(the Fraunces/Inter font pairing, badge colors, the editorial Q&A layout)
is done via a single injected <style> block below rather than by
overriding Streamlit's internal widget CSS classes -- those change across
versions and can't be verified against the exact version installed on
his machine from here, so this only ever styles Claude's own hand-written
HTML, never Streamlit's.
"""

import html
import json
import re
import sys
import time
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
    processing_job_repository,
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

# Sprint 7: a source sitting in any of these is still being worked on by a
# background pipeline thread (app/services/ingestion.py) — used to decide
# whether to keep auto-refreshing the page.
_ACTIVE_STATUSES = {"QUEUED", "DOWNLOADING", "EXTRACTING", "TRANSCRIBING", "PROCESSING", "INDEXING"}

# Sprint 11: which status pill style each source status renders as.
_STATUS_BADGE_VARIANT = {
    "READY": "grounded",
    "FAILED": "failed",
    "CANCELLED": "unverified",
}

# ---------------------------------------------------------------------------
# Sprint 11: presentation-only styling. Nothing below touches app state.
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
.edu-app-title, .edu-qa-question, .edu-detail-subhead {
    font-family: 'Fraunces', Georgia, serif !important;
}

hr { border-color: #E4E0D8 !important; }

/* -- app header (main pane) -- */
.edu-app-title {
    font-size: 2rem;
    font-weight: 600;
    margin-bottom: 0.15rem;
    color: #1F1F1F;
}
.edu-app-subtitle {
    color: #6B6B63;
    font-size: 0.98rem;
    margin-bottom: 1.1rem;
}

/* -- sidebar -- */
.edu-sidebar-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.1rem;
}
.edu-sidebar-subtitle {
    color: #6B6B63;
    font-size: 0.85rem;
    margin-bottom: 0.9rem;
}
.edu-source-title {
    font-weight: 600;
    font-size: 0.97rem;
    margin-bottom: 0.1rem;
    color: #1F1F1F;
}
.edu-source-meta {
    color: #7A776E;
    font-size: 0.8rem;
    margin-bottom: 0.45rem;
}
.edu-source-error {
    color: #8C3B2E;
    font-size: 0.82rem;
    background: #F7EDE9;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    margin-top: 0.3rem;
}

/* -- status / grounding pill badges -- */
.edu-badge {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    line-height: 1.4;
}
.edu-badge--sm { font-size: 0.68rem; padding: 0.12rem 0.5rem; }
.edu-badge--grounded    { background: #E9F0EC; color: #2F5D50; }
.edu-badge--partial     { background: #F7EFDF; color: #8A6D3B; }
.edu-badge--unverified  { background: #EFEDE9; color: #6B6B63; }
.edu-badge--abstained   { background: #EFEDE9; color: #6B6B63; }
.edu-badge--failed      { background: #F7EDE9; color: #8C3B2E; }

/* -- Q&A workspace -- */
.edu-qa-block {
    padding: 1.1rem 0 1.0rem 0;
    border-bottom: 1px solid #ECE8DE;
}
.edu-qa-block:last-child { border-bottom: none; }
.edu-qa-question {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1F1F1F;
    margin-bottom: 0.5rem;
}
.edu-qa-meta { margin-bottom: 0.55rem; }
.edu-qa-answer {
    font-size: 0.98rem;
    line-height: 1.65;
    color: #2B2B27;
}
.edu-qa-answer p { margin: 0 0 0.7rem 0; }
.edu-qa-answer p:last-child { margin-bottom: 0; }
.edu-citation {
    color: #2F5D50;
    font-weight: 600;
    margin-left: 1px;
}

/* -- evidence / footnotes -- */
.edu-evidence-item {
    padding: 0.5rem 0;
    border-bottom: 1px solid #F0EDE6;
}
.edu-evidence-item:last-child { border-bottom: none; }
.edu-evidence-meta {
    font-size: 0.78rem;
    font-weight: 600;
    color: #7A776E;
    margin-bottom: 0.15rem;
}
.edu-evidence-text {
    font-size: 0.9rem;
    color: #45443E;
    line-height: 1.5;
}

/* -- retrieval/grounding diagnostics -- */
.edu-detail-latency {
    font-size: 0.82rem;
    color: #6B6B63;
    margin-bottom: 0.6rem;
}
.edu-detail-subhead {
    font-size: 0.85rem;
    font-weight: 600;
    color: #45443E;
    margin: 0.7rem 0 0.35rem 0;
}
.edu-detail-row {
    font-size: 0.82rem;
    color: #45443E;
    padding: 0.15rem 0;
}
.edu-detail-claim {
    padding: 0.4rem 0;
    border-bottom: 1px solid #F0EDE6;
}
.edu-detail-claim:last-child { border-bottom: none; }
.edu-detail-claim-text {
    font-size: 0.85rem;
    color: #2B2B27;
}
.edu-detail-claim-meta {
    font-size: 0.75rem;
    color: #8C8A80;
    margin-top: 0.1rem;
}
</style>
"""


def _inject_custom_css() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _badge_html(label: str, variant: str, *, small: bool = False) -> str:
    size_class = " edu-badge--sm" if small else ""
    return f'<span class="edu-badge edu-badge--{variant}{size_class}">{html.escape(label)}</span>'


# Citation markers in a generated answer are always "[n]" or "[n, m]" —
# generation.py's system prompt (rule 5, added in Sprint 9) commits the
# model to that shape and nothing else. This regex only ever runs on
# already-HTML-escaped text for DISPLAY purposes; it never touches the
# stored message content, so nothing here can corrupt what's persisted
# or re-affect grounding verification (that runs entirely in
# answering.py/claims.py, well before this file ever sees the answer).
_DISPLAY_CITATION_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def _render_answer_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = _DISPLAY_CITATION_RE.sub(r'<sup class="edu-citation">[\1]</sup>', escaped)
    paragraphs = [p for p in escaped.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [escaped]
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs)


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
    # Self-audit finding: `if not seconds` treats a genuinely meaningful
    # 0 (the very start of a source -- a completely normal start_time for
    # a citation's first chunk, especially once VAD trims leading silence
    # to exactly 0.0) the same as None/unknown, silently dropping the
    # timestamp from that citation. `is None` is the actual "unknown"
    # check; 0 is a real, displayable value ("0:00"), not "no timestamp."
    if seconds is None:
        return ""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _format_latency(latency_ms: int | None) -> str:
    if latency_ms is None:
        return ""
    if latency_ms >= 1000:
        return f"{latency_ms / 1000:.1f}s"
    return f"{latency_ms} ms"


def render_header() -> None:
    st.markdown('<div class="edu-app-title">EduRAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="edu-app-subtitle">Learn from your material. Ask anything. Verify the answer.</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(session_id: str) -> None:
    """Sprint 11: source management moved here (was the top half of the
    main pane) so the main pane can be a pure Q&A workspace. Logic is
    identical to the old render_add_source/render_source_list — only
    where it renders, and how each source's status/metadata looks,
    changed."""
    with st.sidebar:
        st.markdown('<div class="edu-sidebar-title">Your sources</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="edu-sidebar-subtitle">Add a YouTube lecture or a local video to begin.</div>',
            unsafe_allow_html=True,
        )

        tab_youtube, tab_local = st.tabs(["YouTube", "Local video"])

        with tab_youtube:
            with st.form("add_youtube_form", clear_on_submit=True):
                url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                submitted = st.form_submit_button("Add source")
            if submitted and url:
                # Sprint 7: add_youtube_source now only validates + creates
                # the row and hands the real work to a background thread,
                # so this returns almost immediately — no more blocking
                # the whole UI for the length of the video.
                with SessionLocal() as db:
                    service = SourceIngestionService(db)
                    try:
                        service.add_youtube_source(session_id, url)
                    except IngestionError as exc:
                        db.rollback()
                        st.error(str(exc))
                st.rerun()

        with tab_local:
            uploaded = st.file_uploader("Upload a video", type=["mp4", "mkv", "mov", "webm", "avi"])
            if uploaded is not None and st.button("Add this video"):
                with st.spinner("Saving upload..."):
                    with SessionLocal() as db:
                        service = SourceIngestionService(db)
                        try:
                            upload = UploadedFile(
                                name=uploaded.name,
                                read_bytes=uploaded.getvalue(),
                                mime_type=uploaded.type,
                            )
                            service.add_local_video_source(session_id, upload)
                        except IngestionError as exc:
                            db.rollback()
                            st.error(str(exc))
                st.rerun()

        with SessionLocal() as db:
            current_sources = source_repository.list_sources_for_session(db, session_id)

        if not current_sources:
            return

        st.divider()
        for source in current_sources:
            with st.container(border=True):
                title = source.title or source.original_name or source.source_url or "Untitled source"
                meta_bits = [source.source_type.replace("_", " ").title()]
                duration = _format_duration(source.duration_seconds)
                if duration:
                    meta_bits.append(duration)
                if source.language:
                    meta_bits.append(f"Language: {source.language.upper()}")
                st.markdown(f'<div class="edu-source-title">{html.escape(title)}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="edu-source-meta">{html.escape(" · ".join(meta_bits))}</div>',
                    unsafe_allow_html=True,
                )

                status_label = STATUS_LABELS.get(source.status, source.status)
                if source.status == "FAILED":
                    st.markdown(_badge_html(status_label, "failed"), unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="edu-source-error">{html.escape(source.error_message or "Unknown error")}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Remove", key=f"remove_{source.id}"):
                        with SessionLocal() as db:
                            s = source_repository.get_source(db, source.id)
                            source_repository.update_source_status(db, s, "DELETED")
                            db.commit()
                        st.rerun()
                elif source.status == "READY":
                    st.markdown(_badge_html(status_label, "grounded"), unsafe_allow_html=True)
                elif source.status in _ACTIVE_STATUSES:
                    # Sprint 7: this source is being worked on right now by
                    # a background pipeline thread — show real per-stage
                    # progress (already tracked in ProcessingJob) instead
                    # of a static "processing" label with no sense of
                    # movement.
                    with SessionLocal() as db:
                        job = processing_job_repository.latest_job_for_source(db, source.id)
                    stage_label = STATUS_LABELS.get(source.status, source.status)
                    progress = job.progress if job else 0.0
                    st.progress(min(max(progress, 0.0), 1.0), text=stage_label)
                else:
                    st.markdown(_badge_html(status_label, "unverified"), unsafe_allow_html=True)

                # Sprint 2 proof-of-work: once transcription has actually
                # run, show a snippet so it's visible in the UI, not just
                # in job logs.
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

                # Sprint 3/4 proof-of-work: once structuring has actually
                # run, show how many chunks came out of it -- and once
                # indexing has actually completed (READY), that they're
                # searchable, not just chunked.
                if source.status in _STRUCTURED_STATUSES:
                    with SessionLocal() as db:
                        chunk_count = content_repository.count_chunks_for_source(db, source.id)
                    if chunk_count:
                        plural = "s" if chunk_count != 1 else ""
                        if source.status == "READY":
                            st.caption(f"{chunk_count} chunk{plural} indexed and searchable")
                        else:
                            st.caption(f"{chunk_count} chunk{plural} ready for indexing")


def _render_evidence(db, message_id: str) -> None:
    """Footnote-style citation list: numbered, source title + timestamp,
    the raw evidence text underneath. Same data as before (Sprint 6);
    only the rendering changed -- plain st.write/st.caption calls became
    hand-built HTML so this reads like a document's footnotes instead of
    a generic expander dump."""
    evidence_rows = conversation_repository.list_evidence_for_message(db, message_id)
    if not evidence_rows:
        return
    chunks_by_id = content_repository.get_chunks_by_ids(db, [e.chunk_id for e in evidence_rows])
    sources_by_id = source_repository.get_sources_by_ids(db, [e.source_id for e in evidence_rows])
    with st.expander(f"Sources cited ({len(evidence_rows)})"):
        for evidence in evidence_rows:
            chunk = chunks_by_id.get(evidence.chunk_id)
            source = sources_by_id.get(evidence.source_id)
            title = source.title if source and source.title else "Untitled source"
            meta_bits = [f"[{evidence.rank}] {title}"]
            # Self-audit finding: `if evidence.start_time` drops a real
            # 0.0 start (the very first chunk of a source) the same way
            # as a missing one. `is not None` is the correct check here.
            start = _format_duration(int(evidence.start_time)) if evidence.start_time is not None else None
            if start:
                meta_bits.append(f"at {start}")
            text = chunk.text if chunk is not None else ""
            st.markdown(
                f'<div class="edu-evidence-item">'
                f'<div class="edu-evidence-meta">{html.escape(" · ".join(meta_bits))}</div>'
                f'<div class="edu-evidence-text">{html.escape(text)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_retrieval_details(db, message) -> None:
    """Sprint 11 addition: the actual retrieval/rerank/grounding work
    (contextual retrieval, hybrid search + RRF fusion + cross-encoder
    rerank, per-claim NLI verification) has existed since Sprint 5-6 but
    was never visible anywhere except the DB. This surfaces it per
    answer, collapsed by default -- diagnostic, not required reading for
    a learner just trying to study, but real proof of the pipeline
    working rather than a claimed capability."""
    evidence_rows = conversation_repository.list_evidence_for_message(db, message.id)
    verification_results = conversation_repository.list_verification_results_for_message(db, message.id)
    if not evidence_rows and not verification_results:
        return

    claims_by_id = conversation_repository.get_claims_by_ids(
        db, [v.claim_id for v in verification_results]
    )
    evidence_by_id = {e.id: e for e in evidence_rows}

    with st.expander("Retrieval & grounding details"):
        latency_label = _format_latency(message.latency_ms)
        if latency_label:
            st.markdown(f'<div class="edu-detail-latency">Answered in {latency_label}</div>', unsafe_allow_html=True)

        if evidence_rows:
            st.markdown('<div class="edu-detail-subhead">Retrieved passages</div>', unsafe_allow_html=True)
            for evidence in evidence_rows:
                bits = [f"#{evidence.rank}"]
                if evidence.retrieval_score is not None:
                    bits.append(f"retrieval {evidence.retrieval_score:.3f}")
                if evidence.reranker_score is not None:
                    bits.append(f"rerank {evidence.reranker_score:.3f}")
                if evidence.nli_score is not None:
                    bits.append(f"NLI {evidence.nli_score:.3f}")
                st.markdown(
                    f'<div class="edu-detail-row">{html.escape(" · ".join(bits))}</div>',
                    unsafe_allow_html=True,
                )

        if verification_results:
            st.markdown('<div class="edu-detail-subhead">Per-claim verification</div>', unsafe_allow_html=True)
            for result in verification_results:
                claim = claims_by_id.get(result.claim_id)
                evidence = evidence_by_id.get(result.evidence_id)
                claim_text = claim.claim_text if claim is not None else "(claim text unavailable)"
                cites_bit = f"cites #{evidence.rank}" if evidence is not None else ""
                verdict_variant = "grounded" if result.verdict == "ENTAILMENT" else "unverified"
                verdict_label = result.verdict.replace("_", " ").title()
                meta_bits = [b for b in (cites_bit, f"score {result.score:.3f}") if b]
                st.markdown(
                    f'<div class="edu-detail-claim">'
                    f'{_badge_html(verdict_label, verdict_variant, small=True)} '
                    f'<span class="edu-detail-claim-text">{html.escape(claim_text)}</span>'
                    f'<div class="edu-detail-claim-meta">{html.escape(" · ".join(meta_bits))}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


_GROUNDING_BADGE_META = {
    "GROUNDED": ("Grounded in your sources", "grounded"),
    "PARTIALLY_GROUNDED": ("Partially grounded", "partial"),
    "UNVERIFIED": ("Could not be verified", "unverified"),
    "ABSTAINED": ("No answer found", "abstained"),
}


def _render_turn(db, user_message, assistant_message) -> None:
    st.markdown(
        f'<div class="edu-qa-block">'
        f'<div class="edu-qa-question">{html.escape(user_message.content)}</div>',
        unsafe_allow_html=True,
    )
    if assistant_message is not None:
        label, variant = _GROUNDING_BADGE_META.get(
            assistant_message.grounding_status,
            (assistant_message.grounding_status or "", "unverified"),
        )
        if label:
            st.markdown(f'<div class="edu-qa-meta">{_badge_html(label, variant)}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="edu-qa-answer">{_render_answer_html(assistant_message.content)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        _render_evidence(db, assistant_message.id)
        _render_retrieval_details(db, assistant_message)
    else:
        st.markdown("</div>", unsafe_allow_html=True)


def render_ask(session_id: str) -> None:
    """The question/answer workspace: retrieval -> generation -> claim-level
    grounding verification -> a persisted, cited answer (TRD Doc 2 sec
    21-30). Renders the session's whole conversation (not just the latest
    turn), since messages/evidence are now actually persisted rather than
    being a single-shot preview.

    Sprint 11: no longer st.chat_message bubbles -- each USER/ASSISTANT
    message pair renders as one editorial block via _render_turn. The
    retrieval/generation/verification logic this reads from is completely
    unchanged from Sprint 6-10."""
    with SessionLocal() as db:
        sources_for_session = source_repository.list_sources_for_session(db, session_id)
    has_ready_source = any(s.status == "READY" for s in sources_for_session)

    if not has_ready_source:
        if sources_for_session:
            st.info("Your source is still processing — check the sidebar for progress.")
        else:
            st.info("Add a source in the sidebar to start asking questions.")
        return

    st.caption("Answers are generated only from your sources, with every claim checked against the evidence.")

    with SessionLocal() as db:
        conversation = conversation_repository.get_latest_conversation_for_session(db, session_id)
        history = (
            conversation_repository.list_messages_for_conversation(db, conversation.id)
            if conversation is not None
            else []
        )
        pending_user_message = None
        for message in history:
            if message.role == "USER":
                pending_user_message = message
            elif message.role == "ASSISTANT" and pending_user_message is not None:
                _render_turn(db, pending_user_message, message)
                pending_user_message = None
        if pending_user_message is not None:
            # A user turn was persisted but its assistant turn wasn't
            # (mid-flight, or the answering call raised) -- still show the
            # question rather than silently dropping it.
            _render_turn(db, pending_user_message, None)

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


def _has_active_source(session_id: str) -> bool:
    with SessionLocal() as db:
        return any(
            s.status in _ACTIVE_STATUSES
            for s in source_repository.list_sources_for_session(db, session_id)
        )


def main() -> None:
    _ensure_schema()
    _inject_custom_css()
    session_id = _bootstrap_session_id()
    render_header()
    render_sidebar(session_id)
    render_ask(session_id)

    # Sprint 7: while a background pipeline thread is still working on a
    # source, keep re-running this script every couple seconds so its
    # progress (read fresh from the DB each render_sidebar call above)
    # actually moves on screen -- without this the page would sit frozen
    # on whatever stage it was in at page load, even though the thread
    # itself is making real progress underneath it.
    if _has_active_source(session_id):
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
