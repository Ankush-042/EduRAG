"""EduRAG — Streamlit entry point.

Sprint 1: sources can actually be added now (YouTube URL or a local video
upload) and go through real ingestion (download/save -> audio extraction),
with their state-machine status shown per UI/UX spec Doc 3 sec 9-13.
Sprint 2 adds real transcription to that same pipeline, plus a transcript
preview so it's visible that ASR actually ran. The question/answer
workspace and evidence rendering still land in later sprints — this file
keeps growing into `AppShell` incrementally rather than being scaffolded
as dead UI upfront (Doc 3 sec 40).

Sprint 11 (UI polish): sources moved into a persistent sidebar, chat
bubbles replaced with an editorial Q&A layout, custom pill badges,
Fraunces/Inter typography, a per-answer "Retrieval & grounding details"
panel.

Post-Sprint-11 visual redesign (his explicit feedback: the UI "felt like
a normal RAG-over-docs tool" and needed to look nothing like one): a full
presentation-layer pass, still zero changes to DB access, ingestion,
retrieval, generation, or grounding logic below. What changed and why:

  - A real visual identity instead of a generic light theme: a warm
    "paper" background, a forest-green/amber/terracotta status language
    (not default red/yellow/green alert colors), and a third typeface —
    JetBrains Mono — used ONLY for technical/numeric content (timestamps,
    scores, durations). Serif headings + sans body + mono numbers is a
    deliberate signal: "prose is generated, numbers are measured" — which
    is the actual, true story of how this app works, and nothing a
    generic ChatGPT-wrapper RAG demo bothers to visually distinguish.
  - A real brand mark (inline SVG: a play-triangle inside a ring — "video,
    verified") used in the header, the empty state, and nowhere else —
    consistent, not decorative noise.
  - A real empty state (a 3-step "how this works" strip) replacing a bare
    st.info() line — the first thing a visitor with no sources yet sees
    is currently the least distinctive part of the old UI.
  - Status/grounding badges gained an icon glyph (✓ / ◐ / ✕ / – / ?)
    instead of color alone — color-only status communication is both a
    generic-dashboard tell and a real (if minor) accessibility gap.
  - Source cards in the sidebar gained a colored top accent keyed to
    status, and the progress bar is now hand-built (a div with a
    percentage width, not st.progress()) so its exact look is controlled
    rather than inherited from Streamlit's internal (version-fragile)
    widget markup.
  - Evidence citations render as a real footnote rail: a circular rank
    chip, a monospace timecode chip, and the quoted passage in an actual
    blockquote treatment with a left accent bar — instead of a flat
    "meta line + text line" list.
  - The retrieval/grounding diagnostics panel no longer prints raw
    "retrieval 0.031 · rerank 4.207 · NLI 0.812" number soup. NLI score is
    genuinely a 0-1 probability-like value (grounding.py's HHEM output),
    so it gets an honest horizontal meter with a marker at the real 0.5
    decision threshold. Retrieval (RRF-fused, see retrieval.py) and rerank
    (cross-encoder logit, unbounded, see reranking.py) scores are NOT
    0-1-bounded — rendering them as percentage-fill bars would fabricate
    false precision and could visually mis-rank candidates (e.g. a
    reasonable RRF score like 0.03 would draw as a nearly-empty bar).
    Those two stay clean monospace numeric chips instead: a real style
    upgrade over the old plain text without inventing a scale that isn't
    there.
  - Citation markers in answer prose render as small filled chips (still
    "[1]"-shaped, still exactly what's stored) rather than plain colored
    bracket text.

Global color/font baseline lives in .streamlit/config.toml (safe, stable,
officially documented keys) and was re-tuned to match this palette.
Everything more specific is one injected <style> block below, styling
only Claude's own hand-written HTML classes — never Streamlit's internal
widget classes, which change across versions and can't be verified
against the exact version installed on his machine from this sandbox.
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

# Which status pill/accent variant each source status renders as.
_STATUS_BADGE_VARIANT = {
    "READY": "grounded",
    "FAILED": "failed",
    "CANCELLED": "unverified",
}

# ---------------------------------------------------------------------------
# Presentation-only styling + small building blocks. Nothing below this
# banner (down to _ensure_schema) touches app state.
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --edu-bg: #FBF9F3;
    --edu-card: #FFFFFF;
    --edu-ink: #1B1B16;
    --edu-muted: #6E6B5C;
    --edu-faint: #96927F;
    --edu-line: #E9E3D4;
    --edu-accent: #1F6F5C;
    --edu-accent-dark: #164F41;
    --edu-accent-soft: #E4EFEA;
    --edu-amber: #B07A2E;
    --edu-amber-soft: #F6EDDD;
    --edu-terracotta: #A6432E;
    --edu-terracotta-soft: #F6E7E2;
    --edu-slate-soft: #EDEBE3;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--edu-bg) !important;
}

/* Subtle warm vignette instead of a flat fill -- reads as "paper", not
   "default light theme #2". Fixed background so it doesn't scroll/tile. */
[data-testid="stAppViewContainer"] > .main {
    background-image: radial-gradient(circle at 15% 0%, rgba(31,111,92,0.05), transparent 45%),
                       radial-gradient(circle at 100% 20%, rgba(176,122,46,0.045), transparent 40%);
    background-attachment: fixed;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
.edu-app-title, .edu-qa-question, .edu-detail-subhead, .edu-empty-title {
    font-family: 'Fraunces', Georgia, serif !important;
}

.edu-mono {
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
}

hr { border-color: var(--edu-line) !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #DDD6C4; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #CFC7B0; }

/* -- brand mark -- */
.edu-mark { color: var(--edu-accent); flex-shrink: 0; display: block; }

/* -- app header (main pane) -- */
.edu-hero {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.35rem;
}
.edu-app-title {
    font-size: 1.9rem;
    font-weight: 600;
    line-height: 1.1;
    color: var(--edu-ink);
}
.edu-app-subtitle {
    color: var(--edu-muted);
    font-size: 0.98rem;
    margin: 0.1rem 0 0.55rem 0;
}
.edu-hero-tags {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--edu-accent-dark);
    padding-bottom: 1.0rem;
    border-bottom: 1px solid var(--edu-line);
    margin-bottom: 1.2rem;
}
.edu-hero-tags span.sep { color: var(--edu-faint); margin: 0 0.5rem; }

/* -- empty state -- */
.edu-empty {
    text-align: center;
    padding: 2.6rem 1rem 1.6rem 1rem;
}
.edu-empty .edu-mark { margin: 0 auto 0.9rem auto; color: var(--edu-accent); }
.edu-empty-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--edu-ink);
    margin-bottom: 0.35rem;
}
.edu-empty-sub {
    color: var(--edu-muted);
    font-size: 0.95rem;
    max-width: 30rem;
    margin: 0 auto 1.8rem auto;
}
.edu-steps {
    display: flex;
    justify-content: center;
    gap: 0;
    max-width: 34rem;
    margin: 0 auto;
}
.edu-step {
    flex: 1;
    padding: 0 0.6rem;
    position: relative;
}
.edu-step::after {
    content: "";
    position: absolute;
    top: 0.85rem;
    right: -0.1rem;
    width: 0.9rem;
    height: 1px;
    background: var(--edu-line);
}
.edu-step:last-child::after { display: none; }
.edu-step-num {
    width: 1.7rem;
    height: 1.7rem;
    line-height: 1.7rem;
    border-radius: 50%;
    background: var(--edu-accent-soft);
    color: var(--edu-accent-dark);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
    margin: 0 auto 0.5rem auto;
}
.edu-step-label {
    font-size: 0.82rem;
    color: var(--edu-ink);
    font-weight: 500;
    line-height: 1.35;
}

/* -- sidebar -- */
.edu-sidebar-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--edu-ink);
    margin-bottom: 0.1rem;
}
.edu-sidebar-title .edu-mark { width: 18px; height: 18px; }
.edu-sidebar-subtitle {
    color: var(--edu-muted);
    font-size: 0.83rem;
    margin-bottom: 0.9rem;
}
.edu-card-accent {
    height: 3px;
    border-radius: 3px;
    margin: -0.15rem -0.1rem 0.65rem -0.1rem;
}
.edu-source-row {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    margin-bottom: 0.15rem;
}
.edu-source-glyph { color: var(--edu-faint); font-size: 0.75rem; }
.edu-source-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--edu-ink);
}
.edu-source-meta {
    color: var(--edu-faint);
    font-size: 0.78rem;
    margin-bottom: 0.5rem;
}
.edu-source-error {
    color: var(--edu-terracotta);
    font-size: 0.82rem;
    background: var(--edu-terracotta-soft);
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    margin-top: 0.3rem;
}

/* -- hand-built progress bar (not st.progress -- see module docstring) -- */
.edu-progress-label {
    font-size: 0.78rem;
    color: var(--edu-muted);
    margin-bottom: 0.3rem;
    display: flex;
    justify-content: space-between;
}
.edu-progress-label .edu-mono { color: var(--edu-accent-dark); font-weight: 600; }
.edu-progress-track {
    height: 6px;
    border-radius: 4px;
    background: var(--edu-slate-soft);
    overflow: hidden;
}
.edu-progress-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--edu-accent), #2E8A73);
    transition: width 0.6s ease;
}

/* -- status / grounding pill badges (icon + label) -- */
.edu-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.32rem;
    font-size: 0.74rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    padding: 0.22rem 0.62rem;
    border-radius: 999px;
    line-height: 1.4;
}
.edu-badge--sm { font-size: 0.68rem; padding: 0.12rem 0.5rem; }
.edu-badge-icn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.05em;
    height: 1.05em;
    border-radius: 50%;
    font-size: 0.72em;
    line-height: 1;
    flex-shrink: 0;
}
.edu-badge--grounded    { background: var(--edu-accent-soft); color: var(--edu-accent-dark); }
.edu-badge--grounded .edu-badge-icn    { background: var(--edu-accent-dark); color: #fff; }
.edu-badge--partial     { background: var(--edu-amber-soft); color: var(--edu-amber); }
.edu-badge--partial .edu-badge-icn     { background: var(--edu-amber); color: #fff; }
.edu-badge--unverified  { background: var(--edu-slate-soft); color: var(--edu-muted); }
.edu-badge--unverified .edu-badge-icn  { background: var(--edu-muted); color: #fff; }
.edu-badge--abstained   { background: var(--edu-slate-soft); color: var(--edu-muted); }
.edu-badge--abstained .edu-badge-icn   { background: var(--edu-muted); color: #fff; }
.edu-badge--failed      { background: var(--edu-terracotta-soft); color: var(--edu-terracotta); }
.edu-badge--failed .edu-badge-icn      { background: var(--edu-terracotta); color: #fff; }

/* -- Q&A workspace -- */
.edu-qa-block {
    padding: 1.3rem 0 1.15rem 0;
    border-bottom: 1px solid var(--edu-line);
}
.edu-qa-block:last-child { border-bottom: none; }
.edu-qa-question {
    font-size: 1.28rem;
    font-style: italic;
    font-weight: 500;
    color: var(--edu-ink);
    margin-bottom: 0.6rem;
    line-height: 1.3;
}
.edu-qa-meta { margin-bottom: 0.6rem; }
.edu-qa-answer {
    font-size: 0.98rem;
    line-height: 1.68;
    color: #2B2B25;
}
.edu-qa-answer p { margin: 0 0 0.7rem 0; }
.edu-qa-answer p:last-child { margin-bottom: 0; }
.edu-citation {
    display: inline-flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    font-weight: 600;
    background: var(--edu-accent-soft);
    color: var(--edu-accent-dark);
    padding: 0.03rem 0.32rem;
    border-radius: 4px;
    margin-left: 2px;
    vertical-align: text-top;
}

/* -- evidence / citation rail -- */
.edu-evidence-item {
    display: flex;
    gap: 0.6rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--edu-line);
}
.edu-evidence-item:last-child { border-bottom: none; }
.edu-rank-chip {
    flex-shrink: 0;
    width: 1.5rem;
    height: 1.5rem;
    line-height: 1.5rem;
    text-align: center;
    border-radius: 50%;
    background: var(--edu-accent-soft);
    color: var(--edu-accent-dark);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.72rem;
}
.edu-rank-chip--sm { width: 1.2rem; height: 1.2rem; line-height: 1.2rem; font-size: 0.64rem; }
.edu-evidence-body { flex: 1; min-width: 0; }
.edu-evidence-head {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
}
.edu-evidence-source {
    font-size: 0.83rem;
    font-weight: 600;
    color: var(--edu-ink);
}
.edu-timecode {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--edu-accent-dark);
    background: var(--edu-accent-soft);
    padding: 0.05rem 0.4rem;
    border-radius: 4px;
}
.edu-evidence-text {
    font-size: 0.89rem;
    color: #45443A;
    line-height: 1.55;
    margin: 0;
    padding-left: 0.65rem;
    border-left: 2px solid var(--edu-line);
}

/* -- retrieval/grounding diagnostics -- */
.edu-detail-latency {
    font-size: 0.82rem;
    color: var(--edu-muted);
    margin-bottom: 0.7rem;
}
.edu-detail-latency .edu-mono { color: var(--edu-accent-dark); font-weight: 600; }
.edu-detail-subhead {
    font-size: 0.85rem;
    font-weight: 600;
    color: #45443A;
    margin: 0.8rem 0 0.4rem 0;
}
.edu-score-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.2rem 0;
    font-size: 0.78rem;
    color: var(--edu-muted);
}
.edu-score-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #45443A;
    background: var(--edu-slate-soft);
    padding: 0.06rem 0.4rem;
    border-radius: 4px;
}
.edu-meter-wrap { display: flex; align-items: center; gap: 0.35rem; }
.edu-meter-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--edu-faint);
    width: 1.3em;
}
.edu-meter-track {
    position: relative;
    width: 5.5rem;
    height: 5px;
    border-radius: 3px;
    background: var(--edu-slate-soft);
    overflow: visible;
}
.edu-meter-fill {
    position: absolute;
    inset: 0 auto 0 0;
    height: 100%;
    border-radius: 3px;
    background: var(--edu-accent);
}
.edu-meter-threshold {
    position: absolute;
    top: -2px;
    bottom: -2px;
    width: 1px;
    background: var(--edu-faint);
    left: 50%;
}
.edu-detail-claim {
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--edu-line);
}
.edu-detail-claim:last-child { border-bottom: none; }
.edu-detail-claim-text {
    font-size: 0.85rem;
    color: #2B2B25;
}
.edu-detail-claim-meta {
    font-size: 0.74rem;
    color: var(--edu-faint);
    margin-top: 0.15rem;
}

/* -- gentle hover-lift on cards (Streamlit's own bordered container) -- */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 16px -8px rgba(27,27,22,0.18);
}
</style>
"""

# Brand mark: a play-triangle inside a ring -- "video content, verified."
# Reused (never redrawn differently) in the header, sidebar title, and
# empty state, so it reads as one consistent identity rather than
# decoration. currentColor means its color is set entirely by the
# surrounding element's CSS `color`, per the .edu-mark rule above.
def _mark_svg(size: int = 30) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 32 32" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" class="edu-mark">'
        f'<circle cx="16" cy="16" r="14" stroke="currentColor" stroke-width="1.6"/>'
        f'<path d="M13 11.2 L22 16 L13 20.8 Z" fill="currentColor"/>'
        f"</svg>"
    )


# One glyph per badge variant -- paired with color, not a replacement for
# it, since color-alone status communication doesn't hold up for anyone
# with color-vision deficiency and also just reads as a generic dashboard.
_BADGE_ICON = {
    "grounded": "✓",     # check
    "partial": "◐",      # half-filled circle
    "unverified": "?",
    "abstained": "–",    # en dash
    "failed": "✕",       # multiplication x
}

# Status -> accent color for the sidebar source card's top strip.
_STATUS_ACCENT_COLOR = {
    "grounded": "var(--edu-accent)",
    "partial": "var(--edu-amber)",
    "unverified": "var(--edu-faint)",
    "failed": "var(--edu-terracotta)",
}


def _inject_custom_css() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _badge_html(label: str, variant: str, *, small: bool = False) -> str:
    size_class = " edu-badge--sm" if small else ""
    icon = _BADGE_ICON.get(variant, "")
    icon_html = f'<span class="edu-badge-icn">{icon}</span>' if icon else ""
    return (
        f'<span class="edu-badge edu-badge--{variant}{size_class}">'
        f"{icon_html}{html.escape(label)}</span>"
    )


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
    # `is None` (not `if not seconds`) -- a genuinely meaningful 0 (the
    # very start of a source, a completely normal start_time for a
    # citation's first chunk once VAD trims leading silence to exactly
    # 0.0) must not be treated the same as "unknown".
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
    st.markdown(
        f'<div class="edu-hero">{_mark_svg(30)}'
        f'<div><div class="edu-app-title">EduRAG</div>'
        f'<div class="edu-app-subtitle">Learn from your material. Ask anything. Verify the answer.</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="edu-hero-tags">TRANSCRIPT-GROUNDED'
        '<span class="sep">&middot;</span>TIMESTAMPED CITATIONS'
        '<span class="sep">&middot;</span>CLAIM-LEVEL VERIFIED</div>',
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    """Replaces a bare st.info() line. The first thing a visitor with no
    sources yet sees was, until now, the least distinctive part of the
    whole app -- this makes the actual pipeline (add -> verify -> ask)
    visible up front instead of a generic placeholder message."""
    st.markdown(
        f'<div class="edu-empty">{_mark_svg(42)}'
        '<div class="edu-empty-title">Add a lecture to begin</div>'
        '<div class="edu-empty-sub">Drop in a YouTube lecture or a local video from the sidebar. '
        "EduRAG transcribes it, checks every answer against what's actually said, "
        "and cites the exact moment it came from.</div>"
        '<div class="edu-steps">'
        '<div class="edu-step"><div class="edu-step-num">1</div>'
        '<div class="edu-step-label">Add a YouTube link or upload a video</div></div>'
        '<div class="edu-step"><div class="edu-step-num">2</div>'
        '<div class="edu-step-label">We transcribe, chunk & index it</div></div>'
        '<div class="edu-step"><div class="edu-step-num">3</div>'
        '<div class="edu-step-label">Ask — get cited, verified answers</div></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_sidebar(session_id: str) -> None:
    """Source management lives here so the main pane can be a pure Q&A
    workspace. Logic is unchanged from earlier sprints — only how each
    source's status/metadata looks changed."""
    with st.sidebar:
        st.markdown(
            f'<div class="edu-sidebar-title">{_mark_svg(18)}Your sources</div>',
            unsafe_allow_html=True,
        )
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
                status_label = STATUS_LABELS.get(source.status, source.status)
                variant = _STATUS_BADGE_VARIANT.get(
                    source.status, "partial" if source.status in _ACTIVE_STATUSES else "unverified"
                )
                accent_color = _STATUS_ACCENT_COLOR.get(variant, "var(--edu-faint)")
                st.markdown(
                    f'<div class="edu-card-accent" style="background:{accent_color}"></div>',
                    unsafe_allow_html=True,
                )

                title = source.title or source.original_name or source.source_url or "Untitled source"
                meta_bits = [source.source_type.replace("_", " ").title()]
                duration = _format_duration(source.duration_seconds)
                if duration:
                    meta_bits.append(duration)
                if source.language:
                    meta_bits.append(f"Language: {source.language.upper()}")
                st.markdown(
                    f'<div class="edu-source-row"><span class="edu-source-glyph">&#9656;</span>'
                    f'<span class="edu-source-title">{html.escape(title)}</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="edu-source-meta edu-mono">{html.escape(" &middot; ".join(meta_bits))}</div>',
                    unsafe_allow_html=True,
                )

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
                    # progress (already tracked in ProcessingJob). Hand-
                    # built bar (not st.progress()) so its exact look is
                    # ours, not Streamlit's internal widget markup — see
                    # module docstring.
                    with SessionLocal() as db:
                        job = processing_job_repository.latest_job_for_source(db, source.id)
                    progress = min(max(job.progress if job else 0.0, 0.0), 1.0)
                    pct = progress * 100
                    st.markdown(
                        f'<div class="edu-progress-label"><span>{html.escape(status_label)}</span>'
                        f'<span class="edu-mono">{pct:.0f}%</span></div>'
                        f'<div class="edu-progress-track">'
                        f'<div class="edu-progress-fill" style="width:{pct:.1f}%"></div></div>',
                        unsafe_allow_html=True,
                    )
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
    """Footnote-style citation rail: a numbered rank chip, source title +
    a monospace timecode chip, the quoted evidence text in a real
    blockquote treatment underneath."""
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
            # `is not None` (not `if evidence.start_time`) -- a real 0.0
            # start (the very first chunk of a source) must still show.
            start = _format_duration(int(evidence.start_time)) if evidence.start_time is not None else None
            timecode_html = f'<span class="edu-timecode">{html.escape(start)}</span>' if start else ""
            text = chunk.text if chunk is not None else ""
            st.markdown(
                f'<div class="edu-evidence-item">'
                f'<div class="edu-rank-chip">{evidence.rank}</div>'
                f'<div class="edu-evidence-body">'
                f'<div class="edu-evidence-head">'
                f'<span class="edu-evidence-source">{html.escape(title)}</span>{timecode_html}'
                f"</div>"
                f'<blockquote class="edu-evidence-text">{html.escape(text)}</blockquote>'
                f"</div></div>",
                unsafe_allow_html=True,
            )


def _meter_html(label: str, value_0_to_1: float, *, threshold: float | None = None) -> str:
    """Only used for scores that are genuinely 0-1-bounded (HHEM's NLI
    score) -- see module docstring for why retrieval/rerank scores do NOT
    get this treatment. threshold, when given, draws a small tick mark at
    that fraction of the track (the real 0.5 supported/not-supported
    decision boundary grounding.py uses), so the meter shows the actual
    number the app checked against, not just a bare fill."""
    pct = max(0.0, min(1.0, value_0_to_1)) * 100
    threshold_html = ""
    if threshold is not None:
        t_pct = max(0.0, min(1.0, threshold)) * 100
        threshold_html = f'<span class="edu-meter-threshold" style="left:{t_pct:.0f}%"></span>'
    return (
        f'<span class="edu-meter-wrap"><span class="edu-meter-label">{html.escape(label)}</span>'
        f'<span class="edu-meter-track"><span class="edu-meter-fill" style="width:{pct:.0f}%"></span>'
        f"{threshold_html}</span></span>"
    )


def _render_retrieval_details(db, message) -> None:
    """The actual retrieval/rerank/grounding work (contextual retrieval,
    hybrid search + RRF fusion + cross-encoder rerank, per-claim NLI
    verification) has existed since Sprint 5-6 but was never visible
    anywhere except the DB. Surfaced per answer, collapsed by default —
    diagnostic, not required reading for a learner just trying to study,
    but real proof of the pipeline working rather than a claimed
    capability."""
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
            st.markdown(
                f'<div class="edu-detail-latency">Answered in '
                f'<span class="edu-mono">{html.escape(latency_label)}</span></div>',
                unsafe_allow_html=True,
            )

        if evidence_rows:
            st.markdown('<div class="edu-detail-subhead">Retrieved passages</div>', unsafe_allow_html=True)
            for evidence in evidence_rows:
                bits = [f'<div class="edu-rank-chip edu-rank-chip--sm">{evidence.rank}</div>']
                if evidence.retrieval_score is not None:
                    bits.append(f'<span class="edu-score-chip">retrieval {evidence.retrieval_score:.3f}</span>')
                if evidence.reranker_score is not None:
                    bits.append(f'<span class="edu-score-chip">rerank {evidence.reranker_score:.3f}</span>')
                if evidence.nli_score is not None:
                    bits.append(_meter_html("NLI", evidence.nli_score, threshold=0.5))
                st.markdown(
                    f'<div class="edu-score-row">{"".join(bits)}</div>', unsafe_allow_html=True,
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
                    f'<div class="edu-detail-claim-meta edu-mono">{html.escape(" &middot; ".join(meta_bits))}</div>'
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

    Each USER/ASSISTANT message pair renders as one editorial block via
    _render_turn. The retrieval/generation/verification logic this reads
    from is completely unchanged from Sprint 6-10."""
    with SessionLocal() as db:
        sources_for_session = source_repository.list_sources_for_session(db, session_id)
    has_ready_source = any(s.status == "READY" for s in sources_for_session)

    if not has_ready_source:
        if sources_for_session:
            st.info("Your source is still processing — check the sidebar for progress.")
        else:
            _render_empty_state()
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
