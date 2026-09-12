"""EduRAG — Streamlit entry point.

Sprint 0 scope only: boots the app, initializes the DB schema, and renders
the empty-workspace state (UI/UX spec Doc 3 sec 32). Source input, the
question/answer workspace, and evidence rendering land in Sprints 1-8 per
the project's own task list — this file grows into `AppShell` +
`SourceInput`/`ConversationView`/etc. (Doc 3 sec 40) incrementally rather
than being scaffolded as dead UI now.
"""

import streamlit as st

from app.db.base import Base
from app.db.session import engine
from app.db import models  # noqa: F401  (registers tables before create_all)

st.set_page_config(page_title="EduRAG", page_icon=None, layout="centered")


def _ensure_schema() -> None:
    # Sprint 0 convenience only — real migrations run through Alembic
    # (TRD Doc 2 sec 39); this just means a fresh clone works immediately.
    Base.metadata.create_all(bind=engine)


def render_empty_state() -> None:
    st.markdown("## EduRAG")
    st.markdown("Learn from your material. Ask anything. Verify the answer.")
    st.divider()
    st.markdown("### Start with your learning material")
    st.markdown("Add a YouTube lecture, video, or PDF to begin asking questions.")
    st.button("Add source", disabled=True, help="Source ingestion lands in Sprint 1")


def main() -> None:
    _ensure_schema()
    render_empty_state()


if __name__ == "__main__":
    main()
