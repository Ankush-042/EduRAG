"""Sprint 8 — the eval harness (previously "Sprint 11" in the original
code comments; pulled forward here because every accuracy knob after
Sprint 7 -- RRF k, top_k_evidence, real LLM contextual retrieval, BM25
stemming -- needs something to be measured against, or tuning them is
just guessing with extra steps).

Unlike the other scripts/ diagnostics (dump_last_answer.py,
diagnose_hhem_load.py), this one runs the REAL pipeline
(app.services.answering.answer -> retrieval -> generation -> claim
verification) against a fixed set of questions and checks the result
against expectations recorded in a JSON file -- so a change to any
retrieval/generation parameter can be checked against a known-good
baseline instead of "ask a couple of questions in the UI and eyeball it"
(which is how the grounding.py model swap bugs were originally found).

Test cases live in scripts/eval_cases.json (gitignored is NOT set for
this one -- unlike .env, these are meant to be committed and grown over
time as you add more real sources). Each case:
  {
    "id":            short name, just for the printed report
    "question":       the question to ask
    "expect_status":  list of acceptable grounding_status values
                       (default: ["GROUNDED", "PARTIALLY_GROUNDED"])
    "expect_any":     optional list of substrings -- the answer must
                       contain at least one (case-insensitive)
    "expect_none":    optional list of substrings that must NOT appear
                       (catches a specific wrong-answer/hallucination
                       pattern you've seen before)
    "known_issue":    optional free-text note -- if set, a failing case
                       is reported but doesn't affect the overall
                       pass/fail exit code (for a documented, not-yet-
                       fixed gap rather than a silent skip)
    "expect_source_title": optional substring -- the resolved session
                       must have at least one READY source whose title
                       contains this (case-insensitive), or the case
                       fails immediately with a clear "source not in
                       this session" error instead of running the real
                       question against whatever happens to be indexed.

Self-audit finding: without expect_source_title, resolving "whichever
session was most recently active" (see _latest_session_id below) with no
check that the CONTENT this case actually assumes is present means a case
authored against one lecture can silently run against a completely
different (or empty) session -- every content-specific expect_any/
expect_none assertion then fails for the wrong reason (mismatched
content, not a real regression), or worse, happens to pass by accident.
expect_source_title makes that assumption explicit and checked up front.

Run it with the project venv active, from the project root:
    python scripts\\eval_answers.py
    python scripts\\eval_answers.py --session <session_id>   # pin a session
    python scripts\\eval_answers.py --cases scripts\\my_cases.json

Without --session, it evaluates against whichever session was most
recently active (session_repository has no "list all" today -- this
queries the sessions table directly for that, same as the other
scripts/ diagnostics reading the DB directly rather than growing the
app's own repositories just for one-off tooling).

Every run is also saved under scripts/eval_results/<timestamp>.json so
two runs (before/after a tuning change) can be diffed later.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"
_RESULTS_DIR = Path(__file__).resolve().parent / "eval_results"
_DEFAULT_EXPECT_STATUS = ["GROUNDED", "PARTIALLY_GROUNDED"]


def _latest_session_id(db) -> str | None:
    from app.db.models.session import SessionModel

    session = (
        db.query(SessionModel)
        .filter(SessionModel.status == "ACTIVE")
        .order_by(SessionModel.last_active_at.desc())
        .first()
    )
    return session.id if session else None


def _ready_source_titles(db, session_id: str) -> list[str]:
    from app.db.repositories import source_repository

    sources = source_repository.list_sources_for_session(db, session_id)
    return [s.title for s in sources if s.status == "READY" and s.title]


def _run_case(db, session_id: str, case: dict, ready_titles: list[str]) -> dict:
    from app.services.answering import AnsweringError, answer

    expected_title = case.get("expect_source_title")
    if expected_title and not any(expected_title.lower() in t.lower() for t in ready_titles):
        return {
            **case,
            "passed": False,
            "error": (
                f"expected a READY source with title containing "
                f"{expected_title!r}, but session {session_id} only has: "
                f"{ready_titles or '(no READY sources)'} -- re-ingest the "
                f"expected source, or point --session at the right one."
            ),
            "grounding_status": None,
            "answer": None,
            "latency_ms": 0,
        }

    t0 = time.monotonic()
    try:
        result = answer(db, session_id, case["question"])
        db.commit()
    except AnsweringError as exc:
        db.rollback()
        return {
            **case,
            "passed": False,
            "error": str(exc),
            "grounding_status": None,
            "answer": None,
            "latency_ms": int((time.monotonic() - t0) * 1000),
        }

    expect_status = case.get("expect_status", _DEFAULT_EXPECT_STATUS)
    status_ok = result.grounding_status in expect_status

    answer_lower = result.content.lower()
    expect_any = case.get("expect_any")
    any_ok = expect_any is None or any(s.lower() in answer_lower for s in expect_any)

    expect_none = case.get("expect_none", [])
    none_ok = all(s.lower() not in answer_lower for s in expect_none)

    return {
        **case,
        "passed": status_ok and any_ok and none_ok,
        "status_ok": status_ok,
        "any_ok": any_ok,
        "none_ok": none_ok,
        "grounding_status": result.grounding_status,
        "answer": result.content,
        "evidence_count": len(result.evidence),
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=None, help="Session id to evaluate against (default: most recently active)")
    parser.add_argument("--cases", default=str(_DEFAULT_CASES_PATH), help="Path to a JSON test-case file")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"No test-case file at {cases_path}. See scripts/eval_cases.json for the expected shape.")
        return 2
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not cases:
        print(f"{cases_path} has no test cases in it yet.")
        return 2

    from app.db.session import SessionLocal
    from app.db import models  # noqa: F401 — registers tables

    with SessionLocal() as db:
        session_id = args.session or _latest_session_id(db)
        if session_id is None:
            print("No active session found — open the app and add at least one source first.")
            return 2

        ready_titles = _ready_source_titles(db, session_id)
        print(f"Evaluating {len(cases)} case(s) against session {session_id}")
        print(f"READY sources in this session: {ready_titles or '(none)'}\n")
        results = [_run_case(db, session_id, case, ready_titles) for case in cases]

    hard_failures = 0
    for r in results:
        marker = "PASS" if r["passed"] else ("KNOWN ISSUE" if r.get("known_issue") else "FAIL")
        print(f"[{marker}] {r.get('id', r['question'])}  ({r['latency_ms']}ms, status={r['grounding_status']})")
        if not r["passed"]:
            if r.get("error"):
                print(f"         error: {r['error']}")
            else:
                print(f"         answer: {(r['answer'] or '')[:200]}")
            if r.get("known_issue"):
                print(f"         known issue: {r['known_issue']}")
            else:
                hard_failures += 1
    print()

    passed = sum(1 for r in results if r["passed"])
    print(f"{passed}/{len(results)} passed ({hard_failures} unexplained failure(s), "
          f"{sum(1 for r in results if not r['passed'] and r.get('known_issue'))} documented known issue(s))")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = _RESULTS_DIR / f"{stamp}.json"
    out_path.write_text(json.dumps({"session_id": session_id, "results": results}, indent=2), encoding="utf-8")
    print(f"Full report: {out_path}")

    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
