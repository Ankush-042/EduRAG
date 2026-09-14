"""One-off diagnostic -- NOT part of the app, not imported anywhere.

Reads data/edurag.db directly (no need to click through the UI, and
works even after you've closed the Streamlit tab) and prints, for the
most recent conversation's assistant message(s):
  - the generated answer text and its grounding_status
  - every retrieved evidence chunk (in rank order) -- this is exactly
    what the "Evidence (N)" expander in the UI shows
  - every extracted claim, and what it was checked against: which
    evidence chunk scored best, what verdict (ENTAILMENT / NEUTRAL /
    CONTRADICTION) and score it got

This exists to see the actual retrieved text and verdicts directly,
rather than guessing why grounding_status came back UNVERIFIED.

Run it with the project venv active, from the project root:
    python scripts\\dump_last_answer.py
"""

import sqlite3
import textwrap
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "edurag.db"


def wrap(text: str, indent: str = "      ") -> str:
    return textwrap.fill(
        text, width=100, initial_indent=indent, subsequent_indent=indent,
    )


def main() -> None:
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH} -- run this from the project root (D:\\Projects\\EduRAG).")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    conversation = conn.execute(
        "SELECT id FROM conversations ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    if conversation is None:
        print("No conversations found in the database yet.")
        return

    messages = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? AND role = 'ASSISTANT' ORDER BY message_order",
        (conversation["id"],),
    ).fetchall()
    if not messages:
        print("No assistant messages found in the most recent conversation.")
        return

    for message in messages:
        print("=" * 100)
        print(f"ANSWER (grounding_status = {message['grounding_status']}):")
        print(wrap(message["content"]))
        print()

        evidence_rows = conn.execute(
            """
            SELECT evidence.rank, evidence.id AS evidence_id, chunks.text AS chunk_text,
                   evidence.start_time, evidence.reranker_score
            FROM evidence
            JOIN chunks ON chunks.id = evidence.chunk_id
            WHERE evidence.message_id = ?
            ORDER BY evidence.rank
            """,
            (message["id"],),
        ).fetchall()

        print(f"RETRIEVED EVIDENCE ({len(evidence_rows)} chunks):")
        evidence_by_id = {}
        for row in evidence_rows:
            evidence_by_id[row["evidence_id"]] = row
            print(f"  [{row['rank']}] (reranker_score={row['reranker_score']:.4f}, start_time={row['start_time']})")
            print(wrap(row["chunk_text"]))
            print()

        claims = conn.execute(
            "SELECT * FROM claims WHERE message_id = ?", (message["id"],),
        ).fetchall()

        print(f"CLAIMS EXTRACTED FROM THE ANSWER ({len(claims)}):")
        for claim in claims:
            print(f"  - {claim['claim_text']}")
            verifications = conn.execute(
                "SELECT * FROM verification_results WHERE claim_id = ? AND message_id = ?",
                (claim["id"], message["id"]),
            ).fetchall()
            if not verifications:
                print("      (no verification result recorded -- empty claim after citation-stripping)")
                continue
            for verification in verifications:
                evidence = evidence_by_id.get(verification["evidence_id"])
                rank = evidence["rank"] if evidence else "?"
                print(f"      -> verdict={verification['verdict']}  score={verification['score']:.4f}  (best match: evidence [{rank}])")
        print()

    conn.close()


if __name__ == "__main__":
    main()
