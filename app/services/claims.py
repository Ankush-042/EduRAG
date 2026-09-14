"""Sprint 6 (part 3) — splits a generated answer into individual claims for
per-claim NLI verification (TRD Doc 2 sec 27; Data/Schema spec Doc 4 sec
18's Claim table, GENERATED_CLAIM). Deliberately not an LLM call: a second
generative step to decide "what are the claims here" would just add
another place for the pipeline to hallucinate, stacked on top of the one
generation.py already guards carefully. A conservative, rule-based
sentence splitter is cruder but fully deterministic, free, and can't
itself introduce ungrounded content.
"""

from __future__ import annotations

import re

# A short, curated list of abbreviations that must not be treated as
# sentence-final periods — otherwise "e.g. Python" or "Dr. Smith" would get
# split mid-thought. Not exhaustive (a real sentence tokenizer, e.g. nltk's
# punkt or spaCy, would do better) but covers the common cases likely to
# appear in educational/technical answers; a known, documented limitation
# rather than a silent one.
_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "ms.", "fig.", "eq.", "approx.",
)

# Split on a sentence-ending punctuation mark followed by whitespace and
# then a capital letter, digit, quote, or bracket — the shape a new
# sentence (or a "[1] ..." citation-led fragment) actually starts with.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])")

# Sprint 9 finding: a real generation run produced a malformed citation
# ("[x]" instead of a numbered "[1]"), which this regex -- matching only
# digit lists -- left untouched, so it rode straight through
# strip_citation_markers() and into the NLI verifier as literal noise on
# the end of that claim's text (a real contributing factor to that
# claim's grounding false negative, on top of the multi-passage-synthesis
# issue answering.py now handles). Widened to also catch a short
# letter-led bracketed token in citation position -- "[x]", "[Source 2]",
# "[note]" -- since the generation system prompt (generation.py) commits
# the model to ONLY ever using brackets for numbered citations, so any
# other bracketed content appearing is by definition a malformed citation
# attempt, not meaningful prose to preserve.
_CITATION_RE = re.compile(r"\[(?:\d+(?:,\s*\d+)*|[A-Za-z][\w\s]{0,15})\]")

# Below this length a "sentence" is almost certainly a stray fragment
# (leftover punctuation, a lone citation marker) rather than a real,
# independently-verifiable claim.
_MIN_CLAIM_CHARS = 3

# Placeholder token used to protect abbreviation periods from the
# sentence-boundary regex during splitting; an ordinary printable string
# rather than a control character, since a raw NUL/control byte embedded
# in a .py source file is itself a syntax error in CPython. Not something
# that could plausibly occur in real generated text, so no collision risk.
_PERIOD_PLACEHOLDER = "@@PERIOD@@"


_ABBREVIATION_RE = re.compile(
    "|".join(re.escape(abbr) for abbr in _ABBREVIATIONS), flags=re.IGNORECASE
)


def _protect_abbreviation_periods(match: "re.Match[str]") -> str:
    # Swap only the abbreviation's own trailing period for the placeholder,
    # keeping the rest of the matched text byte-for-byte as written (case
    # included) -- "Dr." must come back out as "Dr.", not "dr.".
    matched = match.group(0)
    return matched[:-1] + _PERIOD_PLACEHOLDER


def split_into_claims(answer: str) -> list[str]:
    """Splits an answer into candidate claim sentences, each trimmed and
    non-empty. Extremely short fragments (stray punctuation, bare list
    markers) are dropped since they can't meaningfully be checked against
    evidence on their own."""
    answer = answer.strip()
    if not answer:
        return []

    protected = _ABBREVIATION_RE.sub(_protect_abbreviation_periods, answer)
    raw_sentences = _SENTENCE_BOUNDARY_RE.split(protected)

    claims = []
    for raw in raw_sentences:
        sentence = raw.replace(_PERIOD_PLACEHOLDER, ".").strip()
        if len(sentence) >= _MIN_CLAIM_CHARS:
            claims.append(sentence)
    return claims


def strip_citation_markers(text: str) -> str:
    """Removes inline/trailing [n] or [n, m] citation markers. Used before
    feeding claim text to the NLI verifier (grounding.py), which was never
    trained on bracket-citation syntax — leaving markers in would add
    scoring noise the model doesn't understand rather than signal about
    the claim's actual content."""
    return _CITATION_RE.sub("", text).strip()
