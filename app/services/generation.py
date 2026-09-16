"""Sprint 6 (part 1) — grounded answer generation (TRD Doc 2 sec 21-26).

The prompt is the single most safety-critical piece of this whole sprint:
an "accuracy-first" assistant that quietly answers from the model's own
training knowledge instead of the retrieved evidence is worse than no
assistant at all, since a wrong-but-fluent answer is indistinguishable
from a grounded one until someone checks the citations. The system prompt
below is deliberately blunt about "context-only, cite-or-refuse" rather
than a soft, easily-ignored suggestion — and generation.py never decides
on its own to skip calling the model when there's no evidence; that
decision belongs to answering.py, which abstains before ever reaching
here (see its _ABSTAIN_MESSAGE).

Doc 6 authority / scope decision: a real local-inference fallback (e.g.
llama.cpp, an ONNX runtime) needs a model file on disk and an inference
library that isn't in requirements.txt (TRD Doc 2 sec 49 lists this as an
explicit fallback, not the MVP default — Groq is sec 48's primary path).
Rather than silently faking a "local" path that's actually just Groq
again, LocalGenerator is a clearly-labelled stub that raises
GenerationError telling the caller exactly why, instead of ever answering
un-grounded or pretending a capability exists that doesn't.

Sprint 10 revisit of that decision: re-checked, nothing has changed --
still no model file on disk, still no llama.cpp/ONNX-type runtime in
requirements.txt, and adding one now (most such packages need a native
build step) would be a real, untested new failure mode introduced for a
capability nobody has asked to actually use offline. The stub stays a
stub; what Sprint 10 actually adds is retry/timeout hardening around the
GROQ path itself (see GroqGenerator below), since that's the primary path
every real turn goes through, not the fallback nobody's exercising.
"""

from __future__ import annotations

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.interfaces import Generator

settings = get_settings()

_SYSTEM_PROMPT = (
    "You are EduRAG, a study assistant that answers ONLY from the numbered "
    "context passages given to you below. Rules, no exceptions:\n"
    "1. Use only facts stated in the context. Never use outside knowledge, "
    "even if you are confident it is correct.\n"
    "2. Every factual sentence you write must end with the number(s) of the "
    "context passage(s) it came from, in square brackets — e.g. \"Python "
    "uses # for comments [2].\" or \"...and it supports async [1, 3].\"\n"
    "3. If the context does not contain enough information to answer the "
    "question, say so plainly (for example: \"The material provided doesn't "
    "cover this.\") instead of guessing or filling gaps from general "
    "knowledge.\n"
    "4. Be concise and direct — answer the question, don't pad, restate the "
    "question, or add unrelated commentary.\n"
    "5. Only ever use square brackets for numbered citations like [2] or "
    "[1, 3] — never for anything else (a variable name, a note, an aside). "
    "If you need to refer to a variable named x from the material, write "
    "it as x, not [x].\n"
    "6. Write short, single-idea sentences — one fact or one step per "
    "sentence. If something has multiple steps (like a procedure), give "
    "each step its own sentence instead of joining them with 'and', a "
    "dash, or a semicolon. This is so each sentence can be checked "
    "against the evidence on its own; a sentence that bundles several "
    "facts together is harder to verify than several short ones.\n"
)


class GenerationError(Exception):
    """Raised for any generation failure; the message is what the caller
    (answering.py, and ultimately the UI) surfaces — never a partial,
    unlabelled, or silently-degraded answer."""


def _build_user_prompt(question: str, context: str, language: str) -> str:
    language_line = "" if language == "en" else f"\nAnswer in this language: {language}.\n"
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n"
        f"{language_line}"
        "Answer, following the rules above exactly."
    )


class GroqGenerator(Generator):
    """Concrete Generator (app/core/interfaces.py) backed by Groq's hosted
    inference (TRD Doc 2 sec 48 — the primary generation path, chosen for
    speed on consumer hardware over running a large local model)."""

    def __init__(self, model_name: str, api_key: str):
        self._model_name = model_name
        self._api_key = api_key

    def generate(self, question: str, context: str, language: str = "en") -> str:
        if not self._api_key:
            raise GenerationError(
                "GROQ_API_KEY is not set — add it to your .env file to enable "
                "answer generation (see .env.example)."
            )
        try:
            from groq import (
                APIConnectionError,
                APITimeoutError,
                Groq,
                InternalServerError,
                RateLimitError,
            )
        except ImportError as exc:
            raise GenerationError(
                "The 'groq' package isn't installed — run `pip install -r "
                "requirements.txt` (groq is already listed there)."
            ) from exc

        # max_retries=0: this generator owns retry policy explicitly via
        # the tenacity Retrying below (see module docstring) rather than
        # also leaving the SDK's own implicit retry-on-transient-error
        # behavior active underneath it — one bounded, inspectable policy
        # instead of two stacked ones whose combined worst-case latency
        # isn't obvious from reading either one alone.
        client = Groq(api_key=self._api_key, max_retries=0)

        # Only retry error types that are genuinely transient (a dropped
        # connection, a timeout, a 429, a 5xx) -- retrying a 400/401/403/404
        # (bad request, bad key, no permission, unknown model) would just
        # burn the same bounded time budget on an error that will never
        # succeed no matter how many times it's repeated.
        retryer = Retrying(
            retry=retry_if_exception_type(
                (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
            ),
            stop=stop_after_attempt(settings.generation_max_attempts),
            wait=wait_exponential(multiplier=0.5, max=settings.generation_retry_max_wait_s),
            reraise=True,  # without this, tenacity raises its own RetryError
            # instead of the real underlying exception once attempts are
            # exhausted -- the except clause below needs the original
            # exception (and its message) to build a useful GenerationError.
        )

        try:
            response = retryer(
                client.chat.completions.create,
                model=self._model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(question, context, language)},
                ],
                # Low but nonzero: near-deterministic without the repetition
                # / degenerate-output risk of temperature=0 on some models.
                temperature=0.1,
                max_tokens=1024,
                # Explicit per-request timeout, well under the SDK's own
                # 60s default -- this call sits in the synchronous query
                # path (TRD's "low latency" requirement), so a single hung
                # request should fail fast enough to retry or abstain
                # within a bounded total time, not silently eat a minute.
                timeout=settings.generation_timeout_s,
            )
        except Exception as exc:  # Groq's SDK raises several distinct error
            # types (auth, rate limit, connection, bad request) — all of
            # them mean "no answer was produced," which is the only thing
            # answering.py needs to know to abstain safely rather than
            # crash the UI.
            raise GenerationError(f"Generation request failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else None
        if not content or not content.strip():
            raise GenerationError("The generation provider returned an empty answer.")
        return content.strip()


class LocalGenerator(Generator):
    """Documented stub for the local-fallback generation path (TRD Doc 2
    sec 49). Not implemented in this MVP — see module docstring for why.
    Raises rather than returning a fabricated or silently-degraded answer,
    so a misconfiguration is loud instead of a quiet quality regression."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    def generate(self, question: str, context: str, language: str = "en") -> str:
        raise GenerationError(
            "Local generation isn't implemented yet in this build — set "
            "GENERATION_PROVIDER=groq and GROQ_API_KEY in .env to use the "
            "primary (hosted) generation path."
        )


def get_generator() -> Generator:
    if settings.generation_provider == "groq":
        return GroqGenerator(settings.generation_model, settings.groq_api_key)
    if settings.generation_provider == "local":
        return LocalGenerator(settings.local_generation_model)
    raise GenerationError(f"Unknown GENERATION_PROVIDER '{settings.generation_provider}'.")
