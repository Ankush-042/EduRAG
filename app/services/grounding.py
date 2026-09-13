"""Sprint 6 (part 2) — claim-level grounding verification via a local NLI
(natural language inference) model (TRD Doc 2 sec 27-30). Deliberately not
a second LLM call asking "is this grounded?" — a small, local, purpose-
built entailment classifier is cheap, deterministic, and can't be talked
out of a verdict the way a generative model can be.

Label-order correctness note (read before touching _LABELS): this was
researched, not guessed. cross-encoder/nli-deberta-v3-base's own official
model card (huggingface.co/cross-encoder/nli-deberta-v3-base) documents
its output order as label_mapping = ['contradiction', 'entailment',
'neutral'], selected via argmax() over that fixed order. Getting this
backwards would silently invert every grounding verdict this app ever
produces — the single most dangerous possible bug in an "accuracy-first"
app, because a flipped verdict looks exactly as confident as a correct
one. If nli_model in config.py is ever changed to a different checkpoint,
_LABELS must be re-verified against *that* model's own card before
shipping — never assumed to match this one.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.interfaces import EntailmentResult, EntailmentVerifier

settings = get_settings()

# Same load-once-reuse-per-process pattern as embedding.py / reranking.py.
_MODEL_CACHE: dict[str, object] = {}

# Verified against cross-encoder/nli-deberta-v3-base's official HuggingFace
# card — see module docstring. Index into this list is the model's raw
# output index (the order CrossEncoder.predict(..., apply_softmax=True)
# returns probabilities in), NOT an arbitrary display order.
_LABELS = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]


def _get_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from sentence_transformers import CrossEncoder

    # device explicitly pinned — same reasoning as embedding.py's
    # _get_model: avoids PyTorch opening a second CUDA context alongside
    # ctranslate2's in the same process (config.py's torch_model_device).
    model = CrossEncoder(model_name, device=settings.torch_model_device)
    _MODEL_CACHE[model_name] = model
    return model


class NLIVerifier(EntailmentVerifier):
    """Concrete EntailmentVerifier (app/core/interfaces.py).

    verify(claim, evidence) asks: does `evidence` (the premise — the
    actual source text) entail `claim` (the hypothesis — one sentence from
    the generated answer)? That premise/hypothesis order is intentional
    and matters: reversing it asks a different, wrong question ("does the
    generated claim imply the source text"), which is meaningless for
    grounding verification — a short, specific claim essentially never
    entails a longer, more general source passage even when the claim is
    perfectly grounded in it.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name

    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        model = _get_model(self._model_name)
        scores = model.predict([(evidence, claim)], apply_softmax=True)
        probabilities = scores[0]
        # Avoid relying on a numpy-only method like .argmax() here: this
        # can never be exercised in the sandbox this was written in (no
        # PyPI access to sentence-transformers there), so the safer,
        # library-agnostic form is used deliberately rather than assumed
        # to work against whatever predict() actually returns.
        best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        return EntailmentResult(verdict=_LABELS[best_index], score=float(probabilities[best_index]))


def get_verifier() -> NLIVerifier:
    return NLIVerifier(settings.nli_model)
