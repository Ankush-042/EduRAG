"""Sprint 6 (part 2) — claim-level grounding verification via a local
hallucination-detection model (TRD Doc 2 sec 27-30). Deliberately not a
second LLM call asking "is this grounded?" — a small, local, purpose-built
verifier is cheap, deterministic, and can't be talked out of a verdict the
way a generative model can be.

Model history: this originally used cross-encoder/nli-deberta-v3-base (a
generic SNLI/MultiNLI sentence-pair classifier). Real dumps of production
grounding runs (scripts/dump_last_answer.py) showed it giving confidently
wrong verdicts (NEUTRAL/CONTRADICTION at ~0.97-0.995 "confidence") on
claims that were genuinely well-supported by the retrieved evidence, once
the label-order hypothesis was empirically ruled out
(scripts/verify_nli_labels.py confirmed the label mapping was correct).
Root cause: that model is trained on short, clean sentence pairs, and is a
poor fit for this app's actual inputs — long, disfluent, raw video-
transcript premises paired with paraphrased, multi-clause LLM-generated
claims.

Replacement: vectara/hallucination_evaluation_model (HHEM-2.1-Open),
purpose-built for exactly this task (premise = source document, hypothesis
= generated claim, unlimited context length vs. the old model's 512-token
cap). It does NOT do 3-way NLI (entailment/neutral/contradiction) — it
outputs a single continuous "supported by the premise" score in [0, 1].
0.5 is Vectara's own documented starting threshold
(docs.vectara.com/docs/hallucination-and-evaluation/hallucination-
evaluation) for supported-vs-not. Below that we call it NOT_SUPPORTED
rather than reusing "NEUTRAL"/"CONTRADICTION" — this model can't tell
those apart, and a fabricated three-way distinction would be worse than
naming what's actually known. Nothing downstream keys off the specific
non-ENTAILMENT label (app/services/answering.py's grounding logic only
ever checks `verdict == "ENTAILMENT"`), so this is a safe rename.

trust_remote_code=True is required to load this model — it runs code
shipped in the model repo, not just weights. This is a widely-used,
Vectara-published model, but it's still worth knowing this is happening.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.interfaces import EntailmentResult, EntailmentVerifier

settings = get_settings()

# Same load-once-reuse-per-process pattern as embedding.py / reranking.py.
_MODEL_CACHE: dict[str, object] = {}

# Vectara's own documented guideline (see module docstring) for the
# supported-vs-hallucinated cutoff on HHEM's continuous [0, 1] score.
_SUPPORTED_THRESHOLD = 0.5


def _get_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from transformers import AutoModelForSequenceClassification

    # trust_remote_code=True: HHEM ships its own predict() implementation
    # in the model repo (it isn't a plain classification head), so this
    # has to be enabled to load it at all.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True
    )
    # Same device-pinning reasoning as embedding.py/config.py's
    # torch_model_device: keep PyTorch models off the GPU ctranslate2 is
    # already using in this process, on Windows, to avoid a silent crash.
    # HHEM is a standard HF PreTrainedModel under trust_remote_code, so
    # .to(device) is expected to work the same as any other transformers
    # model -- but this specific model/version combination hasn't been
    # run yet outside this sandbox (no PyPI access here), so treat the
    # first real run on your machine as the actual verification of this.
    model = model.to(settings.torch_model_device)
    _MODEL_CACHE[model_name] = model
    return model


class NLIVerifier(EntailmentVerifier):
    """Concrete EntailmentVerifier (app/core/interfaces.py).

    verify(claim, evidence) asks: is `claim` (the hypothesis — one sentence
    from the generated answer) supported by `evidence` (the premise — the
    actual source text)? That premise/hypothesis order is intentional and
    matters: reversing it asks a different, wrong question.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name

    def verify(self, claim: str, evidence: str) -> EntailmentResult:
        model = _get_model(self._model_name)
        scores = model.predict([(evidence, claim)])
        score = float(scores[0])
        verdict = "ENTAILMENT" if score >= _SUPPORTED_THRESHOLD else "NOT_SUPPORTED"
        return EntailmentResult(verdict=verdict, score=score)


def get_verifier() -> NLIVerifier:
    return NLIVerifier(settings.nli_model)
