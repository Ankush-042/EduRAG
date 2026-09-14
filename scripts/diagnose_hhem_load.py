"""One-off diagnostic -- NOT part of the app, not imported anywhere.

grounding.py's AutoModelForSequenceClassification.from_pretrained(
'vectara/hallucination_evaluation_model', trust_remote_code=True) just
loaded as plain DebertaV2ForSequenceClassification (no .predict()) instead
of the model repo's own HHEMv2ForSequenceClassification class -- even
though the model's own config.json declares
auto_map.AutoModelForSequenceClassification = HHEMv2ForSequenceClassification
and HHEM's underlying config model_type is deberta-v2. That combination is
a known transformers gotcha: some transformers versions resolve a
config's *native* model_type (deberta-v2 is natively supported) before
consulting its auto_map, silently ignoring trust_remote_code. This checks
that theory directly against your installed transformers version instead
of guessing at a fix.

Run it with the project venv active, from the project root:
    python scripts\\diagnose_hhem_load.py
"""

import transformers

print(f"transformers version: {transformers.__version__}")
print()

from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

MODEL_NAME = "vectara/hallucination_evaluation_model"

print("--- AutoConfig ---")
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"type(config) = {type(config)}")
print(f"config.model_type = {getattr(config, 'model_type', None)}")
print(f"config.auto_map = {getattr(config, 'auto_map', None)}")
print()

print("--- AutoModelForSequenceClassification ---")
model_seqcls = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"type(model) = {type(model_seqcls)}")
print(f"has .predict = {hasattr(model_seqcls, 'predict')}")
print()

print("--- AutoModel (fallback theory: HHEM only registers under AutoModel) ---")
try:
    model_auto = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"type(model) = {type(model_auto)}")
    print(f"has .predict = {hasattr(model_auto, 'predict')}")
except Exception as e:
    print(f"AutoModel failed: {e!r}")
