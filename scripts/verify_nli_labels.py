"""One-off diagnostic — NOT part of the app, not imported anywhere.

Empirically checks whether app/services/grounding.py's _LABELS order
(["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]) actually matches what
cross-encoder/nli-deberta-v3-base outputs on THIS machine's installed
version of the model/library, instead of trusting the HuggingFace model
card description. This exists because grounding.py just flagged a
video that IS about swapping as "Could not be verified" for a swapping
question, which should not happen if the label order is correct -- so
the label order is the top suspect and needs to be checked against
real model output, not documentation.

Run it with the project venv active:
    python scripts\\verify_nli_labels.py   (PowerShell/cmd, from D:\\Projects\\EduRAG)
"""

from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# Each case has an unambiguous, textbook-obvious correct label -- not
# subtle at all -- so whichever index comes back highest tells us
# definitively which slot is which, no guessing.
cases = [
    ("A man is eating food.", "A man is eating.", "ENTAILMENT (obviously true given the premise)"),
    ("A man is eating food.", "The man is sleeping.", "CONTRADICTION (directly conflicts)"),
    ("A man is eating food.", "The man is in Paris.", "NEUTRAL (unrelated, not implied either way)"),
]

print("Raw model output order (whatever it is, before any relabeling):")
print()

for premise, hypothesis, expected in cases:
    scores = model.predict([(premise, hypothesis)], apply_softmax=True)
    probs = scores[0]
    best_index = max(range(len(probs)), key=lambda i: probs[i])
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Expected:   {expected}")
    print(f"Raw probabilities: {[round(float(p), 4) for p in probs]}")
    print(f"Highest-probability index: {best_index}")
    print()

print("=" * 70)
print("Current code assumes this index -> label mapping:")
print("  index 0 = CONTRADICTION")
print("  index 1 = ENTAILMENT")
print("  index 2 = NEUTRAL")
print()
print("Compare each case's 'Highest-probability index' above against")
print("its 'Expected' label using that mapping. If they don't line up")
print("for all three cases, the mapping in app/services/grounding.py's")
print("_LABELS is wrong and needs to change to match what you saw here.")
