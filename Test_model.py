#!/usr/bin/env python3
import sys
import io
import os
import pickle

# Force UTF-8 stdout/stderr with safe fallback
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

print("Testing model loading...")
print("=" * 60)

model_path = "models/fake_news_model.pkl"
vectorizer_path = "models/vectorizer.pkl"

if not os.path.exists(model_path):
    print(f"❌ ERROR: {model_path} not found!")
    sys.exit(1)
if not os.path.exists(vectorizer_path):
    print(f"❌ ERROR: {vectorizer_path} not found!")
    sys.exit(1)

# Load model
print("Loading model...")
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

# Load vectorizer
print("Loading vectorizer...")
try:
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    print("✓ Vectorizer loaded successfully!")
except Exception as e:
    print(f"Failed to load vectorizer: {e}")
    raise

# Test prediction
print("\n" + "=" * 60)
print("TESTING PREDICTION:")
print("=" * 60)

test_text = "Scientists announce breakthrough in renewable energy technology"
print(f"\nTest text: {test_text}")

vec = vectorizer.transform([test_text])
prediction = model.predict(vec)[0]

# Safely get probabilities if available
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(vec)[0]
else:
    # fallback: try decision_function -> convert to pseudo-probabilities
    try:
        import numpy as np
        scores = model.decision_function(vec)
        s = np.atleast_1d(scores).astype(float)
        if s.ndim == 1:
            pos = 1 / (1 + np.exp(-s))
            probability = [1 - pos[0], pos[0]]
        else:
            exp = np.exp(s - np.max(s, axis=1, keepdims=True))
            soft = exp / exp.sum(axis=1, keepdims=True)
            probability = soft[0].tolist()
    except Exception:
        probability = [0.0, 0.0]

result = "REAL" if prediction == 1 else "FAKE"
confidence = max(probability) * 100 if len(probability) > 0 else 0.0

print(f"\nPrediction: {result}")
print(f"Confidence: {confidence:.1f}%")
print(f"  - Fake: {probability[0]*100:.1f}%")
print(f"  - Real: {probability[1]*100:.1f}%")

print("\n" + "=" * 60)
print("✅ MODEL TEST SUCCESSFUL!")
print("=" * 60)
