"""
Model tester — loads live_messages.csv and shows prediction + confidence score.
Run from project root:
    python tests/test_model.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

BEST_MODEL_FILE = os.path.join(project_root, "models", "best_model.pkl")
FEATURES_FILE   = os.path.join(project_root, "models", "features.pkl")
LIVE_FILE       = os.path.join(project_root, "data", "live", "live_messages.csv")

print("🔄 Loading model...")
with open(BEST_MODEL_FILE, "rb") as f:
    best_model_info = pickle.load(f)
with open(FEATURES_FILE, "rb") as f:
    feature_data = pickle.load(f)

vectorizer  = feature_data["vectorizer"]
model       = best_model_info["model"]
model_name  = best_model_info["name"]
print(f"✅ Model loaded: {model_name}\n")


def get_confidence(model, X):
    """Return confidence score regardless of model type."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        return float(np.max(proba)), proba
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))
        proba = [1 - confidence, confidence] if score > 0 else [confidence, 1 - confidence]
        return confidence, proba
    else:
        return None, None


def predict(row_id, text, source="unknown"):
    clean = str(text).strip().lower()
    X     = vectorizer.transform([clean])
    pred  = model.predict(X)[0]
    label = "positive 😊" if pred == 1 else "negative 😢"

    confidence, proba = get_confidence(model, X)

    print("─" * 55)
    print(f"ID        : {row_id}  |  Source: {source}")
    print(f"Input     : {text}")
    print(f"Sentiment : {label}")
    if confidence is not None:
        print(f"Confidence: {confidence * 100:.1f}%")
        print(f"  → Negative: {proba[0] * 100:.1f}%  |  Positive: {proba[1] * 100:.1f}%")
    print(f"Model     : {model_name}")


# ── Load and test from live_messages.csv ─────────────────────────────
print(f"📄 Reading: {LIVE_FILE}\n")

df = pd.read_csv(LIVE_FILE)
print(f"Found {len(df)} message(s) to test:\n")

for _, row in df.iterrows():
    predict(
        row_id=row.get("id", "N/A"),
        text=row["text"],
        source=row.get("source", "unknown")
    )

print("─" * 55)

# ── Interactive mode ──────────────────────────────────────────────────
print("\n💬 Enter your own sentences (type 'quit' to exit):\n")
while True:
    user_input = input(">> ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("👋 Bye!")
        break
    if user_input:
        predict("manual", user_input, source="manual")
