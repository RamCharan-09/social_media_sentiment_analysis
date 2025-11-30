"""
Simple Flask API for real-time sentiment prediction.
Loads best model and vectorizer, preprocesses text, returns prediction.
"""

import os
import sys
import pickle
from flask import Flask, request, jsonify


current_dir = os.path.dirname(os.path.abspath(__file__))        # .../src/api
src_dir = os.path.dirname(current_dir)                          # .../src
project_root = os.path.dirname(src_dir)                         # .../social_media_sentiment_analysis
sys.path.append(project_root)                                   # add project root

from config.settings import *



app = Flask(__name__)

# Load best model wrapper (from Step 4) and features
BEST_MODEL_FILE = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
FEATURES_FILE = os.path.join(BASE_DIR, 'models', 'features.pkl')

print("🔄 Loading model and vectorizer...")

with open(BEST_MODEL_FILE, 'rb') as f:
    best_model_info = pickle.load(f)

with open(FEATURES_FILE, 'rb') as f:
    feature_data = pickle.load(f)

vectorizer = feature_data['vectorizer']
best_model = best_model_info['model']
best_model_name = best_model_info['name']

print(f"✅ Loaded best model: {best_model_name}")

def preprocess_text(text: str) -> str:
    """
    Very simple preprocessing for API:
    - strip spaces
    - lower() (full cleaning already done for training)
    """
    return text.strip().lower()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": best_model_name})

@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON input:
    { "text": "I love this brand!" }
    """
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({"error": "Text must be a non-empty string"}), 400

        clean_text = preprocess_text(text)
        X = vectorizer.transform([clean_text])
        pred = best_model.predict(X)[0]  # 0 or 1
        sentiment_label = "negative" if pred == 0 else "positive"

        return jsonify({
            "text": text,
            "clean_text": clean_text,
            "prediction": int(pred),
            "sentiment": sentiment_label,
            "model": best_model_name
        })

    except Exception as e:
        # Basic error handling
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local testing only (not production)
    app.run(host="0.0.0.0", port=5000, debug=True)
