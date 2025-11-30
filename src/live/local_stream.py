"""
Local 'live stream' integration using a CSV file.
"""

import os
import sys
import time
import pickle
import pandas as pd

# -------------------------------------------------------------------
# Add project root to sys.path so 'config' and others can be imported
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../src/live
src_dir     = os.path.dirname(current_dir)                 # .../src
project_root = os.path.dirname(src_dir)                    # .../social_media_sentiment_analysis
sys.path.append(project_root)                              # add project root
# -------------------------------------------------------------------

from config.settings import *

# Paths
BEST_MODEL_FILE = os.path.join(BASE_DIR, "models", "best_model.pkl")
FEATURES_FILE   = os.path.join(BASE_DIR, "models", "features.pkl")
LIVE_FILE       = os.path.join(BASE_DIR, "data", "live", "live_messages.csv")

print("🔄 Loading model and vectorizer...")

with open(BEST_MODEL_FILE, "rb") as f:
    best_model_info = pickle.load(f)

with open(FEATURES_FILE, "rb") as f:
    feature_data = pickle.load(f)

vectorizer      = feature_data["vectorizer"]
best_model      = best_model_info["model"]
best_model_name = best_model_info["name"]

print(f"✅ Loaded best model: {best_model_name}")
print(f"📄 Live file: {LIVE_FILE}")

def preprocess_text(text: str) -> str:
    """Simple preprocessing for live data."""
    return str(text).strip().lower()

def predict_sentiment(text: str):
    """Return (pred_int, label_str)."""
    clean = preprocess_text(text)
    X     = vectorizer.transform([clean])
    pred  = best_model.predict(X)[0]  # 0 or 1
    label = "negative" if pred == 0 else "positive"
    return pred, label, clean

def stream_from_csv(poll_interval: int = 5):
    """
    Repeatedly read live_messages.csv and process new rows.
    - poll_interval: seconds between checks
    """
    if not os.path.exists(LIVE_FILE):
        print("⚠️ Live file does not exist yet. Create data/live/live_messages.csv first.")
        return

    print("🚀 Starting local live stream from CSV...")
    print("💡 Add new rows to live_messages.csv while this is running.")

    # Track last processed row count
    last_count = 0

    total_processed = 0
    total_pos       = 0
    total_neg       = 0

    try:
        while True:
            if os.path.exists(LIVE_FILE):
                df = pd.read_csv(LIVE_FILE)

                # Expect at least 'id' and 'text' columns
                if "text" not in df.columns:
                    print("⚠️ 'text' column not found in live file. Please include it.")
                    time.sleep(poll_interval)
                    continue

                current_count = len(df)

                if current_count > last_count:
                    # New rows detected
                    new_rows = df.iloc[last_count:current_count]
                    print(f"\n📥 Detected {len(new_rows)} new message(s)...")

                    for _, row in new_rows.iterrows():
                        text = row["text"]
                        pred, label, clean = predict_sentiment(text)

                        total_processed += 1
                        if pred == 1:
                            total_pos += 1
                        else:
                            total_neg += 1

                        print("────────────────────────────────────────")
                        print(f"ID: {row.get('id', 'N/A')}")
                        print(f"Source: {row.get('source', 'unknown')}")
                        print(f"Original: {text}")
                        print(f"Cleaned:  {clean}")
                        print(f"Sentiment: {label} (pred={pred})")

                    # Show simple running stats
                    if total_processed > 0:
                        pos_pct = (total_pos / total_processed) * 100
                        neg_pct = (total_neg / total_processed) * 100
                        print("\n📊 Running stats:")
                        print(f"   Total processed: {total_processed}")
                        print(f"   Positive: {total_pos} ({pos_pct:.1f}%)")
                        print(f"   Negative: {total_neg} ({neg_pct:.1f}%)")

                    last_count = current_count

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n🛑 Live streaming stopped by user.")

if __name__ == "__main__":
    stream_from_csv(poll_interval=5)
