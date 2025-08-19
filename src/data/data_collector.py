import pandas as pd
import os
import sys

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config.settings import *

class DataCollector:
    def __init__(self):
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        print("✅ DataCollector ready for 100K samples!")
    
    def load_data(self):
        """Load 100,000 balanced samples from Sentiment140 dataset"""
        
        if not os.path.exists(SENTIMENT140_FILE):
            print("❌ Dataset not found!")
            print(f"📁 Please place sentiment140.csv at: {SENTIMENT140_FILE}")
            print("💡 Download from: https://www.kaggle.com/datasets/kazanova/sentiment140")
            return None
        
        try:
            print("📊 Loading 100K tweets from 1.6M dataset...")
            
            # Load full dataset
            df = pd.read_csv(SENTIMENT140_FILE, encoding='latin-1', names=COLUMNS, header=None)
            print(f"📄 Found {len(df):,} total tweets")
            
            # Get balanced sample of 100K
            negative = df[df['sentiment'] == 0].sample(n=50000, random_state=42)
            positive = df[df['sentiment'] == 4].sample(n=50000, random_state=42)
            
            # Combine and shuffle
            sample_df = pd.concat([negative, positive]).sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"✅ Loaded {len(sample_df):,} balanced tweets")
            print(f"😢 Negative: {len(sample_df[sample_df['sentiment']==0]):,}")
            print(f"😊 Positive: {len(sample_df[sample_df['sentiment']==4]):,}")
            
            return sample_df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

if __name__ == "__main__":
    collector = DataCollector()
    data = collector.load_data()
    if data is not None:
        print(f"\n📝 Sample tweet: {data.iloc[0]['text']}")
