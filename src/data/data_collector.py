"""
Data collector for Sentiment140 dataset - Manual Download Version
Loads 45,000 samples from 1.6M tweet dataset
"""
import pandas as pd
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config.settings import *

class DataCollector:
    """Loads tweets from manually downloaded Sentiment140 dataset"""
    
    def __init__(self):
        # Create folders for storing data
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        print("✅ DataCollector ready!")
    
    def check_dataset_exists(self):
        """Check if manually downloaded dataset exists"""
        if os.path.exists(SENTIMENT140_FILE):
            file_size = os.path.getsize(SENTIMENT140_FILE) / (1024**2)  # Size in MB
            print(f"✅ Dataset found: {file_size:.1f}MB")
            return True
        else:
            print("❌ Dataset not found!")
            print(f"📁 Please place your downloaded file at: {SENTIMENT140_FILE}")
            print("💡 Download from: https://www.kaggle.com/datasets/kazanova/sentiment140")
            return False
    

    # def load_data(self, sample_size=1600000):
    def load_data(self, sample_size=45000):
        """Load 45,000 samples from the 1.6M dataset"""
        
        # Check if dataset exists
        if not self.check_dataset_exists():
            return None
        
        try:
            print(f"📊 Loading Sentiment140 dataset...")
            print(f"🎯 Target sample size: {sample_size:,}")
            
            # Load the massive CSV file
            print("⏳ Reading 1.6M tweets... (this may take a moment)")
            df = pd.read_csv(
                SENTIMENT140_FILE,
                encoding='latin-1',  # Handles special characters
                names=COLUMNS,       # Our column names
                header=None          # No header row in file
            )
            
            print(f"📄 Successfully loaded {len(df):,} total tweets")
            
            # Take a balanced sample
            if sample_size and sample_size < len(df):
                # Get equal numbers of positive and negative tweets
                negative_tweets = df[df['sentiment'] == 0]
                positive_tweets = df[df['sentiment'] == 4]
                
                half_sample = sample_size // 2
                
                # Sample equally from both sentiments
                neg_sample = negative_tweets.sample(n=min(half_sample, len(negative_tweets)), random_state=42)
                pos_sample = positive_tweets.sample(n=min(half_sample, len(positive_tweets)), random_state=42)
                
                # Combine and shuffle
                df = pd.concat([neg_sample, pos_sample], ignore_index=True)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
                
                print(f"✅ Sampled {len(df):,} tweets (balanced)")
            else:
                print(f"✅ Using all {len(df):,} tweets")
            
            # Show what we have
            print(f"📈 Sentiment breakdown:")
            sentiment_counts = df['sentiment'].value_counts().sort_index()
            for sentiment, count in sentiment_counts.items():
                label = "😢 Negative" if sentiment == 0 else "😊 Positive"
                percentage = (count / len(df)) * 100
                print(f"   {label}: {count:,} tweets ({percentage:.1f}%)")
            
            # Show memory usage
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            print(f"💾 Memory usage: {memory_mb:.1f}MB")
            
            return df
            
        except FileNotFoundError:
            print("❌ Dataset file not found!")
            print(f"📁 Expected location: {SENTIMENT140_FILE}")
            return None
        except pd.errors.EmptyDataError:
            print("❌ Dataset file is empty or corrupted!")
            return None
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def get_dataset_info(self):
        """Get information about the dataset without loading it all"""
        if not self.check_dataset_exists():
            return None
        
        try:
            # Read just the first few lines to get info
            sample = pd.read_csv(
                SENTIMENT140_FILE,
                encoding='latin-1',
                names=COLUMNS,
                header=None,
                nrows=1000  # Just first 1000 rows
            )
            
            # Get file size
            file_size = os.path.getsize(SENTIMENT140_FILE) / (1024**2)
            
            print(f"📊 Dataset Information:")
            print(f"📁 File size: {file_size:.1f}MB")
            print(f"📋 Columns: {list(sample.columns)}")
            print(f"📝 Sample text: '{sample.iloc[0]['text']}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading dataset info: {e}")
            return None

# Test the collector
if __name__ == "__main__":
    print("🧪 Testing DataCollector with 45K samples...")
    
    collector = DataCollector()
    
    # Show dataset info
    collector.get_dataset_info()
    
    # Load 45,000 samples
    # data = collector.load_data(sample_size=100000)
    data = collector.load_data(sample_size=45000)
    
    if data is not None:
        print(f"\n📝 First 3 tweets:")
        for i in range(3):
            sentiment = "😢 NEGATIVE" if data.iloc[i]['sentiment'] == 0 else "😊 POSITIVE"
            tweet = data.iloc[i]['text'][:100] + "..." if len(data.iloc[i]['text']) > 100 else data.iloc[i]['text']
            print(f"{i+1}. {sentiment}: {tweet}")
    
    print("\n✅ Test complete!")
