"""
Optimized text cleaner for 45,000 tweets
Fast processing with progress tracking
"""
import pandas as pd
import re
import sys
import os
from tqdm import tqdm  # Progress bar

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config.settings import *

class DataCleaner:
    """Fast cleaner optimized for 45,000 tweets"""
    
    def __init__(self):
        # Comprehensive stopwords list
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
            'not', 'no', 'yes', 'get', 'got', 'go', 'going', 'come', 'came',
            'see', 'saw', 'make', 'made', 'take', 'took', 'say', 'said', 'know',
            'think', 'want', 'like', 'look', 'way', 'time', 'day', 'good', 'new'
        }
        print("✅ DataCleaner ready for 45K tweets!")
    
    def clean_text_fast(self, text):
        """Optimized single text cleaning"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase string
        text = str(text).lower()
        
        # Remove URLs, mentions, hashtags in one pass
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        
        # Remove punctuation, numbers, special chars - keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove stopwords and short words in one pass
        words = text.split()
        filtered_words = [word for word in words 
                         if word not in self.stopwords and len(word) > 2]
        
        return ' '.join(filtered_words)
    
    def clean_dataset(self, df):
        """Clean 45,000 tweets with progress tracking"""
        
        print("🧹 Starting to clean 45,000 tweets...")
        print(f"📊 Input: {len(df):,} tweets")
        
        # Make a copy
        clean_df = df.copy()
        
        # Step 1: Remove empty tweets
        original_count = len(clean_df)
        clean_df = clean_df.dropna(subset=['text'])
        clean_df = clean_df[clean_df['text'].str.strip() != '']
        print(f"🗑️ Removed {original_count - len(clean_df):,} empty tweets")
        
        # Step 2: Clean text with progress bar
        print("🔄 Cleaning text (removing junk, stopwords)...")
        tqdm.pandas(desc="Cleaning")  # Enable progress bar
        clean_df['cleaned_text'] = clean_df['text'].progress_apply(self.clean_text_fast)
        
        # Step 3: Remove tweets that became empty after cleaning
        original_count = len(clean_df)
        clean_df = clean_df[clean_df['cleaned_text'].str.len() > 0]
        print(f"🗑️ Removed {original_count - len(clean_df):,} tweets that became empty")
        
        # Step 4: Add readable sentiment labels
        clean_df['sentiment_label'] = clean_df['sentiment'].map(SENTIMENT_MAP)
        
        # Step 5: Remove duplicates
        original_count = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=['cleaned_text'])
        print(f"🗑️ Removed {original_count - len(clean_df):,} duplicate tweets")
        
        # Step 6: Reset index
        clean_df = clean_df.reset_index(drop=True)
        
        # Final statistics
        print(f"\n✅ Cleaning complete!")
        print(f"📊 Final dataset: {len(clean_df):,} clean tweets")
        print(f"📈 Final sentiment breakdown:")
        sentiment_counts = clean_df['sentiment_label'].value_counts()
        for label, count in sentiment_counts.items():
            emoji = "😢" if label == "negative" else "😊"
            percentage = (count / len(clean_df)) * 100
            print(f"   {emoji} {label.title()}: {count:,} tweets ({percentage:.1f}%)")
        
        # Show some statistics
        avg_length = clean_df['cleaned_text'].str.len().mean()
        print(f"📏 Average tweet length after cleaning: {avg_length:.1f} characters")
        
        return clean_df
    
    def save_data(self, df, filename=None):
        """Save cleaned data with compression"""
        if filename is None:
            filename = CLEANED_DATA_FILE
        
        try:
            # Save with compression to reduce file size
            df.to_csv(filename, index=False, compression='gzip' if filename.endswith('.gz') else None)
            
            file_size = os.path.getsize(filename) / (1024**2)
            print(f"💾 Saved to: {filename}")
            print(f"📏 File size: {file_size:.1f}MB")
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")

# Test the cleaner
if __name__ == "__main__":
    print("🧪 Testing DataCleaner with sample data...")
    
    from data_collector import DataCollector
    
    # Get some data to clean
    collector = DataCollector()
    cleaner = DataCleaner()
    
    # Load 1000 samples for testing
    data = collector.load_data(sample_size=1000)
    
    if data is not None:
        # Clean the data
        cleaned = cleaner.clean_dataset(data)
        
        # Show examples
        print(f"\n🔍 Before vs After examples:")
        for i in range(3):
            original = data.iloc[i]['text']
            cleaned_text = cleaned.iloc[i]['cleaned_text']
            sentiment = cleaned.iloc[i]['sentiment_label']
            
            print(f"\n{i+1}. ORIGINAL: {original}")
            print(f"   CLEANED:  {cleaned_text}")
            print(f"   SENTIMENT: {sentiment}")
    
    print("\n✅ Test complete!")
