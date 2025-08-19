import pandas as pd
import re
import nltk
import sys
import os
from tqdm import tqdm

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config.settings import *

class DataCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        print(" DataCleaner ready with full pipeline!")
    
    def clean_text(self, text):
        """Complete text cleaning pipeline"""
        if pd.isna(text) or text == '':
            return ''
        
        # Step 1: Convert to lowercase
        text = str(text).lower()
        
        # Step 2: Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        
        # Step 3: Remove emojis and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Step 4: Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Step 5: Remove extra spaces
        text = ' '.join(text.split())
        
        # Step 6: Remove stopwords and short words
        words = [word for word in text.split() 
                if word not in self.stop_words and len(word) > 2]
        
        # Step 7: Lemmatization (stemming alternative)
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def clean_dataset(self, df):
        """Clean 100K tweets with full pipeline"""
        print("🧹 Starting complete cleaning pipeline for 100K tweets...")
        
        # Handle missing values
        print(" Handling missing values...")
        original_count = len(df)
        df = df.dropna(subset=['text']).reset_index(drop=True)
        print(f" Removed {original_count - len(df):,} missing values")
        
        # Clean text with progress bar
        print(" Cleaning text (URLs, emojis, normalization, stopwords)...")
        tqdm.pandas(desc="Processing")
        df['cleaned_text'] = df['text'].progress_apply(self.clean_text)
        
        # Remove empty cleaned texts
        original_count = len(df)
        df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        print(f" Removed {original_count - len(df):,} empty texts")
        
        # Handle duplicates
        original_count = len(df)
        df = df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
        print(f" Removed {original_count - len(df):,} duplicates")
        
        # Add sentiment labels
        df['sentiment_label'] = df['sentiment'].map(SENTIMENT_MAP)
        
        print(f"\n Cleaning complete!")
        print(f" Final dataset: {len(df):,} clean tweets")
        
        # Show final distribution
        final_counts = df['sentiment_label'].value_counts()
        for label, count in final_counts.items():
            emoji = "😢" if label == "negative" else "😊"
            print(f"   {emoji} {label.title()}: {count:,}")
        
        return df
    
    def save_data(self, df):
        """Save cleaned data"""
        df.to_csv(CLEANED_DATA_FILE, index=False)
        file_size = os.path.getsize(CLEANED_DATA_FILE) / (1024**2)
        print(f"💾 Saved {len(df):,} tweets to: {CLEANED_DATA_FILE}")
        print(f"📏 File size: {file_size:.1f}MB")

if __name__ == "__main__":
    from data_collector import DataCollector
    
    collector = DataCollector()
    cleaner = DataCleaner()
    
    # Test with 1000 samples
    data = collector.load_data()
    if data is not None:
        # Take small sample for testing
        test_data = data.head(1000)
        cleaned = cleaner.clean_dataset(test_data)
        
        print(f"\n🔍 Example:")
        print(f"BEFORE: {data.iloc[0]['text']}")
        print(f"AFTER:  {cleaned.iloc['cleaned_text']}")
