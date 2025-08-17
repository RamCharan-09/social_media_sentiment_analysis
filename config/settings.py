import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Dataset info
SENTIMENT140_FILE = os.path.join(RAW_DATA_DIR, "sentiment140.csv")
CLEANED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "cleaned_data.csv")

# Column names
COLUMNS = ['sentiment', 'tweet_id', 'date', 'query', 'user', 'text']

# Sentiment mapping
SENTIMENT_MAP = {0: 'negative', 2: 'neutral', 4: 'positive'}

print("✅ Config loaded")
