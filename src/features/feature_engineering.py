"""
Converts text to TF-IDF numerical features for machine learning
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config.settings import *

class FeatureEngineer:
    """Converts cleaned text to numerical features"""
    
    def __init__(self):
        # TF-IDF Vectorizer with optimal settings
        self.vectorizer = TfidfVectorizer(
            max_features=10000,      # Top 10,000 most important words
            min_df=5,                # Word must appear in at least 5 tweets
            max_df=0.95,             # Ignore words in >95% of tweets
            ngram_range=(1, 2),      # Use single words and word pairs
            stop_words='english'     # Remove common English words
        )
        self.is_fitted = False
        print(" FeatureEngineer ready!")
    
    def create_features(self, texts):
        """Convert text to TF-IDF feature matrix"""
        print(f" Converting {len(texts):,} texts to numerical features...")
        
        # Fit and transform texts to TF-IDF matrix
        X = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        
        # Get feature names (vocabulary)
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f" Feature creation complete:")
        print(f"    Matrix shape: ({X.shape[0]:,} tweets × {X.shape[1]:,} features)")
        print(f"    Vocabulary size: {len(feature_names):,} words/phrases")
        print(f"    Approx. memory usage: {X.data.nbytes / (1024**2):.1f}MB")
        
        return X, feature_names
    
    def transform(self, texts):
        """Transform new unseen texts using fitted vectorizer"""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer not fitted yet. Run create_features() first.")
        return self.vectorizer.transform(texts)
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets"""
        print(f" Splitting data: {100-test_size*100:.0f}% train, {test_size*100:.0f}% test")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # fallback if stratify fails (e.g., small dataset)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print(f" Split complete:")
        print(f"    Training: {X_train.shape[0]:,} samples")
        print(f"    Testing: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def analyze_features(self, feature_names, X):
        """Analyze most important features"""
        print(" Analyzing feature importance...")
        
        # Calculate average TF-IDF scores
        mean_scores = np.array(X.mean(axis=0)).flatten()
        
        # Create feature importance DataFrame
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'avg_tfidf': mean_scores
        }).sort_values('avg_tfidf', ascending=False)
        
        print(f" Top 10 most important features:")
        print(feature_df.head(10).to_string(index=False))
        
        return feature_df
    
    def save_features(self, X_train, X_test, y_train, y_test, feature_names):
        """Save processed features for machine learning"""
        models_dir = os.path.join(BASE_DIR, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        feature_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'vectorizer': self.vectorizer
        }
        
        feature_file = os.path.join(models_dir, 'features.pkl')
        with open(feature_file, 'wb') as f:
            pickle.dump(feature_data, f)
        
        file_size = os.path.getsize(feature_file) / (1024**2)
        print(f" Saved features to: {feature_file}")
        print(f" File size: {file_size:.1f}MB")
    
    def load_features(self, filepath):
        """Load previously saved feature data"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Test the feature engineer
if __name__ == "__main__":
    print(" Testing FeatureEngineer...")
    
    sample_texts = [
        "love amazing great wonderful",
        "hate terrible awful horrible", 
        "good nice excellent",
        "bad worst disgusting"
    ]
    sample_labels = [1, 0, 1, 0]
    
    engineer = FeatureEngineer()
    X, features = engineer.create_features(sample_texts)
    X_train, X_test, y_train, y_test = engineer.split_data(X, sample_labels, test_size=0.5)
    
    print("\n FeatureEngineer test complete!")
