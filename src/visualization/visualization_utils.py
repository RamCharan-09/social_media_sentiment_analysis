"""
Visualization utilities for sentiment analysis project.
Creates:
- Sentiment trend over time
- Sentiment distribution charts
- Feature importance plots
- Comparative analysis visualizations
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config.settings import *

class Visualizer:
    def __init__(self):
        sns.set(style="whitegrid")
        print("✅ Visualizer ready!")

    def plot_sentiment_trend(self, df):
        """
        Plot sentiment trend over time.
        Expects df with columns: ['date', 'sentiment_label'].
        """
        if 'date' not in df.columns:
            print("⚠️ No 'date' column in data, skipping trend plot.")
            return

        trend = (df
                 .groupby(['date', 'sentiment_label'])
                 .size()
                 .reset_index(name='count'))

        plt.figure(figsize=(10, 5))
        for label, group in trend.groupby('sentiment_label'):
            plt.plot(group['date'], group['count'], label=label)
        plt.title("Sentiment Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Tweets")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_sentiment_distribution(self, df):
        """
        Plot sentiment distribution (bar + pie).
        Expects 'sentiment_label' column.
        """
        if 'sentiment_label' not in df.columns:
            print("⚠️ No 'sentiment_label' column, skipping distribution plots.")
            return

        counts = df['sentiment_label'].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Bar chart
        ax1.bar(counts.index, counts.values, color=['red', 'green'])
        ax1.set_title("Sentiment Distribution (Bar)")
        ax1.set_ylabel("Count")

        # Pie chart
        ax2.pie(counts.values,
                labels=counts.index,
                autopct='%1.1f%%',
                startangle=90)
        ax2.set_title("Sentiment Distribution (Pie)")

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_analysis_csv, top_n=20):
        """
        Plot top N features by average TF-IDF score.
        feature_analysis_csv: path to CSV created in Step 2.
        """
        if not os.path.exists(feature_analysis_csv):
            print(f"⚠️ File not found: {feature_analysis_csv}")
            return

        fa = pd.read_csv(feature_analysis_csv)
        fa = fa.sort_values("avg_tfidf", ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        plt.barh(fa['feature'], fa['avg_tfidf'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.title(f"Top {top_n} Features by Average TF-IDF")
        plt.xlabel("Average TF-IDF Score")
        plt.tight_layout()
        plt.show()

    def compare_models(self, comparison_csv):
        """
        Plot model accuracy comparison.
        comparison_csv: model_comparison.csv from Step 3 or final_model_comparison.csv from Step 4.
        """
        if not os.path.exists(comparison_csv):
            print(f"⚠️ File not found: {comparison_csv}")
            return

        df = pd.read_csv(comparison_csv)
        plt.figure(figsize=(8, 4))
        plt.bar(df['Model'], df['Accuracy'], color='lightgreen', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy (%)")
        plt.tight_layout()
        plt.show()
