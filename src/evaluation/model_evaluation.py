"""
Model Evaluation & Benchmarking
Compare ML models with VADER baseline and calculate comprehensive metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pickle
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config.settings import *

class ModelEvaluator:
    """Evaluate and compare all trained models including VADER baseline"""
    
    def __init__(self):
        self.metrics_df = None
        self.best_model = None
        print("✅ ModelEvaluator ready!")
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate all performance metrics for a model"""
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred) * 100,
            'Precision': precision_score(y_true, y_pred, zero_division=0) * 100,
            'Recall': recall_score(y_true, y_pred, zero_division=0) * 100,
            'F1-Score': f1_score(y_true, y_pred, zero_division=0) * 100
        }
        return metrics
    
    def evaluate_all_models(self, trained_models, X_test, y_test):
        """Evaluate all trained ML models"""
        print("📊 Evaluating all ML models...")
        
        all_metrics = []
        confusion_matrices = {}
        
        for name, model in trained_models.items():
            print(f"🔄 Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, name)
            all_metrics.append(metrics)
            
            # Store confusion matrix
            confusion_matrices[name] = confusion_matrix(y_test, y_pred)
            
            print(f"✅ {name}: {metrics['Accuracy']:.2f}% accuracy")
        
        # Create metrics DataFrame
        self.metrics_df = pd.DataFrame(all_metrics).sort_values('Accuracy', ascending=False)
        
        return self.metrics_df, confusion_matrices
    
    def evaluate_vader_baseline(self, texts, y_test):
        """Evaluate VADER sentiment analysis as baseline"""
        print("🔄 Evaluating VADER baseline...")
        
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            
            # Initialize VADER
            analyzer = SentimentIntensityAnalyzer()
            
            # Predict sentiments
            vader_predictions = []
            for text in texts:
                scores = analyzer.polarity_scores(str(text))
                # Positive if compound score > 0, else negative
                vader_predictions.append(1 if scores['compound'] >= 0 else 0)
            
            vader_predictions = np.array(vader_predictions)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, vader_predictions, 'VADER (Rule-Based)')
            
            # Create confusion matrix
            vader_cm = confusion_matrix(y_test, vader_predictions)
            
            print(f"✅ VADER: {metrics['Accuracy']:.2f}% accuracy")
            
            return metrics, vader_cm, vader_predictions
            
        except ImportError:
            print("⚠️ VADER not installed. Install with: pip install vaderSentiment")
            return None, None, None
    
    def compare_with_vader(self, vader_metrics):
        """Compare ML models with VADER baseline"""
        if vader_metrics is None:
            return self.metrics_df
        
        # Add VADER to comparison
        vader_df = pd.DataFrame([vader_metrics])
        comparison_df = pd.concat([self.metrics_df, vader_df], ignore_index=True)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def analyze_model_strengths(self, confusion_matrices):
        """Analyze strengths and weaknesses of each model"""
        print("\n🔍 Analyzing model strengths and weaknesses...")
        
        analysis = {}
        
        for name, cm in confusion_matrices.items():
            # Calculate per-class metrics
            tn, fp, fn, tp = cm.ravel()
            
            # Negative class metrics
            negative_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
            negative_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Positive class metrics  
            positive_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            positive_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            analysis[name] = {
                'True Negatives': tn,
                'False Positives': fp,
                'False Negatives': fn,
                'True Positives': tp,
                'Negative Precision': negative_precision * 100,
                'Negative Recall': negative_recall * 100,
                'Positive Precision': positive_precision * 100,
                'Positive Recall': positive_recall * 100
            }
        
        return pd.DataFrame(analysis).T
    
    def select_best_model(self, comparison_df, trained_models):
        """Select the best performing ML model for deployment"""
        # Filter out VADER to select best ML model
        ml_models = comparison_df[~comparison_df['Model'].str.contains('VADER', na=False)]
        
        if len(ml_models) > 0:
            best_model_name = ml_models.iloc[0]['Model']
            best_model_obj = trained_models[best_model_name]
            best_accuracy = ml_models.iloc[0]['Accuracy']
            
            self.best_model = {
                'name': best_model_name,
                'model': best_model_obj,
                'accuracy': best_accuracy,
                'metrics': ml_models.iloc[0].to_dict()
            }
            
            print(f"\n🏆 Best ML Model: {best_model_name}")
            print(f"   Accuracy: {best_accuracy:.2f}%")
            print(f"   Precision: {ml_models.iloc[0]['Precision']:.2f}%")
            print(f"   Recall: {ml_models.iloc[0]['Recall']:.2f}%")
            print(f"   F1-Score: {ml_models.iloc[0]['F1-Score']:.2f}%")
            
            return self.best_model
        
        return None
    
    def save_evaluation_results(self, comparison_df, confusion_matrices, analysis_df):
        """Save all evaluation results"""
        eval_dir = os.path.join(BASE_DIR, 'data', 'processed')
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save comparison CSV
        comparison_file = os.path.join(eval_dir, 'final_model_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        
        # Save analysis CSV
        analysis_file = os.path.join(eval_dir, 'model_analysis.csv')
        analysis_df.to_csv(analysis_file)
        
        # Save confusion matrices and best model
        results_file = os.path.join(BASE_DIR, 'models', 'evaluation_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump({
                'comparison': comparison_df,
                'confusion_matrices': confusion_matrices,
                'analysis': analysis_df,
                'best_model': self.best_model
            }, f)
        
        print(f"💾 Saved comparison to: {comparison_file}")
        print(f"💾 Saved analysis to: {analysis_file}")
        print(f"💾 Saved results to: {results_file}")

# Test the evaluator
if __name__ == "__main__":
    print("🧪 Testing ModelEvaluator...")
    
    # Create sample data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_true, y_pred, 'Test Model')
    
    print(f"\n✅ Test metrics: {metrics}")
    print("✅ ModelEvaluator test complete!")
