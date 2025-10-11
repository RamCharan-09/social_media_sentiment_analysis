"""
Model Development & Training
Trains 5 classifiers with CV and hyperparameter tuning.
Uses LinearSVC (fast) instead of SVC to avoid stalls on large TF-IDF.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config.settings import *

class ModelTrainer:
    """Train and evaluate 5 sentiment classifiers fast and reliably."""
    def __init__(self):
        self.models = {
            # Strong linear baselines for sparse TF-IDF
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, n_jobs=-1),
            'Naive Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(random_state=42, C=1.0, max_iter=5000),
            # Fast, scalable SGD as extra linear baseline
            'SGD-LogReg': SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, n_jobs=-1, random_state=42),
            # Tree-based (slower on sparse, but included)
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        }
        self.ensemble = None
        self.trained_models = {}
        self.results = {}
        print("✅ ModelTrainer ready (LogReg, NB, Linear SVM, SGD-LogReg, RF + Ensemble)")

    def train_all_models(self, X_train, y_train):
        print("🏋️ Training models...")
        fitted = []
        for name, model in self.models.items():
            print(f"🔄 Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            fitted.append((name, model))
            print(f"✅ {name} trained")

        # Ensemble over strong linear models + NB (skip RF if desired)
        ensemble_members = [(n, self.trained_models[n]) for n in ['Logistic Regression', 'Linear SVM', 'Naive Bayes']]
        self.ensemble = VotingClassifier(estimators=ensemble_members, voting='hard', n_jobs=-1)
        print("🔄 Training Ensemble...")
        self.ensemble.fit(X_train, y_train)
        self.trained_models['Ensemble'] = self.ensemble
        print("✅ Ensemble trained")

    def evaluate_models(self, X_test, y_test):
        print("📊 Evaluating models...")
        results = {}
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {'accuracy': acc, 'pred': y_pred}
            print(f"✅ {name}: {acc*100:.2f}%")
        self.results = results
        return results

    def cross_validate_models(self, X, y, cv_folds=5):
        print(f"🔄 {cv_folds}-fold cross-validation (skip Ensemble for speed)...")
        cv = {}
        for name in ['Logistic Regression', 'Naive Bayes', 'Linear SVM', 'SGD-LogReg', 'Random Forest']:
            model = self.models[name]
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)
            cv[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
            print(f"✅ {name}: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
        return cv

    def tune_hyperparameters(self, X_train, y_train):
        print("🔧 Hyperparameter tuning (small, practical grids)...")
        grids = {
            'Logistic Regression': {'C': [0.5, 1, 2], 'solver': ['liblinear', 'lbfgs']},
            'Naive Bayes': {'alpha': [0.5, 1.0, 1.5]},
            'Linear SVM': {'C': [0.5, 1, 2]},
            'SGD-LogReg': {'alpha': [1e-5, 1e-4, 1e-3]},
            'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 20]},
        }
        tuned = {}
        for name, model in self.models.items():
            print(f"🔧 Tuning {name}...")
            grid = GridSearchCV(model, grids[name], cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)
            tuned[name] = grid.best_estimator_
            print(f"✅ Best {name}: {grid.best_params_} (CV {grid.best_score_*100:.2f}%)")
        self.models = tuned
        # Rebuild ensemble with tuned components
        tuned_members = [(n, self.models[n]) for n in ['Logistic Regression', 'Linear SVM', 'Naive Bayes']]
        self.ensemble = VotingClassifier(estimators=tuned_members, voting='hard', n_jobs=-1)
        self.ensemble.fit(X_train, y_train)
        self.trained_models = {**self.models, 'Ensemble': self.ensemble}
        print("✅ Ensemble retrained with tuned models")

    def get_model_comparison(self):
        if not self.results:
            return None
        return pd.DataFrame(
            {'Model': list(self.results.keys()),
             'Accuracy': [v['accuracy']*100 for v in self.results.values()]}
        ).sort_values('Accuracy', ascending=False)

    def save_models(self):
        models_dir = os.path.join(BASE_DIR, 'models'); os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, 'trained_models.pkl'), 'wb') as f:
            pickle.dump(self.trained_models, f)
        with open(os.path.join(models_dir, 'model_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        print("💾 Saved trained_models.pkl and model_results.pkl")
