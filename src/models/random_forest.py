"""
Random Forest Classifier for DNS Anomaly Detection

This module implements a Random Forest classifier for supervised binary 
classification of DNS traffic (Benign vs Malicious).

Random Forest is an ensemble learning method that constructs multiple decision
trees during training and outputs the mode of the classes (classification) or 
mean prediction (regression) of the individual trees.

Advantages:
- Handles high-dimensional data well
- Provides feature importance rankings
- Robust to overfitting
- Works well with imbalanced datasets (with class_weight='balanced')
- Can capture non-linear relationships
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

import warnings
warnings.filterwarnings('ignore')


class RandomForestDetector:
    """
    Random Forest Classifier for DNS Anomaly Detection.
    
    This class wraps scikit-learn's RandomForestClassifier with additional
    functionality for training, evaluation, and visualization specific to
    DNS anomaly detection.
    
    Parameters:
    -----------
    n_estimators : int, default=500
        Number of trees in the forest
    max_depth : int or None, default=None
        Maximum depth of the tree. None means nodes expand until all leaves are pure
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
    max_features : str or int, default='sqrt'
        Number of features to consider when looking for the best split
    class_weight : str or dict, default='balanced'
        Weights associated with classes
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of jobs to run in parallel (-1 uses all processors)
    """
    
    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 class_weight: str = 'balanced',
                 criterion: str = 'gini',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """Initialize Random Forest detector."""
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            criterion=criterion,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self.feature_names = None
        self.training_stats = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'RandomForestDetector':
        """
        Train the Random Forest classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features (n_samples, n_features)
        y : np.ndarray
            Training labels (n_samples,)
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        self : RandomForestDetector
            Fitted estimator
        """
        print(f"\nTraining Random Forest with {self.n_estimators} trees...")
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Train model
        start_time = datetime.now()
        self.model.fit(X, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store training statistics
        self.training_stats = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_trees': self.n_estimators,
            'training_time_seconds': training_time,
            'feature_importances': self.model.feature_importances_.tolist()
        }
        
        if self.oob_score and self.bootstrap:
            self.training_stats['oob_score'] = self.model.oob_score_
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        if self.oob_score and self.bootstrap:
            print(f"  OOB Score: {self.model.oob_score_:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted class labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
            
        Returns:
        --------
        probabilities : np.ndarray
            Class probabilities (n_samples, n_classes)
        """
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to return
            
        Returns:
        --------
        importance_df : pd.DataFrame
            DataFrame with feature names and importance scores
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importance_df = pd.DataFrame({
            'feature': [self.feature_names[i] for i in indices[:top_n]],
            'importance': importances[indices[:top_n]]
        })
        
        return importance_df
    
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 output_dir: Optional[Path] = None,
                 show_plot: bool = False) -> Dict[str, Any]:
        """
        Evaluate the Random Forest model.
        
        Parameters:
        -----------
        X : np.ndarray
            Test features
        y : np.ndarray
            Test labels
        output_dir : Path, optional
            Directory to save plots
        show_plot : bool, default=False
            Whether to display plots
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        print("\nEvaluating Random Forest model...")
        
        # Get predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred, average=None, labels=[0, 1]
        )
        
        metrics['precision_benign'] = precision[0]
        metrics['recall_benign'] = recall[0]
        metrics['f1_benign'] = f1[0]
        
        metrics['precision_malicious'] = precision[1]
        metrics['recall_malicious'] = recall[1]
        metrics['f1_malicious'] = f1[1]
        
        # Overall metrics
        metrics['precision'] = precision_recall_fscore_support(y, y_pred, average='weighted')[0]
        metrics['recall'] = precision_recall_fscore_support(y, y_pred, average='weighted')[1]
        metrics['f1_score'] = precision_recall_fscore_support(y, y_pred, average='weighted')[2]
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        
        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Generate plots
        if output_dir or show_plot:
            self._plot_evaluation(y, y_pred, y_pred_proba, metrics, 
                                 output_dir, show_plot)
        
        return metrics
    
    def _plot_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_pred_proba: np.ndarray, metrics: Dict[str, Any],
                        output_dir: Optional[Path], show_plot: bool):
        """Generate evaluation plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Random Forest Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        
        # Create labels with both counts and class-wise fractions
        cm_labels = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            row_sum = cm[i, :].sum()  # Sum of true class i
            for j in range(cm.shape[1]):
                count = cm[i, j]
                fraction = count / row_sum if row_sum > 0 else 0
                cm_labels[i, j] = f'{count}\n({fraction:.1%})'
        
        sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Benign', 'Malicious'],
                   yticklabels=['Benign', 'Malicious'])
        axes[0, 0].set_title('Confusion Matrix (Count & Class-wise %)')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, 
                       label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        axes[0, 2].plot(recall_curve, precision_curve, 'g-', linewidth=2,
                       label=f'PR (AUC = {metrics["pr_auc"]:.4f})')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature Importance (Top 15)
        importance_df = self.get_feature_importance(top_n=15)
        axes[1, 0].barh(range(len(importance_df)), importance_df['importance'])
        axes[1, 0].set_yticks(range(len(importance_df)))
        axes[1, 0].set_yticklabels(importance_df['feature'], fontsize=8)
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importances')
        axes[1, 0].invert_yaxis()
        
        # 5. Metrics Bar Chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['roc_auc'],
            metrics['pr_auc']
        ]
        
        bars = axes[1, 1].bar(range(len(metric_names)), metric_values, 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        axes[1, 1].set_xticks(range(len(metric_names)))
        axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Class-wise Performance
        classes = ['Benign', 'Malicious']
        precision_vals = [metrics['precision_benign'], metrics['precision_malicious']]
        recall_vals = [metrics['recall_benign'], metrics['recall_malicious']]
        f1_vals = [metrics['f1_benign'], metrics['f1_malicious']]
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[1, 2].bar(x - width, precision_vals, width, label='Precision', color='#1f77b4')
        axes[1, 2].bar(x, recall_vals, width, label='Recall', color='#ff7f0e')
        axes[1, 2].bar(x + width, f1_vals, width, label='F1-Score', color='#2ca02c')
        
        axes[1, 2].set_xlabel('Class')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Per-Class Performance')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(classes)
        axes[1, 2].legend()
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / 'random_forest_evaluation.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Evaluation plots saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, filepath: Path):
        """
        Save the trained Random Forest model.
        
        Parameters:
        -----------
        filepath : Path
            Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'RandomForestDetector':
        """
        Load a trained Random Forest model.
        
        Parameters:
        -----------
        filepath : Path
            Path to the saved model
            
        Returns:
        --------
        detector : RandomForestDetector
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        detector = cls(
            n_estimators=model_data.get('n_estimators', 500),
            max_depth=model_data.get('max_depth', None),
            random_state=model_data.get('random_state', 42)
        )
        
        detector.model = model_data['model']
        detector.feature_names = model_data['feature_names']
        detector.training_stats = model_data['training_stats']
        
        print(f"✓ Model loaded from: {filepath}")
        return detector
