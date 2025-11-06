"""
Unsupervised Anomaly Detection for DNS Traffic
Isolation Forest approach for detecting malicious DNS behavior without labeled data
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import logging

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """
    Isolation Forest for unsupervised anomaly detection in DNS traffic.
    
    Isolation Forest works on the principle that anomalies are:
    1. Few in number (rare)
    2. Have feature values that are very different from normal instances
    
    The algorithm isolates observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and
    minimum values of the selected feature. Anomalies are easier to isolate
    (require fewer splits) than normal points.
    
    Key Parameters:
    - n_estimators: Number of isolation trees in the forest
    - max_samples: Number of samples to draw from X to train each tree
    - contamination: Expected proportion of outliers in the dataset
    - max_features: Number of features to draw from X to train each tree
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, str] = 'auto',
        contamination: float = 0.05,
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 0
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of trees in the forest (default: 100)
            max_samples: Number of samples to draw to train each tree
                        - 'auto': min(256, n_samples)
                        - int: specific number
            contamination: Expected proportion of outliers (0-0.5)
                          Used to define decision threshold
            max_features: Number of features to draw to train each tree
                         - float (0-1): proportion of features
                         - int: specific number
            bootstrap: Whether to use bootstrap sampling
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
            verbose: Verbosity level
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize models
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
        
        # Training metadata
        self.is_fitted = False
        self.feature_names = None
        self.training_stats = {}
        
        logger.info(
            f"Initialized Isolation Forest with n_estimators={n_estimators}, "
            f"contamination={contamination}, max_samples={max_samples}"
        )
    
    def fit(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> 'IsolationForestAnomalyDetector':
        """
        Train the Isolation Forest on normal (benign) traffic only.
        
        Important: X should contain ONLY benign/normal traffic.
        The model learns the characteristics of normal behavior.
        
        Args:
            X: Training features (benign traffic only)
            feature_names: List of feature names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Isolation Forest on {len(X)} samples...")
        
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Store training statistics
        self.training_stats['n_samples'] = len(X)
        self.training_stats['n_features'] = X.shape[1]
        self.training_stats['timestamp'] = datetime.now().isoformat()
        
        # Step 1: Standardize features (zero mean, unit variance)
        logger.info("Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 2: Train Isolation Forest
        logger.info("Training Isolation Forest model...")
        self.model.fit(X_scaled)
        
        # Step 3: Compute training set statistics
        # Get anomaly scores (negative = outlier, positive = inlier)
        anomaly_scores = self.model.score_samples(X_scaled)
        decisions = self.model.decision_function(X_scaled)
        
        self.training_stats['anomaly_score_mean'] = float(np.mean(anomaly_scores))
        self.training_stats['anomaly_score_std'] = float(np.std(anomaly_scores))
        self.training_stats['anomaly_score_min'] = float(np.min(anomaly_scores))
        self.training_stats['anomaly_score_max'] = float(np.max(anomaly_scores))
        
        self.training_stats['decision_mean'] = float(np.mean(decisions))
        self.training_stats['decision_std'] = float(np.std(decisions))
        self.training_stats['decision_min'] = float(np.min(decisions))
        self.training_stats['decision_max'] = float(np.max(decisions))
        
        # Get the offset (threshold for anomaly detection)
        self.training_stats['offset'] = float(self.model.offset_)
        
        # Count predicted outliers in training set
        train_predictions = self.model.predict(X_scaled)
        n_outliers_train = np.sum(train_predictions == -1)
        self.training_stats['n_outliers_in_training'] = int(n_outliers_train)
        self.training_stats['outlier_ratio_training'] = float(n_outliers_train / len(X))
        
        self.is_fitted = True
        logger.info(
            f"Training complete! Detected {n_outliers_train}/{len(X)} outliers "
            f"in training set ({n_outliers_train/len(X):.2%})"
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict whether samples are normal (1) or anomalous (-1).
        
        Args:
            X: Test features
            
        Returns:
            Predictions: 1 = normal/inlier, -1 = anomaly/outlier
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores (shifted opposite of the score_samples).
        
        Values are shifted so that:
        - Positive values = inliers (normal)
        - Negative values = outliers (anomalous)
        
        Args:
            X: Test features
            
        Returns:
            Decision scores (negative = anomaly, positive = normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute raw anomaly scores (average path length).
        
        Lower scores indicate anomalies (easier to isolate).
        Higher scores indicate normal behavior (harder to isolate).
        
        Args:
            X: Test features
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert anomaly scores to pseudo-probabilities.
        
        Args:
            X: Test features
            
        Returns:
            Anomaly probabilities (0-1, higher = more likely anomalous)
        """
        # Get raw anomaly scores
        anomaly_scores = self.score_samples(X)
        
        # Normalize to [0, 1] range based on training distribution
        # Lower scores are more anomalous, so we invert
        min_score = self.training_stats['anomaly_score_min']
        max_score = self.training_stats['anomaly_score_max']
        
        # Invert and normalize: lower scores become high probabilities
        probabilities = 1 - (anomaly_scores - min_score) / (max_score - min_score)
        probabilities = np.clip(probabilities, 0, 1)
        
        return probabilities
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: np.ndarray,
        output_dir: Optional[Path] = None,
        show_plot: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on labeled test data.
        
        Args:
            X_test: Test features
            y_test: True labels (0=benign, 1=malicious)
            output_dir: Directory to save plots and results
            show_plot: Whether to display plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {len(X_test)} test samples...")
        
        # Get predictions and scores
        predictions = self.predict(X_test)  # 1 or -1
        decision_scores = self.decision_function(X_test)
        anomaly_scores = self.score_samples(X_test)
        anomaly_probs = self.predict_proba(X_test)
        
        # Convert predictions: 1 (normal) -> 0 (benign), -1 (anomaly) -> 1 (malicious)
        y_pred = np.where(predictions == 1, 0, 1)
        
        # Compute metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, anomaly_probs)
        roc_auc = auc(fpr, tpr)
        
        # Compute Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, anomaly_probs)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'accuracy': float(np.mean(y_pred == y_test)),
            'precision_benign': report['0']['precision'],
            'recall_benign': report['0']['recall'],
            'f1_benign': report['0']['f1-score'],
            'precision_malicious': report['1']['precision'],
            'recall_malicious': report['1']['recall'],
            'f1_malicious': report['1']['f1-score'],
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'anomaly_score_stats': {
                'mean': float(np.mean(anomaly_scores)),
                'std': float(np.std(anomaly_scores)),
                'min': float(np.min(anomaly_scores)),
                'max': float(np.max(anomaly_scores))
            },
            'decision_score_stats': {
                'mean': float(np.mean(decision_scores)),
                'std': float(np.std(decision_scores)),
                'min': float(np.min(decision_scores)),
                'max': float(np.max(decision_scores))
            }
        }
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Malicious Detection Rate (Recall): {metrics['recall_malicious']:.4f}")
        logger.info(f"False Positive Rate: {1 - metrics['recall_benign']:.4f}")
        
        # Generate visualizations
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png', show_plot)
            self._plot_roc_curve(fpr, tpr, roc_auc, output_dir / 'roc_curve.png', show_plot)
            self._plot_pr_curve(recall, precision, pr_auc, output_dir / 'pr_curve.png', show_plot)
            self._plot_score_distribution(
                anomaly_scores, y_test, 
                output_dir / 'anomaly_score_distribution.png', 
                show_plot
            )
            self._plot_decision_distribution(
                decision_scores, y_test,
                output_dir / 'decision_distribution.png',
                show_plot
            )
            self._plot_anomaly_probability(
                anomaly_probs, y_test,
                output_dir / 'anomaly_probability.png',
                show_plot
            )
            
            # Save metrics
            with open(output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Results saved to {output_dir}")
        
        return metrics
    
    def _plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        save_path: Path,
        show_plot: bool = False
    ):
        """Plot confusion matrix with both counts and percentages."""
        # Create figure with custom size
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations with both count and percentage
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = cm_normalized[i, j] * 100
                annotations[i, j] = f'{count:,}\n({percentage:.1f}%)'
        
        # Plot heatmap
        sns.heatmap(
            cm_normalized,  # Use normalized values for color intensity
            annot=annotations,  # Use custom annotations
            fmt='',  # Empty format since we're using custom strings
            cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'],
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Proportion'}
        )
        
        plt.title('Isolation Forest Confusion Matrix\n(Count and Percentage)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def _plot_roc_curve(
        self, 
        fpr: np.ndarray, 
        tpr: np.ndarray, 
        roc_auc: float,
        save_path: Path,
        show_plot: bool = False
    ):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Isolation Forest ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"ROC curve saved to {save_path}")
    
    def _plot_pr_curve(
        self, 
        recall: np.ndarray, 
        precision: np.ndarray, 
        pr_auc: float,
        save_path: Path,
        show_plot: bool = False
    ):
        """Plot Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Isolation Forest Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"PR curve saved to {save_path}")
    
    def _plot_score_distribution(
        self, 
        anomaly_scores: np.ndarray, 
        y_true: np.ndarray,
        save_path: Path,
        show_plot: bool = False
    ):
        """Plot distribution of anomaly scores (lower = more anomalous)."""
        plt.figure(figsize=(10, 6))
        
        benign_scores = anomaly_scores[y_true == 0]
        malicious_scores = anomaly_scores[y_true == 1]
        
        plt.hist(benign_scores, bins=50, alpha=0.6, label='Benign', color='green', density=True)
        plt.hist(malicious_scores, bins=50, alpha=0.6, label='Malicious', color='red', density=True)
        
        # Add training mean as reference
        plt.axvline(
            x=self.training_stats['anomaly_score_mean'], 
            color='blue', linestyle='--', linewidth=2, 
            label=f"Training Mean ({self.training_stats['anomaly_score_mean']:.3f})"
        )
        
        plt.xlabel('Anomaly Score (lower = more anomalous)')
        plt.ylabel('Density')
        plt.title('Distribution of Isolation Forest Anomaly Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"Anomaly score distribution saved to {save_path}")
    
    def _plot_decision_distribution(
        self, 
        decision_scores: np.ndarray, 
        y_true: np.ndarray,
        save_path: Path,
        show_plot: bool = False
    ):
        """Plot distribution of decision function values."""
        plt.figure(figsize=(10, 6))
        
        benign_scores = decision_scores[y_true == 0]
        malicious_scores = decision_scores[y_true == 1]
        
        plt.hist(benign_scores, bins=50, alpha=0.6, label='Benign', color='green', density=True)
        plt.hist(malicious_scores, bins=50, alpha=0.6, label='Malicious', color='red', density=True)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        
        plt.xlabel('Decision Function Value')
        plt.ylabel('Density')
        plt.title('Distribution of Decision Scores (negative = anomaly)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"Decision distribution saved to {save_path}")
    
    def _plot_anomaly_probability(
        self, 
        anomaly_probs: np.ndarray, 
        y_true: np.ndarray,
        save_path: Path,
        show_plot: bool = False
    ):
        """Plot distribution of anomaly probability scores."""
        plt.figure(figsize=(10, 6))
        
        benign_probs = anomaly_probs[y_true == 0]
        malicious_probs = anomaly_probs[y_true == 1]
        
        plt.hist(benign_probs, bins=50, alpha=0.6, label='Benign', color='green', density=True)
        plt.hist(malicious_probs, bins=50, alpha=0.6, label='Malicious', color='red', density=True)
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        
        plt.xlabel('Anomaly Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Probabilities')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"Anomaly probability distribution saved to {save_path}")
    
    def save_model(self, filepath: Path):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'config': {
                'n_estimators': self.n_estimators,
                'max_samples': self.max_samples,
                'contamination': self.contamination,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'verbose': self.verbose
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'IsolationForestAnomalyDetector':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded IsolationForestAnomalyDetector instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct detector
        config = model_data['config']
        detector = cls(
            n_estimators=config['n_estimators'],
            max_samples=config['max_samples'],
            contamination=config['contamination'],
            max_features=config['max_features'],
            bootstrap=config['bootstrap'],
            n_jobs=config['n_jobs'],
            random_state=config['random_state'],
            verbose=config['verbose']
        )
        
        # Restore trained components
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_names = model_data['feature_names']
        detector.training_stats = model_data['training_stats']
        detector.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return detector
    
    def get_anomaly_ranking(
        self, 
        X: pd.DataFrame, 
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Get top N most anomalous samples.
        
        Args:
            X: Features to rank
            top_n: Number of top anomalies to return
            
        Returns:
            DataFrame with indices and anomaly scores
        """
        anomaly_scores = self.score_samples(X)
        decision_scores = self.decision_function(X)
        anomaly_probs = self.predict_proba(X)
        
        # Create ranking dataframe
        ranking = pd.DataFrame({
            'index': X.index if isinstance(X, pd.DataFrame) else range(len(X)),
            'anomaly_score': anomaly_scores,
            'decision_score': decision_scores,
            'anomaly_probability': anomaly_probs
        })
        
        # Sort by most anomalous (lowest anomaly score = easiest to isolate)
        ranking = ranking.sort_values('anomaly_score', ascending=True)
        
        return ranking.head(top_n)
    
    def explain_anomaly(
        self, 
        X: pd.DataFrame, 
        sample_idx: int
    ) -> Dict[str, Any]:
        """
        Explain why a specific sample was classified as anomalous.
        
        Args:
            X: Feature dataframe
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        sample = X.iloc[[sample_idx]]
        anomaly_score = self.score_samples(sample)[0]
        decision_score = self.decision_function(sample)[0]
        anomaly_prob = self.predict_proba(sample)[0]
        prediction = self.predict(sample)[0]
        
        # Get feature values
        feature_values = sample.iloc[0].to_dict()
        
        # Compare to training statistics
        explanation = {
            'sample_index': sample_idx,
            'anomaly_score': float(anomaly_score),
            'decision_score': float(decision_score),
            'anomaly_probability': float(anomaly_prob),
            'prediction': 'Anomaly' if prediction == -1 else 'Normal',
            'is_anomalous': bool(prediction == -1),
            'feature_values': {k: float(v) for k, v in feature_values.items()},
            'comparison_to_training': {
                'anomaly_score_vs_mean': float(
                    anomaly_score - self.training_stats['anomaly_score_mean']
                ),
                'std_deviations_from_mean': float(
                    (anomaly_score - self.training_stats['anomaly_score_mean']) / 
                    self.training_stats['anomaly_score_std']
                ),
                'easier_to_isolate_than_training_avg': bool(
                    anomaly_score < self.training_stats['anomaly_score_mean']
                )
            },
            'interpretation': self._interpret_score(anomaly_score, decision_score)
        }
        
        return explanation
    
    def _interpret_score(self, anomaly_score: float, decision_score: float) -> str:
        """
        Provide human-readable interpretation of scores.
        
        Args:
            anomaly_score: Raw anomaly score (average path length)
            decision_score: Decision function value
            
        Returns:
            Interpretation string
        """
        training_mean = self.training_stats['anomaly_score_mean']
        training_std = self.training_stats['anomaly_score_std']
        
        std_devs = (anomaly_score - training_mean) / training_std
        
        if decision_score < 0:
            severity = "HIGHLY ANOMALOUS" if std_devs < -2 else "ANOMALOUS"
            return (
                f"{severity}: This sample is easier to isolate than normal traffic "
                f"({abs(std_devs):.2f} std devs below training average). "
                f"Decision score: {decision_score:.4f} (negative = outlier)."
            )
        else:
            return (
                f"NORMAL: This sample has characteristics similar to benign traffic. "
                f"Decision score: {decision_score:.4f} (positive = inlier). "
                f"Anomaly score {std_devs:.2f} std devs from training mean."
            )
    
    def get_feature_importance_proxy(self, X: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
        """
        Approximate feature importance by perturbation analysis.
        
        Note: Isolation Forest doesn't provide native feature importance,
        so we estimate it by measuring how much each feature affects predictions.
        
        Args:
            X: Sample data to analyze
            n_samples: Number of samples to use for analysis
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        logger.info(f"Computing feature importance proxy on {n_samples} samples...")
        
        # Sample data if needed
        if len(X) > n_samples:
            X_sample = X.sample(n=n_samples, random_state=self.random_state)
        else:
            X_sample = X.copy()
        
        # Get baseline scores
        baseline_scores = self.score_samples(X_sample)
        
        # Compute importance by permutation
        importances = []
        feature_names = X_sample.columns if isinstance(X_sample, pd.DataFrame) else [f"feature_{i}" for i in range(X_sample.shape[1])]
        
        for i, feature in enumerate(feature_names):
            # Permute feature
            X_permuted = X_sample.copy()
            if isinstance(X_permuted, pd.DataFrame):
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            else:
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Get scores with permuted feature
            permuted_scores = self.score_samples(X_permuted)
            
            # Importance = change in anomaly score
            importance = np.mean(np.abs(baseline_scores - permuted_scores))
            importances.append(importance)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 important features: {', '.join(importance_df.head(5)['feature'].tolist())}")
        
        return importance_df
