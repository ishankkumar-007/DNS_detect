"""
Unsupervised Anomaly Detection for DNS Traffic
One-Class SVM approach for detecting malicious DNS behavior without labeled data
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


class OneClassSVMAnomalyDetector:
    """
    One-Class SVM for unsupervised anomaly detection in DNS traffic.
    
    This approach assumes normal (benign) traffic is densely distributed
    and learns to identify the boundary. Traffic outside this boundary
    is classified as anomalous (potentially malicious).
    
    Key Parameters:
    - kernel: 'rbf' (Radial Basis Function) for non-linear decision boundary
    - nu: Upper bound on fraction of outliers (0.01-0.1 typical)
    - gamma: Kernel coefficient (auto uses 1/n_features)
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.05,
        gamma: str = 'scale',
        contamination: float = 0.05,
        use_pca: bool = True,
        n_components: int = 30,
        random_state: int = 42
    ):
        """
        Initialize One-Class SVM detector.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            nu: Upper bound on fraction of training errors (0-1)
                Should approximate expected % of outliers in training
            gamma: Kernel coefficient ('scale', 'auto', or float)
            contamination: Expected proportion of outliers (for evaluation)
            use_pca: Whether to apply PCA for dimensionality reduction
            n_components: Number of PCA components if use_pca=True
            random_state: Random seed for reproducibility
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.contamination = contamination
        self.use_pca = use_pca
        self.n_components = n_components
        self.random_state = random_state
        
        # Initialize models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state) if use_pca else None
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )
        
        # Training metadata
        self.is_fitted = False
        self.feature_names = None
        self.training_stats = {}
        
        logger.info(f"Initialized One-Class SVM with kernel={kernel}, nu={nu}, gamma={gamma}")
    
    def fit(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> 'OneClassSVMAnomalyDetector':
        """
        Train the One-Class SVM on normal (benign) traffic only.
        
        Important: X should contain ONLY benign/normal traffic.
        The model learns the characteristics of normal behavior.
        
        Args:
            X: Training features (benign traffic only)
            feature_names: List of feature names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training One-Class SVM on {len(X)} samples...")
        
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
        
        # Step 2: Apply PCA for dimensionality reduction (optional)
        if self.use_pca:
            logger.info(f"Applying PCA to reduce from {X.shape[1]} to {self.n_components} components...")
            X_transformed = self.pca.fit_transform(X_scaled)
            
            # Log explained variance
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA explained variance: {explained_var:.4f}")
            self.training_stats['pca_explained_variance'] = float(explained_var)
        else:
            X_transformed = X_scaled
        
        # Step 3: Train One-Class SVM
        logger.info("Training One-Class SVM model...")
        self.model.fit(X_transformed)
        
        # Step 4: Compute training set statistics
        decisions = self.model.decision_function(X_transformed)
        self.training_stats['decision_mean'] = float(np.mean(decisions))
        self.training_stats['decision_std'] = float(np.std(decisions))
        self.training_stats['decision_min'] = float(np.min(decisions))
        self.training_stats['decision_max'] = float(np.max(decisions))
        
        # Get support vectors
        n_support = len(self.model.support_vectors_)
        self.training_stats['n_support_vectors'] = int(n_support)
        self.training_stats['support_vector_ratio'] = float(n_support / len(X))
        
        self.is_fitted = True
        logger.info(f"Training complete! Support vectors: {n_support}/{len(X)} ({n_support/len(X):.2%})")
        
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
        if self.use_pca:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        return self.model.predict(X_transformed)
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores (decision function values).
        
        Negative values indicate anomalies.
        More negative = more anomalous.
        
        Args:
            X: Test features
            
        Returns:
            Decision scores (negative = anomaly, positive = normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        return self.model.decision_function(X_transformed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert decision function to pseudo-probabilities.
        
        Args:
            X: Test features
            
        Returns:
            Anomaly probabilities (0-1, higher = more likely anomalous)
        """
        decisions = self.decision_function(X)
        
        # Normalize decision scores to [0, 1] range
        # More negative = higher anomaly probability
        min_score = self.training_stats['decision_min']
        max_score = self.training_stats['decision_max']
        
        # Invert and normalize: negative scores become high probabilities
        probabilities = 1 - (decisions - min_score) / (max_score - min_score)
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
            self._plot_decision_distribution(
                decision_scores, y_test, 
                output_dir / 'decision_distribution.png', 
                show_plot
            )
            self._plot_anomaly_scores(
                anomaly_probs, y_test,
                output_dir / 'anomaly_scores.png',
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
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious']
        )
        plt.title('One-Class SVM Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
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
        plt.title('One-Class SVM ROC Curve')
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
        plt.title('One-Class SVM Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"PR curve saved to {save_path}")
    
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
        plt.title('Distribution of Decision Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"Decision distribution saved to {save_path}")
    
    def _plot_anomaly_scores(
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
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        logger.info(f"Anomaly scores saved to {save_path}")
    
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
            'pca': self.pca,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'config': {
                'kernel': self.kernel,
                'nu': self.nu,
                'gamma': self.gamma,
                'contamination': self.contamination,
                'use_pca': self.use_pca,
                'n_components': self.n_components,
                'random_state': self.random_state
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'OneClassSVMAnomalyDetector':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded OneClassSVMAnomalyDetector instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct detector
        config = model_data['config']
        detector = cls(
            kernel=config['kernel'],
            nu=config['nu'],
            gamma=config['gamma'],
            contamination=config['contamination'],
            use_pca=config['use_pca'],
            n_components=config['n_components'],
            random_state=config['random_state']
        )
        
        # Restore trained components
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.pca = model_data['pca']
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
        decision_scores = self.decision_function(X)
        anomaly_probs = self.predict_proba(X)
        
        # Create ranking dataframe
        ranking = pd.DataFrame({
            'index': X.index if isinstance(X, pd.DataFrame) else range(len(X)),
            'decision_score': decision_scores,
            'anomaly_probability': anomaly_probs
        })
        
        # Sort by most anomalous (most negative decision score)
        ranking = ranking.sort_values('decision_score', ascending=True)
        
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
        decision_score = self.decision_function(sample)[0]
        anomaly_prob = self.predict_proba(sample)[0]
        prediction = self.predict(sample)[0]
        
        # Get feature values
        feature_values = sample.iloc[0].to_dict()
        
        # Compare to training statistics
        explanation = {
            'sample_index': sample_idx,
            'decision_score': float(decision_score),
            'anomaly_probability': float(anomaly_prob),
            'prediction': 'Anomaly' if prediction == -1 else 'Normal',
            'is_anomalous': bool(prediction == -1),
            'feature_values': {k: float(v) for k, v in feature_values.items()},
            'deviation_from_normal': {
                'decision_score_vs_mean': float(
                    decision_score - self.training_stats['decision_mean']
                ),
                'std_deviations': float(
                    (decision_score - self.training_stats['decision_mean']) / 
                    self.training_stats['decision_std']
                )
            }
        }
        
        return explanation


def compare_supervised_vs_unsupervised(
    supervised_metrics: Dict[str, float],
    unsupervised_metrics: Dict[str, float],
    output_path: Optional[Path] = None,
    show_plot: bool = False
):
    """
    Compare performance of supervised vs unsupervised approaches.
    
    Args:
        supervised_metrics: Metrics from LightGBM (supervised)
        unsupervised_metrics: Metrics from One-Class SVM (unsupervised)
        output_path: Path to save comparison plot
        show_plot: Whether to display plot
    """
    metrics = ['accuracy', 'precision_malicious', 'recall_malicious', 'f1_malicious', 'roc_auc']
    supervised_values = [supervised_metrics.get(m, 0) for m in metrics]
    unsupervised_values = [unsupervised_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, supervised_values, width, label='Supervised (LightGBM)', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, unsupervised_values, width, label='Unsupervised (One-Class SVM)', color='red', alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Supervised vs Unsupervised DNS Anomaly Detection')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
