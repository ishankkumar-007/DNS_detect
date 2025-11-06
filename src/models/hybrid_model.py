"""
Hybrid DNS Anomaly Detection Model
Combines Supervised (Random Forest) and Unsupervised (Isolation Forest) Learning

This model leverages the strengths of both approaches:
- Random Forest: Excellent precision on known attack patterns
- Isolation Forest: Detects novel/zero-day attacks without labels

Fusion Strategies:
1. Voting: Weighted majority vote or soft voting
2. Anomaly-Aware: Use IF anomaly scores to adjust RF predictions
3. Two-Stage: IF filters suspicious samples, RF classifies them
4. Weighted Average: Combine probability scores with weights
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import logging

from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridDetector:
    """
    Hybrid detector combining supervised and unsupervised models.
    
    This class implements multiple fusion strategies to combine predictions
    from a Random Forest (supervised) and Isolation Forest (unsupervised).
    
    Key Features:
    - Multiple fusion strategies (voting, anomaly-aware, two-stage)
    - Confidence score calibration
    - Explainable predictions
    - Model agreement analysis
    """
    
    def __init__(
        self,
        supervised_model: Any,
        unsupervised_model: Any,
        fusion_strategy: str = "anomaly_aware",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hybrid detector.
        
        Args:
            supervised_model: Trained supervised model (e.g., Random Forest)
            unsupervised_model: Trained unsupervised model (e.g., Isolation Forest)
            fusion_strategy: Strategy to combine predictions
                           Options: "voting", "anomaly_aware", "two_stage", "weighted_average"
            config: Configuration dictionary with fusion parameters
        """
        self.supervised_model = supervised_model
        self.unsupervised_model = unsupervised_model
        self.fusion_strategy = fusion_strategy
        self.config = config or {}
        
        # Extract specific model objects if they're wrapped in dict/class
        self.rf_model = self._extract_model(supervised_model)
        self.if_model = self._extract_model(unsupervised_model)
        
        # Training statistics
        self.training_stats = {
            'supervised_model_type': type(self.rf_model).__name__,
            'unsupervised_model_type': type(self.if_model).__name__,
            'fusion_strategy': fusion_strategy,
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"Initialized Hybrid Detector with fusion strategy: {fusion_strategy}")
        logger.info(f"  Supervised model: {type(self.rf_model).__name__}")
        logger.info(f"  Unsupervised model: {type(self.if_model).__name__}")
    
    def _extract_model(self, model: Any) -> Any:
        """
        Extract the actual model object from various wrapper formats.
        
        Handles:
        - Wrapper classes (RandomForestDetector, IsolationForestAnomalyDetector)
        - Dictionary with 'model' key
        - Object with 'model' attribute  
        - Direct model object
        """
        # Check if it's already a wrapper class with predict and predict_proba
        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            # Check if it's one of our wrapper classes
            class_name = type(model).__name__
            if class_name in ['RandomForestDetector', 'IsolationForestAnomalyDetector']:
                logger.info(f"Using wrapper class directly: {class_name}")
                return model
        
        # Otherwise, try to extract
        if isinstance(model, dict):
            if 'model' in model:
                extracted = model['model']
                logger.info(f"Extracted model from dict: {type(extracted).__name__}")
                return extracted
            else:
                raise ValueError(f"Dictionary doesn't contain 'model' key: {model.keys()}")
        elif hasattr(model, 'model'):
            extracted = model.model
            logger.info(f"Extracted model from attribute: {type(extracted).__name__}")
            return extracted
        else:
            logger.info(f"Using model directly: {type(model).__name__}")
            return model
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels using hybrid approach.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels (0 = benign, 1 = malicious)
        """
        # Get probabilities
        proba = self.predict_proba(X)
        
        # Apply threshold
        threshold = self.config.get('hybrid', {}).get('decision', {}).get('threshold', 0.5)
        return (proba[:, 1] >= threshold).astype(int)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using hybrid approach.
        
        Args:
            X: Features to predict
            
        Returns:
            Probabilities for each class [P(benign), P(malicious)]
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(X_array)
        
        # Get IF predictions - convert from {1, -1} to probabilities
        if_predictions = self.if_model.predict(X_array)
        if_anomaly_scores = self.if_model.score_samples(X_array)
        
        # Convert IF to probabilities using predict_proba if available
        if hasattr(self.if_model, 'predict_proba'):
            if_proba_malicious = self.if_model.predict_proba(X_array)
        else:
            # Convert anomaly scores to probabilities
            # Lower score = more anomalous = higher malicious probability
            if_proba_malicious = self._convert_anomaly_scores_to_proba(if_anomaly_scores)
        
        # Apply fusion strategy
        if self.fusion_strategy == "voting":
            hybrid_proba = self._voting_fusion(rf_proba, if_proba_malicious, if_predictions)
        elif self.fusion_strategy == "anomaly_aware":
            hybrid_proba = self._anomaly_aware_fusion(rf_proba, if_proba_malicious, if_anomaly_scores)
        elif self.fusion_strategy == "two_stage":
            hybrid_proba = self._two_stage_fusion(rf_proba, if_proba_malicious, if_predictions)
        elif self.fusion_strategy == "weighted_average":
            hybrid_proba = self._weighted_average_fusion(rf_proba, if_proba_malicious)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return hybrid_proba
    
    def _convert_anomaly_scores_to_proba(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Convert IF anomaly scores to malicious probabilities.
        
        Isolation Forest scores:
        - Lower (more negative) = more anomalous = higher malicious probability
        - Higher (closer to 0) = more normal = lower malicious probability
        """
        # Get training statistics if available
        if hasattr(self.if_model, 'training_stats'):
            score_min = self.if_model.training_stats.get('anomaly_score_min', -0.7)
            score_max = self.if_model.training_stats.get('anomaly_score_max', -0.35)
        else:
            # Use data-driven min/max
            score_min = anomaly_scores.min()
            score_max = anomaly_scores.max()
        
        # Normalize and invert: lower scores = higher probability
        normalized = (anomaly_scores - score_min) / (score_max - score_min + 1e-10)
        proba_malicious = 1 - normalized  # Invert so low scores = high probability
        
        return np.clip(proba_malicious, 0, 1)
    
    def _voting_fusion(
        self, 
        rf_proba: np.ndarray, 
        if_proba_malicious: np.ndarray,
        if_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Voting-based fusion strategy.
        
        Combines RF and IF predictions using weighted voting.
        """
        voting_config = self.config.get('hybrid', {}).get('voting', {})
        method = voting_config.get('method', 'soft')
        weights = voting_config.get('weights', [0.7, 0.3])
        
        if method == 'soft':
            # Soft voting: weighted average of probabilities
            # Convert IF anomaly to 2-class probability
            if isinstance(if_proba_malicious, np.ndarray) and if_proba_malicious.ndim == 1:
                if_proba = np.column_stack([1 - if_proba_malicious, if_proba_malicious])
            else:
                if_proba = if_proba_malicious
            
            # Weighted average
            hybrid_proba = (weights[0] * rf_proba + weights[1] * if_proba) / sum(weights)
        else:
            # Hard voting: majority vote
            rf_pred = (rf_proba[:, 1] >= 0.5).astype(int)
            if_pred = (if_predictions == -1).astype(int)  # -1 = anomaly = malicious
            
            # Weighted vote
            votes = weights[0] * rf_pred + weights[1] * if_pred
            final_pred = (votes >= (sum(weights) / 2)).astype(int)
            
            # Convert to probabilities
            hybrid_proba = np.column_stack([1 - final_pred, final_pred])
        
        return hybrid_proba
    
    def _anomaly_aware_fusion(
        self, 
        rf_proba: np.ndarray, 
        if_proba_malicious: np.ndarray,
        if_anomaly_scores: np.ndarray
    ) -> np.ndarray:
        """
        Anomaly-aware fusion strategy.
        
        Uses IF anomaly scores to boost RF malicious probability when
        samples are highly anomalous.
        """
        aa_config = self.config.get('hybrid', {}).get('anomaly_aware', {})
        anomaly_threshold = aa_config.get('anomaly_threshold', 0.6)
        boost_factor = aa_config.get('anomaly_boost_factor', 1.5)
        min_rf_confidence = aa_config.get('min_rf_confidence', 0.5)
        
        # Start with RF probabilities
        hybrid_proba = rf_proba.copy()
        
        # Identify highly anomalous samples
        if isinstance(if_proba_malicious, np.ndarray) and if_proba_malicious.ndim == 1:
            is_anomalous = if_proba_malicious > anomaly_threshold
        else:
            is_anomalous = if_proba_malicious[:, 1] > anomaly_threshold
        
        # For anomalous samples with low RF confidence, boost malicious probability
        low_confidence_mask = rf_proba[:, 1] < min_rf_confidence
        boost_mask = is_anomalous & low_confidence_mask
        
        if boost_mask.any():
            # Boost malicious probability
            if isinstance(if_proba_malicious, np.ndarray) and if_proba_malicious.ndim == 1:
                if_mal_prob = if_proba_malicious
            else:
                if_mal_prob = if_proba_malicious[:, 1]
            
            boosted_prob = np.minimum(
                rf_proba[:, 1] + (if_mal_prob * boost_factor * 0.3),  # Scale boost
                1.0
            )
            hybrid_proba[boost_mask, 1] = boosted_prob[boost_mask]
            hybrid_proba[boost_mask, 0] = 1 - hybrid_proba[boost_mask, 1]
        
        return hybrid_proba
    
    def _two_stage_fusion(
        self, 
        rf_proba: np.ndarray, 
        if_proba_malicious: np.ndarray,
        if_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Two-stage fusion strategy.
        
        Stage 1: IF identifies potential anomalies
        Stage 2: RF classifies the flagged samples
        """
        ts_config = self.config.get('hybrid', {}).get('two_stage', {})
        if_threshold = ts_config.get('if_threshold', 0.5)
        logic = ts_config.get('logic', 'if_then_rf')
        
        if logic == 'if_then_rf':
            # Start with all benign
            hybrid_proba = np.zeros_like(rf_proba)
            hybrid_proba[:, 0] = 1.0  # All benign initially
            
            # Identify anomalies detected by IF
            is_anomaly = if_predictions == -1
            
            # For anomalies, use RF classification
            hybrid_proba[is_anomaly] = rf_proba[is_anomaly]
        else:
            # RF with IF filter: only trust RF when IF also suspicious
            if isinstance(if_proba_malicious, np.ndarray) and if_proba_malicious.ndim == 1:
                if_suspicious = if_proba_malicious > if_threshold
            else:
                if_suspicious = if_proba_malicious[:, 1] > if_threshold
            
            hybrid_proba = rf_proba.copy()
            
            # For samples both models agree are suspicious, increase confidence
            rf_suspicious = rf_proba[:, 1] > 0.5
            both_suspicious = if_suspicious & rf_suspicious
            
            if both_suspicious.any():
                # Average the probabilities for higher confidence
                hybrid_proba[both_suspicious, 1] = np.minimum(
                    (rf_proba[both_suspicious, 1] + if_proba_malicious[both_suspicious]) / 2 * 1.2,
                    1.0
                )
                hybrid_proba[both_suspicious, 0] = 1 - hybrid_proba[both_suspicious, 1]
        
        return hybrid_proba
    
    def _weighted_average_fusion(
        self, 
        rf_proba: np.ndarray, 
        if_proba_malicious: np.ndarray
    ) -> np.ndarray:
        """
        Weighted average fusion strategy.
        
        Simple weighted average of probability scores.
        """
        wa_config = self.config.get('hybrid', {}).get('weighted_average', {})
        
        # Get weights
        supervised_weight = self.config.get('hybrid', {}).get('supervised_model', {}).get('weight', 0.7)
        unsupervised_weight = self.config.get('hybrid', {}).get('unsupervised_model', {}).get('weight', 0.3)
        
        normalize = wa_config.get('normalize_weights', True)
        
        # Convert IF to 2-class probability if needed
        if isinstance(if_proba_malicious, np.ndarray) and if_proba_malicious.ndim == 1:
            if_proba = np.column_stack([1 - if_proba_malicious, if_proba_malicious])
        else:
            if_proba = if_proba_malicious
        
        # Weighted average
        if normalize:
            total_weight = supervised_weight + unsupervised_weight
            hybrid_proba = (supervised_weight * rf_proba + unsupervised_weight * if_proba) / total_weight
        else:
            hybrid_proba = supervised_weight * rf_proba + unsupervised_weight * if_proba
        
        return hybrid_proba
    
    def evaluate(
        self, 
        X_test: Union[pd.DataFrame, np.ndarray], 
        y_test: np.ndarray,
        output_dir: Optional[Path] = None,
        show_plot: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate hybrid model on test data.
        
        Args:
            X_test: Test features
            y_test: True labels
            output_dir: Directory to save plots and results
            show_plot: Whether to display plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating hybrid model with {self.fusion_strategy} fusion...")
        
        # Get predictions from all models
        y_pred_hybrid = self.predict(X_test)
        y_proba_hybrid = self.predict_proba(X_test)
        
        # Get individual model predictions for comparison
        X_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_pred_rf = self.rf_model.predict(X_array)
        y_proba_rf = self.rf_model.predict_proba(X_array)
        
        # Get IF predictions and convert properly
        y_pred_if_raw = self.if_model.predict(X_array)
        
        # Debug: Check what IF is returning
        logger.info(f"IF raw predictions - Unique values: {np.unique(y_pred_if_raw)}")
        logger.info(f"IF raw predictions - Counts: {np.bincount(y_pred_if_raw.astype(int) + 1)}")  # +1 to handle -1 index
        
        # Convert IF predictions: 1 (normal/inlier) -> 0 (benign), -1 (anomaly/outlier) -> 1 (malicious)
        y_pred_if = np.where(y_pred_if_raw == -1, 1, 0)  # Fixed: -1 is anomaly (malicious), 1 is normal (benign)
        
        logger.info(f"IF converted predictions - Benign: {(y_pred_if == 0).sum()}, Malicious: {(y_pred_if == 1).sum()}")
        logger.info(f"True labels - Benign: {(y_test == 0).sum()}, Malicious: {(y_test == 1).sum()}")
        
        # Get IF probabilities for ROC-AUC calculation
        if hasattr(self.if_model, 'predict_proba'):
            if_proba_raw = self.if_model.predict_proba(X_array)
            # Check if it's 1D or 2D
            if if_proba_raw.ndim == 1:
                # Single anomaly probability - convert to 2-class format
                y_proba_if = np.column_stack([1 - if_proba_raw, if_proba_raw])
            else:
                y_proba_if = if_proba_raw
        else:
            y_proba_if = None
        
        # Calculate metrics for hybrid model
        metrics = self._calculate_metrics(y_test, y_pred_hybrid, y_proba_hybrid, prefix='hybrid')
        
        # Calculate metrics for base models
        metrics_rf = self._calculate_metrics(y_test, y_pred_rf, y_proba_rf, prefix='rf')
        metrics_if = self._calculate_metrics(y_test, y_pred_if, y_proba_if, prefix='if')
        
        # Add base model metrics for comparison
        metrics['base_models'] = {
            'random_forest': metrics_rf,
            'isolation_forest': metrics_if
        }
        
        # Analyze model agreement
        agreement_stats = self._analyze_model_agreement(
            y_test, y_pred_hybrid, y_pred_rf, y_pred_if
        )
        metrics['agreement_analysis'] = agreement_stats
        
        # Log results
        self._log_evaluation_results(metrics)
        
        # Generate visualizations
        if output_dir or show_plot:
            self._plot_evaluation(
                y_test, y_pred_hybrid, y_proba_hybrid,
                y_pred_rf, y_pred_if,
                metrics, output_dir, show_plot
            )
        
        return metrics
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray],
        prefix: str = ''
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )
        
        metrics['precision_benign'] = float(precision[0])
        metrics['recall_benign'] = float(recall[0])
        metrics['f1_benign'] = float(f1[0])
        
        if len(precision) > 1:
            metrics['precision_malicious'] = float(precision[1])
            metrics['recall_malicious'] = float(recall[1])
            metrics['f1_malicious'] = float(f1[1])
        
        # Overall weighted metrics
        metrics['precision_weighted'] = float(
            precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[0]
        )
        metrics['recall_weighted'] = float(
            precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[1]
        )
        metrics['f1_weighted'] = float(
            precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[2]
        )
        
        # ROC-AUC and PR-AUC (if probabilities available)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                metrics['roc_auc'] = float(auc(fpr, tpr))
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba[:, 1])
                metrics['pr_auc'] = float(average_precision_score(y_true, y_proba[:, 1]))
            except Exception as e:
                logger.warning(f"Could not calculate ROC/PR curves for {prefix}: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # False positive/negative rates
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics['fnr'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def _analyze_model_agreement(
        self,
        y_true: np.ndarray,
        y_pred_hybrid: np.ndarray,
        y_pred_rf: np.ndarray,
        y_pred_if: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze agreement between models."""
        n_samples = len(y_true)
        
        # Agreement between all three
        all_agree = (y_pred_hybrid == y_pred_rf) & (y_pred_rf == y_pred_if)
        
        # Agreement between pairs
        hybrid_rf_agree = y_pred_hybrid == y_pred_rf
        hybrid_if_agree = y_pred_hybrid == y_pred_if
        rf_if_agree = y_pred_rf == y_pred_if
        
        # Disagreement cases
        all_disagree = (y_pred_hybrid != y_pred_rf) & (y_pred_rf != y_pred_if) & (y_pred_hybrid != y_pred_if)
        
        stats = {
            'all_models_agree': {
                'count': int(all_agree.sum()),
                'percentage': float(all_agree.sum() / n_samples),
                'accuracy_when_agree': float(accuracy_score(y_true[all_agree], y_pred_hybrid[all_agree])) if all_agree.any() else 0.0
            },
            'hybrid_rf_agree': {
                'count': int(hybrid_rf_agree.sum()),
                'percentage': float(hybrid_rf_agree.sum() / n_samples)
            },
            'hybrid_if_agree': {
                'count': int(hybrid_if_agree.sum()),
                'percentage': float(hybrid_if_agree.sum() / n_samples)
            },
            'rf_if_agree': {
                'count': int(rf_if_agree.sum()),
                'percentage': float(rf_if_agree.sum() / n_samples)
            },
            'all_models_disagree': {
                'count': int(all_disagree.sum()),
                'percentage': float(all_disagree.sum() / n_samples)
            }
        }
        
        return stats
    
    def _log_evaluation_results(self, metrics: Dict[str, Any]):
        """Log evaluation results to console."""
        logger.info("\n" + "="*70)
        logger.info("HYBRID MODEL EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"Fusion Strategy: {self.fusion_strategy}")
        logger.info("\nHybrid Model Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (Malicious): {metrics.get('precision_malicious', 0):.4f}")
        logger.info(f"  Recall (Malicious): {metrics.get('recall_malicious', 0):.4f}")
        logger.info(f"  F1-Score (Malicious): {metrics.get('f1_malicious', 0):.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  False Positive Rate: {metrics['fpr']:.4f}")
        logger.info(f"  False Negative Rate: {metrics['fnr']:.4f}")
        
        logger.info("\nComparison with Base Models:")
        logger.info(f"  Random Forest Accuracy: {metrics['base_models']['random_forest']['accuracy']:.4f}")
        logger.info(f"  Isolation Forest Accuracy: {metrics['base_models']['isolation_forest']['accuracy']:.4f}")
        logger.info(f"  Hybrid Improvement over RF: {(metrics['accuracy'] - metrics['base_models']['random_forest']['accuracy']):.4f}")
        logger.info(f"  Hybrid Improvement over IF: {(metrics['accuracy'] - metrics['base_models']['isolation_forest']['accuracy']):.4f}")
        
        logger.info("\nModel Agreement Analysis:")
        agree_stats = metrics['agreement_analysis']
        logger.info(f"  All models agree: {agree_stats['all_models_agree']['percentage']:.2%}")
        logger.info(f"  Accuracy when all agree: {agree_stats['all_models_agree']['accuracy_when_agree']:.4f}")
        logger.info(f"  RF-IF agreement: {agree_stats['rf_if_agree']['percentage']:.2%}")
        logger.info("="*70)
    
    def _plot_evaluation(
        self,
        y_true: np.ndarray,
        y_pred_hybrid: np.ndarray,
        y_proba_hybrid: np.ndarray,
        y_pred_rf: np.ndarray,
        y_pred_if: np.ndarray,
        metrics: Dict[str, Any],
        output_dir: Optional[Path],
        show_plot: bool
    ):
        """Generate comprehensive evaluation plots."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix - Hybrid
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_confusion_matrix(metrics['confusion_matrix'], ax1, "Hybrid Model")
        
        # 2. Confusion Matrix - RF
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_confusion_matrix(
            metrics['base_models']['random_forest']['confusion_matrix'], 
            ax2, "Random Forest"
        )
        
        # 3. Confusion Matrix - IF
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_confusion_matrix(
            metrics['base_models']['isolation_forest']['confusion_matrix'],
            ax3, "Isolation Forest"
        )
        
        # 4. Model Comparison Bar Chart
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_model_comparison(metrics, ax4)
        
        # 5. ROC Curves
        if 'roc_auc' in metrics:
            ax5 = fig.add_subplot(gs[1, 0:2])
            self._plot_roc_curves(y_true, y_proba_hybrid, metrics, ax5)
        
        # 6. PR Curves
        if 'pr_auc' in metrics:
            ax6 = fig.add_subplot(gs[1, 2:4])
            self._plot_pr_curves(y_true, y_proba_hybrid, metrics, ax6)
        
        # 7. Model Agreement Venn-style
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_agreement_analysis(metrics['agreement_analysis'], ax7)
        
        # 8. Score Distribution
        ax8 = fig.add_subplot(gs[2, 1:3])
        self._plot_score_distributions(y_true, y_proba_hybrid, y_pred_rf, y_pred_if, ax8)
        
        # 9. Performance Metrics Radar
        ax9 = fig.add_subplot(gs[2, 3])
        self._plot_metrics_comparison(metrics, ax9)
        
        plt.suptitle(f'Hybrid Model Evaluation - {self.fusion_strategy.upper()} Fusion', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save plot
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / 'hybrid_evaluation.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _plot_confusion_matrix(self, cm: list, ax: plt.Axes, title: str):
        """Plot a single confusion matrix."""
        cm_array = np.array(cm)
        cm_normalized = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
        
        annotations = np.empty_like(cm_array, dtype=object)
        for i in range(cm_array.shape[0]):
            for j in range(cm_array.shape[1]):
                count = cm_array[i, j]
                percentage = cm_normalized[i, j] * 100
                annotations[i, j] = f'{count:,}\n({percentage:.1f}%)'
        
        sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap='Blues', ax=ax,
                   xticklabels=['Benign', 'Malicious'],
                   yticklabels=['Benign', 'Malicious'],
                   vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    def _plot_model_comparison(self, metrics: Dict, ax: plt.Axes):
        """Plot accuracy comparison between models."""
        models = ['Random Forest', 'Isolation Forest', 'Hybrid']
        accuracies = [
            metrics['base_models']['random_forest']['accuracy'],
            metrics['base_models']['isolation_forest']['accuracy'],
            metrics['accuracy']
        ]
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray, 
                        metrics: Dict, ax: plt.Axes):
        """Plot ROC curves."""
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, 'b-', linewidth=2, 
               label=f'Hybrid (AUC = {metrics["roc_auc"]:.4f})')
        
        # Add base model ROC if available
        if 'roc_auc' in metrics['base_models']['random_forest']:
            ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pr_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       metrics: Dict, ax: plt.Axes):
        """Plot Precision-Recall curves."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        ax.plot(recall, precision, 'g-', linewidth=2,
               label=f'Hybrid (AUC = {metrics["pr_auc"]:.4f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_agreement_analysis(self, agreement_stats: Dict, ax: plt.Axes):
        """Plot model agreement statistics."""
        categories = ['All Agree', 'RF-IF Agree', 'All Disagree']
        percentages = [
            agreement_stats['all_models_agree']['percentage'] * 100,
            agreement_stats['rf_if_agree']['percentage'] * 100,
            agreement_stats['all_models_disagree']['percentage'] * 100
        ]
        colors = ['green', 'orange', 'red']
        
        bars = ax.bar(categories, percentages, color=colors, alpha=0.7)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Model Agreement', fontweight='bold')
        ax.set_ylim([0, 100])
        
        # Add value labels
        for bar, val in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom')
    
    def _plot_score_distributions(self, y_true: np.ndarray, y_proba_hybrid: np.ndarray,
                                  y_pred_rf: np.ndarray, y_pred_if: np.ndarray, ax: plt.Axes):
        """Plot distribution of hybrid confidence scores."""
        benign_scores = y_proba_hybrid[y_true == 0, 1]
        malicious_scores = y_proba_hybrid[y_true == 1, 1]
        
        ax.hist(benign_scores, bins=50, alpha=0.6, label='Benign (True)', 
               color='green', density=True)
        ax.hist(malicious_scores, bins=50, alpha=0.6, label='Malicious (True)', 
               color='red', density=True)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        
        ax.set_xlabel('Hybrid Malicious Probability')
        ax.set_ylabel('Density')
        ax.set_title('Hybrid Score Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_comparison(self, metrics: Dict, ax: plt.Axes):
        """Plot performance metrics comparison."""
        metric_names = ['Precision\n(Mal)', 'Recall\n(Mal)', 'F1\n(Mal)', 'Accuracy']
        
        hybrid_values = [
            metrics.get('precision_malicious', 0),
            metrics.get('recall_malicious', 0),
            metrics.get('f1_malicious', 0),
            metrics['accuracy']
        ]
        
        rf_values = [
            metrics['base_models']['random_forest'].get('precision_malicious', 0),
            metrics['base_models']['random_forest'].get('recall_malicious', 0),
            metrics['base_models']['random_forest'].get('f1_malicious', 0),
            metrics['base_models']['random_forest']['accuracy']
        ]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        ax.bar(x - width/2, rf_values, width, label='Random Forest', color='orange', alpha=0.7)
        ax.bar(x + width/2, hybrid_values, width, label='Hybrid', color='red', alpha=0.7)
        
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=8)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
    
    def save_model(self, filepath: Path, metadata: Optional[Dict] = None):
        """
        Save hybrid model configuration.
        
        Args:
            filepath: Path to save model
            metadata: Additional metadata
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'supervised_model': self.supervised_model,
            'unsupervised_model': self.unsupervised_model,
            'fusion_strategy': self.fusion_strategy,
            'config': self.config,
            'training_stats': self.training_stats
        }
        
        if metadata:
            model_data['metadata'] = metadata
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Hybrid model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'HybridDetector':
        """
        Load saved hybrid model.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded HybridDetector instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        detector = cls(
            supervised_model=model_data['supervised_model'],
            unsupervised_model=model_data['unsupervised_model'],
            fusion_strategy=model_data['fusion_strategy'],
            config=model_data['config']
        )
        
        detector.training_stats = model_data.get('training_stats', {})
        
        logger.info(f"Hybrid model loaded from: {filepath}")
        return detector
