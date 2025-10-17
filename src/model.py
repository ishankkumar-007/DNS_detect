"""
LightGBM Model Module for DNS Spoofing Detection
Handles training, evaluation, hyperparameter tuning, and model persistence
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Tuple, Optional, List
import joblib
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DNSSpoofingDetector:
    """
    LightGBM-based DNS spoofing detector with comprehensive evaluation
    """
    
    def __init__(self, params: Optional[Dict] = None, random_state: int = 42):
        """
        Initialize DNS spoofing detector
        
        Args:
            params: LightGBM parameters (optional)
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.label_names = None
        self.feature_names = None
        self.training_history = {}
        
        # Default LightGBM parameters optimized for binary DNS detection
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1,
            'is_unbalance': True  # Handle class imbalance
        }
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                     test_size: float = 0.2, stratify: bool = True) -> Tuple:
        """
        Split data into train/test sets
        
        Args:
            X: Features DataFrame
            y: Labels Series
            test_size: Proportion of test data
            stratify: Whether to stratify split by labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data: train={1-test_size:.0%}, test={test_size:.0%}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if stratify else None
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Train label distribution:\n{pd.Series(y_train).value_counts()}")
        logger.info(f"Test label distribution:\n{pd.Series(y_test).value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              early_stopping_rounds: int = 50) -> lgb.Booster:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Trained LightGBM model
        """
        logger.info("Starting model training...")
        
        # Store feature names and label info
        self.feature_names = X_train.columns.tolist()
        unique_labels = np.unique(y_train)
        # Convert numpy int64 keys to native Python int for JSON serialization
        self.label_names = {int(i): f"Class_{i}" for i in unique_labels}
        
        # Update num_class in params
        num_classes = len(unique_labels)
        self.params['num_class'] = num_classes
        
        if num_classes == 2:
            self.params['objective'] = 'binary'
            self.params['metric'] = 'binary_logloss'
            self.params.pop('num_class', None)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds)
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
        logger.info(f"Best score: {self.model.best_score}")
        
        return self.model
    
    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features DataFrame
            return_proba: Return probabilities instead of class labels
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        if return_proba:
            return predictions
        else:
            # Convert probabilities to class labels
            if len(predictions.shape) == 1:
                # Binary classification
                return (predictions > 0.5).astype(int)
            else:
                # Multi-class
                return np.argmax(predictions, axis=1)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict(X_test, return_proba=True)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC (handle both binary and multi-class)
        try:
            if len(np.unique(y_test)) == 2:
                if len(y_pred_proba.shape) == 1:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        # Log results
        logger.info("\n" + "="*50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
        logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
        logger.info("="*50)
        
        return metrics
    
    def plot_confusion_matrix(self, y_test: pd.Series, y_pred: np.ndarray,
                              save_path: Optional[str] = None, show_plot: bool = False):
        """
        Plot confusion matrix
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
            show_plot: Whether to display plot interactively
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                   xticklabels=self.label_names.values(),
                   yticklabels=self.label_names.values())
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(self, importance_type: str = 'gain', 
                                max_features: int = 20,
                                save_path: Optional[str] = None,
                                show_plot: bool = False):
        """
        Plot feature importance
        
        Args:
            importance_type: Type of importance ('gain', 'split')
            max_features: Maximum features to display
            save_path: Path to save plot
            show_plot: Whether to display plot interactively
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(max_features)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {max_features} Features by {importance_type.capitalize()} Importance',
                 fontsize=16, fontweight='bold')
        plt.xlabel(f'{importance_type.capitalize()} Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series,
                       save_path: Optional[str] = None, show_plot: bool = False):
        """
        Plot ROC curve (binary classification only)
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save plot
            show_plot: Whether to display plot interactively
        """
        if len(np.unique(y_test)) != 2:
            logger.warning("ROC curve plotting only supported for binary classification")
            return
        
        y_pred_proba = self.predict(X_test, return_proba=True)
        
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, model_path: str, metadata: Optional[Dict] = None):
        """
        Save model and metadata
        
        Args:
            model_path: Path to save model
            metadata: Additional metadata to save
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata (convert numpy types to native Python types for JSON serialization)
        metadata_dict = {
            'feature_names': self.feature_names,
            'label_names': self.label_names,
            'params': self.params,
            'best_iteration': int(self.model.best_iteration) if self.model.best_iteration else None,
            'best_score': self._convert_to_native_types(self.model.best_score),
        }
        
        if metadata:
            metadata_dict.update(self._convert_to_native_types(metadata))
        
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def _convert_to_native_types(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization
        
        Args:
            obj: Object to convert
            
        Returns:
            Converted object with native Python types
        """
        if isinstance(obj, dict):
            return {self._convert_to_native_types(k): self._convert_to_native_types(v) 
                    for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def load_model(self, model_path: str):
        """
        Load saved model and metadata
        
        Args:
            model_path: Path to saved model
        """
        model_path = Path(model_path)
        
        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(model_path))
        logger.info(f"Model loaded from {model_path}")
        
        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get('feature_names')
            self.label_names = metadata.get('label_names')
            self.params = metadata.get('params')
            
            logger.info(f"Metadata loaded from {metadata_path}")


def main():
    """Test model module"""
    logger.info("Model module ready")


if __name__ == "__main__":
    main()
