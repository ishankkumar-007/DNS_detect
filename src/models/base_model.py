"""
Base Model Abstract Class for DNS Spoofing Detection
====================================================

This module defines the abstract base class that all detection models
must inherit from to ensure consistent interface across different
model types (supervised, unsupervised, deep learning, ensemble).

Author: DNS Spoofing Detection Project
Date: 2024
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime


class BaseDetector(ABC):
    """
    Abstract base class for all DNS spoofing detection models.
    
    This class enforces a consistent interface across different model types
    while allowing flexibility for model-specific implementations.
    
    Attributes:
        model_name (str): Name of the model
        model_type (str): Type of model ('supervised', 'unsupervised', 'deep_learning', 'ensemble')
        config (Dict[str, Any]): Model configuration dictionary
        model: The underlying model object (sklearn, keras, etc.)
        is_trained (bool): Whether the model has been trained
        feature_names (List[str]): Names of features used by the model
        metadata (Dict[str, Any]): Additional metadata (training time, version, etc.)
    """
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        """
        Initialize base detector.
        
        Args:
            config: Configuration dictionary containing model parameters
            model_name: Optional custom name for the model
        """
        self.config = config
        self.model_name = model_name or config.get('model_name', self.__class__.__name__)
        self.model_type = config.get('model_type', 'unknown')
        
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
        # Metadata
        self.metadata = {
            'model_class': self.__class__.__name__,
            'model_type': self.model_type,
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'trained_at': None,
            'training_duration_seconds': None,
            'n_features': None,
            'n_samples_trained': None
        }
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build/initialize the model architecture.
        
        This method should create the underlying model object
        (e.g., LightGBM classifier, OneClassSVM, Keras model, etc.)
        
        Returns:
            The initialized model object
        """
        pass
    
    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Optional[Union[np.ndarray, pd.Series]] = None,
              **kwargs) -> 'BaseDetector':
        """
        Train the model on provided data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (optional for unsupervised models)
            **kwargs: Additional training parameters
        
        Returns:
            Self (for method chaining)
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (if supported).
        
        Args:
            X: Features to predict on (n_samples, n_features)
        
        Returns:
            Predicted probabilities (n_samples, n_classes)
            For unsupervised models, may return anomaly scores
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 **kwargs) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            **kwargs: Additional evaluation parameters
        
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self.metadata['saved_at'] = datetime.now().isoformat()
        
        # Save model object
        model_path = filepath.with_suffix('.pkl')
        joblib.dump(self.model, model_path)
        
        # Save metadata and config
        metadata_path = filepath.with_suffix('.json')
        save_data = {
            'metadata': self.metadata,
            'config': self.config,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Metadata saved to {metadata_path}")
    
    def load(self, filepath: Union[str, Path]) -> 'BaseDetector':
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
        
        Returns:
            Self (for method chaining)
        """
        filepath = Path(filepath)
        
        # Load model object
        model_path = filepath.with_suffix('.pkl')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                save_data = json.load(f)
            
            self.metadata = save_data.get('metadata', {})
            self.model_name = save_data.get('model_name', self.model_name)
            self.model_type = save_data.get('model_type', self.model_type)
            self.is_trained = save_data.get('is_trained', True)
            self.feature_names = save_data.get('feature_names')
        
        print(f"✓ Model loaded from {model_path}")
        return self
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if supported by model).
        
        Returns:
            DataFrame with feature names and importance scores,
            or None if not supported
        """
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata and configuration
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_class': self.__class__.__name__,
            'is_trained': self.is_trained,
            'n_features': self.metadata.get('n_features'),
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'config': self.config
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.model_name}', type='{self.model_type}', status='{status}')"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        info = self.get_model_info()
        lines = [
            f"Model: {info['model_name']}",
            f"Type: {info['model_type']}",
            f"Class: {info['model_class']}",
            f"Status: {'Trained' if info['is_trained'] else 'Untrained'}",
        ]
        
        if info['n_features']:
            lines.append(f"Features: {info['n_features']}")
        
        if info['metadata'].get('trained_at'):
            lines.append(f"Trained: {info['metadata']['trained_at']}")
        
        return '\n'.join(lines)


class SupervisedDetector(BaseDetector):
    """
    Base class for supervised detection models.
    
    Supervised models learn from labeled data (benign vs. malicious).
    """
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        super().__init__(config, model_name)
        self.model_type = 'supervised'
        self.metadata['model_type'] = 'supervised'
        self.classes_ = None
    
    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.Series],
              **kwargs) -> 'SupervisedDetector':
        """Train supervised model (requires labels)."""
        pass


class UnsupervisedDetector(BaseDetector):
    """
    Base class for unsupervised detection models.
    
    Unsupervised models learn patterns from data without labels,
    typically trained on benign traffic to detect anomalies.
    """
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        super().__init__(config, model_name)
        self.model_type = 'unsupervised'
        self.metadata['model_type'] = 'unsupervised'
        self.anomaly_threshold_ = None
    
    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame],
              y: Optional[Union[np.ndarray, pd.Series]] = None,
              **kwargs) -> 'UnsupervisedDetector':
        """Train unsupervised model (labels optional, typically ignored)."""
        pass
    
    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            X: Features to score
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        # Default implementation using predict_proba
        # Subclasses should override if they have native decision_function
        proba = self.predict_proba(X)
        # Assuming predict_proba returns [normal_score, anomaly_score]
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]  # Return anomaly probability
        return proba


# Utility function for loading models
def load_model(filepath: Union[str, Path], 
               model_class: Optional[type] = None) -> BaseDetector:
    """
    Load a saved model from disk.
    
    Args:
        filepath: Path to the saved model
        model_class: Optional model class to instantiate.
                     If None, will try to infer from metadata.
    
    Returns:
        Loaded model instance
    
    Example:
        >>> model = load_model('models/lightgbm_best.pkl')
        >>> predictions = model.predict(X_test)
    """
    filepath = Path(filepath)
    metadata_path = filepath.with_suffix('.json')
    
    # Try to load metadata to get model class
    if metadata_path.exists() and model_class is None:
        with open(metadata_path, 'r') as f:
            save_data = json.load(f)
        
        # Here you would map model_class names to actual classes
        # For now, raise error if model_class not provided
        if model_class is None:
            raise ValueError(
                f"model_class parameter required. "
                f"Model class: {save_data['metadata']['model_class']}"
            )
    
    # Instantiate and load
    if model_class is None:
        raise ValueError("model_class parameter is required")
    
    # Create instance with minimal config (will be overwritten by load)
    instance = model_class(config={})
    instance.load(filepath)
    
    return instance
