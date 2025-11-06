"""
BiLSTM Deep Learning Model for DNS Anomaly Detection

This module implements a two-layer Bidirectional LSTM architecture for
supervised binary classification of DNS traffic (Benign vs Malicious).

Architecture:
- Layer 1: Bidirectional LSTM (30 units, Nadam optimizer, dropout=0.2)
- Layer 2: Bidirectional LSTM (30 units, Adam optimizer, no dropout)
- Output: Dense layer with sigmoid activation for binary classification

Based on the architecture specification with two BiLSTM layers.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, optimizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Model = None  # Prevent NameError in type hints
    print("Warning: TensorFlow not available. Please install: pip install tensorflow")

# Metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_recall_fscore_support, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)

import warnings
warnings.filterwarnings('ignore')


class BiLSTMDetector:
    """
    Two-layer Bidirectional LSTM for supervised DNS anomaly detection.
    
    This model learns temporal patterns in DNS flow sequences through a deep
    BiLSTM architecture and classifies them as benign or malicious.
    
    Architecture Details:
    - Layer 1: BiLSTM (30 units each direction, 60 total output)
              Optimizer: Nadam (lr=0.001), Dropout: 0.2, Activation: tanh
    - Layer 2: BiLSTM (30 units each direction, 60 total output)
              Optimizer: Adam (lr=0.01), No dropout, Activation: tanh
    - Output: Dense(1, sigmoid) for binary classification
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing:
        - sequence: sequence_length, step_size, padding, truncating
        - bilstm: layer1/layer2 configurations
        - training: validation_split, early_stopping, layer_wise_training
    
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, config: Dict[str, Any], random_state: int = 42):
        """Initialize the BiLSTM detector."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for BiLSTMDetector. "
                            "Install with: pip install tensorflow")
        
        self.config = config
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Extract configuration
        self.seq_config = config.get('sequence', {})
        self.bilstm_config = config.get('bilstm', {})
        self.training_config = config.get('training', {})
        
        # Sequence parameters
        self.sequence_length = self.seq_config.get('sequence_length', 50)
        self.step_size = self.seq_config.get('step_size', 10)
        self.padding = self.seq_config.get('padding', 'post')
        self.truncating = self.seq_config.get('truncating', 'post')
        
        # Model components
        self.model = None  # Full BiLSTM model
        self.label_encoder = LabelEncoder()  # For encoding labels
        
        # Training history
        self.history = None
        
        # Feature information
        self.n_features = None
        self.feature_names = None
        
        print(f"✓ BiLSTM Detector initialized")
        print(f"  - Sequence length: {self.sequence_length}")
        print(f"  - Layer 1: {self.bilstm_config.get('layer1', {}).get('lstm_units', 30)} BiLSTM units")
        print(f"  - Layer 2: {self.bilstm_config.get('layer2', {}).get('lstm_units', 30)} BiLSTM units")
        print(f"  - Output: Binary classification (sigmoid)")
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Create sliding window sequences from flow data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        
        Returns:
        --------
        sequences : np.ndarray
            3D array (n_sequences, sequence_length, n_features)
        """
        n_samples, n_features = X.shape
        sequences = []
        
        # Sliding window
        for i in range(0, n_samples - self.sequence_length + 1, self.step_size):
            seq = X[i:i + self.sequence_length]
            sequences.append(seq)
        
        # Handle remaining samples if any
        if len(sequences) == 0:
            # Dataset too small, pad to sequence_length
            if n_samples < self.sequence_length:
                pad_width = self.sequence_length - n_samples
                if self.padding == 'post':
                    seq = np.vstack([X, np.zeros((pad_width, n_features))])
                else:
                    seq = np.vstack([np.zeros((pad_width, n_features)), X])
                sequences.append(seq)
            else:
                sequences.append(X[:self.sequence_length])
        
        return np.array(sequences)
    
    def _create_sequence_labels(self, y: np.ndarray, original_length: int) -> np.ndarray:
        """
        Create labels for sequences (use last label in each sequence).
        
        Parameters:
        -----------
        y : np.ndarray
            Original labels
        original_length : int
            Original data length before sequencing
        
        Returns:
        --------
        sequence_labels : np.ndarray
            Labels for each sequence
        """
        labels = []
        
        # Sliding window
        for i in range(0, original_length - self.sequence_length + 1, self.step_size):
            # Use the label of the last flow in the sequence
            label = y[min(i + self.sequence_length - 1, original_length - 1)]
            labels.append(label)
        
        # Handle case where no sequences were created
        if len(labels) == 0:
            labels.append(y[-1] if len(y) > 0 else 0)
        
        return np.array(labels)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> 'Model':
        """
        Build the two-layer BiLSTM model.
        
        Parameters:
        -----------
        input_shape : tuple
            (sequence_length, n_features)
        
        Returns:
        --------
        model : keras.Model
            Compiled BiLSTM model
        """
        layer1_config = self.bilstm_config.get('layer1', {})
        layer2_config = self.bilstm_config.get('layer2', {})
        
        # Input
        inputs = layers.Input(shape=input_shape, name='input_sequences')
        
        # Layer 1: BiLSTM with dropout
        lstm1 = layers.Bidirectional(
            layers.LSTM(
                units=layer1_config.get('lstm_units', 30),
                activation=layer1_config.get('activation', 'tanh'),
                return_sequences=layer1_config.get('return_sequences', True),
                dropout=layer1_config.get('dropout', 0.2),
                name='lstm_layer1'
            ),
            name='bilstm_layer1'
        )(inputs)
        
        # Layer 2: BiLSTM without dropout
        lstm2 = layers.Bidirectional(
            layers.LSTM(
                units=layer2_config.get('lstm_units', 30),
                activation=layer2_config.get('activation', 'tanh'),
                return_sequences=layer2_config.get('return_sequences', False),
                dropout=layer2_config.get('dropout', 0.0),
                name='lstm_layer2'
            ),
            name='bilstm_layer2'
        )(lstm1)
        
        # Output layer for binary classification
        outputs = layers.Dense(
            units=1,
            activation='sigmoid',  # Binary classification
            name='output'
        )(lstm2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='bilstm_classifier')
        
        return model
    
    def _compile_model(self, model: 'Model', optimizer_name: str, learning_rate: float):
        """
        Compile the model with specified optimizer.
        
        Parameters:
        -----------
        model : keras.Model
            Model to compile
        optimizer_name : str
            'adam' or 'nadam'
        learning_rate : float
            Learning rate
        """
        # Get optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'nadam':
            optimizer = optimizers.Nadam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # For supervised binary classification
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: int = 1) -> Dict[str, Any]:
        """
        Train the BiLSTM classification model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features (n_samples, n_features)
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels
        verbose : int
            Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns:
        --------
        history : dict
            Training history
        """
        print("\n" + "="*80)
        print("TRAINING BiLSTM CLASSIFICATION MODEL")
        print("="*80)
        
        # Store feature information
        self.n_features = X_train.shape[1]
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        if y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
        
        # Step 1: Create sequences
        print(f"\n[Step 1/3] Creating sequences...")
        X_train_seq = self._create_sequences(X_train)
        print(f"  ✓ Training sequences: {X_train_seq.shape}")
        
        # Adjust labels for sequences (use last label in each sequence)
        y_train_seq = self._create_sequence_labels(y_train_encoded, len(X_train))
        print(f"  ✓ Training labels: {y_train_seq.shape}")
        
        if X_val is not None:
            X_val_seq = self._create_sequences(X_val)
            y_val_seq = self._create_sequence_labels(y_val_encoded, len(X_val))
            print(f"  ✓ Validation sequences: {X_val_seq.shape}")
            print(f"  ✓ Validation labels: {y_val_seq.shape}")
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Step 2: Build model
        print(f"\n[Step 2/3] Building BiLSTM architecture...")
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)
        print(f"  ✓ Model built with input shape: {input_shape}")
        self.model.summary()
        
        # Compile model (use Layer 2 optimizer settings)
        layer2_config = self.bilstm_config.get('layer2', {})
        self._compile_model(self.model,
                           layer2_config.get('optimizer', 'adam'),
                           layer2_config.get('learning_rate', 0.01))
        
        # Step 3: Train model
        print(f"\n[Step 3/3] Training model...")
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Train
        history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=layer2_config.get('batch_size', 80),
            epochs=layer2_config.get('epochs', 100),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        self.history = history.history
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80)
        
        return self.history
    
    def _get_callbacks(self) -> List:
        """Get training callbacks."""
        callbacks = []
        
        # Early stopping
        es_config = self.training_config.get('early_stopping', {})
        if es_config:
            callbacks.append(EarlyStopping(
                monitor=es_config.get('monitor', 'val_loss'),
                patience=es_config.get('patience', 15),
                restore_best_weights=es_config.get('restore_best_weights', True),
                verbose=1
            ))
        
        # Reduce learning rate on plateau
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss' if 'val_loss' in es_config.get('monitor', 'val_loss') else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ))
        
        return callbacks
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
        
        Returns:
        --------
        predictions : np.ndarray
            Predicted class labels
        """
        # Create sequences
        X_seq = self._create_sequences(X)
        
        # Predict probabilities
        y_pred_proba = self.model.predict(X_seq, verbose=0)
        
        # Convert to class labels
        y_pred_binary = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(y_pred_binary)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
        
        Returns:
        --------
        probabilities : np.ndarray
            Predicted probabilities for positive class
        """
        X_seq = self._create_sequences(X)
        y_pred_proba = self.model.predict(X_seq, verbose=0)
        return y_pred_proba.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 save_dir: Optional[str] = None, show_plots: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        save_dir : str, optional
            Directory to save plots
        show_plots : bool
            Whether to display plots
        
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        print("\n" + "="*80)
        print("EVALUATING BiLSTM MODEL")
        print("="*80)
        
        # Create sequences from test data
        X_test_seq = self._create_sequences(X_test)
        print(f"Test sequences created: {X_test_seq.shape}")
        
        # Encode labels and align with sequences
        y_test_encoded = self.label_encoder.transform(y_test)
        y_test_seq = self._create_sequence_labels(y_test_encoded, len(X_test))
        print(f"Test sequence labels: {y_test_seq.shape}")
        
        # Get predictions on sequences
        y_pred_proba = self.model.predict(X_test_seq, verbose=0).flatten()
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # Decode predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_binary)
        y_test_aligned = self.label_encoder.inverse_transform(y_test_seq)
        
        # Classification metrics
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_test_aligned, y_pred)
        
        # Find positive class label
        pos_label = 'Malicious' if 'Malicious' in self.label_encoder.classes_ else self.label_encoder.classes_[1]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_aligned, y_pred, average='binary', pos_label=pos_label
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test_seq, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        
        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_seq, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_test_seq, y_pred_proba)
        
        # Print metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_aligned, y_pred))
        
        # Generate plots
        if save_dir or show_plots:
            self._plot_evaluation(X_test_seq, y_test_aligned, y_pred, y_pred_proba,
                                 metrics, save_dir, show_plots)
        
        print("="*80)
        return metrics
    
    def _plot_evaluation(self, X_test: np.ndarray, y_test: np.ndarray, 
                        y_pred: np.ndarray, y_pred_proba: np.ndarray,
                        metrics: Dict[str, float], save_dir: Optional[str], 
                        show_plots: bool):
        """Generate evaluation plots."""
        fig = plt.figure(figsize=(15, 10))
        
        # Encode labels for plotting
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        
        # Add percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.array([[f'{count}\n({pct:.1f}%)' 
                          for count, pct in zip(row_counts, row_pcts)]
                         for row_counts, row_pcts in zip(cm, cm_percent)])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax1,
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_title('Confusion Matrix')
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba)
        ax2.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={metrics["roc_auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = plt.subplot(2, 3, 3)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_encoded, y_pred_proba)
        ax3.plot(recall_curve, precision_curve, linewidth=2, 
                label=f'PR (AUC={metrics["pr_auc"]:.3f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Training History - Loss
        ax4 = plt.subplot(2, 3, 4)
        if self.history:
            epochs = range(1, len(self.history['loss']) + 1)
            ax4.plot(epochs, self.history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in self.history:
                ax4.plot(epochs, self.history['val_loss'], label='Validation Loss', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Loss Curves')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Training History - Accuracy
        ax5 = plt.subplot(2, 3, 5)
        if self.history:
            epochs = range(1, len(self.history['accuracy']) + 1)
            ax5.plot(epochs, self.history['accuracy'], label='Training Accuracy', linewidth=2)
            if 'val_accuracy' in self.history:
                ax5.plot(epochs, self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Accuracy')
            ax5.set_title('Training Accuracy Curves')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Metrics Bar Chart
        ax6 = plt.subplot(2, 3, 6)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('roc_auc', 0),
            metrics.get('pr_auc', 0)
        ]
        
        bars = ax6.barh(metric_names, metric_values, color='steelblue', edgecolor='black')
        ax6.set_xlabel('Score')
        ax6.set_title('Classification Metrics')
        ax6.set_xlim(0, 1)
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax6.text(val + 0.02, i, f'{val:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(save_dir, f'bilstm_evaluation_{timestamp}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plots saved to: {filepath}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, filepath: str):
        """Save the complete model (BiLSTM + label encoder)."""
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save BiLSTM model
        bilstm_path = filepath.replace('.pkl', '_bilstm.h5')
        self.model.save(bilstm_path)
        
        # Save other components
        model_dict = {
            'label_encoder': self.label_encoder,
            'config': self.config,
            'history': self.history,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"✓ Model saved to:")
        print(f"  - BiLSTM: {bilstm_path}")
        print(f"  - Components: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BiLSTMDetector':
        """Load a saved model."""
        # Load components
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Create instance
        instance = cls(model_dict['config'], model_dict['random_state'])
        
        # Load BiLSTM model
        bilstm_path = filepath.replace('.pkl', '_bilstm.h5')
        instance.model = keras.models.load_model(bilstm_path)
        
        # Restore components
        instance.label_encoder = model_dict['label_encoder']
        instance.history = model_dict['history']
        instance.n_features = model_dict['n_features']
        instance.feature_names = model_dict['feature_names']
        
        print(f"✓ Model loaded from:")
        print(f"  - BiLSTM: {bilstm_path}")
        print(f"  - Components: {filepath}")
        
        return instance
