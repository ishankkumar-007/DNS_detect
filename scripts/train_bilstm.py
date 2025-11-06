"""
Training script for BiLSTM Supervised Deep Learning
Trains a two-layer Bidirectional LSTM for DNS anomaly detection
"""

import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import DNSDataPreprocessor
from models.deep_bilstm import BiLSTMDetector
from feature_selection import HybridFeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bilstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(base_config_path: str = 'configs/base_config.yaml', 
                model_config_path: str = 'configs/bilstm_config.yaml') -> dict:
    """
    Load and merge base and model-specific configuration files.
    
    Args:
        base_config_path: Path to base configuration
        model_config_path: Path to model-specific configuration
        
    Returns:
        Merged configuration dictionary (model config overrides base)
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load model-specific config
    try:
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Model config not found: {model_config_path}, using base config only")
        return base_config
    
    # Deep merge: model config overrides base config
    def deep_merge(base_dict, override_dict):
        """Recursively merge dictionaries."""
        result = base_dict.copy()
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_config = deep_merge(base_config, model_config)
    logger.info(f"Loaded base config from: {base_config_path}")
    logger.info(f"Loaded model config from: {model_config_path}")
    
    return merged_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train BiLSTM for supervised DNS anomaly detection'
    )
    
    parser.add_argument(
        '--base-config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to base configuration file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/bilstm_config.yaml',
        help='Path to model-specific configuration file'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='bilstm_detection',
        help='Name for this experiment'
    )
    
    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Fraction of data to use (0-1). None = use full dataset'
    )
    
    parser.add_argument(
        '--lstm-units',
        type=int,
        default=None,
        help='LSTM units per layer. Overrides config.'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=None,
        help='Sequence length for LSTM. Overrides config.'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size. Overrides config.'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Training epochs. Overrides config.'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear existing cache before training'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    
    parser.add_argument(
        '--compare-model',
        type=str,
        default=None,
        help='Path to another model results for comparison'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline for BiLSTM."""
    
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.base_config, args.config)
    
    # Override config with command line arguments
    if args.sample is not None:
        config['data']['sample_fraction'] = args.sample
    
    # Get BiLSTM parameters from config or command line (command line takes precedence)
    lstm_units = (args.lstm_units if args.lstm_units is not None 
                  else config.get('bilstm', {}).get('layer1', {}).get('lstm_units', 30))
    sequence_length = (args.sequence_length if args.sequence_length is not None
                      else config.get('sequence', {}).get('sequence_length', 50))
    batch_size = (args.batch_size if args.batch_size is not None
                 else config.get('bilstm', {}).get('layer1', {}).get('batch_size', 80))
    epochs = (args.epochs if args.epochs is not None
             else config.get('bilstm', {}).get('layer1', {}).get('epochs', 100))
    
    # Update config with overrides
    if args.lstm_units is not None:
        config['bilstm']['layer1']['lstm_units'] = lstm_units
        config['bilstm']['layer2']['lstm_units'] = lstm_units
    if args.sequence_length is not None:
        config['sequence']['sequence_length'] = sequence_length
    if args.batch_size is not None:
        config['bilstm']['layer1']['batch_size'] = batch_size
        config['bilstm']['layer2']['batch_size'] = batch_size
    if args.epochs is not None:
        config['bilstm']['layer1']['epochs'] = epochs
        config['bilstm']['layer2']['epochs'] = epochs
    
    use_cache = not args.no_cache and config['output'].get('use_cache', True)
    
    logger.info("="*80)
    logger.info("BiLSTM SUPERVISED DNS ANOMALY DETECTION")
    logger.info("="*80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"LSTM Units: {lstm_units}, Sequence Length: {sequence_length}")
    logger.info(f"Batch Size: {batch_size}, Epochs: {epochs}")
    logger.info(f"Sample fraction: {config['data'].get('sample_fraction', 'Full dataset')}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.experiment_name}_lstm{lstm_units}_seq{sequence_length}_{timestamp}"
    exp_dir = Path('results') / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    models_dir = exp_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    metrics_dir = exp_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    
    logger.info(f"Results will be saved to: {exp_dir}")
    
    # ============================================================================
    # STEP 1: DATA PREPROCESSING
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*80)
    
    logger.info(f"Cache directory: {config['output'].get('cache_dir', 'cache')}")
    logger.info(f"Cache enabled: {use_cache}")
    logger.info(f"Sample fraction: {config['data'].get('sample_fraction', 'None (full dataset)')}")
    
    
    preprocessor = DNSDataPreprocessor(
        data_dir=config['data']['dataset_path'],
        use_dask=config['data'].get('use_dask', True),
        cache_dir=config['output'].get('cache_dir', 'cache'),
        use_cache=use_cache
    )
    
    if args.clear_cache:
        logger.info("Clearing cache...")
        import shutil
        if Path('cache').exists():
            shutil.rmtree('cache')
        logger.info("Cache cleared")
    
    # Load all data
    df = preprocessor.load_all_data(sample_frac=config['data'].get('sample_fraction'))
    logger.info(f"Raw data loaded: {len(df):,} samples")
    
    # Preprocess features
    X, y = preprocessor.preprocess_features(df)
    
    # Clean up raw data to free memory
    del df
    
    logger.info(f"\nDataset loaded and preprocessed:")
    logger.info(f"  Total samples: {len(X):,}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Benign: {(y == 0).sum():,} ({(y == 0).sum()/len(y):.2%})")
    logger.info(f"  Malicious: {(y == 1).sum():,} ({(y == 1).sum()/len(y):.2%})")
    
    # ============================================================================
    # STEP 2: TRAIN/TEST SPLIT - STRATIFIED
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: TRAIN/TEST SPLIT")
    logger.info("="*80)
    
    from sklearn.model_selection import train_test_split
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        stratify=y,
        random_state=config['model']['random_state']
    )
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"  Benign: {(y_train == 0).sum():,}")
    logger.info(f"  Malicious: {(y_train == 1).sum():,}")
    logger.info(f"Test set: {len(X_test):,} samples")
    logger.info(f"  Benign: {(y_test == 0).sum():,}")
    logger.info(f"  Malicious: {(y_test == 1).sum():,}")
    
    # ============================================================================
    # STEP 2.5: FEATURE SELECTION (OPTIONAL)
    # ============================================================================
    if config.get('feature_selection', {}).get('enable', False):
        logger.info("\n" + "="*80)
        logger.info("STEP 2.5: FEATURE SELECTION")
        logger.info("="*80)
        
        fs_config = config['feature_selection']
        
        if fs_config.get('method') == 'hybrid':
            logger.info("Using hybrid feature selection (SelectKBest + SHAP)")
            
            # Initialize feature selector
            selector = HybridFeatureSelector(
                k_best=fs_config['selectkbest']['k'],
                shap_top_n=fs_config['shap']['top_n'],
                random_state=config['model']['random_state'],
                cache_dir=config['output'].get('cache_dir', 'cache'),
                use_cache=use_cache,
                sample_fraction=config['data'].get('sample_fraction')
            )
            
            # Perform feature selection
            X_train_selected, X_test_selected = selector.fit_transform(
                X_train, y_train,
                X_test, y_test,
                method=fs_config['selectkbest']['score_func']
            )
            
            # Update X_train and X_test with selected features
            X_train = X_train_selected
            X_test = X_test_selected
            
            logger.info(f"Features after selection: {X_train.shape[1]}")
            logger.info(f"Selected features: {selector.get_feature_names()[:10]}...")
            
            # Save feature selection plots if configured
            if config['output'].get('save_plots', True):
                plots_dir.mkdir(exist_ok=True)
                selector.plot_feature_importance(
                    save_path=plots_dir / 'feature_importance.png',
                    show_plot=args.show_plots
                )
        elif fs_config.get('method') == 'selectkbest':
            logger.info("Using SelectKBest feature selection only")
            from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
            
            score_func_map = {
                'mutual_info': mutual_info_classif,
                'chi2': chi2,
                'f_classif': f_classif
            }
            score_func = score_func_map.get(fs_config['selectkbest']['score_func'], mutual_info_classif)
            
            selector = SelectKBest(score_func=score_func, k=fs_config['selectkbest']['k'])
            X_train = pd.DataFrame(
                selector.fit_transform(X_train, y_train),
                columns=X_train.columns[selector.get_support()],
                index=X_train.index
            )
            X_test = pd.DataFrame(
                selector.transform(X_test),
                columns=X_train.columns,
                index=X_test.index
            )
            
            logger.info(f"Features after SelectKBest: {X_train.shape[1]}")
        else:
            logger.info("Feature selection disabled")
    
    # ============================================================================
    # STEP 3: INITIALIZE BiLSTM MODEL
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: INITIALIZE BiLSTM MODEL")
    logger.info("="*80)
    
    detector = BiLSTMDetector(
        config=config,
        random_state=config['model']['random_state']
    )
    
    # ============================================================================
    # STEP 4: TRAIN BiLSTM MODEL
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAIN BiLSTM MODEL")
    logger.info("="*80)
    
    # Convert to numpy arrays
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    # Train on both benign and malicious data (supervised learning)
    history = detector.fit(
        X_train_np, y_train_np,
        X_val=X_test_np, y_val=y_test_np,
        verbose=1
    )
    
    logger.info(f"\nTraining Statistics:")
    logger.info(f"  Final training loss: {history['loss'][-1]:.4f}")
    logger.info(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  Final training accuracy: {history['accuracy'][-1]:.4f}")
    logger.info(f"  Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # ============================================================================
    # STEP 5: EVALUATE ON TEST SET
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: EVALUATE ON TEST SET")
    logger.info("="*80)
    
    metrics = detector.evaluate(
        X_test_np, y_test_np,
        save_dir=str(plots_dir),
        show_plots=args.show_plots
    )
    
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    
    # ============================================================================
    # STEP 6: SAVE MODEL
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: SAVE MODEL")
    logger.info("="*80)
    
    model_path = models_dir / 'bilstm_detector.pkl'
    detector.save_model(str(model_path))
    
    # Save experiment configuration
    exp_config = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'model_type': 'BiLSTM',
        'lstm_units': lstm_units,
        'sequence_length': sequence_length,
        'batch_size': batch_size,
        'epochs': epochs,
        'sample_fraction': config['data'].get('sample_fraction'),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_count': X.shape[1],
        'test_metrics': metrics,
        'training_history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    with open(metrics_dir / 'experiment_config.json', 'w') as f:
        json.dump(exp_config, f, indent=4)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_dir / 'metrics.csv', index=False)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Configuration saved to: {metrics_dir / 'experiment_config.json'}")
    logger.info(f"Metrics saved to: {metrics_dir / 'metrics.csv'}")
    
    # ============================================================================
    # STEP 7: MODEL COMPARISON (OPTIONAL)
    # ============================================================================
    if args.compare_model:
        logger.info("\n" + "="*80)
        logger.info("STEP 7: MODEL COMPARISON")
        logger.info("="*80)
        
        try:
            # Load comparison model metrics
            compare_dir = Path(args.compare_model)
            compare_metrics_path = compare_dir / 'metrics' / 'metrics.csv'
            
            if compare_metrics_path.exists():
                compare_metrics = pd.read_csv(compare_metrics_path).iloc[0].to_dict()
                
                logger.info(f"\nComparison with {compare_dir.name}:")
                logger.info("-" * 70)
                
                # Compare common metrics
                common_metrics = set(metrics.keys()) & set(compare_metrics.keys())
                comparison_data = []
                
                for metric in sorted(common_metrics):
                    current_val = metrics[metric]
                    compare_val = compare_metrics[metric]
                    diff = current_val - compare_val
                    pct_change = (diff / compare_val * 100) if compare_val != 0 else 0
                    
                    comparison_data.append({
                        'Metric': metric,
                        'BiLSTM': f'{current_val:.4f}',
                        'Baseline': f'{compare_val:.4f}',
                        'Difference': f'{diff:+.4f}',
                        'Change %': f'{pct_change:+.2f}%'
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                logger.info("\n" + comparison_df.to_string(index=False))
                
                # Save comparison
                comparison_path = metrics_dir / 'model_comparison.csv'
                comparison_df.to_csv(comparison_path, index=False)
                logger.info(f"\nComparison saved to: {comparison_path}")
            else:
                logger.warning(f"Metrics file not found: {compare_metrics_path}")
        
        except Exception as e:
            logger.error(f"Could not compare with baseline model: {e}")
    
    # ============================================================================
    # STEP 8: PREDICTION ANALYSIS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 8: PREDICTION ANALYSIS")
    logger.info("="*80)
    
    # Create sequences and aligned labels for prediction analysis
    # (same as what happens in evaluate())
    X_test_seq = detector._create_sequences(X_test_np)
    y_test_encoded = detector.label_encoder.transform(y_test_np)
    y_test_seq = detector._create_sequence_labels(y_test_encoded, len(X_test_np))
    y_test_aligned = detector.label_encoder.inverse_transform(y_test_seq)
    
    logger.info(f"Test data: {len(X_test_np):,} samples → {len(X_test_seq):,} sequences")
    
    # Get predictions and probabilities on sequences
    y_pred_proba = detector.model.predict(X_test_seq, verbose=0).flatten()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    y_pred = detector.label_encoder.inverse_transform(y_pred_binary)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'true_label': y_test_aligned,
        'predicted_label': y_pred,
        'prediction_probability': y_pred_proba
    })
    
    # Add prediction confidence
    predictions_df['confidence'] = predictions_df['prediction_probability'].apply(
        lambda x: x if x > 0.5 else 1 - x
    )
    
    logger.info(f"\nPredictions generated:")
    logger.info(f"  Total predictions: {len(predictions_df):,}")
    logger.info(f"  Mean confidence: {predictions_df['confidence'].mean():.4f}")
    logger.info(f"  Median confidence: {predictions_df['confidence'].median():.4f}")
    
    # Analyze high-confidence errors
    errors = predictions_df[predictions_df['true_label'] != predictions_df['predicted_label']]
    if len(errors) > 0:
        high_conf_errors = errors[errors['confidence'] > 0.9]
        logger.info(f"\n  High-confidence errors (>0.9): {len(high_conf_errors)}/{len(errors)}")
        if len(high_conf_errors) > 0:
            logger.info(f"  - Mean confidence of high-conf errors: {high_conf_errors['confidence'].mean():.4f}")
    
    # Save predictions
    predictions_path = metrics_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nExperiment: {exp_name}")
    logger.info(f"Results directory: {exp_dir}")
    logger.info(f"\nKey Findings:")
    sample_info = f" ({config['data'].get('sample_fraction', 1.0)*100:.1f}% sample)" if config['data'].get('sample_fraction') else ""
    logger.info(f"  - Trained on {len(X_train):,} samples{sample_info}")
    logger.info(f"  - Test set size: {len(X_test):,} samples")
    logger.info(f"  - Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - Precision: {metrics['precision']:.4f}")
    logger.info(f"  - Recall: {metrics['recall']:.4f}")
    logger.info(f"  - F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"\nAdvantages of BiLSTM:")
    logger.info(f"  ✓ Learns temporal patterns in DNS traffic sequences")
    logger.info(f"  ✓ Bidirectional context captures forward and backward dependencies")
    logger.info(f"  ✓ Deep architecture with 2 layers for complex feature learning")
    logger.info(f"  ✓ Supervised learning with labeled benign and malicious data")
    logger.info(f"\nLimitations:")
    logger.info(f"  ✗ Requires substantial labeled training data")
    logger.info(f"  ✗ Computationally expensive (GPU recommended)")
    logger.info(f"  ✗ Less interpretable than tree-based models")
    logger.info(f"  ✗ May not detect novel attack patterns not in training data")
    logger.info("="*80)


if __name__ == '__main__':
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)
