"""
Training script for Hybrid DNS Anomaly Detection Model
Combines Supervised (Random Forest) and Unsupervised (Isolation Forest) Learning

This script loads pre-trained RF and IF models, combines them using various
fusion strategies, and evaluates the hybrid ensemble performance.
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
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import DNSDataPreprocessor
from models.hybrid_model import HybridDetector
from models.random_forest import RandomForestDetector
from models.unsupervised_iforest import IsolationForestAnomalyDetector
from feature_selection import HybridFeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(base_config_path: str = 'configs/base_config.yaml', 
                model_config_path: str = 'configs/hybrid_config.yaml') -> dict:
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
        description='Train hybrid model combining Random Forest and Isolation Forest'
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
        default='configs/hybrid_config.yaml',
        help='Path to hybrid model configuration file'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='hybrid_detection',
        help='Name for this experiment'
    )
    
    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Fraction of data to use (0-1). None = use full dataset'
    )
    
    parser.add_argument(
        '--fusion-strategy',
        type=str,
        choices=['voting', 'anomaly_aware', 'two_stage', 'weighted_average'],
        default=None,
        help='Fusion strategy to combine models. Overrides config.'
    )
    
    parser.add_argument(
        '--rf-model',
        type=str,
        default=None,
        help='Path to trained Random Forest model. Overrides config.'
    )
    
    parser.add_argument(
        '--if-model',
        type=str,
        default=None,
        help='Path to trained Isolation Forest model. Overrides config.'
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
    
    return parser.parse_args()


def load_pretrained_model(model_path: Path, model_name: str):
    """
    Load a pre-trained model from disk.
    
    Args:
        model_path: Path to saved model
        model_name: Name of the model for logging
        
    Returns:
        Loaded model object
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Pre-trained {model_name} model not found at {model_path}")
    
    logger.info(f"Loading {model_name} from {model_path}...")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info(f"✓ Successfully loaded {model_name}")
        
        # Log model information
        if isinstance(model_data, dict):
            logger.info(f"  Model type: {type(model_data.get('model', 'Unknown')).__name__}")
            if 'training_stats' in model_data:
                stats = model_data['training_stats']
                logger.info(f"  Training samples: {stats.get('n_samples', 'Unknown')}")
                logger.info(f"  Features: {stats.get('n_features', 'Unknown')}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        raise


def main():
    """Main training pipeline for hybrid model."""
    
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.base_config, args.config)
    
    # Override config with command line arguments
    if args.sample is not None:
        config['data']['sample_fraction'] = args.sample
    
    if args.fusion_strategy is not None:
        config['hybrid']['fusion_strategy'] = args.fusion_strategy
    
    if args.rf_model is not None:
        config['hybrid']['supervised_model']['path'] = args.rf_model
    
    if args.if_model is not None:
        config['hybrid']['unsupervised_model']['path'] = args.if_model
    
    use_cache = not args.no_cache and config['output'].get('use_cache', True)
    
    fusion_strategy = config['hybrid']['fusion_strategy']
    
    logger.info("="*80)
    logger.info("HYBRID DNS ANOMALY DETECTION - SUPERVISED + UNSUPERVISED")
    logger.info("="*80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Fusion Strategy: {fusion_strategy}")
    logger.info(f"Sample fraction: {config['data'].get('sample_fraction', 'Full dataset')}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.experiment_name}_{fusion_strategy}_{timestamp}"
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
    # STEP 1: LOAD PRE-TRAINED BASE MODELS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD PRE-TRAINED BASE MODELS")
    logger.info("="*80)
    
    # Load Random Forest (Supervised) - use class method for proper reconstruction
    rf_model_path = config['hybrid']['supervised_model']['path']
    logger.info(f"Loading Random Forest from {rf_model_path}...")
    rf_model = RandomForestDetector.load_model(rf_model_path)
    logger.info(f"✓ Random Forest loaded successfully")
    
    # Load Isolation Forest (Unsupervised) - use class method for proper reconstruction
    if_model_path = config['hybrid']['unsupervised_model']['path']
    logger.info(f"Loading Isolation Forest from {if_model_path}...")
    if_model = IsolationForestAnomalyDetector.load_model(if_model_path)
    logger.info(f"✓ Isolation Forest loaded successfully")
    
    logger.info("\n✓ Both base models loaded successfully")
    
    # ============================================================================
    # STEP 2: DATA PREPROCESSING
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("="*80)
    
    preprocessor = DNSDataPreprocessor(
        data_dir=config['data']['dataset_path'],
        use_dask=config['data'].get('use_dask', True),
        cache_dir=config['output'].get('cache_dir', 'cache'),
        use_cache=use_cache
    )
    
    if args.clear_cache:
        logger.info("Clearing cache...")
        import shutil
        cache_path = Path(config['output'].get('cache_dir', 'cache'))
        if cache_path.exists():
            shutil.rmtree(cache_path)
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
    # STEP 3: TRAIN/TEST SPLIT - STRATIFIED
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: TRAIN/TEST SPLIT")
    logger.info("="*80)
    
    from sklearn.model_selection import train_test_split
    
    # Split into train and test (using same split as base models)
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
    # STEP 3.5: FEATURE SELECTION (MUST MATCH BASE MODELS)
    # ============================================================================
    if config.get('feature_selection', {}).get('enable', False):
        logger.info("\n" + "="*80)
        logger.info("STEP 3.5: FEATURE SELECTION (Matching Base Models)")
        logger.info("="*80)
        
        fs_config = config['feature_selection']
        
        if fs_config.get('method') == 'hybrid':
            logger.info("Using hybrid feature selection (SelectKBest + SHAP)")
            logger.info("⚠️  CRITICAL: Using SAME features as base models")
            logger.info("⚠️  FORCING use of FULL dataset feature selection to match base models")
            
            # Initialize feature selector
            # CRITICAL: Always use full dataset feature selection (sample_fraction=None)
            # to match the features used when training base models, regardless of
            # what sample size we use for hybrid model testing
            selector = HybridFeatureSelector(
                k_best=fs_config['selectkbest']['k'],
                shap_top_n=fs_config['shap']['top_n'],
                random_state=config['model']['random_state'],
                cache_dir=config['output'].get('cache_dir', 'cache'),
                use_cache=use_cache,
                sample_fraction=None  # Force full dataset feature selection
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
            logger.info(f"Selected features: {selector.get_feature_names()}")
            
            # Save feature selection plots if configured
            if config['output'].get('save_plots', True):
                selector.plot_feature_importance(
                    save_path=plots_dir / 'feature_importance.png',
                    show_plot=args.show_plots
                )
        else:
            logger.info("Feature selection disabled")
    
    # ============================================================================
    # STEP 4: CREATE HYBRID MODEL
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: CREATE HYBRID MODEL")
    logger.info("="*80)
    
    detector = HybridDetector(
        supervised_model=rf_model,
        unsupervised_model=if_model,
        fusion_strategy=fusion_strategy,
        config=config
    )
    
    logger.info(f"✓ Hybrid detector created with {fusion_strategy} fusion")
    
    # ============================================================================
    # STEP 5: EVALUATE HYBRID MODEL ON TEST SET
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: EVALUATE HYBRID MODEL ON TEST SET")
    logger.info("="*80)
    
    metrics = detector.evaluate(
        X_test=X_test.values,
        y_test=y_test.values,
        output_dir=plots_dir,
        show_plot=args.show_plots
    )
    
    # ============================================================================
    # STEP 6: DETAILED PERFORMANCE ANALYSIS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: DETAILED PERFORMANCE ANALYSIS")
    logger.info("="*80)
    
    # Confusion matrix breakdown
    cm = np.array(metrics['confusion_matrix'])
    tn, fp, fn, tp = cm.ravel()
    
    logger.info(f"\nHybrid Model Confusion Matrix Breakdown:")
    logger.info(f"  True Negatives (Benign correctly identified): {tn:,}")
    logger.info(f"  False Positives (Benign misclassified as Malicious): {fp:,}")
    logger.info(f"  False Negatives (Malicious missed): {fn:,}")
    logger.info(f"  True Positives (Malicious correctly detected): {tp:,}")
    logger.info(f"  False Positive Rate: {fp/(tn+fp):.4f}")
    logger.info(f"  False Negative Rate: {fn/(fn+tp):.4f}")
    
    # Compare with base models
    rf_metrics = metrics['base_models']['random_forest']
    if_metrics = metrics['base_models']['isolation_forest']
    
    logger.info(f"\nRandom Forest Performance:")
    logger.info(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    logger.info(f"  Malicious Recall: {rf_metrics.get('recall_malicious', 0):.4f}")
    logger.info(f"  Malicious Precision: {rf_metrics.get('precision_malicious', 0):.4f}")
    
    logger.info(f"\nIsolation Forest Performance:")
    logger.info(f"  Accuracy: {if_metrics['accuracy']:.4f}")
    logger.info(f"  Malicious Recall: {if_metrics.get('recall_malicious', 0):.4f}")
    logger.info(f"  Malicious Precision: {if_metrics.get('precision_malicious', 0):.4f}")
    
    logger.info(f"\nHybrid Model Improvements:")
    acc_improvement_rf = metrics['accuracy'] - rf_metrics['accuracy']
    acc_improvement_if = metrics['accuracy'] - if_metrics['accuracy']
    logger.info(f"  Accuracy vs RF: {acc_improvement_rf:+.4f} ({acc_improvement_rf/rf_metrics['accuracy']*100:+.2f}%)")
    logger.info(f"  Accuracy vs IF: {acc_improvement_if:+.4f} ({acc_improvement_if/if_metrics['accuracy']*100:+.2f}%)")
    
    # ============================================================================
    # STEP 7: SAVE HYBRID MODEL AND RESULTS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 7: SAVE HYBRID MODEL AND RESULTS")
    logger.info("="*80)
    
    # Save hybrid model
    model_path = models_dir / 'hybrid_detector.pkl'
    metadata = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'fusion_strategy': fusion_strategy,
        'base_models': {
            'random_forest': str(rf_model_path),
            'isolation_forest': str(if_model_path)
        },
        'sample_fraction': config['data'].get('sample_fraction'),
        'test_metrics': metrics
    }
    detector.save_model(model_path, metadata)
    
    # Save experiment configuration
    exp_config = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'model_type': 'Hybrid',
        'fusion_strategy': fusion_strategy,
        'supervised_model_path': str(rf_model_path),
        'unsupervised_model_path': str(if_model_path),
        'sample_fraction': config['data'].get('sample_fraction'),
        'test_samples': len(X_test),
        'feature_count': X_test.shape[1],
        'metrics': metrics
    }
    
    with open(metrics_dir / 'experiment_config.json', 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        json.dump(convert_to_serializable(exp_config), f, indent=4)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'model': 'hybrid',
        'fusion_strategy': fusion_strategy,
        **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    }])
    metrics_df.to_csv(metrics_dir / 'metrics.csv', index=False)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Configuration saved to: {metrics_dir / 'experiment_config.json'}")
    logger.info(f"Metrics saved to: {metrics_dir / 'metrics.csv'}")
    
    # ============================================================================
    # STEP 8: GENERATE COMPARISON REPORT
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 8: GENERATE COMPARISON REPORT")
    logger.info("="*80)
    
    comparison_data = []
    
    # Hybrid model
    comparison_data.append({
        'Model': 'Hybrid',
        'Accuracy': metrics['accuracy'],
        'Precision (Mal)': metrics.get('precision_malicious', 0),
        'Recall (Mal)': metrics.get('recall_malicious', 0),
        'F1 (Mal)': metrics.get('f1_malicious', 0),
        'ROC-AUC': metrics.get('roc_auc', 0),
        'FPR': metrics['fpr'],
        'FNR': metrics['fnr']
    })
    
    # Random Forest
    comparison_data.append({
        'Model': 'Random Forest',
        'Accuracy': rf_metrics['accuracy'],
        'Precision (Mal)': rf_metrics.get('precision_malicious', 0),
        'Recall (Mal)': rf_metrics.get('recall_malicious', 0),
        'F1 (Mal)': rf_metrics.get('f1_malicious', 0),
        'ROC-AUC': rf_metrics.get('roc_auc', 0),
        'FPR': rf_metrics['fpr'],
        'FNR': rf_metrics['fnr']
    })
    
    # Isolation Forest
    comparison_data.append({
        'Model': 'Isolation Forest',
        'Accuracy': if_metrics['accuracy'],
        'Precision (Mal)': if_metrics.get('precision_malicious', 0),
        'Recall (Mal)': if_metrics.get('recall_malicious', 0),
        'F1 (Mal)': if_metrics.get('f1_malicious', 0),
        'ROC-AUC': if_metrics.get('roc_auc', 0),
        'FPR': if_metrics['fpr'],
        'FNR': if_metrics['fnr']
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = metrics_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    
    logger.info("\nModel Comparison:")
    logger.info("\n" + comparison_df.to_string(index=False))
    logger.info(f"\nComparison saved to: {comparison_path}")
    
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
    logger.info(f"  - Test set size: {len(X_test):,} samples{sample_info}")
    logger.info(f"  - Hybrid accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - Malicious detection rate (recall): {metrics.get('recall_malicious', 0):.4f}")
    logger.info(f"  - False positive rate: {metrics['fpr']:.4f}")
    logger.info(f"  - ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    
    logger.info(f"\nFusion Strategy: {fusion_strategy}")
    if fusion_strategy == 'voting':
        logger.info(f"  - Combines predictions via weighted voting")
    elif fusion_strategy == 'anomaly_aware':
        logger.info(f"  - Uses IF anomaly scores to boost RF predictions")
    elif fusion_strategy == 'two_stage':
        logger.info(f"  - IF filters suspicious samples, RF classifies them")
    elif fusion_strategy == 'weighted_average':
        logger.info(f"  - Weighted average of prediction probabilities")
    
    logger.info(f"\nAdvantages of Hybrid Approach:")
    logger.info(f"  ✓ Combines strengths of supervised and unsupervised learning")
    logger.info(f"  ✓ Better detection of both known and novel attacks")
    logger.info(f"  ✓ More robust predictions through model consensus")
    logger.info(f"  ✓ Reduced false positives via cross-validation")
    logger.info(f"  ✓ Anomaly scores provide additional context for analysts")
    
    logger.info(f"\nPerformance vs Base Models:")
    logger.info(f"  • Hybrid vs RF: {acc_improvement_rf:+.4f} accuracy improvement")
    logger.info(f"  • Hybrid vs IF: {acc_improvement_if:+.4f} accuracy improvement")
    logger.info(f"  • Best overall accuracy: {'Hybrid' if metrics['accuracy'] >= max(rf_metrics['accuracy'], if_metrics['accuracy']) else 'Base model'}")
    
    logger.info("="*80)


if __name__ == '__main__':
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)
