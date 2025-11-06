"""
Training script for One-Class SVM Unsupervised Anomaly Detection
Trains on benign traffic only to learn normal behavior patterns
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
from models.unsupervised_ocsvm import OneClassSVMAnomalyDetector, compare_supervised_vs_unsupervised
from feature_selection import HybridFeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unsupervised_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(base_config_path: str = 'configs/base_config.yaml', 
                model_config_path: str = 'configs/ocsvm_config.yaml') -> dict:
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
        description='Train One-Class SVM for unsupervised DNS anomaly detection'
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
        default='configs/ocsvm_config.yaml',
        help='Path to model-specific configuration file'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='ocsvm_detection',
        help='Name for this experiment'
    )
    
    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Fraction of data to use (0-1). None = use full dataset'
    )
    
    parser.add_argument(
        '--nu',
        type=float,
        default=None,
        help='One-Class SVM nu parameter (expected outlier fraction). Overrides config.'
    )
    
    parser.add_argument(
        '--kernel',
        type=str,
        default=None,
        choices=['rbf', 'linear', 'poly', 'sigmoid'],
        help='Kernel type for One-Class SVM. Overrides config.'
    )
    
    parser.add_argument(
        '--use-pca',
        action='store_true',
        default=None,
        help='Use PCA for dimensionality reduction. Overrides config.'
    )
    
    parser.add_argument(
        '--no-pca',
        action='store_true',
        help='Disable PCA for dimensionality reduction. Overrides config.'
    )
    
    parser.add_argument(
        '--pca-components',
        type=int,
        default=None,
        help='Number of PCA components. Overrides config.'
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
        '--compare-supervised',
        type=str,
        default=None,
        help='Path to supervised model results for comparison'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline for One-Class SVM."""
    
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.base_config, args.config)
    
    # Override config with command line arguments
    if args.sample is not None:
        config['data']['sample_fraction'] = args.sample
    
    # Get OCSVM parameters from config or command line (command line takes precedence)
    kernel = args.kernel if args.kernel is not None else config.get('ocsvm', {}).get('kernel', 'rbf')
    nu = args.nu if args.nu is not None else config.get('ocsvm', {}).get('nu', 0.05)
    
    # Handle PCA arguments
    if args.no_pca:
        use_pca = False
    elif args.use_pca:
        use_pca = True
    else:
        use_pca = config.get('dimensionality_reduction', {}).get('use_pca', True)
    
    pca_components = (args.pca_components if args.pca_components is not None 
                     else config.get('dimensionality_reduction', {}).get('n_components', 30))
    
    use_cache = not args.no_cache and config['output'].get('use_cache', True)
    
    logger.info("="*80)
    logger.info("ONE-CLASS SVM UNSUPERVISED DNS ANOMALY DETECTION")
    logger.info("="*80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Kernel: {kernel}, nu: {nu}")
    logger.info(f"PCA: {use_pca}, Components: {pca_components if use_pca else 'N/A'}")
    logger.info(f"Sample fraction: {config['data'].get('sample_fraction', 'Full dataset')}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.experiment_name}_nu{int(nu*100)}_{'pca' if use_pca else 'nopca'}_{timestamp}"
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
            
            # Get cache dir for feature selection
            fs_cache_dir = config['output'].get('cache_dir', 'cache')
            logger.info(f"Feature selection cache directory: {fs_cache_dir}")
            
            # Initialize feature selector
            selector = HybridFeatureSelector(
                k_best=fs_config['selectkbest']['k'],
                shap_top_n=fs_config['shap']['top_n'],
                random_state=config['model']['random_state'],
                cache_dir=fs_cache_dir,
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
    # STEP 3: EXTRACT BENIGN TRAINING DATA ONLY
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: EXTRACT BENIGN DATA FOR UNSUPERVISED TRAINING")
    logger.info("="*80)
    
    # One-Class SVM trains on benign traffic ONLY
    X_train_benign = X_train[y_train == 0]
    
    logger.info(f"Training on BENIGN traffic only:")
    logger.info(f"  Benign samples: {len(X_train_benign):,}")
    logger.info(f"  Features: {X_train_benign.shape[1]}")
    logger.info(f"\nNote: Model will learn what 'normal' looks like from benign data")
    logger.info(f"      Then detect deviations (anomalies) as potential threats")
    
    # ============================================================================
    # STEP 4: TRAIN ONE-CLASS SVM
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAIN ONE-CLASS SVM MODEL")
    logger.info("="*80)
    
    detector = OneClassSVMAnomalyDetector(
        kernel=kernel,
        nu=nu,
        gamma=config.get('ocsvm', {}).get('gamma', 'scale'),
        use_pca=use_pca,
        n_components=pca_components if use_pca else None,
        random_state=config['model']['random_state']
    )
    
    # Train on benign data only
    detector.fit(X_train_benign, feature_names=X.columns.tolist())
    
    logger.info(f"\nTraining Statistics:")
    logger.info(f"  Support vectors: {detector.training_stats['n_support_vectors']:,}")
    logger.info(f"  Support vector ratio: {detector.training_stats['support_vector_ratio']:.4f}")
    logger.info(f"  Decision score range: [{detector.training_stats['decision_min']:.4f}, "
                f"{detector.training_stats['decision_max']:.4f}]")
    
    if use_pca:
        logger.info(f"  PCA explained variance: {detector.training_stats['pca_explained_variance']:.4f}")
    
    # ============================================================================
    # STEP 5: EVALUATE ON TEST SET
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: EVALUATE ON TEST SET")
    logger.info("="*80)
    
    metrics = detector.evaluate(
        X_test, y_test,
        output_dir=plots_dir,
        show_plot=args.show_plots
    )
    
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"\nBenign Detection (Specificity):")
    logger.info(f"  Precision: {metrics['precision_benign']:.4f}")
    logger.info(f"  Recall: {metrics['recall_benign']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_benign']:.4f}")
    logger.info(f"\nMalicious Detection (Sensitivity):")
    logger.info(f"  Precision: {metrics['precision_malicious']:.4f}")
    logger.info(f"  Recall: {metrics['recall_malicious']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_malicious']:.4f}")
    
    # Confusion matrix breakdown
    cm = np.array(metrics['confusion_matrix'])
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"\nConfusion Matrix Breakdown:")
    logger.info(f"  True Negatives (Benign correctly identified): {tn:,}")
    logger.info(f"  False Positives (Benign misclassified as Malicious): {fp:,}")
    logger.info(f"  False Negatives (Malicious missed): {fn:,}")
    logger.info(f"  True Positives (Malicious correctly detected): {tp:,}")
    logger.info(f"  False Positive Rate: {fp/(tn+fp):.4f}")
    logger.info(f"  False Negative Rate: {fn/(fn+tp):.4f}")
    
    # ============================================================================
    # STEP 6: SAVE MODEL
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: SAVE MODEL")
    logger.info("="*80)
    
    model_path = models_dir / 'ocsvm_detector.pkl'
    detector.save_model(model_path)
    
    # Save experiment configuration
    exp_config = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'model_type': 'OneClassSVM',
        'kernel': kernel,
        'nu': nu,
        'gamma': config.get('ocsvm', {}).get('gamma', 'scale'),
        'use_pca': use_pca,
        'pca_components': pca_components if use_pca else None,
        'sample_fraction': config['data'].get('sample_fraction'),
        'training_samples': len(X_train_benign),
        'test_samples': len(X_test),
        'feature_count': X.shape[1],
        'training_stats': detector.training_stats,
        'test_metrics': metrics
    }
    
    with open(metrics_dir / 'experiment_config.json', 'w') as f:
        json.dump(exp_config, f, indent=4)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Configuration saved to: {metrics_dir / 'experiment_config.json'}")
    
    # ============================================================================
    # STEP 7: COMPARE WITH SUPERVISED APPROACH (OPTIONAL)
    # ============================================================================
    if args.compare_supervised:
        logger.info("\n" + "="*80)
        logger.info("STEP 7: COMPARE WITH SUPERVISED MODEL")
        logger.info("="*80)
        
        try:
            supervised_metrics_path = Path(args.compare_supervised) / 'metrics' / 'evaluation_metrics.json'
            with open(supervised_metrics_path, 'r') as f:
                supervised_metrics = json.load(f)
            
            # Normalize metric names for comparison
            supervised_comparison = {
                'accuracy': supervised_metrics['accuracy'],
                'precision_malicious': supervised_metrics['classification_report']['1']['precision'],
                'recall_malicious': supervised_metrics['classification_report']['1']['recall'],
                'f1_malicious': supervised_metrics['classification_report']['1']['f1-score'],
                'roc_auc': supervised_metrics['roc_auc']
            }
            
            unsupervised_comparison = {
                'accuracy': metrics['accuracy'],
                'precision_malicious': metrics['precision_malicious'],
                'recall_malicious': metrics['recall_malicious'],
                'f1_malicious': metrics['f1_malicious'],
                'roc_auc': metrics['roc_auc']
            }
            
            logger.info("\nPerformance Comparison:")
            logger.info(f"{'Metric':<25} {'Supervised':<15} {'Unsupervised':<15} {'Difference':<15}")
            logger.info("-" * 70)
            
            for metric in supervised_comparison.keys():
                sup_val = supervised_comparison[metric]
                unsup_val = unsupervised_comparison[metric]
                diff = unsup_val - sup_val
                logger.info(f"{metric:<25} {sup_val:<15.4f} {unsup_val:<15.4f} {diff:+.4f}")
            
            # Generate comparison plot
            compare_supervised_vs_unsupervised(
                supervised_comparison,
                unsupervised_comparison,
                output_path=plots_dir / 'supervised_vs_unsupervised.png',
                show_plot=args.show_plots
            )
            
            logger.info(f"\nComparison plot saved to: {plots_dir / 'supervised_vs_unsupervised.png'}")
            
        except Exception as e:
            logger.error(f"Could not compare with supervised model: {e}")
    
    # ============================================================================
    # STEP 8: ANOMALY RANKING
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 8: TOP ANOMALIES ANALYSIS")
    logger.info("="*80)
    
    # Get top 50 most anomalous samples
    top_anomalies = detector.get_anomaly_ranking(X_test, top_n=50)
    
    # Add true labels - need to align indices properly
    # The 'index' column contains original X_test indices
    top_anomalies['true_label'] = top_anomalies['index'].map(lambda idx: y_test.iloc[X_test.index.get_loc(idx)] if idx in X_test.index else None)
    top_anomalies['true_class'] = top_anomalies['true_label'].map({0: 'Benign', 1: 'Malicious'})
    
    logger.info(f"\nTop 50 Most Anomalous Samples:")
    logger.info(f"  Actual Malicious: {(top_anomalies['true_label'] == 1).sum()}")
    logger.info(f"  Actual Benign: {(top_anomalies['true_label'] == 0).sum()}")
    logger.info(f"  Detection rate in top-50: {(top_anomalies['true_label'] == 1).sum() / 50:.2%}")
    
    # Save top anomalies
    top_anomalies.to_csv(metrics_dir / 'top_anomalies.csv', index=False)
    logger.info(f"Top anomalies saved to: {metrics_dir / 'top_anomalies.csv'}")
    
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
    logger.info(f"  - Trained on {len(X_train_benign):,} benign samples{sample_info}")
    logger.info(f"  - Test set size: {len(X_test):,} samples")
    logger.info(f"  - Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - Malicious detection rate: {metrics['recall_malicious']:.4f}")
    logger.info(f"  - False positive rate: {fp/(tn+fp):.4f}")
    logger.info(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"\nAdvantages of One-Class SVM:")
    logger.info(f"  ✓ No labeled malicious data needed for training")
    logger.info(f"  ✓ Can detect novel/zero-day attacks")
    logger.info(f"  ✓ Learns from normal behavior patterns only")
    logger.info(f"\nLimitations:")
    logger.info(f"  ✗ May have higher false positive rate than supervised")
    logger.info(f"  ✗ Performance depends on quality of 'normal' training data")
    logger.info(f"  ✗ Less interpretable than tree-based supervised models")
    logger.info("="*80)


if __name__ == '__main__':
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)
