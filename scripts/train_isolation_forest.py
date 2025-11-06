"""
Training script for Isolation Forest Unsupervised Anomaly Detection
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
from models.unsupervised_iforest import IsolationForestAnomalyDetector
from feature_selection import HybridFeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/iforest_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(base_config_path: str = 'configs/base_config.yaml', 
                model_config_path: str = 'configs/isolation_forest_config.yaml') -> dict:
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
        description='Train Isolation Forest for unsupervised DNS anomaly detection'
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
        default='configs/isolation_forest_config.yaml',
        help='Path to model-specific configuration file'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='iforest_detection',
        help='Name for this experiment'
    )
    
    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Fraction of data to use (0-1). None = use full dataset'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=None,
        help='Number of isolation trees. Overrides config.'
    )
    
    parser.add_argument(
        '--contamination',
        type=float,
        default=None,
        help='Expected proportion of outliers (0-0.5). Overrides config.'
    )
    
    parser.add_argument(
        '--max-samples',
        type=str,
        default=None,
        help='Number of samples per tree (int or "auto"). Overrides config.'
    )
    
    parser.add_argument(
        '--max-features',
        type=float,
        default=None,
        help='Proportion of features per tree (0-1). Overrides config.'
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
        help='Path to another model results for comparison (e.g., OCSVM)'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline for Isolation Forest."""
    
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.base_config, args.config)
    
    # Override config with command line arguments
    if args.sample is not None:
        config['data']['sample_fraction'] = args.sample
    
    # Get Isolation Forest parameters from config or command line (command line takes precedence)
    n_estimators = (args.n_estimators if args.n_estimators is not None 
                   else config.get('isolation_forest', {}).get('n_estimators', 100))
    contamination = (args.contamination if args.contamination is not None 
                    else config.get('isolation_forest', {}).get('contamination', 0.05))
    max_samples = (args.max_samples if args.max_samples is not None 
                  else config.get('isolation_forest', {}).get('max_samples', 'auto'))
    max_features = (args.max_features if args.max_features is not None 
                   else config.get('isolation_forest', {}).get('max_features', 1.0))
    
    # Convert max_samples if it's a string number
    if isinstance(max_samples, str) and max_samples != 'auto':
        try:
            max_samples = int(max_samples)
        except ValueError:
            pass
    
    use_cache = not args.no_cache and config['output'].get('use_cache', True)
    
    logger.info("="*80)
    logger.info("ISOLATION FOREST UNSUPERVISED DNS ANOMALY DETECTION")
    logger.info("="*80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Trees: {n_estimators}, Contamination: {contamination}")
    logger.info(f"Max samples: {max_samples}, Max features: {max_features}")
    logger.info(f"Sample fraction: {config['data'].get('sample_fraction', 'Full dataset')}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.experiment_name}_n{n_estimators}_cont{int(contamination*100)}_{timestamp}"
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
    # STEP 3: EXTRACT BENIGN TRAINING DATA ONLY
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: EXTRACT BENIGN DATA FOR UNSUPERVISED TRAINING")
    logger.info("="*80)
    
    # Isolation Forest trains on benign traffic ONLY
    X_train_benign = X_train[y_train == 0]
    
    logger.info(f"Training on BENIGN traffic only:")
    logger.info(f"  Benign samples: {len(X_train_benign):,}")
    logger.info(f"  Features: {X_train_benign.shape[1]}")
    logger.info(f"\nNote: Isolation Forest learns normal patterns by isolating them")
    logger.info(f"      Anomalies are easier to isolate (require fewer tree splits)")
    
    # ============================================================================
    # STEP 4: TRAIN ISOLATION FOREST
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAIN ISOLATION FOREST MODEL")
    logger.info("="*80)
    
    detector = IsolationForestAnomalyDetector(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=config.get('isolation_forest', {}).get('bootstrap', False),
        n_jobs=config.get('isolation_forest', {}).get('n_jobs', -1),
        random_state=config['model']['random_state'],
        verbose=config.get('isolation_forest', {}).get('verbose', 0)
    )
    
    # Train on benign data only
    detector.fit(X_train_benign, feature_names=X.columns.tolist())
    
    logger.info(f"\nTraining Statistics:")
    logger.info(f"  Trees in forest: {n_estimators}")
    logger.info(f"  Samples per tree: {max_samples}")
    logger.info(f"  Features per tree: {max_features}")
    logger.info(f"  Outliers in training: {detector.training_stats['n_outliers_in_training']:,} "
                f"({detector.training_stats['outlier_ratio_training']:.4f})")
    logger.info(f"  Anomaly score range: [{detector.training_stats['anomaly_score_min']:.4f}, "
                f"{detector.training_stats['anomaly_score_max']:.4f}]")
    logger.info(f"  Decision score range: [{detector.training_stats['decision_min']:.4f}, "
                f"{detector.training_stats['decision_max']:.4f}]")
    
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
    
    model_path = models_dir / 'iforest_detector.pkl'
    detector.save_model(model_path)
    
    # Save experiment configuration
    exp_config = {
        'experiment_name': exp_name,
        'timestamp': timestamp,
        'model_type': 'IsolationForest',
        'n_estimators': n_estimators,
        'contamination': contamination,
        'max_samples': str(max_samples),
        'max_features': max_features,
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
    # STEP 7: COMPARE WITH OTHER MODEL (OPTIONAL)
    # ============================================================================
    if args.compare_model:
        logger.info("\n" + "="*80)
        logger.info("STEP 7: COMPARE WITH OTHER MODEL")
        logger.info("="*80)
        
        try:
            # Try to load metrics from other model
            compare_metrics_path = Path(args.compare_model) / 'metrics' / 'experiment_config.json'
            if not compare_metrics_path.exists():
                # Try alternative path
                compare_metrics_path = Path(args.compare_model) / 'metrics' / 'metrics.json'
            
            with open(compare_metrics_path, 'r') as f:
                compare_data = json.load(f)
            
            # Extract metrics (handle different formats)
            if 'test_metrics' in compare_data:
                compare_metrics = compare_data['test_metrics']
                compare_model_name = compare_data.get('model_type', 'Other Model')
            else:
                compare_metrics = compare_data
                compare_model_name = 'Other Model'
            
            logger.info(f"\nComparing with: {compare_model_name}")
            logger.info(f"{'Metric':<25} {'Isolation Forest':<20} {compare_model_name:<20} {'Difference':<15}")
            logger.info("-" * 80)
            
            comparison_metrics = ['accuracy', 'precision_malicious', 'recall_malicious', 'f1_malicious', 'roc_auc']
            for metric in comparison_metrics:
                if metric in metrics and metric in compare_metrics:
                    iforest_val = metrics[metric]
                    compare_val = compare_metrics[metric]
                    diff = iforest_val - compare_val
                    logger.info(f"{metric:<25} {iforest_val:<20.4f} {compare_val:<20.4f} {diff:+.4f}")
            
        except Exception as e:
            logger.error(f"Could not compare with other model: {e}")
    
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
    
    # Example: Explain one anomaly
    if len(top_anomalies) > 0:
        # Get the actual index from the ranking
        most_anomalous_original_idx = top_anomalies.iloc[0]['index']
        
        # Find position in X_test
        if most_anomalous_original_idx in X_test.index:
            pos = X_test.index.get_loc(most_anomalous_original_idx)
            explanation = detector.explain_anomaly(X_test, pos)
            
            logger.info(f"\nExample: Most Anomalous Sample Analysis")
            logger.info(f"  Original Index: {most_anomalous_original_idx}")
            logger.info(f"  Prediction: {explanation['prediction']}")
            logger.info(f"  Anomaly Score: {explanation['anomaly_score']:.4f}")
            logger.info(f"  Anomaly Probability: {explanation['anomaly_probability']:.4f}")
            logger.info(f"  Interpretation: {explanation['interpretation']}")
        else:
            logger.warning(f"Most anomalous index {most_anomalous_original_idx} not found in X_test")
    
    # ============================================================================
    # STEP 9: FEATURE IMPORTANCE PROXY (OPTIONAL)
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 9: FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    
    # Note: This is computationally expensive for large datasets
    sample_size = min(1000, len(X_test))
    logger.info(f"Computing feature importance on {sample_size} samples...")
    logger.info("(This may take a few minutes...)")
    
    try:
        importance_df = detector.get_feature_importance_proxy(X_test, n_samples=sample_size)
        
        logger.info(f"\nTop 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.6f}")
        
        # Save importance
        importance_df.to_csv(metrics_dir / 'feature_importance.csv', index=False)
        logger.info(f"\nFeature importance saved to: {metrics_dir / 'feature_importance.csv'}")
        
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
    
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
    logger.info(f"\nAdvantages of Isolation Forest:")
    logger.info(f"  ✓ Very fast training and prediction")
    logger.info(f"  ✓ Scales well to large datasets (linear complexity)")
    logger.info(f"  ✓ No distance/density calculations needed")
    logger.info(f"  ✓ Handles high-dimensional data well")
    logger.info(f"  ✓ Less sensitive to outliers in training data")
    logger.info(f"  ✓ Can explain anomalies via isolation paths")
    logger.info(f"\nLimitations:")
    logger.info(f"  ✗ May struggle with anomalies that appear in clusters")
    logger.info(f"  ✗ Performance depends on contamination parameter")
    logger.info(f"  ✗ Less effective when anomalies are not sparse")
    logger.info("="*80)


if __name__ == '__main__':
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)
