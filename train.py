"""
Main Training Script for DNS Spoofing Detection
End-to-end pipeline: Data Loading → Preprocessing → Feature Selection → Training → Evaluation
"""

import sys
import yaml
import argparse
from pathlib import Path
import logging
import pickle
import hashlib
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import DNSDataPreprocessor
from feature_selection import HybridFeatureSelector
from model import DNSSpoofingDetector
from utils import (
    setup_logging, save_metrics, plot_class_distribution,
    print_summary_stats, create_experiment_folder
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DNS Spoofing Detection Model")
    
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--sample', type=float, default=None,
        help='Sample fraction of data (e.g., 0.1 for 10%)'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Name for this experiment'
    )
    parser.add_argument(
        '--skip-feature-selection', action='store_true',
        help='Skip feature selection and use all features'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Ignore cached data and reprocess from scratch'
    )
    parser.add_argument(
        '--clear-cache', action='store_true',
        help='Clear all cached data before running'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_cache_key(config: dict) -> str:
    """
    Generate cache key based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Hash string for cache identification
    """
    # Create a unique key based on data settings
    cache_params = {
        'sample_fraction': config['data']['sample_fraction'],
        'dataset_path': config['data']['dataset_path'],
        'test_size': config['data']['test_size'],
        'random_seed': config['experiment']['random_seed']
    }
    
    key_string = str(sorted(cache_params.items()))
    return hashlib.md5(key_string.encode()).hexdigest()[:12]


def save_to_cache(data: dict, cache_path: Path, description: str):
    """
    Save data to cache
    
    Args:
        data: Dictionary of data to cache
        cache_path: Path to cache file
        description: Description for logging
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = cache_path.stat().st_size / (1024 ** 2)
    logger.info(f"[CACHE] Saved {description} to {cache_path} ({size_mb:.2f} MB)")


def load_from_cache(cache_path: Path, description: str) -> dict:
    """
    Load data from cache
    
    Args:
        cache_path: Path to cache file
        description: Description for logging
        
    Returns:
        Cached data dictionary
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        size_mb = cache_path.stat().st_size / (1024 ** 2)
        logger.info(f"[CACHE] Loaded {description} from cache ({size_mb:.2f} MB)")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None


def clear_cache_directory(cache_dir: Path):
    """
    Clear all cached data
    
    Args:
        cache_dir: Cache directory path
    """
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info(f"[CACHE] Cleared cache directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)


def main():
    """Main training pipeline"""
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.sample:
        config['data']['sample_fraction'] = args.sample
    if args.skip_feature_selection:
        config['feature_selection']['enable'] = False
    if args.no_cache:
        config['output']['use_cache'] = False
    
    # Setup cache directory
    cache_dir = Path(config['output']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache_directory(cache_dir)
    
    # Generate cache key
    cache_key = get_cache_key(config)
    preprocessed_cache_path = cache_dir / f"preprocessed_{cache_key}.pkl"
    selected_features_cache_path = cache_dir / f"selected_features_{cache_key}.pkl"
    
    # Setup logging
    setup_logging(
        log_file=config['logging']['log_file'],
        level=getattr(logging, config['logging']['level'])
    )
    
    logger.info("="*80)
    logger.info("DNS SPOOFING DETECTION - TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info("="*80)
    
    # Build experiment name with feature selection parameters
    base_exp_name = args.experiment_name or config['experiment']['name']
    
    # Add feature selection info to experiment name if enabled
    if config['feature_selection']['enable']:
        k_best = config['feature_selection']['selectkbest']['k']
        shap_n = config['feature_selection']['shap']['top_n']
        exp_name = f"{base_exp_name}_k{k_best}_shap{shap_n}"
    else:
        exp_name = base_exp_name
    
    # Create experiment folder
    exp_folder = create_experiment_folder(
        base_path=config['output']['results_dir'],
        experiment_name=exp_name
    )
    
    # ========== STEP 1: DATA LOADING & PREPROCESSING ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA LOADING & PREPROCESSING")
    logger.info("="*80)
    
    # Try to load from cache
    cached_data = None
    if config['output']['use_cache'] and config['output']['cache_preprocessed']:
        logger.info(f"Cache key: {cache_key}")
        cached_data = load_from_cache(preprocessed_cache_path, "preprocessed features")
    
    if cached_data:
        # Use cached data
        X = cached_data['X']
        y = cached_data['y']
        logger.info("[CACHE] Using cached preprocessed data - skipping data loading and preprocessing")
        print_summary_stats(X, "Preprocessed Features (from cache)")
    else:
        # Load and preprocess from scratch
        logger.info("Loading and preprocessing data from scratch...")
        
        preprocessor = DNSDataPreprocessor(
            data_dir=config['data']['dataset_path'],
            use_dask=config['data']['use_dask']
        )
        
        # Load all data
        df = preprocessor.load_all_data(sample_frac=config['data']['sample_fraction'])
        print_summary_stats(df, "Raw Data")
        
        # Plot class distribution
        if 'label' in df.columns:
            plot_class_distribution(
                df['label'],
                title="Class Distribution (Raw Data)",
                save_path=exp_folder / "plots" / "class_distribution_raw.png"
            )
        
        # Preprocess features
        X, y = preprocessor.preprocess_features(df)
        print_summary_stats(X, "Preprocessed Features")
        
        # Clean up raw data to free memory
        del df
        
        # Cache preprocessed data
        if config['output']['cache_preprocessed']:
            cache_data = {'X': X, 'y': y}
            save_to_cache(cache_data, preprocessed_cache_path, "preprocessed features")
    
    # ========== STEP 3: TRAIN/TEST SPLIT ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 3: TRAIN/TEST SPLIT")
    logger.info("="*80)
    
    detector = DNSSpoofingDetector(
        params=config['model'],
        random_state=config['experiment']['random_seed']
    )
    
    X_train, X_test, y_train, y_test = detector.prepare_data(
        X, y,
        test_size=config['data']['test_size'],
        stratify=config['training']['stratify_split']
    )
    
    # Further split train into train/validation
    val_size = config['data']['validation_size']
    if val_size > 0:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=config['experiment']['random_seed'],
            stratify=y_train if config['training']['stratify_split'] else None
        )
    else:
        X_val, y_val = None, None
    
    # ========== STEP 4: FEATURE SELECTION ==========
    feature_selector = None  # Initialize to None
    selected_feature_names = None  # Initialize to None
    
    if config['feature_selection']['enable']:
        logger.info("\n" + "="*80)
        logger.info("STEP 4: HYBRID FEATURE SELECTION")
        logger.info("="*80)
        
        # Try to load cached feature selection
        cached_features = None
        if config['output']['use_cache'] and config['output']['cache_selected_features']:
            cached_features = load_from_cache(selected_features_cache_path, "selected features")
        
        if cached_features and set(cached_features['selected_features']).issubset(set(X_train.columns)):
            # Use cached feature selection
            logger.info("[CACHE] Using cached feature selection")
            selected_feature_names = cached_features['selected_features']
            
            X_train = X_train[selected_feature_names]
            X_test = X_test[selected_feature_names]
            if X_val is not None:
                X_val = X_val[selected_feature_names]
            
            logger.info(f"Final feature count: {X_train.shape[1]}")
            logger.info(f"Selected features: {selected_feature_names}")
        else:
            # Perform feature selection
            logger.info("Performing feature selection from scratch...")
            
            feature_selector = HybridFeatureSelector(
                k_best=config['feature_selection']['selectkbest']['k'],
                shap_top_n=config['feature_selection']['shap']['top_n'],
                random_state=config['experiment']['random_seed']
            )
            
            # Apply feature selection
            X_train_selected, X_val_selected = feature_selector.fit_transform(
                X_train, y_train,
                X_val, y_val,
                method=config['feature_selection']['selectkbest']['score_func']
            )
            
            # Transform test set
            X_test_selected = feature_selector.transform(X_test)
            
            # Plot feature importance
            if config['evaluation']['plot_shap_summary']:
                show_plots = config['evaluation'].get('show_plots', False)
                save_plots = config['evaluation'].get('save_plots', True)
                feature_selector.plot_feature_importance(
                    save_path=exp_folder / "plots" / "feature_importance_shap.png" if save_plots else None,
                    show_plot=show_plots
                )
            
            # Cache selected features
            if config['output']['cache_selected_features']:
                feature_cache_data = {
                    'selected_features': feature_selector.get_feature_names(),
                    'feature_importance': feature_selector.feature_importance_df.to_dict() if feature_selector.feature_importance_df is not None else None
                }
                save_to_cache(feature_cache_data, selected_features_cache_path, "selected features")
            
            # Use selected features
            X_train = X_train_selected
            X_val = X_val_selected
            X_test = X_test_selected
            
            logger.info(f"Final feature count: {X_train.shape[1]}")
            logger.info(f"Selected features: {feature_selector.get_feature_names()}")
    
    else:
        logger.info("\n" + "="*80)
        logger.info("STEP 4: FEATURE SELECTION SKIPPED")
        logger.info("="*80)
    
    # ========== STEP 5: MODEL TRAINING ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 5: MODEL TRAINING")
    logger.info("="*80)
    
    detector.train(
        X_train, y_train,
        X_val, y_val,
        early_stopping_rounds=config['model']['early_stopping_rounds']
    )
    
    # ========== STEP 6: MODEL EVALUATION ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 6: MODEL EVALUATION")
    logger.info("="*80)
    
    # Evaluate on test set
    metrics = detector.evaluate(X_test, y_test)
    
    # Save metrics
    if config['output']['save_metrics']:
        save_metrics(metrics, exp_folder / "metrics" / "evaluation_metrics.json")
    
    # ========== STEP 7: VISUALIZATIONS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 7: GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    show_plots = config['evaluation'].get('show_plots', False)
    save_plots = config['evaluation'].get('save_plots', True)
    
    # Confusion matrix
    if config['evaluation']['plot_confusion_matrix']:
        y_pred = detector.predict(X_test)
        detector.plot_confusion_matrix(
            y_test, y_pred,
            save_path=exp_folder / "plots" / "confusion_matrix.png" if save_plots else None,
            show_plot=show_plots
        )
    
    # Feature importance
    if config['evaluation']['plot_feature_importance']:
        detector.plot_feature_importance(
            importance_type='gain',
            max_features=20,
            save_path=exp_folder / "plots" / "feature_importance_lgbm.png" if save_plots else None,
            show_plot=show_plots
        )
    
    # ROC curve (binary only)
    if config['evaluation']['plot_roc_curve'] and len(y_test.unique()) == 2:
        detector.plot_roc_curve(
            X_test, y_test,
            save_path=exp_folder / "plots" / "roc_curve.png" if save_plots else None,
            show_plot=show_plots
        )
    
    # ========== STEP 8: MODEL SAVING ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 8: SAVING MODEL")
    logger.info("="*80)
    
    if config['output']['save_best_model']:
        model_path = exp_folder / "models" / f"{config['output']['model_name']}.txt"
        
        # Save feature selection info if used
        metadata = {}
        if config['feature_selection']['enable']:
            # Use feature_selector if available, otherwise use cached feature names
            if feature_selector is not None:
                metadata['selected_features'] = feature_selector.get_feature_names()
            elif selected_feature_names is not None:
                metadata['selected_features'] = selected_feature_names
            metadata['feature_selection_method'] = config['feature_selection']['method']
        
        detector.save_model(model_path, metadata=metadata)
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Experiment folder: {exp_folder}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Score (macro): {metrics['f1_macro']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info("="*80)
    
    return metrics


if __name__ == "__main__":
    try:
        metrics = main()
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)
