"""
Hybrid Feature Selection Module for DNS Spoofing Detection
Implements SelectKBest + SHAP feature selection strategy
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import MinMaxScaler
import shap
import lightgbm as lgb
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridFeatureSelector:
    """
    Hybrid feature selection using SelectKBest and SHAP
    Two-stage approach:
    1. SelectKBest: Statistical filter to reduce dimensionality
    2. SHAP: Model-based explainability for fine-tuning
    """
    
    def __init__(self, k_best: int = 50, shap_top_n: int = 30, random_state: int = 42,
                 cache_dir: str = 'cache', use_cache: bool = True, sample_fraction: Optional[float] = None):
        """
        Initialize hybrid feature selector
        
        Args:
            k_best: Number of features to select in SelectKBest stage
            shap_top_n: Number of top features to select from SHAP stage
            random_state: Random seed for reproducibility
            cache_dir: Directory to store cached feature selections
            use_cache: Whether to use caching
            sample_fraction: Sample fraction used for data (for cache key)
        """
        self.k_best = k_best
        self.shap_top_n = shap_top_n
        self.random_state = random_state
        self.sample_fraction = sample_fraction
        
        self.selectkbest_selector = None
        self.selected_features_kbest = None
        self.shap_values = None
        self.final_features = None
        self.feature_importance_df = None
        
        # Cache configuration
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info(f"Feature selection caching enabled: {self.cache_dir}")
        
    def stage1_selectkbest(self, X: pd.DataFrame, y: pd.Series, 
                           method: str = 'mutual_info') -> pd.DataFrame:
        """
        Stage 1: SelectKBest feature selection
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('mutual_info', 'chi2', 'f_classif')
            
        Returns:
            DataFrame with top k features
        """
        logger.info(f"Stage 1: SelectKBest with {method} method (k={self.k_best})")
        
        # Ensure all features are non-negative for chi2
        if method == 'chi2':
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Select scoring function
        score_func_map = {
            'chi2': chi2,
            'mutual_info': mutual_info_classif,
            'f_classif': f_classif
        }
        score_func = score_func_map.get(method, mutual_info_classif)
        
        # Apply SelectKBest
        self.selectkbest_selector = SelectKBest(
            score_func=score_func,
            k=min(self.k_best, X.shape[1])
        )
        
        X_selected = self.selectkbest_selector.fit_transform(X_scaled, y)
        
        # Get selected feature names
        selected_mask = self.selectkbest_selector.get_support()
        self.selected_features_kbest = X.columns[selected_mask].tolist()
        
        logger.info(f"Selected {len(self.selected_features_kbest)} features from Stage 1")
        
        # Return original values (not scaled) for selected features
        return X[self.selected_features_kbest]
    
    def stage2_shap(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> List[str]:
        """
        Stage 2: SHAP-based feature selection
        
        Args:
            X_train: Training features (after SelectKBest)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            List of top N feature names based on SHAP importance
        """
        logger.info(f"Stage 2: SHAP analysis (selecting top {self.shap_top_n} features)")
        
        # Train a LightGBM model for SHAP analysis
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Determine if binary or multi-class
        num_classes = len(np.unique(y_train))
        
        if num_classes == 2:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
        else:
            params = {
                'objective': 'multiclass',
                'num_class': num_classes,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
        
        # Train model
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
        else:
            model = lgb.train(params, train_data, num_boost_round=100)
        
        # Calculate SHAP values
        logger.info("Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        
        # Use a sample for SHAP calculation to speed up
        sample_size = min(1000, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=self.random_state)
        
        self.shap_values = explainer.shap_values(X_sample)
        
        # Debug: Log SHAP values structure
        if isinstance(self.shap_values, list):
            logger.info(f"SHAP values: list with {len(self.shap_values)} classes")
            logger.info(f"Each class shape: {self.shap_values[0].shape}")
        else:
            logger.info(f"SHAP values shape: {self.shap_values.shape}")
        
        # Calculate mean absolute SHAP values for feature importance
        if isinstance(self.shap_values, list):
            # Multi-class as list: each element is (n_samples, n_features) for each class
            # Average the absolute values across samples for each class, then average across classes
            class_importances = []
            for i, class_shap in enumerate(self.shap_values):
                # class_shap shape: (n_samples, n_features)
                logger.debug(f"Class {i} SHAP shape: {class_shap.shape}")
                # Take mean across samples: (n_features,)
                class_importance = np.abs(class_shap).mean(axis=0)
                logger.debug(f"Class {i} importance shape after mean: {class_importance.shape}")
                class_importances.append(class_importance)
            # Stack and average across classes: (n_features,)
            class_importances = np.array(class_importances)  # Shape: (n_classes, n_features)
            logger.info(f"Stacked class importances shape: {class_importances.shape}")
            shap_importance = np.mean(class_importances, axis=0)
            logger.info(f"Final SHAP importance shape after averaging classes: {shap_importance.shape}")
        elif len(self.shap_values.shape) == 3:
            # Multi-class as 3D array: (n_samples, n_features, n_classes)
            logger.info(f"3D SHAP array detected: {self.shap_values.shape}")
            # Take mean absolute value across samples (axis=0) and classes (axis=2)
            # Result should be (n_features,)
            shap_importance = np.abs(self.shap_values).mean(axis=(0, 2))
            logger.info(f"Final SHAP importance shape after averaging: {shap_importance.shape}")
        else:
            # Binary or regression: (n_samples, n_features)
            shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Ensure shap_importance is 1D and has correct length
        shap_importance = np.asarray(shap_importance).flatten()
        
        # Verify lengths match
        if len(shap_importance) != len(X_train.columns):
            logger.error(f"SHAP importance length {len(shap_importance)} != feature count {len(X_train.columns)}")
            raise ValueError(f"SHAP values shape mismatch: got {len(shap_importance)}, expected {len(X_train.columns)}")
        
        # Create feature importance DataFrame
        self.feature_importance_df = pd.DataFrame({
            'feature': X_train.columns.tolist(),
            'shap_importance': shap_importance.tolist()
        }).sort_values('shap_importance', ascending=False)
        
        # Select top N features
        self.final_features = self.feature_importance_df.head(self.shap_top_n)['feature'].tolist()
        
        logger.info(f"Selected {len(self.final_features)} features from Stage 2")
        logger.info(f"Top 10 features: {self.final_features[:10]}")
        
        return self.final_features
    
    def _get_cache_key(self, method: str, n_features: int) -> str:
        """
        Generate cache key for feature selection results.
        
        Args:
            method: Selection method used
            n_features: Number of input features
            
        Returns:
            Cache filename
        """
        # Include sample fraction in cache key
        sample_str = f"{self.sample_fraction}" if self.sample_fraction is not None else "full"
        cache_key = f"feature_selection_{method}_k{self.k_best}_shap{self.shap_top_n}_nf{n_features}_sample{sample_str}.pkl"
        return cache_key
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Load feature selection from cache if it exists.
        
        Args:
            cache_key: Cache filename
            
        Returns:
            Cached feature selection dict or None if not found
        """
        if not self.use_cache:
            return None
        
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            logger.info(f"Loading feature selection from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"✓ Loaded {len(cached_data['final_features'])} selected features from cache")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load feature selection cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str):
        """
        Save feature selection results to cache.
        
        Args:
            cache_key: Cache filename
        """
        if not self.use_cache:
            return
        
        cache_path = self.cache_dir / cache_key
        logger.info(f"Saving feature selection to cache: {cache_path}")
        try:
            cached_data = {
                'selected_features_kbest': self.selected_features_kbest,
                'final_features': self.final_features,
                'feature_importance_df': self.feature_importance_df,
                'k_best': self.k_best,
                'shap_top_n': self.shap_top_n
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"✓ Cached feature selection with {len(self.final_features)} features")
        except Exception as e:
            logger.warning(f"Failed to save feature selection cache: {e}")
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame = None, y_val: pd.Series = None,
                      method: str = 'mutual_info') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Complete hybrid feature selection pipeline with caching support.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            method: SelectKBest method
            
        Returns:
            Tuple of (selected training features, selected validation features)
        """
        logger.info("Starting hybrid feature selection pipeline...")
        logger.info(f"Initial features: {X_train.shape[1]}")
        
        # Try to load from cache
        cache_key = self._get_cache_key(method, X_train.shape[1])
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            # Restore from cache
            self.selected_features_kbest = cached_data['selected_features_kbest']
            self.final_features = cached_data['final_features']
            self.feature_importance_df = cached_data['feature_importance_df']
            
            # Return selected features
            X_train_final = X_train[self.final_features]
            X_val_final = X_val[self.final_features] if X_val is not None else None
            
            logger.info(f"Using cached feature selection: {len(self.final_features)} features")
            return X_train_final, X_val_final
        
        # Cache miss - perform feature selection
        logger.info("Cache miss - performing feature selection...")
        
        # Stage 1: SelectKBest
        X_train_kbest = self.stage1_selectkbest(X_train, y_train, method=method)
        
        if X_val is not None:
            X_val_kbest = X_val[self.selected_features_kbest]
        else:
            X_val_kbest = None
        
        # Stage 2: SHAP
        self.stage2_shap(X_train_kbest, y_train, X_val_kbest, y_val)
        
        # Save to cache
        self._save_to_cache(cache_key)
        
        # Return final selected features
        X_train_final = X_train_kbest[self.final_features]
        
        if X_val is not None:
            X_val_final = X_val_kbest[self.final_features]
        else:
            X_val_final = None
        
        logger.info(f"Final features: {X_train_final.shape[1]}")
        logger.info("Feature selection complete!")
        
        return X_train_final, X_val_final
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using selected features
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with selected features only
        """
        if self.final_features is None:
            raise ValueError("Must call fit_transform first")
        
        return X[self.final_features]
    
    def plot_feature_importance(self, save_path: str = None, show_plot: bool = False):
        """
        Plot feature importance from SHAP analysis
        
        Args:
            save_path: Path to save plot (optional)
            show_plot: Whether to display plot interactively
        """
        if self.feature_importance_df is None:
            raise ValueError("Must run stage2_shap first")
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance_df.head(20)
        
        sns.barplot(data=top_features, y='feature', x='shap_importance', palette='viridis')
        plt.title('Top 20 Features by SHAP Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_shap_summary(self, X_sample: pd.DataFrame, save_path: str = None, show_plot: bool = False):
        """
        Plot SHAP summary plot
        
        Args:
            X_sample: Sample data for visualization
            save_path: Path to save plot (optional)
            show_plot: Whether to display plot interactively
        """
        if self.shap_values is None:
            raise ValueError("Must run stage2_shap first")
        
        plt.figure(figsize=(10, 8))
        
        # For multi-class, use the first class or average
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.summary_plot(
            shap_vals,
            X_sample,
            plot_type="bar",
            show=False,
            max_display=20
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_feature_names(self) -> List[str]:
        """
        Get final selected feature names
        
        Returns:
            List of selected feature names
        """
        if self.final_features is None:
            raise ValueError("Must call fit_transform first")
        
        return self.final_features


def main():
    """Test feature selection pipeline"""
    # This would be run with actual data
    logger.info("Feature selection module ready")


if __name__ == "__main__":
    main()
