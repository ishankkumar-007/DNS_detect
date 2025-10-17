"""
Utility Functions for DNS Spoofing Detection Project
Common helpers for metrics, visualization, and logging
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )


def save_metrics(metrics: Dict, output_path: str):
    """
    Save evaluation metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save metrics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            serializable_metrics[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {output_path}")


def load_metrics(metrics_path: str) -> Dict:
    """
    Load metrics from JSON file
    
    Args:
        metrics_path: Path to metrics file
        
    Returns:
        Dictionary of metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"Metrics loaded from {metrics_path}")
    return metrics


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution",
                           save_path: Optional[str] = None, show_plot: bool = False):
    """
    Plot class distribution
    
    Args:
        y: Target labels
        title: Plot title
        save_path: Path to save plot
        show_plot: Whether to display plot interactively
    """
    plt.figure(figsize=(10, 6))
    
    value_counts = y.value_counts().sort_index()
    
    ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
    
    # Add value labels on bars
    for i, v in enumerate(value_counts.values):
        ax.text(i, v + max(value_counts.values)*0.01, str(v), 
               ha='center', va='bottom', fontweight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_feature_correlation(df: pd.DataFrame, top_n: int = 20,
                            save_path: Optional[str] = None, show_plot: bool = False):
    """
    Plot correlation heatmap for top features
    
    Args:
        df: Features DataFrame
        top_n: Number of features to include
        save_path: Path to save plot
        show_plot: Whether to display plot interactively
    """
    # Calculate correlation with target if available
    if 'label' in df.columns:
        target_corr = df.corr()['label'].abs().sort_values(ascending=False)
        top_features = target_corr.head(top_n + 1).index.tolist()
        df_subset = df[top_features]
    else:
        # Just take first top_n features
        df_subset = df.iloc[:, :top_n]
    
    plt.figure(figsize=(12, 10))
    correlation = df_subset.corr()
    
    sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title(f'Feature Correlation Heatmap (Top {top_n} Features)', 
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_missing_values(df: pd.DataFrame, save_path: Optional[str] = None, show_plot: bool = False):
    """
    Plot missing values analysis
    
    Args:
        df: DataFrame to analyze
        save_path: Path to save plot
        show_plot: Whether to display plot interactively
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'feature': missing.index,
        'missing_count': missing.values,
        'missing_percentage': missing_pct.values
    })
    
    # Filter features with missing values
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
        'missing_percentage', ascending=False
    )
    
    if len(missing_df) == 0:
        logger.info("No missing values found")
        return
    
    plt.figure(figsize=(12, max(6, len(missing_df) * 0.3)))
    
    ax = sns.barplot(data=missing_df.head(30), y='feature', x='missing_percentage', 
                    palette='Reds_r')
    
    plt.title('Missing Values Analysis (Top 30 Features)', fontsize=16, fontweight='bold')
    plt.xlabel('Missing Percentage (%)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Missing values plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_models(metrics_dict: Dict[str, Dict], metric_name: str = 'accuracy',
                  save_path: Optional[str] = None, show_plot: bool = False):
    """
    Compare multiple models on a specific metric
    
    Args:
        metrics_dict: Dictionary of {model_name: metrics_dict}
        metric_name: Metric to compare
        save_path: Path to save plot
        show_plot: Whether to display plot interactively
    """
    model_names = list(metrics_dict.keys())
    metric_values = [metrics_dict[name].get(metric_name, 0) for name in model_names]
    
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(x=model_names, y=metric_values, palette='viridis')
    
    # Add value labels
    for i, v in enumerate(metric_values):
        ax.text(i, v + max(metric_values)*0.01, f'{v:.4f}', 
               ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Model Comparison: {metric_name.replace("_", " ").title()}',
             fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(metric_values) * 1.1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    plt.show()


def print_summary_stats(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print summary statistics for DataFrame
    
    Args:
        df: DataFrame to summarize
        name: Name for display
    """
    logger.info("\n" + "="*50)
    logger.info(f"{name} Summary Statistics")
    logger.info("="*50)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"\nData types:\n{df.dtypes.value_counts()}")
    logger.info(f"\nMissing values: {df.isnull().sum().sum()}")
    
    if 'label' in df.columns:
        logger.info(f"\nLabel distribution:\n{df['label'].value_counts()}")
    
    logger.info("="*50)


def create_experiment_folder(base_path: str = "results", 
                            experiment_name: Optional[str] = None) -> Path:
    """
    Create timestamped experiment folder
    
    Args:
        base_path: Base directory for experiments
        experiment_name: Optional experiment name
        
    Returns:
        Path to experiment folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        folder_name = f"{experiment_name}_{timestamp}"
    else:
        folder_name = f"experiment_{timestamp}"
    
    exp_path = Path(base_path) / folder_name
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_path / "plots").mkdir(exist_ok=True)
    (exp_path / "metrics").mkdir(exist_ok=True)
    (exp_path / "models").mkdir(exist_ok=True)
    
    logger.info(f"Created experiment folder: {exp_path}")
    
    return exp_path


def save_dataframe_sample(df: pd.DataFrame, output_path: str, n_samples: int = 1000):
    """
    Save a sample of DataFrame for inspection
    
    Args:
        df: DataFrame to sample
        output_path: Output path
        n_samples: Number of samples
    """
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.csv':
        sample.to_csv(output_path, index=False)
    elif output_path.suffix == '.parquet':
        sample.to_parquet(output_path, index=False)
    else:
        sample.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(sample)} samples to {output_path}")


def calculate_memory_usage(df: pd.DataFrame) -> Dict:
    """
    Calculate detailed memory usage of DataFrame
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory statistics
    """
    mem_usage = df.memory_usage(deep=True)
    
    stats = {
        'total_mb': mem_usage.sum() / 1024**2,
        'per_column_mb': {col: mem_usage[col] / 1024**2 
                         for col in df.columns},
        'top_memory_columns': mem_usage.sort_values(ascending=False).head(10).to_dict()
    }
    
    return stats


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get comprehensive feature statistics
    
    Args:
        df: Features DataFrame
        
    Returns:
        DataFrame with feature statistics
    """
    stats = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null_count': df.isnull().sum(),
        'null_percentage': (df.isnull().sum() / len(df)) * 100,
        'unique_values': df.nunique(),
        'memory_mb': df.memory_usage(deep=True) / 1024**2
    })
    
    # Add numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats.loc[numeric_cols, 'mean'] = df[numeric_cols].mean()
        stats.loc[numeric_cols, 'std'] = df[numeric_cols].std()
        stats.loc[numeric_cols, 'min'] = df[numeric_cols].min()
        stats.loc[numeric_cols, 'max'] = df[numeric_cols].max()
    
    return stats


def main():
    """Test utility functions"""
    logger.info("Utility functions module ready")


if __name__ == "__main__":
    main()
