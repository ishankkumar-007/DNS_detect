"""
Data Preprocessing Module for DNS Spoofing Detection
Handles large-scale CSV loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from sklearn.preprocessing import LabelEncoder
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DNSDataPreprocessor:
    """
    Preprocessor for BCCC-CIC-Bell-DNS-2024 dataset
    Handles memory-efficient loading and preprocessing of large DNS traffic data
    """
    
    def __init__(self, data_dir: str, use_dask: bool = True, cache_dir: str = 'cache', use_cache: bool = True):
        """
        Initialize preprocessor
        
        Args:
            data_dir: Path to BCCC-CIC-Bell-DNS-2024 directory
            use_dask: Use Dask for large files (>50MB)
            cache_dir: Directory to store cached preprocessed data
            use_cache: Whether to use caching
        """
        self.data_dir = Path(data_dir)
        self.use_dask = use_dask
        self.label_encoder = LabelEncoder()
        self.sample_frac = None  # Store sample fraction for use during loading
        
        # Cache configuration
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info(f"Caching enabled: {self.cache_dir}")
        
        # Define file paths
        self.exf_dir = self.data_dir / "BCCC-CIC-Bell-DNS-EXF"
        self.mal_dir = self.data_dir / "BCCC-CIC-Bell-DNS-Mal"
        
        # Feature columns to drop (non-predictive identifiers)
        self.drop_columns = ['flow_id', 'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port']
        
        # Categorical features that need encoding
        self.categorical_features = [
            'protocol', 'dns_domain_name', 'dns_top_level_domain', 
            'dns_second_level_domain'
        ]
        
    def load_csv_efficiently(self, file_path: Path, chunksize: int = 10000) -> pd.DataFrame:
        """
        Load CSV file efficiently based on size
        
        Args:
            file_path: Path to CSV file
            chunksize: Chunk size for large files
            
        Returns:
            Loaded DataFrame (sampled if sample_frac is set)
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"Loading {file_path.name} ({file_size_mb:.2f} MB)")
        
        if file_size_mb > 50 and self.use_dask:
            # Use Dask for large files with dtype specification to handle 'not a dns flow' strings
            logger.info(f"Using Dask for large file: {file_path.name}")
            
            # Read with all object dtypes first to avoid dtype inference issues
            ddf = dd.read_csv(file_path, assume_missing=True, dtype=object)
            
            # Sample in Dask if needed (more efficient)
            if self.sample_frac is not None and self.sample_frac < 1.0:
                ddf = ddf.sample(frac=self.sample_frac, random_state=42)
            
            df = ddf.compute()
            
            # Convert numeric columns manually, handling 'not a dns flow' strings
            for col in df.columns:
                if col not in ['label', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'protocol',
                              'dns_domain_name', 'dns_top_level_domain', 'dns_second_level_domain',
                              'uni_gram_domain_name', 'bi_gram_domain_name', 'tri_gram_domain_name',
                              'character_distribution', 'query_resource_record_type', 
                              'ans_resource_record_type', 'query_resource_record_class',
                              'ans_resource_record_class']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        else:
            # Use pandas with chunking for medium files
            if file_size_mb > 20:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
                    # Sample chunks if needed
                    if self.sample_frac is not None and self.sample_frac < 1.0:
                        chunk = chunk.sample(frac=self.sample_frac, random_state=42)
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path, low_memory=False)
                # Sample if needed
                if self.sample_frac is not None and self.sample_frac < 1.0:
                    df = df.sample(frac=self.sample_frac, random_state=42)
                return df
    
    def load_exfiltration_data(self) -> pd.DataFrame:
        """
        Load all exfiltration data from BCCC-CIC-Bell-DNS-EXF directory
        
        Returns:
            Combined DataFrame with exfiltration data
        """
        logger.info("Loading exfiltration data...")
        dataframes = []
        
        exf_files = list(self.exf_dir.glob("*.csv"))
        
        for file_path in exf_files:
            df = self.load_csv_efficiently(file_path)
            dataframes.append(df)
            logger.info(f"Loaded {file_path.name}: {len(df)} rows")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Total exfiltration data: {len(combined_df)} rows")
        return combined_df
    
    def load_malicious_data(self) -> pd.DataFrame:
        """
        Load all malicious DNS traffic from BCCC-CIC-Bell-DNS-Mal directory
        
        Returns:
            Combined DataFrame with malicious data
        """
        logger.info("Loading malicious data...")
        dataframes = []
        
        mal_files = list(self.mal_dir.glob("*.csv"))
        
        for file_path in mal_files:
            df = self.load_csv_efficiently(file_path)
            dataframes.append(df)
            logger.info(f"Loaded {file_path.name}: {len(df)} rows")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Total malicious data: {len(combined_df)} rows")
        return combined_df
    
    def _get_cache_key(self, stage: str) -> str:
        """
        Generate cache key based on data configuration and stage.
        
        Args:
            stage: Processing stage ('raw' or 'preprocessed')
            
        Returns:
            Cache filename
        """
        # Create a deterministic hash from sample fraction
        sample_str = f"{self.sample_frac}" if self.sample_frac is not None else "full"
        cache_key = f"dns_data_{stage}_{sample_str}.pkl"
        return cache_key
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if it exists.
        
        Args:
            cache_key: Cache filename
            
        Returns:
            Cached DataFrame or None if not found
        """
        if not self.use_cache:
            return None
        
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"✓ Loaded {len(data):,} rows from cache")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """
        Save data to cache.
        
        Args:
            data: DataFrame to cache
            cache_key: Cache filename
        """
        if not self.use_cache:
            return
        
        cache_path = self.cache_dir / cache_key
        logger.info(f"Saving to cache: {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Cached {len(data):,} rows ({cache_size_mb:.2f} MB)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def load_all_data(self, sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Load all data from both directories with caching support.
        
        Args:
            sample_frac: Fraction of data to sample (for testing/debugging)
                        Sampling is done during loading for efficiency
            
        Returns:
            Combined DataFrame with all data (sampled if requested)
        """
        logger.info("Loading all data...")
        
        # Store sample fraction for use during file loading
        self.sample_frac = sample_frac
        
        if sample_frac is not None and sample_frac < 1.0:
            logger.info(f"Will sample {sample_frac*100:.1f}% of data from each file during loading")
        
        # Try to load from cache first
        cache_key = self._get_cache_key('raw')
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Load from source files
        logger.info("Cache miss - loading from source files...")
        exf_df = self.load_exfiltration_data()
        mal_df = self.load_malicious_data()
        
        combined_df = pd.concat([exf_df, mal_df], ignore_index=True)
        logger.info(f"Total combined data: {len(combined_df):,} rows")
        
        # Save to cache
        self._save_to_cache(combined_df, cache_key)
        
        return combined_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values...")
        
        # Log missing value statistics
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df)) * 100
        features_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(features_with_missing) > 0:
            logger.info(f"Features with missing values:\n{features_with_missing.head(10)}")
        
        # Strategy: Fill numeric NaN with 0 (common for statistical features with insufficient data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Fill categorical NaN with 'unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('unknown')
        
        logger.info("Missing values handled")
        return df
    
    def process_dns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process DNS-specific features (n-grams, character distributions)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed DNS features
        """
        logger.info("Processing DNS-specific features...")
        
        # Convert string representations of lists to actual lengths
        ngram_features = ['uni_gram_domain_name', 'bi_gram_domain_name', 'tri_gram_domain_name']
        
        for feature in ngram_features:
            if feature in df.columns:
                # Extract length from string representation
                df[f'{feature}_count'] = df[feature].astype(str).str.count(',') + 1
                # Drop original string representation (too high cardinality)
                df = df.drop(columns=[feature])
        
        # Process character_distribution (convert string dict to numeric features)
        if 'character_distribution' in df.columns:
            # Drop original - too high cardinality for direct encoding
            df = df.drop(columns=['character_distribution'])
        
        logger.info("DNS features processed")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        # Drop high-cardinality domain names (already have n-gram counts)
        domain_features = ['dns_domain_name', 'dns_second_level_domain']
        for feature in domain_features:
            if feature in df.columns:
                df = df.drop(columns=[feature])
        
        # Encode dns_top_level_domain (relatively low cardinality)
        if 'dns_top_level_domain' in df.columns:
            # Use frequency encoding for TLD
            tld_freq = df['dns_top_level_domain'].value_counts(normalize=True).to_dict()
            df['tld_frequency'] = df['dns_top_level_domain'].map(tld_freq)
            df = df.drop(columns=['dns_top_level_domain'])
        
        # Protocol is always DNS, can drop
        if 'protocol' in df.columns:
            df = df.drop(columns=['protocol'])
        
        logger.info("Categorical features encoded")
        return df
    
    def preprocess_features(self, df: pd.DataFrame, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline with caching support.
        
        Args:
            df: Input DataFrame
            use_cache: Whether to use cached preprocessed features
            
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        # Try to load preprocessed data from cache
        if use_cache and self.use_cache:
            cache_key = self._get_cache_key('preprocessed')
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                # Split cached data back into X and y
                if 'label' in cached_data.columns:
                    y = cached_data['label'].copy()
                    X = cached_data.drop(columns=['label'])
                    logger.info(f"✓ Using cached preprocessed features: {X.shape[1]} features")
                    return X, y
        
        logger.info("Starting feature preprocessing...")
        
        # Separate features and labels
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
        
        y = df['label'].copy()
        X = df.drop(columns=['label'])
        
        # Drop identifier columns
        X = X.drop(columns=[col for col in self.drop_columns if col in X.columns])
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Process DNS features
        X = self.process_dns_features(X)
        
        # Encode categorical features
        X = self.encode_categorical_features(X)
        
        # Final check: ensure all columns are numeric
        logger.info("Ensuring all features are numeric...")
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Converting non-numeric column {col} to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce')
                # Fill any resulting NaN with 0
                X[col] = X[col].fillna(0)
        
        # Verify no non-numeric values remain
        non_numeric_cols = X.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            logger.error(f"Non-numeric columns remain: {non_numeric_cols.tolist()}")
            raise ValueError(f"Failed to convert columns to numeric: {non_numeric_cols.tolist()}")
        
        # Sanitize column names (LightGBM doesn't support special JSON characters)
        logger.info("Sanitizing feature names for LightGBM compatibility...")
        X.columns = [self._sanitize_feature_name(col) for col in X.columns]
        
        # Convert all numeric columns to float64 to avoid dtype issues
        logger.info("Converting all features to float64...")
        for col in X.columns:
            X[col] = X[col].astype('float64')
        
        # Handle infinite values and extreme numbers
        logger.info("Handling infinite and extreme values...")
        # Replace infinity with NaN, then fill with 0
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Count and log infinite values
        inf_counts = X.isnull().sum()
        inf_features = inf_counts[inf_counts > 0]
        if len(inf_features) > 0:
            logger.warning(f"Found infinite/NaN values in {len(inf_features)} features")
            logger.debug(f"Top features with inf/NaN: {inf_features.head(10).to_dict()}")
        
        # Fill NaN (including converted inf) with 0
        X = X.fillna(0)
        
        # Clip extreme values to prevent overflow
        # Use robust scaling bounds (e.g., 99.9th percentile)
        for col in X.columns:
            if X[col].std() > 0:  # Only clip if there's variation
                upper_bound = X[col].quantile(0.999)
                lower_bound = X[col].quantile(0.001)
                if np.isfinite(upper_bound) and np.isfinite(lower_bound):
                    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Convert labels to binary (Benign vs. Malicious)
        logger.info("Converting labels to binary classification...")
        logger.info(f"Original label distribution:\n{y.value_counts()}")
        
        # Create binary labels: 0 = Benign, 1 = Malicious
        y_binary = y.apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
        
        logger.info(f"Binary label distribution:\n{y_binary.value_counts()}")
        logger.info(f"Binary label mapping: 0=Benign, 1=Malicious")
        
        # Store label encoder mapping for binary classification
        self.label_encoder.classes_ = np.array(['Benign', 'Malicious'])
        
        logger.info(f"Final feature shape: {X.shape}")
        
        # Save preprocessed data to cache
        if self.use_cache:
            # Combine X and y for caching
            cached_df = X.copy()
            cached_df['label'] = y_binary
            cache_key = self._get_cache_key('preprocessed')
            self._save_to_cache(cached_df, cache_key)
        
        return X, y_binary
    
    def _sanitize_feature_name(self, name: str) -> str:
        """
        Sanitize feature name to be compatible with LightGBM
        Removes special JSON characters: []{}":,
        
        Args:
            name: Original feature name
            
        Returns:
            Sanitized feature name
        """
        # Replace special characters with underscores
        special_chars = ['[', ']', '{', '}', '"', ':', ',', '<', '>', '\\', '/']
        sanitized = name
        for char in special_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove multiple consecutive underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        return sanitized
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature names after preprocessing
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            List of feature names
        """
        return df.columns.tolist()


def main():
    """Test preprocessing pipeline"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = DNSDataPreprocessor(
        data_dir=config['data']['dataset_path'],
        use_dask=config['data']['use_dask']
    )
    
    # Load and preprocess data (with sampling for testing)
    df = preprocessor.load_all_data(sample_frac=0.1)
    X, y = preprocessor.preprocess_features(df)
    
    logger.info(f"Preprocessing complete!")
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Samples: {X.shape[0]}")


if __name__ == "__main__":
    main()
