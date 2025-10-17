# Caching System Documentation

## Overview

The DNS Spoofing Detection system includes an intelligent caching mechanism that dramatically reduces training time for iterative experimentation by saving preprocessed features and feature selections.

## How It Works

### Cache Key Generation

The system generates a unique cache key based on:
- `sample_fraction`: Data sampling ratio
- `dataset_path`: Location of the dataset
- `test_size`: Train/test split ratio
- `random_seed`: Random seed for reproducibility

**Cache key format**: MD5 hash (first 12 characters)

Example: `a3f2b8c91e4d`

### What Gets Cached

1. **Preprocessed Features** (`preprocessed_{cache_key}.pkl`):
   - Cleaned feature matrix (X)
   - Encoded labels (y)
   - Saves: 2-5 minutes of data loading and preprocessing

2. **Selected Features** (`selected_features_{cache_key}.pkl`):
   - List of selected feature names
   - Feature importance scores
   - Saves: 2-4 minutes of feature selection computation

### Cache Directory Structure

```
cache/
├── preprocessed_a3f2b8c91e4d.pkl    # Preprocessed features
├── selected_features_a3f2b8c91e4d.pkl # Selected features
└── preprocessed_b7d4e1f29c8a.pkl    # Different configuration
```

## Usage

### Enable Caching (Default)

Caching is enabled by default in `config.yaml`:

```yaml
output:
  cache_dir: "cache"
  cache_preprocessed: true
  cache_selected_features: true
  use_cache: true
```

### Command Line Options

```bash
# Use cache (default behavior)
python train.py --sample 0.1 --experiment-name run1

# Ignore cache and reprocess
python train.py --sample 0.1 --experiment-name run1 --no-cache

# Clear all cache before running
python train.py --clear-cache --experiment-name fresh_start
```

### Configuration Options

```yaml
output:
  cache_dir: "cache"              # Cache directory location
  cache_preprocessed: true        # Enable/disable preprocessing cache
  cache_selected_features: true   # Enable/disable feature selection cache
  use_cache: true                 # Master switch for cache usage
```

## Performance Comparison

### Without Cache (First Run)

```
STEP 1: DATA LOADING & PREPROCESSING
  - Loading CSVs: 2-3 min
  - Preprocessing: 1-2 min
  - Total: ~3-5 min

STEP 4: FEATURE SELECTION
  - SelectKBest: 1-2 min
  - SHAP analysis: 1-2 min
  - Total: ~2-4 min

TOTAL TIME: ~5-9 min
```

### With Cache (Subsequent Runs)

```
STEP 1: DATA LOADING & PREPROCESSING
  ⚡ Using cached preprocessed data
  - Load from cache: 5-10 sec
  - Total: ~10 sec

STEP 4: FEATURE SELECTION
  ⚡ Using cached feature selection
  - Load from cache: <1 sec
  - Total: ~1 sec

TOTAL TIME: ~30 sec - 1 min (10-30x faster!)
```

## Cache Invalidation

The cache is automatically invalidated (new cache key generated) when:

✅ **Sample fraction changes**:
- `--sample 0.1` → `--sample 0.2` (different cache)

✅ **Dataset path changes**:
- Different data directory

✅ **Random seed changes**:
- Different `random_seed` in config

✅ **Test/train split changes**:
- Different `test_size` in config

❌ **Does NOT invalidate cache** (same cache reused):
- Experiment name changes
- Model hyperparameters change
- Feature selection method changes (k_best, shap_top_n)
- Output directory changes

## Best Practices

### 1. Hyperparameter Tuning
```bash
# First run: cache preprocessed data
python train.py --sample 0.1 --experiment-name lr_001

# Fast iterations with different hyperparameters
# (Edit config.yaml between runs)
python train.py --sample 0.1 --experiment-name lr_005
python train.py --sample 0.1 --experiment-name lr_01
```

### 2. Feature Selection Experiments
```bash
# Cache preprocessing, try different feature counts
# (Edit feature_selection.shap.top_n in config.yaml)
python train.py --sample 0.1 --experiment-name features_20
python train.py --sample 0.1 --experiment-name features_30
python train.py --sample 0.1 --experiment-name features_50
```

### 3. Fresh Start
```bash
# Clear cache when switching datasets or major changes
python train.py --clear-cache --experiment-name new_baseline
```

### 4. Disable Cache for Production
```bash
# Ensure fresh processing for final model
python train.py --no-cache --experiment-name final_production_model
```

## Disk Space Considerations

### Cache File Sizes (Approximate)

- **10% sample**:
  - Preprocessed: 50-100 MB
  - Selected features: 1-5 MB
  - Total: ~55-105 MB

- **Full dataset**:
  - Preprocessed: 500-1000 MB
  - Selected features: 1-5 MB
  - Total: ~500-1000 MB

### Cleanup Strategies

```bash
# Manual cleanup
rm -rf cache/

# Programmatic cleanup
python train.py --clear-cache

# Clean old cache files (Windows PowerShell)
Get-ChildItem cache/*.pkl | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item
```

## Troubleshooting

### Cache Not Being Used

**Symptom**: System reprocesses data every run

**Solutions**:
1. Check `use_cache: true` in config.yaml
2. Verify cache directory exists: `cache/`
3. Ensure cache key is stable (same config parameters)
4. Check for pickle errors in logs

### Cache Corruption

**Symptom**: Errors loading from cache

**Solution**:
```bash
python train.py --clear-cache --experiment-name recovery
```

### Stale Cache

**Symptom**: Using old cached data after dataset changes

**Solution**:
```bash
# Clear cache when dataset files change
python train.py --clear-cache
```

### Memory Issues

**Symptom**: Large cache files cause memory problems

**Solution**:
```yaml
# Disable caching for large datasets
output:
  cache_preprocessed: false
  cache_selected_features: true  # Keep this, much smaller
```

## Advanced: Cache Inspection

```python
import pickle
from pathlib import Path

# Load and inspect cached data
cache_file = Path('cache/preprocessed_a3f2b8c91e4d.pkl')

with open(cache_file, 'rb') as f:
    cached_data = pickle.load(f)

print(f"Features shape: {cached_data['X'].shape}")
print(f"Labels shape: {cached_data['y'].shape}")
print(f"Feature names: {cached_data['X'].columns.tolist()}")
```

## Security Considerations

⚠️ **Warning**: Cache files use Python pickle format

- Do not load cache files from untrusted sources
- Cache files can execute arbitrary code during unpickling
- Only use cache generated by your own training runs

## Future Enhancements

Potential improvements for the caching system:

1. **Compression**: Compress cache files to save disk space
2. **Metadata**: Add cache metadata (timestamp, config hash, version)
3. **Incremental Updates**: Update cache instead of full regeneration
4. **Distributed Cache**: Share cache across team members
5. **Cloud Storage**: Store cache in cloud for CI/CD pipelines

---

**🚀 The caching system is designed to accelerate your ML experimentation workflow while maintaining data integrity and reproducibility!**
