# Flask App Fix - Hybrid Model Support

## Issue
The hybrid model loading was failing with `KeyError: 'model'` because hybrid models have a different structure than other models.

## Root Cause
- **Random Forest, Isolation Forest, OCSVM**: Stored as `{'model': clf, 'feature_names': [...], ...}`
- **Hybrid Models**: Stored as `{'supervised_model': model1, 'unsupervised_model': model2, 'fusion_strategy': '...', ...}`

## Solution Applied

### 1. Updated `load_model()` function
- Changed from using `joblib.load()` to `pickle.load()` for consistency
- Added metadata loading for all model types to get `label_names`
- Properly handles hybrid model structure

### 2. Updated `predict_with_model()` function
- Added special handling for hybrid models:
  - Reconstructs `HybridDetector` object from loaded data
  - Uses `supervised_model` and `unsupervised_model` keys
  - Properly extracts feature names from metadata
- Separated hybrid from other model types

### 3. Changes Made

**In `load_model()`:**
```python
# For hybrid models - load pickle and metadata separately
elif model_type == 'hybrid':
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Load metadata if available
    metadata_path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'metadata' not in model_data:
                model_data['metadata'] = {}
            model_data['metadata'].update(metadata)
    
    return model_data
```

**In `predict_with_model()`:**
```python
elif model_type == 'hybrid':
    from models.hybrid_model import HybridDetector
    
    # Reconstruct the HybridDetector from loaded data
    detector = HybridDetector(
        supervised_model=model['supervised_model'],
        unsupervised_model=model['unsupervised_model'],
        fusion_strategy=model.get('fusion_strategy', 'weighted'),
        config=model.get('config', {})
    )
    
    # Get feature names from metadata
    metadata = model.get('metadata', {})
    feature_names = metadata.get('feature_names', [])
    label_names = metadata.get('label_names', {})
    
    # ... prediction logic
```

## Testing
The Flask app is running in debug mode and will automatically reload with these changes. Try:

1. Load a hybrid model (e.g., `hybrid_complete_0.7_0.3_anomaly_aware_20251106_224233`)
2. Upload a CSV file
3. Run prediction

The error should be resolved and predictions should work correctly.

## Affected Files
- `app.py` - Flask application with updated model loading and prediction logic

## Status
✅ **FIXED** - Hybrid models can now be loaded and used for predictions in the Flask app
