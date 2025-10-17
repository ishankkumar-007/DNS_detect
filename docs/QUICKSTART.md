# Quick Start Guide - DNS Spoofing Detection

## 🚀 Setup & Installation

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Verify Dataset
Ensure `BCCC-CIC-Bell-DNS-2024/` directory contains:
- `BCCC-CIC-Bell-DNS-EXF/` (18 CSV files)
- `BCCC-CIC-Bell-DNS-Mal/` (7 CSV files)

## 📊 Exploratory Data Analysis

### Run EDA Notebook
```powershell
jupyter notebook notebooks/exploratory_analysis.ipynb
```

This notebook will help you understand:
- Dataset structure and quality
- Class distribution (imbalance analysis)
- Missing value patterns
- DNS-specific feature characteristics
- Feature correlations

## 🎯 Training the Model

### Quick Test (10% sample)
```powershell
python train.py --sample 0.1 --experiment-name quick_test
```

### Full Training
```powershell
python train.py --experiment-name production_model
```

### Fast Iterations with Cache
```powershell
# First run: processes and caches data (~5-10 minutes)
python train.py --sample 0.1 --experiment-name run1

# Subsequent runs: reuses cache (~30 seconds!)
python train.py --sample 0.1 --experiment-name run2

# Clear cache if needed
python train.py --clear-cache --experiment-name fresh_start
```

### Training Output
The training script will:
1. Load and preprocess data
2. Apply hybrid feature selection (SelectKBest + SHAP)
3. Train LightGBM model with early stopping
4. Generate comprehensive evaluation metrics
5. Create visualizations (confusion matrix, feature importance, ROC curves)
6. Save model and metadata

Results will be saved in: `results/<experiment_name>_<timestamp>/`

## 🔍 Real-Time Detection

### Run Predictions on New Data
```powershell
python predict.py --model models/dns_spoofing_detector.txt --input path/to/new_data.csv --output predictions.csv
```

### Streaming Mode
```powershell
python predict.py --model models/dns_spoofing_detector.txt --input path/to/data.csv --stream
```

## 📁 Project Structure

```
.
├── src/
│   ├── preprocessing.py          # Data loading & preprocessing
│   ├── feature_selection.py      # SelectKBest + SHAP
│   ├── model.py                  # LightGBM training
│   ├── real_time_detection.py    # Inference pipeline
│   └── utils.py                  # Helper functions
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA notebook
├── config.yaml                    # Configuration
├── train.py                       # Training pipeline
├── predict.py                     # Inference script
└── README.md                      # Full documentation
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
data:
  sample_fraction: null  # Set to 0.1 for 10% sample
  test_size: 0.2

feature_selection:
  selectkbest:
    k: 50  # Initial feature count
  shap:
    top_n: 30  # Final feature count

model:
  learning_rate: 0.05
  n_estimators: 500
  num_leaves: 31

output:
  cache_preprocessed: true  # Cache processed features
  cache_selected_features: true  # Cache feature selection
  use_cache: true  # Use cached data
```

### 🚀 Performance Optimization with Caching

The system automatically caches preprocessed features and feature selections to dramatically speed up subsequent training runs:

**Without Cache (First Run):**
- Data loading: ~2-5 min
- Preprocessing: ~1-3 min
- Feature selection: ~2-4 min
- **Total**: ~5-12 min

**With Cache (Subsequent Runs):**
- Load from cache: ~5-10 sec
- Feature selection (if cached): skip
- **Total**: ~30 sec - 2 min

**Cache is automatically invalidated when:**
- Sample fraction changes
- Dataset path changes
- Random seed changes
- Test/train split ratio changes

## 📈 Expected Performance

- **Accuracy**: High accuracy across benign and malicious classes
- **F1-Score**: Balanced performance for imbalanced data
- **Inference Latency**: <100ms per prediction
- **Features**: 30 selected features (from 121 original)

## 🐛 Common Issues

### Memory Error
- Use `sample_fraction: 0.1` in config
- Enable Dask: `use_dask: true`

### Slow Training
- Reduce `n_estimators: 100`
- Increase `learning_rate: 0.1`

### Import Errors
```powershell
pip install --upgrade -r requirements.txt
```

## 📚 Module Usage Examples

### Preprocessing
```python
from src.preprocessing import DNSDataPreprocessor

preprocessor = DNSDataPreprocessor(data_dir='BCCC-CIC-Bell-DNS-2024')
df = preprocessor.load_all_data(sample_frac=0.1)
X, y = preprocessor.preprocess_features(df)
```

### Feature Selection
```python
from src.feature_selection import HybridFeatureSelector

selector = HybridFeatureSelector(k_best=50, shap_top_n=30)
X_selected, _ = selector.fit_transform(X_train, y_train)
```

### Model Training
```python
from src.model import DNSSpoofingDetector

detector = DNSSpoofingDetector()
detector.train(X_train, y_train, X_val, y_val)
metrics = detector.evaluate(X_test, y_test)
```

### Real-Time Detection
```python
from src.real_time_detection import RealTimeDNSDetector

detector = RealTimeDNSDetector(model_path='models/dns_spoofing_detector.txt')
result = detector.predict_single(flow_data)
print(f"Prediction: {result['predicted_label']} ({result['confidence']:.2%})")
```

## 🎓 Learning Resources

- **Research Paper**: Shafi et al., "Unveiling Malicious DNS Behavior Profiling", 2024
- **LightGBM Docs**: https://lightgbm.readthedocs.io/
- **SHAP Docs**: https://shap.readthedocs.io/

## 💡 Tips for Best Results

1. **Start with EDA**: Run the Jupyter notebook first to understand the data
2. **Test with samples**: Use `--sample 0.1` for quick iterations
3. **Monitor metrics**: Check F1-score (macro) for imbalanced data
4. **Feature importance**: Review SHAP plots to understand model decisions
5. **Hyperparameter tuning**: Adjust `config.yaml` based on validation results

