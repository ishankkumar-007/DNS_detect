# DNS Spoofing Detection using LightGBM

Real-time AI-based DNS threat analysis system using LightGBM with hybrid feature selection (SelectKBest + SHAP) on the BCCC-CIC-Bell-DNS-2024 dataset.

## 📋 Project Overview

This project implements a comprehensive DNS spoofing detection system focused on:
- **Malicious DNS behavior profiling** through application layer traffic analysis
- **Hybrid feature selection** combining statistical filtering (SelectKBest) and model-based explainability (SHAP)
- **Real-time inference** with <100ms latency constraint
- **Multi-class detection** for various DNS attack types (malware, phishing, spam, data exfiltration)

## 📊 Dataset

**BCCC-CIC-Bell-DNS-2024** (~4.3GB, 26 CSV files)
- Generated using ALFlowLyzer from CIC-Bell-DNS-2021 and CIC-Bell-DNS-EXF-2021
- **121 features** including network flow metrics, statistical features, and DNS-specific attributes
- **Two categories**:
  - `BCCC-CIC-Bell-DNS-EXF/`: Data exfiltration traffic (18 files)
  - `BCCC-CIC-Bell-DNS-Mal/`: Malicious DNS traffic (7 files)

**Citation:**
> Shafi, MohammadMoein, Arash Habibi Lashkari, Hardhik Mohanty. "Unveiling Malicious DNS Behavior Profiling and Generating Benchmark Dataset through Application Layer Traffic Analysis". *Computers and Electrical Engineering*, 2024.

## 🏗️ Project Structure

```
project_root/
├── BCCC-CIC-Bell-DNS-2024/        # Dataset directory
│   ├── BCCC-CIC-Bell-DNS-EXF/     # Exfiltration traffic
│   └── BCCC-CIC-Bell-DNS-Mal/     # Malicious traffic
├── src/
│   ├── preprocessing.py           # Data loading & preprocessing
│   ├── feature_selection.py       # Hybrid SelectKBest + SHAP
│   ├── model.py                   # LightGBM training & evaluation
│   ├── real_time_detection.py     # Real-time inference pipeline
│   └── utils.py                   # Helper functions
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA notebook
├── models/                        # Saved models
├── results/                       # Evaluation results
│   ├── plots/                     # Visualizations
│   └── metrics/                   # Performance metrics
├── logs/                          # Training logs
├── config.yaml                    # Configuration file
├── train.py                       # Main training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- 8GB+ RAM recommended (dataset is ~4.3GB)
- Windows/Linux/MacOS

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "c:\Users\Ishank\Desktop\cns project 2024 dataset"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify dataset location:**
   Ensure `BCCC-CIC-Bell-DNS-2024/` directory contains the CSV files.

### Quick Start

1. **Run training pipeline (with 10% sample for testing):**
   ```bash
   python train.py --sample 0.1 --experiment-name test_run
   ```

2. **Run full training:**
   ```bash
   python train.py --experiment-name full_training
   ```

3. **Skip feature selection (use all features):**
   ```bash
   python train.py --skip-feature-selection
   ```

4. **Use cached data for faster iterations:**
   ```bash
   # First run processes and caches data
   python train.py --sample 0.1 --experiment-name run1
   
   # Subsequent runs reuse cached preprocessed features (much faster!)
   python train.py --sample 0.1 --experiment-name run2
   ```

5. **Clear cache and start fresh:**
   ```bash
   python train.py --clear-cache --experiment-name fresh_run
   ```

## 🔧 Configuration

Edit `config.yaml` to customize:

### Key Settings

```yaml
data:
  sample_fraction: null  # null for full dataset, 0.1 for 10% sample
  test_size: 0.2        # Train/test split ratio

feature_selection:
  selectkbest:
    k: 50               # Initial feature reduction
  shap:
    top_n: 30           # Final feature count

model:
  learning_rate: 0.05
  n_estimators: 500
  num_leaves: 31
  max_depth: -1
```

## 📈 Training Pipeline

The training pipeline follows these steps:

1. **Data Loading**: Memory-efficient loading using pandas/Dask
2. **Preprocessing**: Missing value handling, feature encoding, data merging
3. **Train/Test Split**: Stratified split maintaining class distribution
4. **Feature Selection**:
   - Stage 1: SelectKBest (statistical filtering)
   - Stage 2: SHAP (model-based importance)
5. **Model Training**: LightGBM with early stopping
6. **Evaluation**: Comprehensive metrics and visualizations
7. **Model Saving**: Persistence with metadata

## 🎯 Model Performance

Expected metrics on test set:
- **Accuracy**: High accuracy across all classes
- **Precision/Recall/F1**: Balanced performance for imbalanced classes
- **ROC-AUC**: Strong discrimination capability
- **Inference Latency**: <100ms per prediction

## 🔍 Real-Time Detection

Use the trained model for real-time detection:

```python
from src.real_time_detection import RealTimeDNSDetector

# Load model
detector = RealTimeDNSDetector(model_path="models/dns_spoofing_detector.txt")

# Predict single flow
flow_data = {...}  # Dictionary with DNS flow features
result = detector.predict_single(flow_data)

print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Latency: {result['latency_ms']:.2f}ms")
```

## 📊 Exploratory Data Analysis

Run the EDA notebook to explore:
- Class distribution and imbalance
- Feature correlations
- Missing value patterns
- DNS-specific feature analysis

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## 🛠️ Module Documentation

### `preprocessing.py`
- `DNSDataPreprocessor`: Handles large-scale CSV loading, cleaning, and feature engineering
- Memory-efficient processing with Dask support
- DNS n-gram feature extraction

### `feature_selection.py`
- `HybridFeatureSelector`: Two-stage feature selection
- SelectKBest: Statistical filtering (mutual info, chi2, f_classif)
- SHAP: Model-based explainability

### `model.py`
- `DNSSpoofingDetector`: LightGBM model wrapper
- Training, evaluation, hyperparameter tuning
- Comprehensive visualization methods

### `real_time_detection.py`
- `RealTimeDNSDetector`: Optimized inference pipeline
- Latency monitoring (<100ms target)
- Batch and streaming prediction support

### `utils.py`
- Visualization helpers
- Metrics saving/loading
- Experiment management

## 📝 Command Line Options

```bash
python train.py --help

Options:
  --config CONFIG            Path to configuration file (default: config.yaml)
  --sample SAMPLE            Sample fraction (e.g., 0.1 for 10%)
  --experiment-name NAME     Name for this experiment
  --skip-feature-selection   Skip feature selection, use all features
  --no-cache                 Ignore cached data and reprocess from scratch
  --clear-cache              Clear all cached data before running
```

## 🐛 Troubleshooting

### Memory Issues
- Use `sample_fraction: 0.1` in config for testing
- Enable Dask: `use_dask: true` in config
- Reduce `k_best` and `shap_top_n` values

### Slow Training
- Reduce `n_estimators` in model config
- Increase `learning_rate`
- Use smaller sample for experimentation

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (3.8+ required)

## 📚 Key Features

✅ **Memory-efficient data loading** (handles 4.3GB dataset)  
✅ **Hybrid feature selection** (SelectKBest + SHAP)  
✅ **LightGBM optimization** for DNS detection  
✅ **Real-time inference** (<100ms latency)  
✅ **Comprehensive evaluation** (multiple metrics + visualizations)  
✅ **Reproducible experiments** (seeded randomness)  
✅ **Professional code structure** (modular, documented, tested)

## 🤝 Contributing

This is a research project for DNS spoofing detection. Contributions welcome!

