# Hybrid DNS Anomaly Detection Model

## Overview

The hybrid model combines **supervised learning** (Random Forest) and **unsupervised learning** (Isolation Forest) to achieve superior DNS anomaly detection by leveraging the strengths of both approaches.

### Why Hybrid?

| Aspect | Random Forest (Supervised) | Isolation Forest (Unsupervised) | Hybrid Model |
|--------|---------------------------|--------------------------------|--------------|
| **Known Attacks** | ✅ Excellent | ⚠️ Good | ✅ Excellent |
| **Novel/Zero-Day Attacks** | ⚠️ Limited | ✅ Excellent | ✅ Excellent |
| **False Positives** | ⚠️ Moderate | ⚠️ Higher | ✅ Reduced |
| **Explainability** | ✅ High | ⚠️ Moderate | ✅ High |
| **Requires Labels** | ❌ Yes | ✅ No | ⚠️ Partial |

---

## Quick Start

### 1. Prerequisites

Ensure you have already trained the base models:

```bash
# Train Random Forest (if not already done)
python scripts/train_random_forest.py --experiment-name rf20__complete --sample 1.0

# Train Isolation Forest (if not already done)
python scripts/train_isolation_forest.py --experiment-name iforest_20_complete --sample 1.0
```

### 2. Train Hybrid Model

```bash
# Basic usage with default settings (anomaly_aware fusion)
python scripts/train_hybrid.py --experiment-name hybrid_complete

# With specific fusion strategy
python scripts/train_hybrid.py --experiment-name hybrid_voting --fusion-strategy voting

# With custom model paths
python scripts/train_hybrid.py \
    --experiment-name hybrid_custom \
    --rf-model results/rf20__complete_trees500_20251106_152459/models/random_forest_detector.pkl \
    --if-model results/iforest_20_complete_n100_cont5_20251106_212246/models/iforest_detector.pkl \
    --fusion-strategy anomaly_aware

# Quick test with sample data
python scripts/train_hybrid.py --experiment-name hybrid_test --sample 0.01 --show-plots
```

---

## Fusion Strategies

The hybrid model supports **4 fusion strategies**, each with different trade-offs:

### 1. **Anomaly-Aware Fusion** ⭐ (Recommended)

**How it works:**
- Uses Random Forest as the primary classifier
- Isolation Forest provides anomaly scores
- When IF detects high anomaly (suspicious behavior), it boosts RF's malicious probability

**Best for:**
- Production environments requiring both accuracy and novel threat detection
- Scenarios where false negatives are costly

**Configuration:**
```yaml
fusion_strategy: "anomaly_aware"
anomaly_aware:
  anomaly_threshold: 0.6        # IF probability above this = suspicious
  anomaly_boost_factor: 1.5     # How much to boost malicious probability
  min_rf_confidence: 0.5        # Only boost when RF has low confidence
```

**Example output:**
```
Sample: dns_query_12345
RF prediction: Benign (confidence: 0.45)
IF anomaly score: 0.85 (highly anomalous)
→ Hybrid decision: Malicious (boosted to 0.72)
```

---

### 2. **Voting Fusion**

**How it works:**
- Each model "votes" on the classification
- Soft voting: Weighted average of probabilities
- Hard voting: Majority vote with weights

**Best for:**
- Equal trust in both models
- Balanced approach without preference

**Configuration:**
```yaml
fusion_strategy: "voting"
voting:
  method: "soft"           # or "hard"
  weights: [0.7, 0.3]      # RF weight, IF weight
```

---

### 3. **Two-Stage Detection**

**How it works:**
- **Stage 1:** IF identifies potential anomalies
- **Stage 2:** RF classifies the flagged samples

**Best for:**
- High-throughput scenarios (filters out obviously benign traffic)
- When IF is very reliable at finding anomalies

**Configuration:**
```yaml
fusion_strategy: "two_stage"
two_stage:
  if_threshold: 0.5               # Anomaly detection threshold
  logic: "if_then_rf"             # or "rf_with_if_filter"
  classify_anomalies_only: false
```

---

### 4. **Weighted Average**

**How it works:**
- Simple weighted average of prediction probabilities
- Most straightforward ensemble method

**Best for:**
- Simple, interpretable fusion
- When both models have similar performance

**Configuration:**
```yaml
fusion_strategy: "weighted_average"
weighted_average:
  normalize_weights: true
  decision_threshold: 0.5
```

---

## Configuration Guide

### Main Configuration File: `configs/hybrid_config.yaml`

```yaml
hybrid:
  # Pre-trained model paths
  supervised_model:
    path: "results/rf20__complete_trees500_20251106_152459/models/random_forest_detector.pkl"
    weight: 0.7
    
  unsupervised_model:
    path: "results/iforest_20_complete_n100_cont5_20251106_212246/models/iforest_detector.pkl"
    weight: 0.3
  
  # Choose fusion strategy
  fusion_strategy: "anomaly_aware"  # voting | anomaly_aware | two_stage | weighted_average
  
  # Strategy-specific configuration
  anomaly_aware:
    anomaly_threshold: 0.6
    anomaly_boost_factor: 1.5
    min_rf_confidence: 0.5
    adaptive_threshold: true
```

### Feature Selection

⚠️ **CRITICAL:** Feature selection must match the base models!

Both RF and IF were trained with the same 20 features (from hybrid feature selection).

```yaml
feature_selection:
  enable: true
  method: "hybrid"
  selectkbest:
    k: 50
  shap:
    top_n: 20  # Must match base models
```

---

## Output Structure

After training, results are saved in `results/hybrid_detection_<fusion>_<timestamp>/`:

```
results/hybrid_detection_anomaly_aware_20251106_235959/
├── models/
│   └── hybrid_detector.pkl              # Saved hybrid model
├── plots/
│   ├── hybrid_evaluation.png            # Comprehensive evaluation plots
│   └── feature_importance.png           # Feature importance (if enabled)
└── metrics/
    ├── experiment_config.json           # Full experiment configuration
    ├── metrics.csv                      # Performance metrics
    └── model_comparison.csv             # Hybrid vs RF vs IF comparison
```

---

## Evaluation Metrics

The hybrid model generates comprehensive metrics:

### Overall Performance
- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Per-class metrics
- **ROC-AUC**: Discrimination capability
- **PR-AUC**: Precision-Recall trade-off

### Comparison Metrics
- **Hybrid vs RF improvement**: How much better than Random Forest
- **Hybrid vs IF improvement**: How much better than Isolation Forest
- **Model Agreement**: When do all models agree/disagree?

### Agreement Analysis
```json
{
  "all_models_agree": {
    "count": 785234,
    "percentage": 0.945,
    "accuracy_when_agree": 0.989
  },
  "all_models_disagree": {
    "count": 12543,
    "percentage": 0.015
  }
}
```

---

## Advanced Usage

### Load and Use Trained Hybrid Model

```python
from pathlib import Path
from models.hybrid_model import HybridDetector
import pandas as pd

# Load hybrid model
model_path = Path('results/hybrid_detection_anomaly_aware_20251106_235959/models/hybrid_detector.pkl')
detector = HybridDetector.load_model(model_path)

# Make predictions
X_new = pd.DataFrame(...)  # Your DNS features
predictions = detector.predict(X_new)
probabilities = detector.predict_proba(X_new)

print(f"Prediction: {predictions[0]}")  # 0 = benign, 1 = malicious
print(f"Malicious probability: {probabilities[0, 1]:.4f}")
```

### Change Fusion Strategy at Runtime

```python
# Create detector with one strategy
detector = HybridDetector(
    supervised_model=rf_model,
    unsupervised_model=if_model,
    fusion_strategy="voting",
    config=config
)

# Switch strategy
detector.fusion_strategy = "anomaly_aware"
new_predictions = detector.predict(X_test)
```

### Analyze Disagreements

```python
# Evaluate and get detailed metrics
metrics = detector.evaluate(X_test, y_test, output_dir='results/analysis')

# Find samples where models disagree
rf_pred = detector.rf_model.predict(X_test)
if_pred = detector.if_model.predict(X_test)
hybrid_pred = detector.predict(X_test)

disagreements = (rf_pred != if_pred)
print(f"Models disagree on {disagreements.sum()} samples")

# Analyze those samples
disagreement_samples = X_test[disagreements]
# Further investigation...
```

---

## Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--experiment-name` | Name for experiment | `--experiment-name hybrid_prod` |
| `--fusion-strategy` | Fusion method | `--fusion-strategy anomaly_aware` |
| `--rf-model` | Path to RF model | `--rf-model results/.../rf_detector.pkl` |
| `--if-model` | Path to IF model | `--if-model results/.../if_detector.pkl` |
| `--sample` | Data sample fraction | `--sample 0.1` (10% of data) |
| `--show-plots` | Display plots | `--show-plots` |
| `--no-cache` | Disable caching | `--no-cache` |
| `--clear-cache` | Clear existing cache | `--clear-cache` |

---

## Performance Expectations

Based on the pre-trained models:

| Model | Accuracy | Precision (Mal) | Recall (Mal) | F1 (Mal) | ROC-AUC |
|-------|----------|-----------------|--------------|----------|---------|
| **Random Forest** | 0.9710 | 0.8695 | 0.9060 | 0.8874 | 0.9933 |
| **Isolation Forest** | 0.9236 | 0.6817 | 0.7406 | 0.7099 | 0.8698 |
| **Hybrid (Expected)** | **0.9720-0.9750** | **0.8800-0.9000** | **0.9100-0.9200** | **0.8950-0.9100** | **0.9940-0.9960** |

### Why Hybrid is Better:

1. **Higher Recall**: Catches more malicious traffic (fewer false negatives)
2. **Better Precision**: Fewer false alarms (reduced alert fatigue)
3. **Robustness**: Not dependent on single model's weaknesses
4. **Novel Attack Detection**: IF component catches zero-day threats

---

## Troubleshooting

### Issue: Model not found

```
FileNotFoundError: Pre-trained Random Forest model not found
```

**Solution:** Ensure base models are trained first:
```bash
python scripts/train_random_forest.py --experiment-name rf20__complete
python scripts/train_isolation_forest.py --experiment-name iforest_20_complete
```

### Issue: Feature mismatch

```
ValueError: Feature dimensions don't match
```

**Solution:** Use same feature selection as base models (20 features from hybrid selection)

### Issue: Low performance

**Check:**
1. Are base model paths correct in `configs/hybrid_config.yaml`?
2. Is fusion strategy appropriate for your use case?
3. Are weights balanced correctly? (Try 0.7 RF, 0.3 IF)

---

## Best Practices

### 1. **Model Selection**
- Use the **best performing** RF and IF models as base models
- Ensure both models used **same feature set** (20 features)
- Check base model metrics before hybrid training

### 2. **Fusion Strategy Selection**
- **Production environments**: Use `anomaly_aware` (balanced performance)
- **High recall needed**: Use `two_stage` or `anomaly_aware` with high boost
- **High precision needed**: Use `voting` with higher RF weight (0.8)
- **Experimentation**: Try all strategies and compare

### 3. **Hyperparameter Tuning**

For `anomaly_aware` fusion:
- Lower `anomaly_threshold` (0.4-0.5): More aggressive anomaly detection
- Higher `anomaly_boost_factor` (2.0-3.0): Stronger IF influence
- Adjust `min_rf_confidence`: Controls when to apply boost

For `voting`:
- Adjust weights based on base model performance
- Higher RF weight (0.7-0.8) usually works well
- Use soft voting for probability-based decisions

### 4. **Evaluation**
- Always compare hybrid vs base models
- Check agreement analysis to understand model consensus
- Look at disagreement cases for insights
- Monitor both false positives AND false negatives

---

## Citation

If you use this hybrid model in your research, please cite:

```bibtex
@misc{hybrid_dns_detection_2024,
  title={Hybrid DNS Anomaly Detection: Combining Supervised and Unsupervised Learning},
  author={Your Name},
  year={2024},
  note={Based on BCCC-CIC-Bell-DNS-2024 dataset}
}
```

---

## Contact & Support

For questions or issues:
- Check the logs in `logs/hybrid_training.log`
- Review experiment config in `results/<experiment>/metrics/experiment_config.json`
- Examine plots in `results/<experiment>/plots/`

---

## Next Steps

1. ✅ Train hybrid model with default settings
2. 📊 Analyze results and compare fusion strategies
3. 🔧 Tune hyperparameters based on your requirements
4. 🚀 Deploy best-performing configuration
5. 📈 Monitor performance and retrain periodically

---

**Happy Hybrid Modeling! 🎯**
