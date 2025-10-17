# Quick Reference: Supervised vs Unsupervised DNS Detection

## Overview

This project provides **two complementary approaches** for DNS threat detection:

| Aspect | **Supervised (LightGBM)** | **Unsupervised (One-Class SVM)** |
|--------|---------------------------|----------------------------------|
| **Training Data** | Labeled benign + malicious | Benign only (no labels) |
| **Algorithm** | Gradient Boosting Trees | Support Vector Machine |
| **Accuracy** | 96.88% | 85-92% |
| **ROC-AUC** | 99.36% | 0.85-0.95 |
| **Detection Rate** | 91.93% | 70-85% |
| **False Positive Rate** | 2.40% | 5-15% |
| **Zero-Day Detection** | Limited | Excellent |
| **Interpretability** | High (SHAP) | Low |
| **Training Time** | 8-10 min | 3-5 min |
| **Inference Speed** | <30ms | <50ms |

---

## When to Use Each Approach

### Use Supervised (LightGBM) when:
✅ You have labeled malicious samples  
✅ Maximum accuracy is required  
✅ Known attack types to detect  
✅ Model interpretability is important  
✅ Low false positive rate is critical  

**Best for**: Production deployment with known threats

### Use Unsupervised (One-Class SVM) when:
✅ Limited or no labeled malicious samples  
✅ Zero-day/novel attack detection needed  
✅ Establishing baseline normal behavior  
✅ Complementary to supervised detection  
✅ Anomaly investigation workflow  

**Best for**: Research, zero-day detection, anomaly investigation

### Use Both (Hybrid Defense-in-Depth):
```
Incoming DNS Traffic
        ↓
┌───────────────────┐
│ One-Class SVM     │ → Flag anomalies (broad net)
│ (First Filter)    │
└───────────────────┘
        ↓
  If Anomalous?
        ↓
┌───────────────────┐
│ LightGBM          │ → Classify attack type (precise)
│ (Second Filter)   │
└───────────────────┘
        ↓
  Alert + Response
```

---

## Quick Start Commands

### Supervised Training
```bash
# Full dataset
python train.py --experiment-name supervised_full

# Quick test (10% sample)
python train.py --sample 0.1 --experiment-name supervised_test

# With feature selection
python train.py --experiment-name supervised_fs
```

### Unsupervised Training
```bash
# Full dataset
python train_unsupervised.py --experiment-name unsupervised_full

# Quick test (10% sample)
python train_unsupervised.py --sample 0.1 --experiment-name unsupervised_test

# Adjust sensitivity
python train_unsupervised.py --nu 0.05 --experiment-name unsupervised_balanced
```

### Comparison
```bash
# Train unsupervised and compare with existing supervised model
python train_unsupervised.py \
    --experiment-name comparison \
    --compare-supervised results/complete_k50_shap30_20251017_033122
```

---

## File Structure

```
src/
├── model.py                    # Supervised LightGBM detector
├── unsupervised_detection.py  # Unsupervised One-Class SVM detector
├── preprocessing.py            # Shared data preprocessing
├── feature_selection.py        # Supervised feature selection
├── real_time_detection.py      # Real-time inference
└── utils.py                    # Shared utilities

train.py                        # Supervised training script
train_unsupervised.py          # Unsupervised training script

docs/
├── mid-report.md              # Comprehensive project report
├── dataset_description.md     # Dataset documentation
└── unsupervised_approach.md   # One-Class SVM detailed guide
```

---

## Key Differences in Approach

### Supervised (LightGBM)

**Training Process:**
1. Load benign + malicious labeled data
2. Apply feature selection (SelectKBest + SHAP)
3. Train gradient boosting trees
4. Optimize for accuracy and interpretability

**Prediction:**
- Direct classification: Benign vs Malicious
- Confidence scores via tree voting
- Feature importance via SHAP

**Strengths:**
- High accuracy on known attacks
- Interpretable (can explain why)
- Low false positive rate
- Attack type classification

**Weaknesses:**
- Requires labeled malicious data
- Limited zero-day detection
- May overfit to training attacks
- Vulnerable to adversarial evasion

---

### Unsupervised (One-Class SVM)

**Training Process:**
1. Load benign traffic only
2. Apply PCA for dimensionality reduction
3. Train One-Class SVM on normal behavior
4. Learn decision boundary around benign data

**Prediction:**
- Anomaly detection: Normal vs Anomaly
- Anomaly scores (distance from boundary)
- Binary classification only

**Strengths:**
- No labeled attacks needed
- Excellent zero-day detection
- Detects novel patterns
- Robust to new threats

**Weaknesses:**
- Lower accuracy overall
- Higher false positive rate
- Less interpretable
- Cannot classify attack types

---

## Configuration Files

### Supervised Config (`config.yaml`)
```yaml
feature_selection:
  enable: true
  selectkbest:
    k: 50               # Reduce to 50 features
  shap:
    top_n: 30           # Final 30 features

model:
  objective: binary
  learning_rate: 0.05
  n_estimators: 500
```

### Unsupervised Parameters (CLI)
```bash
--nu 0.05              # Expected outlier fraction (5%)
--kernel rbf           # Radial basis function kernel
--use-pca              # Enable PCA
--pca-components 30    # Reduce to 30 dimensions
```

---

## Expected Performance

### Supervised (LightGBM)
```
Test Set: 830,753 flows (87% benign, 13% malicious)

Confusion Matrix:
                Predicted
              Benign  Malicious
Actual Benign  708,449   17,438  (97.6% correct)
       Malicious  8,460   96,406  (91.9% detected)

Metrics:
  Accuracy:          96.88%
  ROC-AUC:           99.36%
  Precision (Mal):   84.68%
  Recall (Mal):      91.93%
  F1-Score (Mal):    88.16%
  False Pos Rate:    2.40%
```

### Unsupervised (One-Class SVM)
```
Test Set: 830,753 flows (87% benign, 13% malicious)
Training: 2.3M benign samples only

Confusion Matrix (Expected):
                Predicted
              Normal  Anomaly
Actual Benign  640,000  85,000  (88% correct)
       Malicious 25,000  80,000  (76% detected)

Metrics:
  Accuracy:          88-92%
  ROC-AUC:           0.90-0.93
  Precision (Mal):   48-52%
  Recall (Mal):      75-82%
  F1-Score (Mal):    58-62%
  False Pos Rate:    8-12%
```

**Trade-off**: Unsupervised has 15% lower detection rate but requires 0 labeled attacks

---

## Output Files

### Supervised Training Output
```
results/{experiment_name}_k{k}_shap{n}_{timestamp}/
├── models/
│   ├── dns_spoofing_detector.txt           # LightGBM model
│   └── dns_spoofing_detector_metadata.json # Config + features
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance_lgbm.png
│   └── feature_importance_shap.png
└── metrics/
    └── evaluation_metrics.json
```

### Unsupervised Training Output
```
results/{experiment_name}_nu{nu}_{pca/nopca}_{timestamp}/
├── models/
│   └── ocsvm_detector.pkl                  # One-Class SVM model
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── decision_distribution.png
│   ├── anomaly_scores.png
│   └── supervised_vs_unsupervised.png      # Comparison (if --compare-supervised)
└── metrics/
    ├── experiment_config.json
    ├── metrics.json
    └── top_anomalies.csv
```

---

## Python API Usage

### Supervised Prediction
```python
from src.model import DNSSpoofingDetector

# Load model
detector = DNSSpoofingDetector.load_model('models/dns_spoofing_detector.txt')

# Predict
predictions = detector.predict(X_test)  # 0 or 1
probabilities = detector.predict_proba(X_test)  # [prob_benign, prob_malicious]
```

### Unsupervised Prediction
```python
from src.unsupervised_detection import OneClassSVMAnomalyDetector

# Load model
detector = OneClassSVMAnomalyDetector.load_model('models/ocsvm_detector.pkl')

# Predict
predictions = detector.predict(X_test)  # 1 (normal) or -1 (anomaly)
anomaly_scores = detector.decision_function(X_test)  # Real values
probabilities = detector.predict_proba(X_test)  # 0-1 range

# Get top anomalies
top_anomalies = detector.get_anomaly_ranking(X_test, top_n=100)

# Explain specific anomaly
explanation = detector.explain_anomaly(X_test, sample_idx=42)
```

---

## Hyperparameter Tuning

### Supervised (LightGBM)
```bash
# Tune learning rate
python train.py --config config_lr001.yaml  # learning_rate: 0.01
python train.py --config config_lr01.yaml   # learning_rate: 0.1

# Tune feature selection
python train.py --config config_k30_shap20.yaml   # 30 → 20 features
python train.py --config config_k100_shap50.yaml  # 100 → 50 features
```

### Unsupervised (One-Class SVM)
```bash
# Tune nu (sensitivity)
python train_unsupervised.py --nu 0.01  # Strict (1% outliers)
python train_unsupervised.py --nu 0.05  # Balanced (5% outliers)
python train_unsupervised.py --nu 0.10  # Permissive (10% outliers)

# Tune kernel
python train_unsupervised.py --kernel rbf     # Non-linear (best)
python train_unsupervised.py --kernel linear  # Linear (fast)
python train_unsupervised.py --kernel poly    # Polynomial
```

---

## Troubleshooting

### Problem: Unsupervised has too many false positives

**Solution:**
```bash
# Decrease nu parameter (stricter boundary)
python train_unsupervised.py --nu 0.02

# Or: Increase PCA components (retain more information)
python train_unsupervised.py --pca-components 50
```

### Problem: Unsupervised missing too many attacks

**Solution:**
```bash
# Increase nu parameter (looser boundary)
python train_unsupervised.py --nu 0.10

# Or: Add more diverse benign training samples
# Ensure training data represents all normal patterns
```

### Problem: Want best of both worlds

**Solution:**
Use hybrid approach - train both models and ensemble predictions:
```python
# Predict with both
lgbm_pred = supervised_detector.predict(X)
ocsvm_pred = unsupervised_detector.predict(X)

# Alert if EITHER flags as malicious
final_pred = (lgbm_pred == 1) | (ocsvm_pred == -1)

# Or: Weight by confidence
lgbm_conf = supervised_detector.predict_proba(X)[:, 1]
ocsvm_conf = unsupervised_detector.predict_proba(X)
combined = 0.7 * lgbm_conf + 0.3 * ocsvm_conf  # 70-30 weighting
```

---

## Further Reading

- **Project Report**: `docs/mid-report.md` - Comprehensive project documentation
- **Dataset Guide**: `docs/dataset_description.md` - Dataset schema and statistics
- **Unsupervised Details**: `docs/unsupervised_approach.md` - One-Class SVM deep dive
- **README**: `README.md` - Project overview and quick start

---

## Recommendations

### For Research/Experimentation:
1. Start with supervised approach (higher accuracy, better understood)
2. Add unsupervised for comparison
3. Analyze failure cases of each
4. Consider hybrid ensemble

### For Production Deployment:
1. Use supervised as primary detector (low false positives)
2. Add unsupervised as anomaly flag (zero-day protection)
3. Route unsupervised alerts to security analysts
4. Continuously retrain both models

### For Zero-Day Detection:
1. Use unsupervised as primary (broad coverage)
2. Add supervised for attack type classification
3. Implement analyst feedback loop
4. Update models with confirmed attacks

---

**Last Updated**: October 17, 2025  
**Version**: 1.0  
**Contact**: DNS Spoofing Detection Project Team
