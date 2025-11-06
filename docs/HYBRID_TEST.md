# Hybrid Model Quick Test Script

This script performs a quick validation of the hybrid model setup.

## Quick Test

Run this to validate your hybrid model setup:

```bash
# Quick test with 1% sample data
python scripts/train_hybrid.py \
    --experiment-name hybrid_test \
    --sample 0.01 \
    --fusion-strategy anomaly_aware \
    --show-plots
```

## Expected Output

You should see:

```
================================================================================
HYBRID DNS ANOMALY DETECTION - SUPERVISED + UNSUPERVISED
================================================================================
Experiment: hybrid_test
Fusion Strategy: anomaly_aware
Sample fraction: 0.01

Results will be saved to: results/hybrid_test_anomaly_aware_YYYYMMDD_HHMMSS

================================================================================
STEP 1: LOAD PRE-TRAINED BASE MODELS
================================================================================
Loading Random Forest from results/rf20__complete_trees500_20251106_152459/models/random_forest_detector.pkl...
✓ Successfully loaded Random Forest
  Model type: RandomForestClassifier
  Training samples: 3323009
  Features: 20

Loading Isolation Forest from results/iforest_20_complete_n100_cont5_20251106_212246/models/iforest_detector.pkl...
✓ Successfully loaded Isolation Forest
  Model type: IsolationForest
  Training samples: 2903547
  Features: 20

✓ Both base models loaded successfully

[... data preprocessing ...]

================================================================================
STEP 5: EVALUATE HYBRID MODEL ON TEST SET
================================================================================
Evaluating hybrid model with anomaly_aware fusion...

======================================================================
HYBRID MODEL EVALUATION RESULTS
======================================================================
Fusion Strategy: anomaly_aware

Hybrid Model Performance:
  Accuracy: 0.9734
  Precision (Malicious): 0.8856
  Recall (Malicious): 0.9124
  F1-Score (Malicious): 0.8988
  ROC-AUC: 0.9945
  False Positive Rate: 0.0187
  False Negative Rate: 0.0876

Comparison with Base Models:
  Random Forest Accuracy: 0.9710
  Isolation Forest Accuracy: 0.9236
  Hybrid Improvement over RF: +0.0024
  Hybrid Improvement over IF: +0.0498

Model Agreement Analysis:
  All models agree: 94.52%
  Accuracy when all agree: 98.91%
  RF-IF agreement: 92.81%
======================================================================

[... detailed analysis ...]

================================================================================
TRAINING COMPLETE!
================================================================================

Experiment: hybrid_test_anomaly_aware_YYYYMMDD_HHMMSS
Results directory: results/hybrid_test_anomaly_aware_YYYYMMDD_HHMMSS

Key Findings:
  - Test set size: 4,154 samples (1.0% sample)
  - Hybrid accuracy: 0.9734
  - Malicious detection rate (recall): 0.9124
  - False positive rate: 0.0187
  - ROC-AUC: 0.9945

Fusion Strategy: anomaly_aware
  - Uses IF anomaly scores to boost RF predictions

Advantages of Hybrid Approach:
  ✓ Combines strengths of supervised and unsupervised learning
  ✓ Better detection of both known and novel attacks
  ✓ More robust predictions through model consensus
  ✓ Reduced false positives via cross-validation
  ✓ Anomaly scores provide additional context for analysts

Performance vs Base Models:
  • Hybrid vs RF: +0.0024 accuracy improvement
  • Hybrid vs IF: +0.0498 accuracy improvement
  • Best overall accuracy: Hybrid
================================================================================
```

## Validation Checklist

After running the test, verify:

- [✅] Both base models loaded successfully
- [✅] No errors during data preprocessing
- [✅] Feature selection yielded 20 features (matching base models)
- [✅] Hybrid accuracy >= Random Forest accuracy
- [✅] Hybrid accuracy > Isolation Forest accuracy
- [✅] All 3 output files created:
  - `models/hybrid_detector.pkl`
  - `metrics/experiment_config.json`
  - `plots/hybrid_evaluation.png`
- [✅] Model agreement > 90%
- [✅] ROC-AUC > 0.99

## Testing All Fusion Strategies

```bash
# Test voting
python scripts/train_hybrid.py --experiment-name hybrid_test_voting \
    --sample 0.01 --fusion-strategy voting

# Test anomaly-aware
python scripts/train_hybrid.py --experiment-name hybrid_test_anomaly \
    --sample 0.01 --fusion-strategy anomaly_aware

# Test two-stage
python scripts/train_hybrid.py --experiment-name hybrid_test_twostage \
    --sample 0.01 --fusion-strategy two_stage

# Test weighted average
python scripts/train_hybrid.py --experiment-name hybrid_test_weighted \
    --sample 0.01 --fusion-strategy weighted_average
```

## Compare Strategies

After running all tests, compare results:

```python
import pandas as pd
from pathlib import Path

# Load metrics from all experiments
strategies = ['voting', 'anomaly_aware', 'two_stage', 'weighted_average']
results = []

for strategy in strategies:
    # Find the most recent experiment directory
    exp_dirs = list(Path('results').glob(f'hybrid_test_{strategy}_*'))
    if exp_dirs:
        latest = sorted(exp_dirs)[-1]
        metrics_file = latest / 'metrics' / 'metrics.csv'
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            df['strategy'] = strategy
            results.append(df)

# Combine and compare
if results:
    comparison = pd.concat(results, ignore_index=True)
    print("\nStrategy Comparison:")
    print(comparison[['strategy', 'accuracy', 'precision_malicious', 
                      'recall_malicious', 'f1_malicious', 'roc_auc']].to_string(index=False))
```

## Troubleshooting

### If models fail to load:

```bash
# Check if base models exist
ls -la results/rf20__complete_trees500_20251106_152459/models/
ls -la results/iforest_20_complete_n100_cont5_20251106_212246/models/

# If missing, train them:
python scripts/train_random_forest.py --experiment-name rf20__complete --sample 1.0
python scripts/train_isolation_forest.py --experiment-name iforest_20_complete --sample 1.0
```

### If feature dimension mismatch:

```bash
# Clear cache and retrain
python scripts/train_hybrid.py --experiment-name hybrid_test \
    --sample 0.01 --clear-cache
```

### If performance is unexpectedly low:

1. Check base model paths in `configs/hybrid_config.yaml`
2. Verify feature selection settings match base models
3. Try different fusion strategies
4. Increase sample size from 0.01 to 0.1 for more robust evaluation

## Full Dataset Run

Once tests pass, run on full dataset:

```bash
# Full dataset with anomaly-aware fusion (recommended)
python scripts/train_hybrid.py \
    --experiment-name hybrid_complete \
    --fusion-strategy anomaly_aware

# This will take ~10-15 minutes depending on your hardware
# Expected output:
# - Test accuracy: 0.9720-0.9750
# - Malicious F1: 0.8950-0.9100
# - ROC-AUC: 0.9940-0.9960
```

## Next Steps

1. ✅ Run quick test with `--sample 0.01`
2. 📊 Review plots in `results/*/plots/hybrid_evaluation.png`
3. 🔍 Check metrics in `results/*/metrics/model_comparison.csv`
4. 🎯 Choose best fusion strategy for your use case
5. 🚀 Run full dataset training
6. 📈 Deploy hybrid model to production

---

**Good luck with your hybrid model! 🎉**
