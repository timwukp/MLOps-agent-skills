---
name: model-drift-detection
description: >
  Detect and handle data drift, concept drift, and model degradation. Covers statistical drift detection methods
  (PSI, KS test, chi-squared, Wasserstein distance, KL divergence, Jensen-Shannon divergence), sequential methods
  (Page-Hinkley, ADWIN, DDM, CUSUM), multivariate drift (MMD, domain classifier), Evidently drift reports,
  Alibi Detect, NannyML, feature-level vs dataset-level drift, reference window strategies, drift for different
  data types (numerical, categorical, text, image), automated retraining triggers, drift severity assessment,
  false positive management, and drift root cause analysis. Use when detecting distribution shifts, setting up
  drift monitoring, or configuring retraining triggers.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Model Drift Detection

## Overview

Drift occurs when the statistical properties of data change over time, causing model
performance to degrade. Detecting drift early enables proactive retraining before
business impact.

## Types of Drift

```
┌─────────────────────────────────────────────────────┐
│                    Types of Drift                    │
├──────────────┬──────────────┬───────────────────────┤
│  Data Drift  │ Concept Drift│   Label Drift         │
│  P(X) changes│ P(Y|X) changes│  P(Y) changes        │
│              │              │                       │
│  Features    │  Relationship│  Target distribution  │
│  shift       │  between X,Y │  shifts               │
│              │  changes     │                       │
│  Detectable  │  Hard to     │  Detectable           │
│  without     │  detect w/o  │  with ground          │
│  labels      │  ground truth│  truth                │
└──────────────┴──────────────┴───────────────────────┘
```

## When to Use This Skill

- Setting up continuous drift monitoring for production models
- Investigating sudden model performance drops
- Configuring automated retraining pipelines
- Understanding why a model is degrading
- Choosing the right statistical tests for drift detection

## Step-by-Step Instructions

### 1. Population Stability Index (PSI)

```python
import numpy as np

def calculate_psi(reference, current, n_bins=10):
    """Calculate Population Stability Index.

    PSI < 0.1: No significant shift
    PSI 0.1-0.2: Moderate shift (investigate)
    PSI > 0.2: Significant shift (retrain)
    """
    # Create bins from reference distribution
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Avoid zero counts
    ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
    cur_pct = (cur_counts + 1) / (len(current) + n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi
```

### 2. Kolmogorov-Smirnov Test

```python
from scipy import stats

def ks_drift_test(reference, current, significance=0.05):
    """KS test for numerical feature drift."""
    statistic, p_value = stats.ks_2samp(reference, current)
    return {
        "test": "ks",
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < significance,
        "significance": significance,
    }
```

### 3. Chi-Squared Test (Categorical Features)

```python
def chi2_drift_test(reference, current, significance=0.05):
    """Chi-squared test for categorical feature drift."""
    # Get all categories
    categories = set(reference.unique()) | set(current.unique())

    ref_counts = reference.value_counts().reindex(categories, fill_value=0)
    cur_counts = current.value_counts().reindex(categories, fill_value=0)

    # Normalize
    ref_freq = ref_counts / ref_counts.sum()
    cur_freq = cur_counts / cur_counts.sum()

    # Expected counts based on reference distribution
    expected = ref_freq * cur_counts.sum()
    expected = expected.clip(lower=1)  # Avoid zero expected

    statistic, p_value = stats.chisquare(cur_counts, f_exp=expected)
    return {
        "test": "chi2",
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < significance,
    }
```

### 4. Evidently Drift Detection

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    ColumnDriftMetric, DatasetDriftMetric, DataDriftTable
)

# Quick drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=cur_df)

# Custom drift configuration
from evidently.metrics import DataDriftTable
report = Report(metrics=[
    DataDriftTable(
        num_stattest="ks",           # KS test for numerical
        cat_stattest="chisquare",    # Chi-squared for categorical
        num_stattest_threshold=0.05,
        cat_stattest_threshold=0.05,
    ),
])
report.run(reference_data=ref_df, current_data=cur_df)
result = report.as_dict()

# Extract per-feature drift results
drift_results = result["metrics"][0]["result"]
drifted_columns = [
    col for col, info in drift_results["drift_by_columns"].items()
    if info["drift_detected"]
]
```

### 5. Multivariate Drift (Domain Classifier)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def domain_classifier_drift(reference, current, threshold=0.55):
    """Detect multivariate drift using a domain classifier.

    Train a classifier to distinguish reference from current data.
    If AUC > threshold, distributions are distinguishable = drift.
    """
    ref_labeled = reference.copy()
    ref_labeled["_domain"] = 0

    cur_labeled = current.copy()
    cur_labeled["_domain"] = 1

    combined = pd.concat([ref_labeled, cur_labeled], ignore_index=True)
    X = combined.drop(columns=["_domain"])
    y = combined["_domain"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    mean_auc = scores.mean()

    return {
        "test": "domain_classifier",
        "auc": mean_auc,
        "drift_detected": mean_auc > threshold,
        "threshold": threshold,
    }
```

### 6. Comprehensive Drift Detection Pipeline

```python
class DriftDetector:
    def __init__(self, reference_data, config):
        self.reference = reference_data
        self.config = config
        self.history = []

    def detect(self, current_data):
        """Run all configured drift tests."""
        results = {"timestamp": datetime.utcnow(), "features": {}}

        for col in self.config["features"]:
            feature_type = self.config["features"][col]["type"]

            if feature_type == "numerical":
                psi = calculate_psi(self.reference[col], current_data[col])
                ks = ks_drift_test(self.reference[col], current_data[col])
                results["features"][col] = {
                    "psi": psi,
                    "ks_statistic": ks["statistic"],
                    "ks_p_value": ks["p_value"],
                    "drift_detected": psi > 0.2 or ks["drift_detected"],
                }
            elif feature_type == "categorical":
                chi2 = chi2_drift_test(self.reference[col], current_data[col])
                results["features"][col] = {
                    "chi2_statistic": chi2["statistic"],
                    "chi2_p_value": chi2["p_value"],
                    "drift_detected": chi2["drift_detected"],
                }

        # Overall drift score
        drifted = sum(1 for f in results["features"].values() if f["drift_detected"])
        total = len(results["features"])
        results["drift_score"] = drifted / total if total > 0 else 0
        results["drift_detected"] = results["drift_score"] > self.config.get("threshold", 0.3)

        self.history.append(results)
        return results

    def recommend_action(self, results):
        """Recommend action based on drift severity."""
        score = results["drift_score"]
        if score > 0.5:
            return "RETRAIN: Significant drift detected in >50% of features"
        elif score > 0.3:
            return "INVESTIGATE: Moderate drift in 30-50% of features"
        elif score > 0.1:
            return "MONITOR: Minor drift detected, increase monitoring frequency"
        return "OK: No significant drift"
```

### 7. Automated Retraining Trigger

```python
def should_retrain(drift_results, performance_metrics=None, config=None):
    """Decide whether to trigger retraining based on drift and performance."""
    config = config or {
        "drift_threshold": 0.3,
        "performance_drop_threshold": 0.05,
        "min_days_since_last_train": 7,
    }

    reasons = []

    # Drift-based trigger
    if drift_results["drift_score"] > config["drift_threshold"]:
        reasons.append(f"Drift score {drift_results['drift_score']:.2f} > {config['drift_threshold']}")

    # Performance-based trigger
    if performance_metrics:
        baseline = performance_metrics.get("baseline_accuracy", 0.9)
        current = performance_metrics.get("current_accuracy", 0.9)
        if baseline - current > config["performance_drop_threshold"]:
            reasons.append(f"Accuracy dropped by {baseline - current:.4f}")

    return {
        "retrain": len(reasons) > 0,
        "reasons": reasons,
    }
```

## Statistical Test Selection Guide

| Test | Data Type | Best For | Sensitivity |
|------|-----------|----------|-------------|
| PSI | Numerical | Production monitoring | Medium |
| KS Test | Numerical | General drift | High |
| Chi-Squared | Categorical | Category frequency shifts | Medium |
| Wasserstein | Numerical | Magnitude of shift | High |
| KL Divergence | Both | Information loss | High |
| JS Divergence | Both | Symmetric comparison | Medium |
| MMD | Multivariate | High-dimensional drift | High |
| Domain Classifier | Multivariate | Complex drift patterns | Very High |

## Best Practices

1. **Test each feature independently** AND test multivariate drift
2. **Use multiple tests** - No single test catches all types of drift
3. **Set reference windows carefully** - Use a stable, representative period
4. **Adjust for multiple testing** - Bonferroni correction reduces false positives
5. **Track drift severity** over time, not just binary detection
6. **Distinguish seasonal patterns** from true drift
7. **Automate but review** - Human review before automated retraining
8. **Monitor the monitor** - Track false positive/negative rates of drift detection
9. **Different thresholds per feature** - Critical features need tighter monitoring

## Scripts

- `scripts/detect_drift.py` - Multi-test drift detection pipeline
- `scripts/drift_monitor.py` - Continuous drift monitoring service

## References

See [references/REFERENCE.md](references/REFERENCE.md) for test comparisons and threshold guides.
