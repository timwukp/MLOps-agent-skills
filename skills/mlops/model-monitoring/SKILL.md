---
name: model-monitoring
description: >
  Monitor ML model performance in production. Covers Evidently AI reports and tests, Whylogs profiling, NannyML
  performance estimation, prediction distribution monitoring, ground truth collection, feature importance tracking,
  latency and throughput monitoring, error rate tracking, custom business metrics, alerting rules and thresholds
  (static, dynamic, adaptive), Grafana dashboards, Prometheus metrics, prediction logging, batch vs real-time
  monitoring, SLA compliance, and monitoring infrastructure. Use when setting up production model monitoring,
  creating alerts, building dashboards, or debugging model performance degradation.
license: Apache-2.0
metadata:
  author: mlops-skills
  version: "1.0"
  category: mlops
---

# Model Monitoring

## Overview

Model monitoring tracks the health and performance of ML models in production,
detecting degradation before it impacts business outcomes.

## When to Use This Skill

- Setting up monitoring for newly deployed models
- Investigating model performance drops
- Creating alerting rules and dashboards
- Implementing ground truth feedback loops
- Meeting SLA requirements for model quality

## Step-by-Step Instructions

### 1. Evidently AI Monitoring

```python
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset, DataQualityPreset, TargetDriftPreset,
    ClassificationPreset, RegressionPreset
)
from evidently.test_suite import TestSuite
from evidently.tests import *

# Data Drift Report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_df, current_data=production_df)
drift_report.save_html("drift_report.html")
drift_report_dict = drift_report.as_dict()

# Classification Performance Report
perf_report = Report(metrics=[ClassificationPreset()])
perf_report.run(reference_data=reference_df, current_data=production_df)

# Test Suite (pass/fail checks)
test_suite = TestSuite(tests=[
    TestNumberOfDriftedColumns(lt=3),
    TestShareOfDriftedColumns(lt=0.3),
    TestColumnDrift("prediction"),
    TestAccuracyScore(gt=0.85),
    TestF1Score(gt=0.80),
])
test_suite.run(reference_data=reference_df, current_data=production_df)
results = test_suite.as_dict()
```

### 2. Whylogs Profiling

```python
import whylogs as why
from whylogs.core.constraints import ConstraintsBuilder
from whylogs.core.constraints.factories import (
    greater_than_number, no_missing_values, is_in_range
)

# Profile production data
result = why.log(production_df)
profile = result.profile()

# Save profile for comparison
result.writer("local").write(dest="profiles/")

# Define constraints
builder = ConstraintsBuilder(dataset_profile_view=profile.view())
builder.add_constraint(no_missing_values(column_name="user_id"))
builder.add_constraint(greater_than_number(column_name="score", number=0.0))
builder.add_constraint(is_in_range(column_name="age", lower=0, upper=150))

constraints = builder.build()
report = constraints.generate_constraints_report()
print(f"Passed: {report.passed}, Failed: {report.failed}")
```

### 3. Performance Monitoring

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, mean_absolute_error
)
import numpy as np

class ModelPerformanceMonitor:
    def __init__(self, model_name, metric_thresholds):
        self.model_name = model_name
        self.thresholds = metric_thresholds
        self.history = []

    def evaluate(self, y_true, y_pred, y_prob=None):
        """Compute and check performance metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted"),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
        }
        if y_prob is not None:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")

        # Check thresholds
        violations = []
        for metric, value in metrics.items():
            if metric in self.thresholds and value < self.thresholds[metric]:
                violations.append({
                    "metric": metric,
                    "value": value,
                    "threshold": self.thresholds[metric],
                    "severity": "critical" if value < self.thresholds[metric] * 0.9 else "warning",
                })

        self.history.append({"timestamp": datetime.utcnow(), **metrics})
        return metrics, violations
```

### 4. Prometheus Metrics Integration

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Define metrics
PREDICTION_TOTAL = Counter(
    "model_predictions_total",
    "Total predictions made",
    ["model_name", "model_version", "status"]
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency",
    ["model_name"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of prediction values",
    ["model_name"]
)
MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy",
    ["model_name", "model_version"]
)

def track_prediction(model_name, version, prediction, latency, success=True):
    status = "success" if success else "error"
    PREDICTION_TOTAL.labels(model_name, version, status).inc()
    PREDICTION_LATENCY.labels(model_name).observe(latency)
    PREDICTION_VALUE.labels(model_name).observe(prediction)
```

### 5. Alerting Configuration

```yaml
# alerts/model_alerts.yaml
alerts:
  - name: model_accuracy_degraded
    metric: model_accuracy
    condition: "< 0.85"
    severity: warning
    window: 1h
    channels: [slack, email]
    message: "Model accuracy dropped below 85%"

  - name: model_accuracy_critical
    metric: model_accuracy
    condition: "< 0.75"
    severity: critical
    window: 30m
    channels: [slack, pagerduty]
    message: "CRITICAL: Model accuracy below 75%"

  - name: high_latency
    metric: model_prediction_latency_p99
    condition: "> 0.5"
    severity: warning
    window: 15m
    channels: [slack]
    message: "P99 latency exceeds 500ms"

  - name: prediction_volume_drop
    metric: model_predictions_total_rate
    condition: "< 100"
    severity: warning
    window: 30m
    channels: [slack]
    message: "Prediction volume dropped significantly"
```

### 6. Ground Truth Collection

```python
class GroundTruthCollector:
    """Collect ground truth labels for production predictions."""

    def __init__(self, storage_backend):
        self.storage = storage_backend

    def log_prediction(self, prediction_id, features, prediction, timestamp):
        """Log prediction for later ground truth matching."""
        self.storage.save({
            "prediction_id": prediction_id,
            "features": features,
            "prediction": prediction,
            "timestamp": timestamp,
            "ground_truth": None,
            "feedback_received": False,
        })

    def add_ground_truth(self, prediction_id, ground_truth, feedback_timestamp):
        """Match ground truth with earlier prediction."""
        record = self.storage.get(prediction_id)
        record["ground_truth"] = ground_truth
        record["feedback_received"] = True
        record["feedback_delay_hours"] = (
            feedback_timestamp - record["timestamp"]
        ).total_seconds() / 3600
        self.storage.update(record)

    def compute_metrics(self, start_date, end_date):
        """Compute metrics for predictions with ground truth."""
        records = self.storage.query(
            start_date=start_date, end_date=end_date, feedback_received=True
        )
        y_true = [r["ground_truth"] for r in records]
        y_pred = [r["prediction"] for r in records]
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "sample_size": len(records),
            "avg_feedback_delay_hours": np.mean([r["feedback_delay_hours"] for r in records]),
        }
```

## Metrics Catalog

| Model Type | Key Metrics |
|------------|-------------|
| Classification | Accuracy, F1, Precision, Recall, AUC-ROC, Log Loss |
| Regression | RMSE, MAE, MAPE, R², Residual distribution |
| Ranking | NDCG, MAP, MRR, Hit Rate |
| Recommendation | Precision@K, Recall@K, Diversity, Coverage |

## Best Practices

1. **Monitor inputs AND outputs** - Drift in inputs predicts future output degradation
2. **Use multiple metrics** - No single metric captures all failure modes
3. **Set dynamic thresholds** - Adaptive thresholds reduce false alerts
4. **Monitor by segment** - Overall metrics can hide subgroup degradation
5. **Track prediction volume** - Sudden drops indicate upstream issues
6. **Collect ground truth** - Estimated metrics are no substitute for actuals
7. **Alert on trends** - Gradual degradation matters, not just threshold violations
8. **Dashboard hierarchy** - Overview → model detail → segment detail
9. **Log predictions** for debugging and retraining
10. **Review alerts weekly** - Tune thresholds to reduce noise

## Scripts

- `scripts/monitor_model.py` - Evidently + Whylogs monitoring pipeline
- `scripts/setup_alerts.py` - Alert configuration and Prometheus rules

## References

See [references/REFERENCE.md](references/REFERENCE.md) for tool comparisons and Grafana templates.
