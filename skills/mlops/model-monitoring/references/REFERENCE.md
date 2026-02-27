# Model Monitoring Reference Guide

## Monitoring Tools Comparison

| Feature              | Evidently           | Whylogs             | NannyML             | Arize               | Fiddler             |
|----------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Open Source          | Yes                 | Yes                 | Yes (core)          | No                  | No                  |
| Data Drift Detection| PSI, KS, Wasserstein| Statistical profiling| Multivariate drift  | PSI, KL, JS, KS    | JSD, PSI, KS       |
| No-label Monitoring | Limited             | No                  | Yes (best-in-class) | Yes                 | Yes                 |
| Real-time Support   | Batch-oriented      | Streaming profiles  | Batch-oriented      | Real-time + batch   | Real-time + batch   |
| Dashboards          | HTML reports, UI    | WhyLabs platform    | Built-in UI         | Full platform       | Full platform       |
| Explainability      | No                  | No                  | No                  | SHAP integration    | Built-in XAI        |
| Cost                | Free                | Free / paid SaaS    | Free / paid         | Paid                | Paid                |

### When to Choose What

- **Evidently**: Best starting point; rich open-source reports, excellent for batch monitoring.
- **Whylogs**: Best for high-volume streaming data profiling with minimal overhead.
- **NannyML**: Unique capability to estimate performance without ground truth labels (CBPE).
- **Arize**: Enterprise-grade real-time monitoring with root cause analysis.
- **Fiddler**: Strong explainability, good for regulated industries needing transparency.

## What to Monitor

### Data Quality

| Metric                   | Description                                 | Alert Threshold Example       |
|--------------------------|---------------------------------------------|-------------------------------|
| Missing value rate       | % of null/NaN values per feature            | > 5% or 2x increase          |
| Schema violations        | Unexpected types, new categories            | Any violation                 |
| Volume anomalies         | Prediction request count changes            | +/- 30% from rolling average  |
| Feature range violations | Values outside training distribution        | > 1% out-of-range             |
| Cardinality changes      | Unique values for categorical features      | Change > 20% from baseline    |

### Model Performance

Track classification metrics (accuracy, precision, recall, F1, AUC-ROC, calibration), regression metrics (MAE, RMSE, MAPE, R-squared), and ranking metrics (NDCG@k, MRR, CTR) over sliding time windows.

### Feature Drift Detection

| Method                         | Type             | Best For                       | Sensitivity  |
|-------------------------------|------------------|--------------------------------|-------------|
| Population Stability Index    | Binned           | Production monitoring standard | Medium      |
| Kolmogorov-Smirnov (KS)      | Non-parametric   | Continuous features            | High        |
| Jensen-Shannon Divergence     | Information      | Probability distributions      | Medium      |
| Wasserstein Distance          | Optimal transport| Continuous, sensitive to shifts | High        |
| Chi-squared Test              | Statistical      | Categorical features           | Medium      |
| Page-Hinkley Test             | Sequential       | Streaming / online detection   | Configurable|

### Prediction Drift

Monitor model output distributions over time, independent of ground truth. Prediction drift often precedes performance degradation.

```python
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset

report = Report(metrics=[TargetDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)
report.save_html("prediction_drift_report.html")
```

## Alerting Strategies and Thresholds

### Tiered Alerting

| Severity | Condition                                    | Response Time | Channel          |
|----------|----------------------------------------------|---------------|------------------|
| Critical | Serving errors > 1%, monitoring failure      | 15 min        | PagerDuty, SMS   |
| High     | Performance drop > 5%, PSI > 0.25            | 1 hour        | Slack, email     |
| Medium   | Feature drift (PSI 0.1-0.25), volume anomaly | 4 hours       | Slack channel    |
| Low      | Minor shift (PSI < 0.1), slight latency rise | Next day      | Dashboard, digest|

### PSI Interpretation

| PSI Value  | Interpretation               | Action                              |
|------------|------------------------------|-------------------------------------|
| < 0.1      | No significant change        | No action needed                    |
| 0.1 - 0.25| Moderate shift               | Investigate, consider retraining    |
| 0.25 - 0.5| Significant shift            | Retrain, root cause analysis        |
| > 0.5     | Severe distribution change   | Immediate investigation, fallback   |

```yaml
# Alert configuration example
alerts:
  drift:
    feature_drift_psi: { warning: 0.1, critical: 0.25 }
    prediction_drift_ks: { warning: 0.05, critical: 0.1 }
  performance:
    accuracy_drop: { warning: 0.02, critical: 0.05 }
    latency_p99_ms: { warning: 200, critical: 500 }
  volume:
    request_count: { low_warning_pct: -30, high_warning_pct: 50 }
```

## Monitoring Dashboard Design

### Key Panels

**Overview**: Model health status (green/yellow/red), prediction volume, overall metric trend.

**Drift**: PSI heatmap per feature over time, top drifted features, prediction distribution overlay.

**Performance**: Primary metric trend with confidence intervals, per-segment breakdown.

**Data Quality**: Missing value rates, schema validation pass/fail, feature distribution histograms.

**Operational**: Request latency (p50/p95/p99), error rates, model version, resource utilization.

```
+---------------------------+---------------------------+
|     MODEL HEALTH          |    PREDICTION VOLUME      |
+---------------------------+---------------------------+
|     FEATURE DRIFT         |    MODEL PERFORMANCE      |
|   [PSI Heatmap]           |   [Metric Trend + CI]     |
+---------------------------+---------------------------+
|     DATA QUALITY          |    OPERATIONAL METRICS    |
|   [Missing Rate Trends]   |   [Latency Distribution]  |
+---------------------------+---------------------------+
```

## Performance Degradation Root Cause Analysis

### Systematic Workflow

1. **Check Data Quality** -- Missing values increased? Schema changes? Volume anomalies?
2. **Check Feature Drift** -- Which features drifted most? Gradual or sudden? External events?
3. **Check Prediction Drift** -- Output distribution shifted? Certain classes affected more?
4. **Check Upstream Systems** -- Pipeline changes? Feature engineering code changes?
5. **Segment Analysis** -- Which user segments affected? Geographic/temporal patterns?
6. **Remediate** -- Short-term: rollback or rule override. Medium: retrain. Long: fix root cause.

### Common Root Causes

| Symptom                    | Likely Cause                          | Quick Fix                      |
|---------------------------|---------------------------------------|--------------------------------|
| All features drifted      | Upstream data pipeline change         | Contact data engineering       |
| Single feature drifted    | Feature engineering bug               | Fix feature, retrain           |
| Sudden performance drop   | Data pipeline failure, schema change  | Rollback, fix pipeline         |
| Gradual performance decay | Concept drift, changing behavior      | Scheduled retraining           |
| Performance drop on segment| New user segment                     | Add segment to training data   |

## Ground Truth Delay Handling

| Domain              | Typical Delay        | Strategy                          |
|---------------------|---------------------|-----------------------------------|
| Fraud detection     | 30-90 days          | Proxy labels, NannyML CBPE        |
| Credit risk         | 6-24 months         | Early indicators, cohort analysis  |
| Recommendations     | Minutes to hours    | Implicit feedback (clicks, views)  |
| Churn prediction    | 30-90 days          | Early engagement signals           |

### Strategies

**CBPE (Confidence-Based Performance Estimation):**
```python
import nannyml as nml
estimator = nml.CBPE(y_pred_proba="pred_proba", y_pred="prediction",
    y_true="target", metrics=["roc_auc"], problem_type="classification_binary")
estimator.fit(reference_df)
results = estimator.estimate(analysis_df)  # no labels needed
```

**Proxy Labels**: Use approximate labels available sooner (e.g., "reported fraud" in 7 days vs "confirmed fraud" in 90 days). Track proxy-to-true-label correlation.

**Cohort Backtesting**: When labels arrive, compare actual performance to drift metrics from that period to calibrate thresholds.

## Monitoring Infrastructure Patterns

### Batch Monitoring

```
Prediction Logs --> Data Warehouse --> Scheduled Job (Airflow) --> Reports/Alerts
```

Lower cost, simpler. Acceptable when hours of detection delay is tolerable. Tools: Evidently + Airflow.

### Streaming Monitoring

```
Prediction Events (Kafka) --> Stream Processor (Flink) --> Real-time Metrics --> Alerts
```

Sub-minute detection. Higher cost and complexity. Required for high-stakes systems. Tools: Whylogs, Arize.

### Hybrid (Recommended)

Stream critical operational metrics (latency, errors, volume). Batch deeper statistical analysis (drift, performance, fairness). Balance cost and speed based on business requirements.

## Common Pitfalls

1. **Monitoring only accuracy**: Track data quality, drift, and operational metrics alongside performance.
2. **Static thresholds**: Adapt alert thresholds to seasonal patterns and business cycles.
3. **Alert fatigue**: Start with fewer, high-confidence alerts. Tune thresholds based on false alert rates.
4. **Ignoring reference data staleness**: Update reference dataset periodically as the world changes.
5. **Not monitoring the monitor**: Ensure monitoring pipelines themselves have health checks.
6. **One-size-fits-all drift detection**: Different features need different statistical tests.

## Further Reading

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Whylogs Documentation](https://whylogs.readthedocs.io/)
- [NannyML Documentation](https://nannyml.readthedocs.io/)
- [Arize AI Documentation](https://docs.arize.com/)
- [Failing Loudly: Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953)
- [Monitoring ML Models in Production (Made With ML)](https://madewithml.com/courses/mlops/monitoring/)
- [Hidden Technical Debt in ML Systems (Sculley et al.)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
